import os
import math
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import BinaryEncoder

import src.data_handling as data_handling
import src.batch as batch
from src import TICKER
from src._utils import main_logger


class BaselineModel:
    def __init__(
            self,
            input_size: int = 10,
            hidden_size: int = 64,
            ticker: str = TICKER,

            # batch learning metrics
            batch_size: int = 32,
            min_delta: float = 1e-3,
            patience: int = 10,
            lr: float = 1e-4,
        ):
        self.ticker = ticker

        # model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model = nn.Sequential(nn.Linear(self.input_size, self.hidden_size),nn.ReLU(), nn.Linear(self.hidden_size, 1))

        # spark session
        self.spark = data_handling.config_and_start_spark_session()

        # preprocess
        self.num_cols = ['open', 'high', 'low', 'volume', 'ave_open', 'ave_high', 'ave_low', 'ave_close', 'total_volume', '30_day_ma_close', 'timestamp_in_ms']
        self.cat_cols = ['dt', 'year', 'month', 'date']
        self._setup_preprocessor()

        # training
        self.device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(self.device_type)

        self.batch_size = batch_size
        self.min_delta = min_delta
        self.patience = patience

        self.criterion = nn.MSELoss() # mse for logged closing price (target val)
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.steps_trained = 0

        # filepath
        self.model_filepath = os.path.join('artifacts', 'base_model', self.ticker, 'model.pt')
        self.preprocessor_filepath = os.path.join('artifacts', 'preprocessor', self.ticker, 'preprocessor.pkl')


    def _create_and_define_filepaths(self):
        os.makedirs(os.path.dirname(self.model_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(self.preprocessor_filepath), exist_ok=True)


    def _setup_preprocessor(self):
        num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        cat_transformer = Pipeline(steps=[('encoder', BinaryEncoder(cols=self.cat_cols))])
        preprocessor = ColumnTransformer(
            transformers=[('num', num_transformer, self.num_cols), ('cat', cat_transformer, self.cat_cols)],
            remainder='passthrough'
        )
        self.preprocessor = preprocessor


    def _save_trained_preprocessor(self):
        joblib.dump(self.preprocessor, self.preprocessor_filepath)
        main_logger.info(f'... trained preprocessor saved at {self.preprocessor_filepath} ...')


    def _load_trained_preprocessor(self):
        return joblib.load(self.preprocessor_filepath)


    def _create_tensor_data_loader(self, X, y) -> DataLoader:
        X_np = X.values if isinstance(X, pd.Series) else X.to_numpy() if isinstance(X, pd.DataFrame) else X
        X_final_tensor = torch.tensor(X_np, dtype=torch.float32)

        y_np = y.values if isinstance(y, pd.Series) else y.to_numpy() if isinstance(y, pd.DataFrame)  else y
        y_final_tensor = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

        dataset_final = TensorDataset(X_final_tensor, y_final_tensor)
        data_loader = DataLoader(dataset_final, batch_size=self.batch_size)

        return data_loader


    def _save_model(self, checkpoint: dict):
        torch.save(checkpoint, self.model_filepath)
        main_logger.info(f"... base model trained and saved at {self.model_filepath} ...")


    def _load_model_and_optimizer(self):
        checkpoint = torch.load(self.model_filepath, weights_only=False, map_location=self.device)
        state_dict = checkpoint['state_dict']
        input_dim = checkpoint['input_dim']
        model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1) # output size = 1
        )
        model.load_state_dict(state_dict, strict=True)
        self.model = model
        self.model.to(self.device)
        self.optimizer.load_state_dict(state_dict=checkpoint['optimizer_state_dict'])
        main_logger.info(f'... base model loaded and reconstructed ...')


    def initial_batch_learning(self, num_epochs: int = 5000):
        df, _ = batch.run_lakehouse(ticker=self.ticker)

        df = df.toPandas() if not isinstance(df, pd.DataFrame) else df

        y = df['close']
        X = df.copy().drop(columns='close', axis=1)

        X_tv, X_test, y_tv, _ = train_test_split(X, y, test_size=2000, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=2000, shuffle=False)

        X_train = self.preprocessor.fit_transform(X_train)
        X_val = self.preprocessor.transform(X_val)
        X_test = self.preprocessor.transform(X_test)

        # save trained preprocessor
        self._save_trained_preprocessor()

        self.input_size = X_train.shape[-1]
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1) # output size = 1
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        train_data_loader = self._create_tensor_data_loader(X=X_train, y=y_train)
        val_data_loader = self._create_tensor_data_loader(X=X_val, y=y_val)

        # start training - with validation and early stopping
        best_val_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(num_epochs):
            main_logger.info(f'... start epoch {epoch + 1} ...')
            self.model.train()
            for batch_X, batch_y in train_data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()

                try:
                    # pytorch's AMP system automatically handles the casting of tensors to Float16 or keeps them in Float32
                    with torch.autocast(device_type=self.device_type):
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
                            main_logger.error('pytorch model returns nan or inf. break the training loop.')
                            break

                        if not math.isfinite(loss.item()):
                            main_logger.error('loss is nan or inf. break the training loop.')
                            break

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                except:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()

            if (epoch + 1) % 10 == 0: main_logger.info(f"epoch [{epoch+1}/{num_epochs}], loss: {loss.item():.4f}")

            # validate on a validation dataset (subset of the entire training dataset)
            self.model.eval()
            val_loss = 0.0

            # switch the grad mode
            with torch.inference_mode():
                for batch_X_val, batch_y_val in val_data_loader:
                    batch_X_val, batch_y_val = batch_X_val.to(self.device), batch_y_val.to(self.device)
                    outputs_val = self.model(batch_X_val)
                    val_loss += self.criterion(outputs_val, batch_y_val).item()

            val_loss /= len(val_data_loader)

            # early stopping
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    main_logger.info(f'early stopping at epoch {epoch + 1} as validation loss did not improve for {self.patience} epochs.')
                    break

        checkpoint = dict(
            state_dict=self.model.state_dict(),
            input_dim=X_train.shape[1],
            optimizer_name='adam',
            optimizer_state_dict=self.optimizer.state_dict(),
            batch_size=self.batch_size,
            lr=self.lr,
        )
        self._save_model(checkpoint=checkpoint)


    def online_learning(self, current_batch: List[Dict[str, Any]]):
        if not current_batch: return

        # data transformation (imputation, structuring, feature engineering)
        df = self.spark.createDataFrame(current_batch, schema=data_handling.silver.SILVER_SCHEMA)
        df = data_handling.gold.transform(delta_table=df, spark=self.spark)

        df_pandas = df.toPandas()
        y = df_pandas['close']
        X = df_pandas.copy().drop(columns='close', axis=1)

        try:
            self.preprocessor = self._load_trained_preprocessor()
            X = self.preprocessor.transform(X)
        except:
            X = self.preprocessor.fit_transform(X)

        # convert to pytorch tensors
        X = torch.from_numpy(X).float()
        X = X.to(self.device)
        y = torch.from_numpy(y.to_numpy()).float().reshape(-1, 1)
        y = y.to(self.device)

        # load the model and optimizer
        self._load_model_and_optimizer()

        # start online learning
        self.optimizer.zero_grad()
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()
        self.steps_trained += 1

        # pred result
        epsilon = 0
        try: y_pred_actual = np.exp(y_pred + epsilon)
        except: y_pred_actual = np.exp(y_pred.cpu().detach().numpy() + epsilon)

        main_logger.info(f"... pytorch model trained (step {self.steps_trained}) ...\n - batch size {len(current_batch)}\n - mse loss {loss.item():.6f}\n - new closing price predicted: {np.mean(y_pred_actual):.2f}")
