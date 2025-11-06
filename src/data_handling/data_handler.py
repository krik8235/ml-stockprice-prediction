# DataHandler class to manage data from extraction to loader

import os
import json
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import BinaryEncoder
from deltalake import DeltaTable

import src.data_handling as data_handling
from src import TICKER
from src._utils import main_logger


class DataHandler:
    def __init__(
            self,
            num_cols: list = list(),
            cat_cols: list = list(),
            target_col: str = 'close',
            ticker: str = TICKER,
            # model_name: str = 'gru',# model_name options: mlp, mlp_ensemble, lstm, gru, gru_swa
        ):
        self.ticker = ticker

        # spark session
        self.spark = data_handling.config_and_start_spark_session()

        # preprocess
        self.num_cols = num_cols if num_cols else ['open', 'high', 'low', 'volume', 'ave_open', 'ave_high', 'ave_low', 'ave_close', 'total_volume', '30_day_ma_close', 'timestamp_in_ms']
        self.cat_cols = cat_cols if cat_cols else ['dt', 'year', 'month', 'date']
        self.target_col = target_col
        self._setup_preprocessor()

        # device
        self.device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(self.device_type)

        # filepath
        self.data_gold_df_filepath = os.path.join('data', self.ticker, 'gold_df.parquet')
        self.data_x_train_filepath = os.path.join('data', self.ticker, 'x_train_df.parquet')
        self.data_x_val_filepath = os.path.join('data', self.ticker, 'x_val_df.parquet')
        self.data_x_test_filepath = os.path.join('data', self.ticker, 'x_test_df.parquet')
        self.data_y_train_filepath = os.path.join('data', self.ticker, 'y_train_df.parquet')
        self.data_y_val_filepath = os.path.join('data', self.ticker, 'y_val_df.parquet')
        self.data_y_test_filepath = os.path.join('data', self.ticker, 'y_test_df.parquet')
        self.data_x_train_processed_filepath = os.path.join('data', self.ticker, 'x_train_processed.parquet')
        self.data_x_val_processed_filepath = os.path.join('data', self.ticker, 'x_val_processed.parquet')
        self.data_x_test_processed_filepath = os.path.join('data', self.ticker, 'x_test_processed.parquet')
        # self.model_filepath = os.path.join('artifacts', 'models', self.ticker, f'{model_name}.pt')
        self.preprocessor_filepath = os.path.join('artifacts', self.ticker, 'preprocessor.pkl')
        self.feature_name_filepath = os.path.join('artifacts', self.ticker, 'feature_names.json')
        self._create_and_define_filepaths()

        self.s3_bucket_name = 'ml-stockprice-pred'
        self.s3_store_address = f's3a://{self.s3_bucket_name}/'
        self.gold_s3_path =  f's3a://{self.s3_bucket_name}/data/{self.ticker}/gold'


    def _create_and_define_filepaths(self):
        os.makedirs(os.path.dirname(self.data_x_train_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(self.preprocessor_filepath), exist_ok=True)
        # os.makedirs(os.path.dirname(self.model_filepath), exist_ok=True)


    def _setup_preprocessor(self):
        num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        cat_transformer = Pipeline(steps=[('encoder', BinaryEncoder(cols=self.cat_cols))])
        preprocessor = ColumnTransformer(
            transformers=[('num', num_transformer, self.num_cols), ('cat', cat_transformer, self.cat_cols)],
            remainder='passthrough'
        )
        self.preprocessor = preprocessor


    def save_trained_preprocessor(self):
        joblib.dump(self.preprocessor, self.preprocessor_filepath)
        with open(self.feature_name_filepath, 'w') as f:
            feature_names = self.preprocessor.get_feature_names_out()
            json.dump(feature_names.tolist(), f)
        main_logger.info(f'... trained preprocessor and feature names saved at {os.path.dirname(self.preprocessor_filepath)} ...')


    def load_trained_preprocessor(self):
        return joblib.load(self.preprocessor_filepath)


    def _get_full_date_index(self) -> np.ndarray:
        """util func for x axis of the line graph"""
        df = self.run_lakehouse(should_refresh=False)
        return df['dt'].to_numpy()


    def run_lakehouse(self, current_batch = None, should_refresh: bool = False) -> pd.DataFrame:
        from src.data_handling.run_lakehouse import run_lakehouse

        if current_batch:
            df = self.spark.createDataFrame(current_batch, schema=data_handling.silver.SILVER_SCHEMA)
            df = data_handling.gold.transform(delta_table=df, spark=self.spark)
            return df.toPandas() if not isinstance(df, pd.DataFrame) else df

        else:
            if should_refresh:
                df, _, _ = run_lakehouse(ticker=self.ticker)
                return df

            else:
                gold_table = DeltaTable(self.gold_s3_path)
                df = gold_table.to_pandas()

                if df is None:
                    df, _, _ = run_lakehouse(ticker=self.ticker)
                return df


    def preprocess(self, test_size: int = 2000, target_col: str = 'close', should_refresh: bool = False) -> tuple:
        df = self.run_lakehouse(should_refresh=should_refresh)
        y = df[target_col]
        X = df.copy().drop(columns=target_col, axis=1)

        X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=test_size, shuffle=False)

        X_train.to_parquet(self.data_x_train_filepath, index=False)
        X_val.to_parquet(self.data_x_val_filepath, index=False)
        X_test.to_parquet(self.data_x_test_filepath, index=False)
        y_train.to_frame(name=target_col).to_parquet(self.data_y_train_filepath, index=False)
        y_val.to_frame(name=target_col).to_parquet(self.data_y_val_filepath, index=False)
        y_test.to_frame(name=target_col).to_parquet(self.data_y_test_filepath, index=False)

        # preprocess
        X_train = self.preprocessor.fit_transform(X_train)
        X_val = self.preprocessor.transform(X_val)
        X_test = self.preprocessor.transform(X_test)

        pd.DataFrame(X_train).to_parquet(self.data_x_train_processed_filepath, index=False) # type: ignore
        pd.DataFrame(X_val).to_parquet(self.data_x_val_processed_filepath, index=False) # type: ignore
        pd.DataFrame(X_test).to_parquet(self.data_x_test_processed_filepath, index=False) # type: ignore

        # save trained preprocessor
        self.save_trained_preprocessor()

        return X_train, X_val, X_test, y_train, y_val, y_test


    def create_tensor_data_loader(self, X, y, batch_size: int = 32) -> DataLoader:
        X_np = X.values if isinstance(X, pd.Series) else X.to_numpy() if isinstance(X, pd.DataFrame) else X
        X_final_tensor = torch.tensor(X_np, dtype=torch.float32)

        y_np = y.values if isinstance(y, pd.Series) else y.to_numpy() if isinstance(y, pd.DataFrame)  else y
        y_final_tensor = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

        dataset_final = TensorDataset(X_final_tensor, y_final_tensor)
        data_loader = DataLoader(dataset_final, batch_size=batch_size)

        return data_loader
