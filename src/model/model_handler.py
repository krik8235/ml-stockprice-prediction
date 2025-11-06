import numpy as np
import torch
import torch.nn as nn
import torch.optim.swa_utils as swa_utils
from typing import List, Dict, Any

from src import TICKER
from src._utils import main_logger
from src.data_handling.data_handler import DataHandler


class ModelHandler:
    def __init__(
            self,
            ticker: str = TICKER,
            should_refresh: bool = False,

            # model
            model_name: str = 'mlp', # options: mlp, mlp_ensemble, lstm, gru, gru_swa, transformer
            model = None,
            input_size: int = 10,
            **kwargs
        ):
        self.ticker = ticker
        self.should_refresh: bool = should_refresh
        self.data_handler = DataHandler(ticker=self.ticker)
        self.datasets = list()
        self._fetch_datasets()

        # model
        self.base_model_names = ['mlp', 'lstm', 'gru']
        self.model_name = model_name
        self.input_size = input_size
        self.model = model if model else self._instatiate_model(**kwargs)
        self.candidate_models = list() # list({model_name, model, optimizer}
        self.production_model = None # selected best model out of candidate models

        # device
        self.device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(self.device_type)

        # optimizer
        self.optimizer = None
        self.production_optimizer = None # selected best optimizer for the production model
        self.steps_trained = 0 # count online training
        self.criterion = nn.MSELoss()

        # model store filepath
        self.s3_bucket_name = 'ml-stockprice-pred'
        self.model_filepath = f's3a://{self.s3_bucket_name}/artifacts/{self.ticker}/models/{self.model_name}'
        self.parent_dir = f's3a://{self.s3_bucket_name}/artifacts/{self.ticker}/models'
        self.fs = None
        self._create_model_store_dir()


    def _fetch_datasets(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_handler.preprocess(should_refresh=self.should_refresh)
        self.datasets = [X_train, X_val, X_test, y_train, y_val, y_test]


    def _instatiate_model(self, **kwargs):
        from src.model.instantiate import instatiate
        return instatiate(model_name=self.model_name, input_size=self.input_size, **kwargs)


    def _create_model_store_dir(self):
        import fsspec
        self.fs = fsspec.filesystem('s3')
        self.fs.mkdir(self.parent_dir, create_parents=True)


    def _save_checkpoint(self, checkpoint: dict):
        with self.fs.open(self.model_filepath, 'wb') as f: # type: ignore
            torch.save(checkpoint, f)


    def _load_checkpoint(self) -> dict:
        with self.fs.open(self.model_filepath, 'rb') as f: # type: ignore
            checkpoint = torch.load(f, weights_only=False, map_location=self.device)
            return checkpoint


    def tune(self, R: int, halving_factor: int, train_window: int = 1000, val_window: int = 500, buffer: int = 100) -> dict:
        from src.model.tuning import run_hyperband, walk_forward_validation, search_space

        X_train, y_train = self.datasets[0], self.datasets[3]

        main_logger.info(f'... start tuning {self.model_name} ...')
        best_config, best_mse, best_mae, best_model, optimizer = run_hyperband(
            search_space_fn=search_space,
            val_fn=lambda c, b: walk_forward_validation(
                X=X_train, y=y_train,
                config=c,
                budget=b,
                train_window=train_window,
                val_window=val_window,
                buffer=buffer,
                model_name=self.model_name,
            ),
            R=R,
            halving_factor=halving_factor,
            model_name=self.model_name,
        )
        checkpoint = dict(
            state_dict=best_model.state_dict(),
            input_dim=X_train.shape[1],
            best_config=best_config,
            optimizer_name=best_config['optimizer_name'] if best_config else None,
            optimizer_state_dict=optimizer.state_dict() if optimizer else None,
            best_mse=best_mse,
            best_mae=best_mae,
        )
        self.model = best_model
        self.optimizer = optimizer
        return checkpoint


    def define_candidate_models(self):
        from src.model.instantiate import AdaptiveEnsemble

        candidate_models = list()

        for model_name in self.base_model_names:
            model, optimizer = self._reconsruct_model_and_optimizer()
            candidate_models.append(dict(model_name=model_name, model=model, optimizer=optimizer))

            if model_name == 'mlp':
                ada = AdaptiveEnsemble(num_experts=5, model=model).to(self.device)
                candidate_models.append(dict(model_name='adaptive_ensemble', model=ada, optimizer=optimizer))

            if model_name == 'gru':
                gru_swa = swa_utils.AveragedModel(model)
                candidate_models.append(dict(model_name='gru_swa', model=gru_swa, optimizer=optimizer))
        self.candidate_models = candidate_models


    def _reconsruct_model_and_optimizer(self) -> tuple:
        from src.model.reconstruct import reconstruct_model_and_optimizer
        model, optimizer = reconstruct_model_and_optimizer(model_filepath=self.model_filepath, model_name=self.model_name)
        self.model = model
        self.optimizer = optimizer
        return model, optimizer


    def inference(self, model, model_name, X, y):
        from src.model.inference import inference
        y_pred_actual_single, y_pred_actual_all, mse = inference(model_name=model_name, model=model, X=X, y=y)
        return y_pred_actual_single, y_pred_actual_all, mse


    def select_production_model(self, validation_window: int = 7, should_plot: bool = True):
        if not self.candidate_models: self.define_candidate_models()

        X_val, y_val = self.datasets[1], self.datasets[4]
        X_val_sample, y_val_sample = X_val[-validation_window: ], y_val[-validation_window: ]
        X_val_tensor = torch.from_numpy(X_val_sample).float()
        X_val_sample = X_val_tensor.unsqueeze(1).to(self.device)
        best_mse = 0
        best_model_index = 0

        for i, item in enumerate(self.candidate_models):
            model_name, model = item['model_name'], item['model']
            _, y_pred_actual_all, mse = self.inference(model=model, model_name=model_name, X=X_val_sample, y=y_val_sample)
            if mse < best_mse:
                best_mse = mse
                best_model_index = i

        main_logger.info(f'best model found: {self.candidate_models[best_model_index]['model_name']} with mse loss {best_mse:,.4f}')
        production_model = self.candidate_models[best_model_index]['model']
        production_optimizer = self.candidate_models[best_model_index]['optimizer']

        if should_plot:
            from src.model.inference import plot_predictions_and_discrepancy

            dates = self.data_handler._get_full_date_index()[-validation_window:]
            plot_predictions_and_discrepancy(
                dates=dates, # type: ignore
                y_true_log=y_val_sample,
                y_pred_actual=y_pred_actual_all,
                model_name=self.candidate_models[best_model_index]['model_name']
            )

        self.production_model = production_model
        self.production_optimizer = production_optimizer


    def online_learning(self, current_batch: List[Dict[str, Any]], target_col: str = 'close'):
        if not current_batch: return
        if not self.production_model or not self.production_optimizer: self.select_production_model()
        main_logger.info('... production model and optimizer are set ...')

        if self.production_model and self.production_optimizer:
            df = self.data_handler.run_lakehouse(current_batch=current_batch)
            y = df[target_col]
            X = df.copy().drop(columns=target_col, axis=1)

            preprocessor = self.data_handler.load_trained_preprocessor()
            try: X = preprocessor.transform(X)
            except: preprocessor.fit_transform(X)

            # convert to pytorch tensors
            X = torch.from_numpy(X).float().to(self.device)
            y = torch.from_numpy(y.to_numpy()).float().reshape(-1, 1).to(self.device)


            self.production_optimizer.zero_grad()
            y_pred = self.production_model(X)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.production_optimizer.step()
            self.steps_trained += 1

            # pred result
            epsilon = 0
            try: y_pred_actual = np.exp(y_pred + epsilon)
            except: y_pred_actual = np.exp(y_pred.cpu().detach().numpy() + epsilon)

            main_logger.info(f"... pytorch model trained (step {self.steps_trained}) ...\n - batch size {len(current_batch)}\n - mse loss {loss.item():.6f}\n - new closing price predicted: {np.mean(y_pred_actual):.2f}")
