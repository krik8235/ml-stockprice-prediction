# hyperband tuning

import random
import argparse
import math
from math import log, floor
import fsspec
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

from src import TICKER
from src._utils import main_logger
from src.data_handling.data_handler import DataHandler
from src.model.instantiate import LSTM, GRU, MLP
from src.model.inference import _calculate_metrics


def _handle_optimizer(optimizer_name, model, lr, **kwargs):
    optimizer = None
    match optimizer_name.lower():
        case 'adam': optimizer = optim.Adam(model.parameters(), lr=lr, **kwargs)
        case 'adamw': optimizer = optim.AdamW(model.parameters(), lr=lr, **kwargs)
        case 'adamax': optimizer = optim.Adamax(model.parameters(), lr=lr, **kwargs)
        case 'adadelta': optimizer = optim.Adadelta(model.parameters(), lr=lr, **kwargs)
        case 'adafactor': optimizer = optim.Adafactor(model.parameters(), lr=lr, **kwargs)
        case 'rmsprop': optimizer = optim.RMSprop(model.parameters(), lr=lr, **kwargs)
        case 'radam': optimizer = optim.RAdam(model.parameters(), lr=lr, **kwargs)
        case 'rprop': optimizer = optim.Rprop(model.parameters(), lr=lr, **kwargs)
        case 'sgd': optimizer = optim.SGD(model.parameters(), lr=lr, **kwargs)
        case _: optimizer = optim.Adam(model.parameters(), lr=lr, **kwargs)
    return optimizer


def search_space(model_name: str = 'lstm'):
    match model_name:
        case 'mlp':
            num_layers = random.randint(1, 20)
            search_space = dict()
            for i in range(0, num_layers):
                search_space[f'dropout_rate_layer_{i}'] = random.uniform(0.0, 0.6)
                search_space[f'n_units_layer_{i}'] = random.uniform(8, 256)

            search_space.update(
                num_layers=num_layers,
                batch_norm=random.choice([True, False]),
                optimizer_name=random.choice(['adam', 'rmsprop', 'sgd', 'adamw', 'adamax', 'adadelta', 'radam', 'rprop']),
                lr=random.uniform(1e-5, 1e-2),
                batch_size=random.choice([16, 32, 64, 128, 256]),
            )
            return search_space
        case _:
            return dict(
                num_layers=random.randint(1, 20),
                hidden_size=random.choice([16, 32, 64, 128, 256]),
                bias=random.choice([True, False]),
                bidirectional=random.choice([True, False]),
                dropout=random.uniform(0.0, 0.6),

                optimizer_name=random.choice(['adam', 'rmsprop', 'sgd', 'adamw', 'adamax', 'adadelta', 'radam', 'rprop']),
                lr=random.uniform(1e-5, 1e-2),
                batch_size=random.choice([16, 32, 64, 128, 256]),
            )


def _train(
        X_train, y_train, X_val, y_val,
        config: dict,
        budget: int,
        patience: int = 10,
        tol: float = 1e-5,
        model_name: str = 'lstm',
        mlflow_run_name: str = 'train_trial'
    ):
    criterion = nn.MSELoss()
    device = DataHandler().device
    model = None
    match model_name:
        case 'lstm': model = LSTM(input_size=X_train.shape[-1], **config)
        case 'gru': model = GRU(input_size=X_train.shape[-1], **config)
        case _: model = MLP(input_size=X_train.shape[-1], **config)
    model.to(device=device)

    # optimizer
    lr = config['lr']
    optimizer_name = config['optimizer_name']
    optimizer = _handle_optimizer(optimizer_name=optimizer_name, model=model, lr=lr)

    loss_history = []
    best_val_mse = float('inf') # primary loss
    best_val_mae = float('inf')
    epochs_no_improve = 0
    best_model = None
    batch_size = config['batch_size']
    train_data_loader = DataHandler().create_tensor_data_loader(X=X_train, y=y_train, batch_size=batch_size)
    val_data_loader = DataHandler().create_tensor_data_loader(X=X_val, y=y_val, batch_size=batch_size)

    # mlflow: Start a nested run for this trial
    with mlflow.start_run(nested=True, run_name=mlflow_run_name) as child_run:
        mlflow.log_params(config) # Log all hyperparameters

        # training loop for the specified budget
        for epoch in range(int(budget)):
            main_logger.info(f'... start epoch {epoch + 1} ...')
            model.train()

            for batch_X, batch_y in train_data_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                try:
                    with torch.autocast(device_type=DataHandler().device_type):
                        batch_X = batch_X.unsqueeze(1)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
                            main_logger.error('... pytorch model returns nan or inf. break the training loop ...')
                            break
                        if not math.isfinite(loss.item()):
                            main_logger.error('... loss is nan or inf. break the training loop ...')
                            break
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                except:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # val
            model.to(device).eval()
            with torch.inference_mode():
                val_mse_sum, val_mae_sum, batch_count = 0, 0, 0

                # batch loop
                for batch_X, batch_y in val_data_loader:
                    batch_X = batch_X.unsqueeze(1).to(device)  # reshape (batch_size, sequence_length, input_dim)
                    outputs_val = model(batch_X)

                    # metrics
                    batch_mse, batch_mae = _calculate_metrics(outputs_val, batch_y)
                    val_mse_sum += batch_mse
                    val_mae_sum += batch_mae
                    batch_count += 1

                # log metrics
                avg_val_mse = val_mse_sum / batch_count
                avg_val_mae = val_mae_sum / batch_count
                loss_history.append(avg_val_mse)
                mlflow.log_metrics({'val_mse': avg_val_mse, 'val_mae': avg_val_mae}, step=epoch)

                # early stopping
                if avg_val_mse + tol < best_val_mse:
                    best_val_mse = avg_val_mse
                    best_val_mae = avg_val_mae
                    epochs_no_improve = 0
                    best_model = model
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        main_logger.info(f'... early stopping at epoch {epoch+1} due to no improvement in val loss ...')
                        if not best_model: best_model = model
                        break

            if epoch % 50 == 0: main_logger.info(f'... epoch {epoch + 1}: loss {avg_val_mse:.4f} ...')

    # log best metrics after full budget run
    mlflow.log_metric("best_val_mse", best_val_mse)
    mlflow.log_metric("best_val_mae", best_val_mae)

    return best_val_mse, best_val_mae, loss_history, best_model, optimizer


def walk_forward_validation(
        config: dict,
        budget: int,
        X, y,
        train_window: int = 1000,
        val_window: int = 500,
        buffer: int = 100,
        model_name: str = 'lstm' # or gru
    ):

    best_model, optimizer = None, None
    total_val_mse, total_val_mae = 0, 0
    all_loss_histories = []
    num_folds = (X.shape[0] - (train_window + buffer + val_window)) // val_window + 1

    for i in range(num_folds):
        train_start = i * val_window
        train_end = train_start + train_window
        val_start = train_end + buffer
        val_end = val_start + val_window

        # ensure not to past the last item
        if val_end > X.shape[0]:
            break

        X_train_fold = X[train_start:train_end]
        y_train_fold = y[train_start:train_end]
        X_val_fold = X[val_start:val_end]
        y_val_fold = y[val_start:val_end]

        main_logger.info(f'... running walk-forward validation - fold {i+1}/{num_folds}: training on samples from #{train_start} to #{train_end - 1}, validating on samples from #{val_start} to #{val_end - 1}')

        fold_val_mse, fold_val_mae, fold_loss_history, best_model, optimizer = _train(
            X_train=X_train_fold,
            y_train=y_train_fold,
            X_val=X_val_fold,
            y_val=y_val_fold,
            config=config,
            budget=budget,
            model_name=model_name,
            mlflow_run_name=f'model_{model_name}_fold_{i+1}_budget_{budget}',
        )
        total_val_mse += fold_val_mse
        total_val_mae += fold_val_mae
        all_loss_histories.append(fold_loss_history)

    avg_val_mse = total_val_mse / num_folds
    avg_val_mae = total_val_mae / num_folds

    return avg_val_mse, avg_val_mae, all_loss_histories, best_model, optimizer


def run_hyperband(
        search_space_fn,
        val_fn,
        R: int =100,
        halving_factor: int = 4,
        model_name: str = 'lstm'
    ) -> tuple:

    with mlflow.start_run(run_name=f'hyperband_search_for_{model_name}') as parent_run:
        mlflow.set_tag('model_type', model_name)
        mlflow.log_param('R', R)
        mlflow.log_param('halving_factor', halving_factor)

        s_max = int(log(R, halving_factor))
        overall_best_config = None
        overall_best_mse = float('inf')
        overall_best_mae = float('inf')
        best_model, optimizer = None, None

        # outer loop: iterate s_max brackets
        for s in range(s_max, -1, -1):
            n = int(R / halving_factor**s)
            r = int(R / n)
            main_logger.info(f'☑️ bracket {s}/{s_max}: {n} configurations, initial budget={r} ...')

            # geerate n random hyperparameter configurations
            configs = [search_space_fn(model_name=model_name) for _ in range(n)]

            # sha loop
            for i in range(s + 1):
                budget = r * (halving_factor**i)
                main_logger.info(f'... training {len(configs)} configurations for budget {budget} epochs ...')

                results = []
                for config in configs:
                    avg_val_mse, avg_val_mae, loss_history, best_model, optimizer = val_fn(config, budget)
                    results.append((config, avg_val_mse, avg_val_mae, loss_history, best_model, optimizer))

                # sort and select top configurations
                results.sort(key=lambda x: x[1])

                # keep track of the best configuration found so far
                if results and results[0][1] < overall_best_mse:
                    overall_best_config = results[0][0]
                    overall_best_mse = results[0][1]
                    overall_best_mae = results[0][2]
                    best_model = results[0][4]
                    optimizer = results[0][5]

                num_survivors = floor(len(configs) / halving_factor)
                configs = [result[0] for result in results[:num_survivors]]

                if not configs:
                    break

        main_logger.info(f'... best config found: {overall_best_config} with mse {overall_best_mse:,.4f}, mae {overall_best_mae:,.4f}')

        # log final best metrics for the overall hyperband run
        if overall_best_config: mlflow.log_params(overall_best_config)
        mlflow.log_metric("final_best_avg_val_mse", overall_best_mse)
        mlflow.log_metric("final_best_avg_val_mae", overall_best_mae)

    return overall_best_config, overall_best_mse, overall_best_mae, best_model, optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default=TICKER, help=f'ticker. default = {TICKER}')
    parser.add_argument('--r', type=int, default=10, help=f'budget R. default = 10')
    args = parser.parse_args()

    ticker = args.ticker
    R = args.r

    X_train, X_val, X_test, y_train, y_val, y_test = DataHandler(ticker=ticker).preprocess(should_refresh=False)
    model_names = ['mlp', 'lstm', 'gru']

    for model_name in model_names:
        main_logger.info(f'... start tuning {model_name} ...')

        # tuning
        halving_factor = 4
        train_window, val_window, buffer = 1000, 500, 100

        best_config, best_mse, best_mae, best_model, optimizer = run_hyperband(
            search_space_fn=search_space,
            val_fn=lambda c, b: walk_forward_validation(
                X=X_train, y=y_train,
                config=c,
                budget=b,
                train_window=train_window,
                val_window=val_window,
                buffer=buffer,
                model_name=model_name,
            ),
            R=R,
            halving_factor=halving_factor,
            model_name=model_name,
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

        # storage
        s3_bucket_name = 'ml-stockprice-pred'
        model_filepath = f's3a://{s3_bucket_name}/artifacts/{ticker}/models/{model_name}'
        parent_dir = f's3a://{s3_bucket_name}/artifacts/{ticker}/models'
        fs = fsspec.filesystem('s3')
        fs.mkdir(parent_dir, create_parents=True)
        with fs.open(model_filepath, 'wb') as f:
            torch.save(checkpoint, f)

        main_logger.info(f'✅ {model_name} tuned.')

        import time
        time.sleep(30)
