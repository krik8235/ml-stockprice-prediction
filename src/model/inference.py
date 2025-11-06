# perform inference, compute metrics, plot pred results

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import mlflow

from src.data_handling.data_handler import DataHandler
from src._utils import main_logger



def _calculate_metrics(y_pred, y) -> tuple[float, float]:
    if isinstance(y, pd.Series): y = y.to_numpy()
    if isinstance(y, np.ndarray): y = torch.from_numpy(y).float()
    if not isinstance(y, torch.Tensor): y = torch.tensor(y).float()
    y = y.to(DataHandler().device)
    if y_pred.shape != y.shape: y = y.view(y_pred.shape) # adjust dims

    mse = F.mse_loss(y_pred, y).item()
    mae_log = F.l1_loss(y_pred, y).item()
    mae_actual = np.exp(mae_log)
    return mse, mae_actual



def inference(model_name: str, model, X, y) -> tuple:
    model.eval()

    with torch.inference_mode():
        epsilon = 0
        y_pred = model(X)
        y_pred_log_single = y_pred.mean().item()
        y_pred_actual_single = np.exp(y_pred_log_single + epsilon)
        mse, mae = _calculate_metrics(y_pred=y_pred, y=y)
        y_pred_actual_all = torch.exp(y_pred).cpu().numpy().flatten()
        main_logger.info(f'{model_name}: {y_pred_actual_single:,.4f} mse: {mse:,.4f}, mae {mae:,.4f}')
        return y_pred_actual_single, y_pred_actual_all, mse



def plot_predictions_and_discrepancy(
        dates: pd.DatetimeIndex,
        y_true_log: pd.Series,
        y_pred_actual: np.ndarray,
        model_name: str,
        title: str = "Stock Price Prediction vs. True Value"
    ):
    import matplotlib.pyplot as plt
    y_true_actual = np.exp(y_true_log.to_numpy()).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true_actual, label='True Price', color='darkblue', linewidth=2)
    plt.plot(dates, y_pred_actual, label='Predicted Price', color='darkred', linewidth=2)

    # over pred
    plt.fill_between(
        dates,
        y_pred_actual,
        y_true_actual,
        where=(y_pred_actual >= y_true_actual), # type: ignore
        facecolor='salmon',
        alpha=0.4,
        label='over-prediction'
    )

    # under pred
    plt.fill_between(
        dates,
        y_pred_actual,
        y_true_actual,
        where=(y_pred_actual < y_true_actual), # type: ignore
        facecolor='lightblue',
        alpha=0.4,
        label='under-prediction'
    )

    plt.title(f'{title} by {model_name}', fontsize=16)
    plt.xlabel('Date/Time', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    try:
        if mlflow.active_run(): mlflow.log_figure(plt.gcf(), f"predictions_plot_{model_name}.png")
    except NameError:
        pass

    fig_filepath = os.path.join('fig')
    os.makedirs(fig_filepath, exist_ok=True)

    filename = f'fig-{model_name}'
    plt.savefig(os.path.join(fig_filepath, filename))
    # plt.show()
