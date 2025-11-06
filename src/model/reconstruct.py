# reconstruct model and optimizer
import torch
import fsspec

from src.model.instantiate import LSTM, GRU, MLP
from src.data_handling.data_handler import DataHandler
from src.model.tuning import _handle_optimizer


def reconstruct_model_and_optimizer(model_filepath: str, model_name: str = 'lstm') -> tuple:
    fs = fsspec.filesystem('s3')

    # load
    with fs.open(model_filepath, 'rb') as f:
        checkpoint = torch.load(f, weights_only=False, map_location=DataHandler().device)

    # model
    input_dim = checkpoint['input_dim']
    hparams = checkpoint['best_config']
    model = None
    match model_name:
        case 'gru': model = GRU(input_size=input_dim, **hparams)
        case 'lstm': model = LSTM(input_size=input_dim, **hparams)
        case _: model = MLP(input_size=input_dim, **hparams)

    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.to(DataHandler().device)

    # optimizer
    lr = checkpoint['best_config']['lr']
    optimizer_name = checkpoint['best_config']['optimizer_name']
    optimizer = _handle_optimizer(optimizer_name=optimizer_name, model=model, lr=lr)
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    optimizer.load_state_dict(optimizer_state_dict)

    return model, optimizer
