import torch
import torch.nn as nn
import torch.optim.swa_utils as swa_utils


class MLP(nn.Module):
    def __init__(self, input_size = None, **kwargs):
        super(MLP, self).__init__()
        layers = []
        num_layers = kwargs.get('num_layers', 1)
        hidden_units_per_layer = kwargs.get('hidden_units_per_layer', [64] * num_layers)
        batch_norm = kwargs.get('batch_norm', False)
        dropout_rates = kwargs.get('dropout_rates', [0.1 for _ in range(0, num_layers)])
        current_input_size = input_size if input_size is not None else 32

        for i in range(num_layers):
            output_dim = hidden_units_per_layer[i]
            layers.append(nn.Linear(current_input_size, output_dim))
            if batch_norm: layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rates[i]))

            # align with the ouput dimensions
            current_input_size = output_dim

        layers.append(nn.Linear(current_input_size, 1))
        self.model_layers = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2: x = x.flatten(start_dim=1)
        return self.model_layers(x)


class AdaptiveEnsemble(nn.Module):
    def __init__(self, num_experts: int, input_size: int = 10, model = None, **kwargs):
        super().__init__()
        model = model if model else MLP(input_size, **kwargs)
        self.experts = nn.ModuleList([model for _ in range(num_experts)])
        self.num_experts = num_experts
        output_dim = 1
        self.combiner = nn.Linear(in_features=num_experts * output_dim, out_features=output_dim)
        self.gating_network = nn.Sequential(nn.Linear(input_size, num_experts), nn.Softmax(dim=-1))

    def forward(self, x):
        predictions = [expert(x) for expert in self.experts]

        # stacking outputs - [(128, 1), (128, 1), ... ] -> (128, 5)
        stacked_outputs = torch.cat(predictions, dim=1)

        # combiner learns to weigh/transform the stacked expert outputs.
        final_output = self.combiner(stacked_outputs)
        return final_output


class LSTM(nn.Module):
    def __init__(self, input_size: int, **kwargs):
        super(LSTM, self).__init__()
        # kwargs
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.num_layers = kwargs.get('num_layers', 10)
        self.dropout = kwargs.get('dropout', 0.1)
        self.bias = kwargs.get('bias', True)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.num_directions = 2 if self.bidirectional else 1
        self.output_size = 1

        # instantiate
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.bias,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.fc = nn.Linear(in_features=self.hidden_size * self.num_directions, out_features=self.output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        o_t, _ = self.lstm(x, (h0.detach(), c0.detach()))
        o_final = self.fc(o_t[:, -1, :])
        return o_final


class GRU(nn.Module):
    def __init__(self, input_size: int, **kwargs):
        super(GRU, self).__init__()

        self.hidden_size = kwargs.get('hidden_size', 256)
        self.num_layers = kwargs.get('num_layers', 10)
        self.dropout = kwargs.get('dropout', 0.1)
        self.bias = kwargs.get('bias', True)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.num_directions = 2 if self.bidirectional else 1
        self.output_size = 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.bias,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.fc = nn.Linear(in_features=self.hidden_size * self.num_directions, out_features=self.output_size)

    def forward(self, x): # (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        # hidden state shape - (num_layers * num_directions, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        o, _ = self.gru(x, h0)  # o shape - (batch_size, seq_len, hidden_size * num_directions)
        o_final = self.fc(o[:, -1, :])  # last time step = o[:, -1, :]
        return o_final


def instatiate(model_name: str, input_size: int = 10, **kwargs):
    model = None
    match model_name.lower():
        case 'gru': model = GRU(input_size=input_size, **kwargs)
        case 'lstm': model = LSTM(input_size=input_size, **kwargs)
        case _: model = MLP(input_size=input_size, **kwargs)
    return model


def instatiate_all_models(input_size: int, num_experts: int = 5, **kwargs) -> list:
    mlp = MLP(input_size=input_size, **kwargs)
    adaptive = AdaptiveEnsemble(num_experts=num_experts, input_size=input_size, **kwargs)
    lstm = LSTM(input_size=input_size, **kwargs)
    gru = GRU(input_size=input_size, **kwargs)
    gru_swa = swa_utils.AveragedModel(gru)
    return [mlp, adaptive, lstm, gru, gru_swa]
