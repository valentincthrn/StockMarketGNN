import math
import torch
from pathlib import Path
import dataclasses
import yaml


from src.configs import RunConfiguration


def mape_loss(x, y):
    # Ensure no division by zero
    epsilon = 1e-6
    loss = torch.abs(100 * (x - y) / (y + epsilon))
    return torch.mean(loss)


def calculate_mape(true_values, pred_values):
    # Avoid division by zero
    mask = true_values != 0
    return 100 * (abs((true_values - pred_values) / true_values)[mask].mean())


def save_yaml_config(config: RunConfiguration, MODEL_PATH_RID: Path):
    # Write the config file as yaml
    config_dict = dataclasses.asdict(config)
    yaml_str = yaml.dump(config_dict)

    # Write the YAML string to a file
    with open(MODEL_PATH_RID / "run_config.yml", "w") as file:
        file.write(yaml_str)


class PositionalEncoding(torch.nn.Module):
    """Encoder that applies positional based encoding.
    Encoder that considers data temporal position in the time series' tensor to provide
    a encoding based on harmonic functions.
    Attributes:
    hidden_size (int): size of hidden representation
    dropout (float): dropout rate
    max_len (int): maximum quantity of features
    """

    def __init__(self, hidden_size, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.hidden_size = hidden_size
        self.div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2).float()
            * (-math.log(10000.0) / self.hidden_size)
        )

    def forward(self, position):
        pe = torch.empty(position.shape[0], self.hidden_size, device=position.device)
        pe[:, 0::2] = torch.sin(position * self.div_term.to(position.device))
        pe[:, 1::2] = torch.cos(position * self.div_term.to(position.device))
        return self.dropout(pe)
