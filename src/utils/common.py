import math
import torch
from pathlib import Path
import dataclasses
import yaml
import pickle as pkl


from src.configs import RunConfiguration


def mape_loss(x, y):
    # Ensure no division by zero
    epsilon = 1e-6
    loss = torch.abs(100 * (x - y) / (y + epsilon))
    return torch.mean(loss)

def index_agreement_torch(s: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """
    Index of Agreement
    Willmott (1981, 1982)

    This function is adapted for batch processing, where each column represents a different company.

    Args:
        s: Simulated or predicted values (e.g., stock prices), shape [90, 5] where 90 is days and 5 is companies.
        o: Observed or true values, shape [90, 5].

    Returns:
        ia: Index of Agreement averaged over all companies.
    """
    o_mean = torch.mean(o, dim=1, keepdim=True)
    numerator = torch.sum((o - s) ** 2, dim=1)
    denominator = torch.sum((torch.abs(s - o_mean) + torch.abs(o - o_mean)) ** 2, dim=1)
    ia = 1 - numerator / denominator

    return - ia.mean() + 1


def calculate_mape(true_values, pred_values):
    # Avoid division by zero
    mask = true_values != 0
    return 100 * (abs((true_values - pred_values) / true_values)[mask].mean())


def save_yaml_config(config: RunConfiguration, MODEL_PATH_RID: Path, file_name: str = "run_config.yml"):
    # Write the config file as yaml
    config_dict = dataclasses.asdict(config)
    yaml_str = yaml.dump(config_dict)

    # Write the YAML string to a file
    with open(MODEL_PATH_RID / file_name, "w") as file:
        file.write(yaml_str)
        
def save_pickle(dictio, MODEL_PATH_RID: Path, file_name: str = "normalization_config.pkl"):

    with open(MODEL_PATH_RID / file_name, "wb") as file:
        pkl.dump(dictio, file)
        
def load_pickle(pkl_path: Path):

    with open(pkl_path, "rb") as file:
        dictio = pkl.load(file)
    return dictio


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
