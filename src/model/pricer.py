import torch
from torch_geometric.data import Data, Batch

from src.configs import RunConfiguration
from src.utils.db import DBInterface


class GraphStockPricer(torch.nn.Module):
    def __init__(
        self,
        config: RunConfiguration,
        db: DBInterface,
    ):
        super(GraphStockPricer, self).__init__()
        self.config_data_prep: dict = config.model
        self.config_encoding: dict = config.encoding
        self.config_model: dict = config.model
        self.targets: list = config.targets_to_predict
        self.db: DBInterface = db

    def data_prep(self):
        return 1, 2

    def train(self):
        return

    def evaluate(self):
        return

    def postprocess(self):
        return
