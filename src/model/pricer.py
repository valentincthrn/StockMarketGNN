import torch
from torch_geometric.data import Data, Batch
import pandas as pd
import functools as ft


from src.model.data_manage import df_prep, dataset_prep
from src.configs import RunConfiguration
from src.utils.db import DBInterface


class GraphStockPricer(torch.nn.Module):
    def __init__(
        self,
        config: RunConfiguration,
        db: DBInterface,
    ):
        super(GraphStockPricer, self).__init__()
        self.config_data_prep: dict = config.data_prep
        self.config_encoding: dict = config.encoding
        self.config_model: dict = config.model
        self.targets: list = config.targets_to_predict
        self.db: DBInterface = db

    def data_prep(self):
        data = df_prep(self.db, self.targets)

        self.trainingset, self.testingset = dataset_prep(data, self.config_data_prep)

    def train(self):
        return

    def evaluate(self):
        return

    def postprocess(self):
        return
