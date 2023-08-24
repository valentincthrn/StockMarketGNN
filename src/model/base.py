import torch
from torch_geometric.data import Data, Batch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import mlflow
import pandas as pd
import functools as ft
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import abc
from random import shuffle
from typing import List, Optional, Dict
import collections


from src.utils.common import mape_loss, calculate_mape
from src.model.data_manage import df_prep, dataset_prep
from src.model.mygnn import GNN, testing
from src.configs import RunConfiguration, DATE_FORMAT
from src.utils.db import DBInterface

import torch
import torch.nn.functional as F


class _BaseEXP(torch.nn.Module, abc.ABC):
    targets: List[str]
    features: str
    encoding_method: Optional[None]

    def __init__(
        self,
        config: RunConfiguration,
        db: DBInterface,
    ):
        super(_BaseEXP, self).__init__()

        # global config
        self.data_config: dict = config.data_prep
        self.model_config: dict = config.model

        self.targets = self._targets
        self.features = self._features
        self.encoding_method = self._encodings
        self.exp_model = self._exp_model

        self.db: DBInterface = db

        assert (
            self.data_config["history_length"] == self.model_config["history_length"]
        ), "History Length of data ingestion should be equal"

        mlflow.log_param("Target Companies", self.targets)
        mlflow.log_param("Nbr Companies", len(self.targets))
        mlflow.log_params(self.data_config)
        mlflow.log_params(self.model_config)
        mlflow.log_param("Encoding", self.encoding_method)

    @property
    @abc.abstractmethod
    def _targets(self) -> List[str]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _encodings(self) -> List[str]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _features(self) -> Dict[str, bool]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _exp_model(self) -> Dict[str, bool]:
        raise NotImplementedError

    def data_prep(self):
        self.data_price, self.data_roc = df_prep(
            self.db, self.targets, price_to_roc=self.data_config["price_to_roc"]
        )

        self.trainingset, self.testingset = dataset_prep(
            self.data_price, self.data_roc, self.data_config
        )

    def train(self):
        self.model = GNN(model_config=self.model_config, exp_config=self.exp_model)
        mlflow.log_params(self.exp_model)

        self.data_price.columns = [col + "_true" for col in self.data_price.columns]

        # Define a suitable optimizer
        optimizer = Adam(self.model.parameters(), lr=0.001)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        # Define loss function - mean squared error loss
        # loss_func = torch.nn.MSELoss()
        data_list = self.trainingset.to_data_list()
        # Training loop
        for epoch in range(self.model_config["epoch"]):
            loss_list = []
            print("EPOCH > ", epoch)
            shuffle(data_list)
            for data in tqdm(data_list):  # Iterate over each graph in the batch
                self.model.train()
                optimizer.zero_grad()

                # Forward pass
                out = self.model(data).squeeze()

                # Calculate loss
                loss = mape_loss(out, data.y)
                loss_list.append(loss.item())

                # Backward pass
                loss.backward()
                optimizer.step()

                self.model.eval()
            scheduler.step

            print("MAPE Training Loss: ", np.mean(loss_list))
            mlflow.log_metric("MAPE Training Loss", np.mean(loss_list))

            testing(
                self.model,
                self.testingset,
                self.data_price,
                self.data_config["test_days"],
                during_training=True,
            )

    def evaluate(self):
        self.result = testing(
            self.model,
            self.testingset,
            self.data_price,
            self.data_config["test_days"],
            during_training=False,
        )

        # Add the naive predictions
        pred_col_naive = [
            col.replace("_true", "_pred_naive") for col in self.data_price.columns
        ]

        self.result[pred_col_naive] = self.result[self.data_price.columns].shift(1)

        # Creating the MultiIndex
        multi_columns = [
            (col.split("_", 2)[1], col.split("_", 2)[2]) for col in self.result.columns
        ]
        multi_index = pd.MultiIndex.from_tuples(multi_columns)

        # Set the columns of your dataframe to this MultiIndex
        self.result.columns = multi_index
        # Sort the columns based on the first level of the MultiIndex
        self.result.sort_index(axis=1, level=0, inplace=True)

    def postprocess(self):
        for col in self.result.columns.get_level_values(0).unique():
            true_values = self.result[col, "true"]
            pred_naive_values = self.result[col, "pred_naive"]
            naive_error = calculate_mape(true_values, pred_naive_values)
            mlflow.log_metric(col + "_naive", naive_error)

        self.result.columns = [
            " ".join(col).strip() for col in self.result.columns.values
        ]

        self.result.to_csv("output.csv")
        mlflow.log_artifact("output.csv")
