import torch
from torch_geometric.data import Data, Batch
from torch.optim import Adam
import pandas as pd
import functools as ft
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import abc
from typing import List, Optional, Dict

from src.utils.common import mape_loss
from src.model.data_manage import df_prep, dataset_prep
from src.model.mygnn import GNN
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

        self.db: DBInterface = db

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

    def data_prep(self):
        self.data_price, self.data_roc = df_prep(
            self.db, self.targets, price_to_roc=self.data_config["price_to_roc"]
        )

        self.trainingset, self.testingset = dataset_prep(
            self.data_price, self.data_roc, self.data_config
        )

    def train(self):
        self.model = GNN(
            num_features=self.data_config["history_length"],
            hidden_channels=self.model_config["hidden_size"],
            num_classes=1,
        )

        # Define a suitable optimizer
        optimizer = Adam(self.model.parameters(), lr=0.001)

        # Define loss function - mean squared error loss
        # loss_func = torch.nn.MSELoss()

        # Training loop
        for epoch in range(self.model_config["epoch"]):
            loss_list = []
            print("EPOCH > ", epoch)
            for i, data in tqdm(
                enumerate(self.trainingset.to_data_list())
            ):  # Iterate over each graph in the batch
                self.model.train()
                optimizer.zero_grad()

                # Forward pass
                out = self.model(data).squeeze()

                # Calculate loss
                loss = mape_loss(out, data.y)
                loss_list.append(loss.item())
                # if (i == 10) & (epoch == 1):
                #     print("Exemple data for the 10 points")
                #     print("input data", data.x)
                #     print("label", data.y)
                #     print("output model", out)

                # Backward pass
                loss.backward()
                optimizer.step()

                self.model.eval()
            print("Mean Loss: ", np.mean(loss_list))

    def evaluate(self):
        # evaluate predictions
        preds = []

        # Iterate over each graph in the batch
        for data in tqdm(self.testingset.to_data_list()):
            out = list(self.model(data).detach().numpy())

            preds.append(out)

        pred_col_gnn = [col + "_pred_gnn" for col in self.data_price.columns]
        self.result = pd.DataFrame(
            np.squeeze(np.array(preds)),
            columns=pred_col_gnn,
            index=self.data_price.iloc[-self.data_config["test_days"] :].index,
        )

        if self.data_config["price_to_roc"]:
            first_price = pd.DataFrame(
                data=self.data_price.iloc[-self.data_config["test_days"] - 1]
            ).T
            first_price.columns = self.result.columns

            # GNN prediction
            self.result = pd.concat(
                [
                    first_price,
                    1 + self.result / 100,
                ]
            ).cumprod()

        # Add the close price
        self.result = self.result.join(self.data_price)

        # Add the naive predictions
        naive_cols = [col + "_pred_naive" for col in self.data_price.columns]
        self.result[naive_cols] = self.result[self.data_price.columns].shift(1)

    def postprocess(self):
        multi_index = []
        for s in self.data_price.columns:
            s_without_sa = s.lower().replace(".sa", "")
            multi_index.append((s_without_sa, "pred_gnn"))
        for s in self.data_price.columns:
            s_without_sa = s.lower().replace(".sa", "")
            multi_index.append((s_without_sa, "true"))
        for s in self.data_price.columns:
            s_without_sa = s.lower().replace(".sa", "")
            multi_index.append((s_without_sa, "pred_naive"))
        cols = pd.MultiIndex.from_tuples(multi_index)
        self.result.columns = cols

        symbol = []
        rmse_gnn = []
        rmse_naive = []

        # outputing results
        for s in self.data_price:
            s_without_sa = s.lower().replace(".sa", "")
            symbol.append(s_without_sa)
            df_s = self.result[s_without_sa]
            df_s.plot()
            rmse_gnn.append(((df_s.true - df_s.pred_gnn) ** 2).mean() ** 0.5)
            rmse_naive.append(((df_s.true - df_s.pred_naive) ** 2).mean() ** 0.5)

        self.output = pd.DataFrame(
            {
                "symbol": symbol,
                "rmse_gnn": rmse_gnn,
                "rmse_naive": rmse_naive,
            }
        )

        print(self.output)

        return

    def save(self):
        self.result.to_csv("test.csv")
