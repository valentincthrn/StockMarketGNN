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


from src.model.data_manage import df_prep, dataset_prep
from src.model.mygnn import GNN
from src.configs import RunConfiguration
from src.utils.db import DBInterface


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
        self.data = df_prep(self.db, self.targets)

        self.trainingset, self.testingset = dataset_prep(self.data, self.data_config)

    def train(self):
        self.model = GNN(
            num_features=self.data_config["history_length"],
            hidden_channels=self.model_config["hidden_size"],
            num_classes=1,
        )

        # Define a suitable optimizer
        optimizer = Adam(self.model.parameters(), lr=0.001)

        # Define loss function - mean squared error loss
        loss_func = torch.nn.MSELoss()

        # Training loop
        for epoch in range(self.model_config["epoch"]):
            loss_list = []
            print("EPOCH > ", epoch)
            for data in tqdm(
                self.trainingset.to_data_list()
            ):  # Iterate over each graph in the batch
                self.model.train()
                optimizer.zero_grad()

                # print("data: ", data)
                # Forward pass
                out = self.model(data).squeeze()

                # print("out: ", out)

                # Calculate loss
                loss = loss_func(out, data.y)
                loss_list.append(loss.item())

                # Backward pass
                loss.backward()
                optimizer.step()

                self.model.eval()
            print("Mean Loss: ", np.mean(loss_list))

    def evaluate(self):
        preds = []

        # Iterate over each graph in the batch
        for data in tqdm(self.testingset.to_data_list()):
            out = list(self.model(data).detach().numpy())

            preds.append(out)

        pred_col_gnn = [col + "_pred_gnn" for col in self.data.columns]
        df_pred = pd.DataFrame(
            np.squeeze(np.array(preds)),
            columns=pred_col_gnn,
            index=self.data.iloc[-self.data_config["test_days"] :].index,
        )

        self.result = pd.concat(
            [self.data.iloc[-self.data_config["test_days"] :], df_pred],
            axis=1,
        )

        naive_cols = [col + "_pred_naive" for col in self.data.columns]
        self.result[naive_cols] = self.result[self.data.columns].shift(1)

    def postprocess(self):
        multi_index = []
        for s in self.targets:
            s_without_sa = s.lower().replace(".sa", "")
            multi_index.append((s_without_sa, "true"))
        for s in self.targets:
            s_without_sa = s.lower().replace(".sa", "")
            multi_index.append((s_without_sa, "pred_gnn"))
        for s in self.targets:
            s_without_sa = s.lower().replace(".sa", "")
            multi_index.append((s_without_sa, "pred_naive"))
        cols = pd.MultiIndex.from_tuples(multi_index)
        self.result.columns = cols

        symbol = []
        rmse_gnn = []
        rmse_naive = []

        # outputing results
        for s in self.targets:
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
