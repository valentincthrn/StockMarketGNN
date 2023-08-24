from tqdm import tqdm
import numpy as np
import pandas as pd
import mlflow
from typing import Dict
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.data import Batch

from src.utils.common import calculate_mape


class GNN(torch.nn.Module):
    def __init__(self, model_config: Dict, exp_config: Dict):
        super(GNN, self).__init__()

        self.nbr_gcn_hidden = exp_config["nbr_gcn_hidden"]
        self.nbr_mlp_hidden = exp_config["nbr_mlp_hidden"]
        self.has_dropout = exp_config["has_dropout"]

        self.conv_first = GraphConv(
            model_config["history_length"], model_config["hidden_size"]
        )
        self.conv_hidden = GraphConv(
            model_config["hidden_size"], model_config["hidden_size"]
        )

        self.dropout = torch.nn.Dropout(0.1)  # Dropout layer

        self.fc_hidden = torch.nn.Linear(
            model_config["hidden_size"], model_config["hidden_size"]
        )
        self.fc_last = torch.nn.Linear(
            model_config["hidden_size"], model_config["horizon_forecast"]
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv_first(x, edge_index, edge_attr)
        x = F.relu(x)

        if self.has_dropout:
            x = self.dropout(x)

        for _ in range(self.nbr_gcn_hidden):
            x = self.conv_hidden(x, edge_index, edge_attr)
            x = F.relu(x)

        for _ in range(self.nbr_mlp_hidden):
            x = self.fc_hidden(x)
            x = F.relu(x)

        x = self.fc_last(x)

        return x


def testing(
    model: GNN,
    testingset: Batch,
    data_price: pd.DataFrame,
    test_days: int,
    during_training: bool = False,
) -> pd.DataFrame:
    # evaluate predictions
    preds = []

    # Iterate over each graph in the batch
    for data in tqdm(testingset.to_data_list()):
        out = list(model(data).detach().numpy())

        preds.append(out)

    pred_col_gnn = [col.replace("_true", "_pred_gnn") for col in data_price.columns]

    result = pd.DataFrame(
        np.squeeze(np.array(preds)),
        columns=pred_col_gnn,
        index=data_price.iloc[-test_days:].index,
    )

    # Add the close price
    result = result.join(data_price)

    if during_training:
        # Creating the MultiIndex
        multi_columns = [
            (col.split("_", 2)[1], col.split("_", 2)[2]) for col in result.columns
        ]
        multi_index = pd.MultiIndex.from_tuples(multi_columns)

        # Set the columns of your dataframe to this MultiIndex
        result.columns = multi_index
        # Sort the columns based on the first level of the MultiIndex
        result.sort_index(axis=1, level=0, inplace=True)

        mape_values = {}
        for col in result.columns.get_level_values(0).unique():
            true_values = result[col, "true"]
            pred_values = result[col, "pred_gnn"]
            mape_values[col + "_test_mape"] = calculate_mape(true_values, pred_values)

        mape = sum(mape_values.values()) / len(mape_values)
        mlflow.log_metrics(mape_values)
        mlflow.log_metric("MAPE Testing Loss", mape)
        print("MAPE Testing Loss: ", mape)

    else:
        return result
