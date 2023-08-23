from tqdm import tqdm
import numpy as np
import pandas as pd
import mlflow

import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.data import Batch


class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(num_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.dropout = torch.nn.Dropout(0.1)  # Dropout layer
        self.fc1 = torch.nn.Linear(hidden_channels, num_classes)  # MLP hidden layer

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.fc1(x)
        return x


def testing(
    model: GNN, testingset: Batch, data_price: pd.DataFrame, test_days: int
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

    mlflow.log_metrics(mape_values)
    mlflow.log_metric("Mean Testing Loss", sum(mape_values.values()) / len(mape_values))


def calculate_mape(true_values, pred_values):
    # Avoid division by zero
    mask = true_values != 0
    return 100 * (abs((true_values - pred_values) / true_values)[mask].mean())
