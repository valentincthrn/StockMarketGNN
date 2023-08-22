from tqdm import tqdm
import pandas as pd
from torch_geometric.data import Data, Batch
import networkx as nx
import torch

from src.utils.db import DBInterface


def df_prep(db: DBInterface, targets: list) -> pd.DataFrame:
    df = db.read_sql(query="SELECT * FROM stocks")

    df_symbol = df[lambda f: f["symbol"].isin(targets)]

    assert not df_symbol.empty, "The dataframe is empty"

    df_pivot = df.pivot(
        index="quote_date",
        columns="symbol",
        values=["open", "close", "high", "low", "volume"],
    )
    df_pivot.columns = [
        "_".join(col).lower().replace(".sa", "") for col in df_pivot.columns.values
    ]

    # focus only on close price for now
    df_close = df_pivot.filter(like="close")

    # fillna 0 for now
    df_close = df_close.fillna(0)

    return df_close


def dataset_prep(data: pd.DataFrame, config: dict) -> [Batch, Batch]:
    """From the dataframe, create the graph dataset that the model
    will train on

    :return: the training and testing set as a sequence of graphs
    """

    past_k = config["history_length"]
    test_days = config["test_days"]
    # Create a mapping of node names (stocks) to integers
    node_mapping = {node: i for i, node in enumerate(data.columns)}

    training_list = []
    testing_list = []

    N = len(data)

    for t in tqdm(range(past_k, N)):
        # Step 1: Create correlation matrix and graph for this time step
        correlation_matrix = data.iloc[:t].corr().fillna(0)
        graph = nx.from_pandas_adjacency(correlation_matrix)
        graph.remove_edges_from(nx.selfloop_edges(graph))

        # Step 2: Convert edges to integer tuples and store as tensor
        edges = list(graph.edges)
        edges_mapped = [(node_mapping[u], node_mapping[v]) for u, v in edges]
        edge_indices = torch.tensor(edges_mapped, dtype=torch.long).t().contiguous()

        # Step 3: Store edge weights as tensor
        edge_weights = torch.tensor(
            [graph[u][v]["weight"] for u, v in edges], dtype=torch.float
        )

        # Step 4: Store closing prices as node features (assume last column of df is 'Close')
        node_features = torch.tensor(
            data.iloc[t - past_k : t, :].T.values, dtype=torch.float
        )

        # Step 5: Store prices at t+1 as labels
        labels = torch.tensor(data.iloc[t, :], dtype=torch.float)

        if t < N - test_days:
            training_list.append(
                Data(
                    x=node_features,
                    edge_index=edge_indices,
                    edge_attr=edge_weights,
                    y=labels,
                )
            )
        else:
            testing_list.append(
                Data(
                    x=node_features,
                    edge_index=edge_indices,
                    edge_attr=edge_weights,
                    y=labels,
                )
            )

    # Convert data_list into a Batch for feeding into your model
    training_batch = Batch.from_data_list(training_list)
    testing_batch = Batch.from_data_list(testing_list)

    return training_batch, testing_batch
