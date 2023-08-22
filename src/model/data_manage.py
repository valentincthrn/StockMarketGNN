import pandas as pd
from src.utils.db import DBInterface

from torch_geometric.data import Data, Batch


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
    print(data)
    print(config)
    return
