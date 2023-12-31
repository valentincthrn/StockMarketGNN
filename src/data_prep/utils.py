import pandas as pd
from pathlib import Path
from typing import Dict


def add_time_component(
    df: pd.DataFrame,
    time: int,
    df_raw_prices: pd.DataFrame,
    history: int,
    horizon: int,
    min_points: int,
) -> pd.DataFrame:
    """Based on the price database,
    add the time component positional encoding

    :param df: the dataframe in the window
    :type df: pd.DataFrame
    """

    if history is None:
        # Filter the dataframe
        df_range = df.iloc[time - horizon :]
    else:
        df_range = df.iloc[time - horizon : time + history]

    # get prices
    df_range_prices = df_range.xs("price", axis=1, level=1, drop_level=True)

    # Keep symbol where there is all predicted prices AND at least one history price
    exist_min_price_history = df_range_prices.iloc[horizon:].notna().sum() >= min_points
    exist_all_predicted_price = ~df_range_prices.iloc[:horizon].isna().any()
    keep = exist_min_price_history & exist_all_predicted_price
    keep = pd.concat([keep, pd.Series({"diff": True})], axis=0)

    df_range = df_range[keep.index[keep]]
    df_raw_prices_keep = df_raw_prices[keep.index[keep]]

    # Companies
    companies = list(set(df_range.columns.get_level_values(0)) - {"diff"})

    # Define y
    y_list = (
        df_range_prices.iloc[:horizon]
        .sort_index()[companies]
        .round(8)
        .T.values.tolist()
    )

    last_price = df_raw_prices_keep.iloc[time][companies].values.tolist()

    # cumulated the time component
    df_range.iloc[:horizon, -1] = 0
    df_order = df_range.assign(order=lambda x: x["diff"].cumsum())

    return df_order, y_list, last_price, companies


def get_pred_data_with_time_comp(
    df: pd.DataFrame,
    history: int,
    snapshot,
) -> pd.DataFrame:
    """Based on the price database,
    add the time component positional encoding

    :param df: the dataframe in the window
    :type df: pd.DataFrame
    """
    if snapshot is not None:
        df = df.loc[lambda f: f.index <= snapshot]

    df_range = df.iloc[:history]

    # Companies
    companies = list(set(df_range.columns.get_level_values(0)) - {"diff"})

    # cumulated the time component
    df_range.iloc[0, -1] = 0
    df_order = df_range.assign(order=lambda x: x["diff"].cumsum())

    return df_order, companies


def normalize_and_diff(df_prices_raw: pd.DataFrame, test_days: int = 0):

    df = df_prices_raw.copy()
    # Extract the second level of columns which contain 'price'
    price_cols = [col for col in df.columns if "price" in col[1]]
    means_stds = {}

    # Standardize each 'price' column
    for col in price_cols:
        # mean = df.iloc[test_days:][col].mean()
        # std = df.iloc[test_days:][col].std()
        df[col] = (df[col] / df[col].shift(-1)) - 1
        # df[col] = df[col] - df[col].shift(-1)
        # means_stds[col[0]] = (mean, std)

    return df, means_stds
