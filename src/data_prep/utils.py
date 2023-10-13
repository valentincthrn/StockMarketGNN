import pandas as pd


def add_time_component(
    df: pd.DataFrame, time: int, history: int, horizon: int
) -> pd.DataFrame:
    """Based on the price database,
    add the time component positional encoding

    :param df: the dataframe in the window
    :type df: pd.DataFrame
    """

    # Filter the dataframe
    df_range = df[time - horizon : time + history]

    # get prices
    df_range_prices = df_range.xs("price", axis=1, level=1, drop_level=True)

    # Keep symbol where there is all predicted prices AND at least one history price
    exist_price_history = ~df_range_prices.isna().all()
    exist_predicted_price = ~df_range_prices.isna().any()
    keep = exist_price_history & exist_predicted_price
    keep = pd.concat([keep, pd.Series({"diff": True})], axis=0)

    df_range = df_range[keep.index[keep]]

    # Companies
    companies = list(set(df_range.columns.get_level_values(0)) - {"diff"})

    # Define y
    y_list = df_range_prices.iloc[:horizon][companies].round(5).T.values.tolist()

    # cumulated the time component
    df_range.iloc[:horizon, -1] = 0
    df_order = df_range.assign(order=lambda x: x["diff"].cumsum())

    return df_order, y_list, companies
