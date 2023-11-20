from pandas.tseries.offsets import BDay
import streamlit as st
import pandas as pd

from src.utils.db import DBInterface
from src.configs import DATE_FORMAT
from datetime import datetime


def extract_current_stocks_data():
    # Create a database instance
    db = DBInterface()

    # Query to get the latest quote date for each stock
    query_stocks = """
            SELECT symbol, MAX(quote_date) as last_quote_date
            FROM stocks
            GROUP BY symbol
            """

    query_metadata = """
            SELECT symbol, name
            FROM stocks_metadata
            """

    query_macro = """
            SELECT indicators, MAX(quote_date) as last_quote_date
            FROM macro
            GROUP BY indicators
            """

    df_stocks = db.read_sql(query=query_stocks)
    df_metadata = db.read_sql(query=query_metadata)
    df_macro = db.read_sql(query=query_macro)

    # Calculate the last business day
    last_bday = (datetime.now() - BDay(1)).strftime(DATE_FORMAT)

    # Check if each stock is up to date
    df_stocks["is_prices_uptodate"] = df_stocks["last_quote_date"] >= last_bday
    df_macro["is_prices_uptodate"] = df_macro["last_quote_date"] >= last_bday

    # get company name
    df_stocks_name = df_stocks.merge(df_metadata, on="symbol", how="left")

    # Fill NaN
    df_stocks_name["name"] = df_stocks_name["name"].fillna("Not Available")

    # Change date format
    df_stocks_name["last_quote_date"] = pd.to_datetime(
        df_stocks_name["last_quote_date"]
    )
    df_stocks_name["last_quote_date"] = df_stocks_name["last_quote_date"].dt.strftime(
        "%d-%b-%y"
    )

    # Reorder columns
    df_stocks_name = df_stocks_name[
        ["symbol", "name", "last_quote_date", "is_prices_uptodate"]
    ]

    return df_stocks_name, df_macro
