from pathlib import Path
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime
from urllib.error import HTTPError
from typing import List

from src.utils.db import DBInterface
from src.configs import RunConfiguration, DATE_FORMAT

logger = logging.getLogger(__name__)


def ingest_data_local(target_list: List[str], db: DBInterface) -> None:
    """
    Set up a new local instance SQLite database
    and update it with the most recent files

    :param config_path: path to the configuration file
    :type config_path: Path
    :param force: flag if we should force re-generating the database, defaults to False
    :type force: bool, optional
    """

    # get the already exist
    symbol_metadata_existing = [
        res[0]
        for res in db.execute_statement("SELECT DISTINCT symbol FROM stocks_metadata")
    ]
    symbol_stocks_existing = [
        res[0] for res in db.execute_statement("SELECT DISTINCT symbol FROM stocks")
    ]

    # for each symbol we want to ingest
    for s in target_list:
        # get the ticker
        ticker = yf.Ticker(s)

        # check if the ticker exists
        try:
            info = ticker.info
        except HTTPError:
            logger.info(f"{s} is not an existing symbol in yahoo finance")
            continue

        # new symbol to load metadata
        if s not in symbol_metadata_existing:
            # loading metadata if not exist
            db.commit_many(
                "INSERT INTO stocks_metadata "
                + "(symbol,name,industry,sector,business_summary) "
                + "values (?,?,?,?,?)",
                (
                    (
                        s,
                        info.get("longName"),
                        info.get("industry"),
                        info.get("sector"),
                        info.get("longBusinessSummary"),
                    ),
                ),
            )
            logger.info(f"{s} metadata loaded succesfully")

        # new symbol to load stocks information
        if s not in symbol_stocks_existing:
            # get stock prices with the max period
            df_history = prepare_history(ticker)
            db.df_to_sql(
                pdf=df_history, tablename="stocks", if_exists="append", index=False
            )
            logger.info(
                f"{s}: {len(df_history)} timestamps added from {df_history['quote_date'].min()} loaded succesfully"
            )

        # otherwise, update the stocks informations
        else:
            update_db(db, ticker)


def prepare_history(ticker: yf.Ticker) -> pd.DataFrame:
    """Given a ticker, prepare the data to fill in the SQL database

    :param ticker: the ticker to prepare the stock data
    :type ticker: yf.Ticker
    :return: the dataframe prepared to be added
    :rtype: pd.DataFrame
    """

    df_history = ticker.history(period="max").reset_index()[
        ["Date", "Open", "Close", "High", "Low", "Volume"]
    ]
    df_history.columns = ["quote_date", "open", "close", "high", "low", "volume"]
    df_history["quote_date"] = df_history["quote_date"].dt.strftime(DATE_FORMAT)
    df_history.insert(loc=0, column="symbol", value=ticker.info.get("symbol"))

    current_date_minus_2d = (datetime.now() - pd.Timedelta(days=2)).strftime(
        DATE_FORMAT
    )

    df_history = df_history[df_history["quote_date"] <= current_date_minus_2d]

    return df_history


def update_db(db: DBInterface, ticker: yf.Ticker, limit: int = 2) -> None:
    """Given a ticker, update its prices in the SQL if needed.
    We load the prices until 2 days ago to ensure that the prices ingested are reliable

    :param db: the SQL instance
    :type db: DBInterface
    :param ticker: the name of the ticker to update
    :type ticker: yf.Ticker
    """

    # Get the officiel ticker name based on his symbol
    symbol = ticker.info.get("symbol")

    # Get the last date existing in the database of the ticker
    quote_dates = [
        res[0]
        for res in db.execute_statement(
            f"SELECT quote_date FROM stocks WHERE symbol='{symbol}'"
        )
    ]

    # Set the maximum date we will ingest: TODAY - 2 Days
    current_date_minus_2d = (datetime.now() - pd.Timedelta(days=limit)).strftime(
        DATE_FORMAT
    )

    # If this last day has already be ingested: pass
    if max(quote_dates) == current_date_minus_2d:
        logger.info(f"{symbol} stocks prices up-to-date")
        return

    # otherwise, update it to the
    df_history = prepare_history(ticker)
    df_to_add = df_history.loc[~df_history["quote_date"].isin(quote_dates)]
    db.df_to_sql(pdf=df_to_add, tablename="stocks", if_exists="append", index=False)
    logger.info(f"{len(df_to_add)} new timestamps add for {symbol} updated succesfully")
