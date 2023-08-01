from pathlib import Path
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime
from urllib.error import HTTPError

from src.utils.db import DBInterface
from src.configs import RunConfiguration, DATE_FORMAT

logger = logging.getLogger(__name__)


def ingest_data_local(config_path: Path, force: bool = False):
    """
    Set up a new local instance SQLite database
    and update it with the most recent files

    :param config_path: path to the configuration file
    :type config_path: Path
    :param force: flag if we should force re-generating the database, defaults to False
    :type force: bool, optional
    """

    # initialize the database
    db = DBInterface()
    db.initialize_db(force)

    config = RunConfiguration.from_yaml(config_path)

    # get the already exist
    symbol_metadata_existing = [
        res[0]
        for res in db.execute_statement("SELECT DISTINCT symbol FROM stocks_metadata")
    ]
    symbol_stocks_existing = [
        res[0] for res in db.execute_statement("SELECT DISTINCT symbol FROM stocks")
    ]

    # for each symbol
    for s in config.targets_to_predict:
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
    """Given a ticker, prepare the data to fill in the database

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

    return df_history


def update_db(db: DBInterface, ticker: yf.Ticker):
    symbol = ticker.info.get("symbol")

    # get last date
    quote_dates = [
        res[0]
        for res in db.execute_statement(
            f"SELECT quote_date FROM stocks WHERE symbol='{symbol}'"
        )
    ]

    current_date = datetime.now().strftime(DATE_FORMAT)

    # if the same date, it is fully updated
    if max(quote_dates) == current_date:
        logger.info(f"{symbol} stocks prices up-to-date")
        return

    # otherwise, update
    df_history = prepare_history(ticker)
    df_to_add = df_history.loc[~(lambda f: f["quote_date"].isin(quote_dates))]
    db.df_to_sql(pdf=df_to_add, tablename="stocks", if_exists="append", index=False)
    logger.info(f"{len(df_to_add)} new timestamps add for {symbol} updated succesfully")
