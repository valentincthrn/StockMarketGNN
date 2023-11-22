from pathlib import Path
import logging

from src.utils.db import DBInterface
from src.configs import RunConfiguration
from src.ingest.stock import ingest_data_local
from src.ingest.macroeco import ingest_macroeco_data

logger = logging.getLogger(__name__)


def ingest_data(config: RunConfiguration, force: bool = False) -> None:
    """
    Set up a new local instance SQLite database
    and update stocks prices and macroeconomical indicators
    with the most recent files

    :param config_path: path to the configuration file
    :type config_path: Path
    :param force: flag if we should force re-generating the database, defaults to False
    :type force: bool, optional
    """
    # initialize the database
    db = DBInterface()
    db.initialize_db(force)

    logger.info("> INGEST DATA PRICES")
    ingest_data_local(target_list=config.ingest["target_stocks"], db=db)

    if config.ingest["macro_indicators"] is None:
        logger.info("> NO MACROECO INDICATORS!")
        pass

    else:
        logger.info("> INGEST MACROECO INDICATORS")
        ingest_macroeco_data(target_indicators=config.ingest["macro_indicators"], db=db)
