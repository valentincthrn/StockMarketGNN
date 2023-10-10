import logging

from typing import List
from src.utils.db import DBInterface

logger = logging.getLogger(__name__)


def ingest_macroeco_data(target_indicators: List[str], db: DBInterface) -> None:
    # get the already exist
    macrodata_existing = db.execute_statement(
        "SELECT indicators, MAX(quote_date) FROM macro GROUPBY indicators"
    )
    dict_macrodata_max_date = {res[0]: res[1] for res in macrodata_existing}

    if len(macrodata_existing) == 0:
        logger.info("First time loading macro data => LOADING ALL")

    return
