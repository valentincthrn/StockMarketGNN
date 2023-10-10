import logging
from tqdm import tqdm
from DadosAbertosBrasil import ipea
import pandas as pd
from datetime import datetime

from typing import Dict
from src.utils.db import DBInterface

logger = logging.getLogger(__name__)


def ingest_macroeco_data(target_indicators: Dict[str, str], db: DBInterface) -> None:
    # get the already exist
    macrodata_existing = db.execute_statement(
        "SELECT indicators, MAX(quote_date) FROM macro GROUP BY indicators"
    )
    dict_macrodata_max_date = {res[0]: res[1] for res in macrodata_existing}

    for ind, codigo in tqdm(target_indicators.items(), desc="MacroEco Indicators"):
        # Get Data
        df_ind = pd.DataFrame(ipea.Serie(codigo).valores)
        df_ind = df_ind.rename(columns={"data": "quote_date"}).dropna()
        df_ind["quote_date"] = pd.to_datetime(df_ind["quote_date"], errors="coerce")
        df_ind["indicators"] = ind
        df_ind = df_ind[["quote_date", "indicators", "valor"]].sort_values("quote_date")

        # forward fill missing values
        df_ind["valor"].fillna(method="ffill", inplace=True)

        # transform into a variation
        df_ind["valor"] = (
            ((df_ind["valor"] - df_ind["valor"].shift(1)) / df_ind["valor"].shift(1))
            .mul(100)
            .round(3)
        )

        # filter to get new points
        max_date_indic = dict_macrodata_max_date.get(ind)
        df_ind_new = df_ind.copy()

        if not max_date_indic is None:
            # max_date_indic = datetime.strptime(max_date_indic, "%Y-%m-%d")
            df_ind_new = df_ind.loc[lambda f: f["quote_date"] > max_date_indic]

        if df_ind_new.empty:
            logger.info(f"{ind} macroeconomic data already up-to-date")
        else:
            db.df_to_sql(
                pdf=df_ind_new, tablename="macro", if_exists="append", index=False
            )
            logger.info(
                f"{ind}: {len(df_ind_new)} timestamps added from {df_ind_new['quote_date'].min()} loaded succesfully"
            )
