import logging
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

from src.utils.db import DBInterface
from src.configs import RunConfiguration
from src.data_prep.utils import add_time_component
from src.utils.common import PositionalEncoding

logger = logging.getLogger(__name__)


class DataPrep:
    CACHE_DATA: dict

    def __init__(self) -> None:
        project_path = Path(os.getcwd())
        self.data_path = project_path / "data"

    def get_data(self, run_config: RunConfiguration, db: DBInterface):
        if self.CACHE_DATA.get(run_config) is None:
            logger.info("No data found based on this config, creating it...")
            data = self.extract_data(run_config, db)
            self.CACHE_DATA[run_config] = data
        return self.CACHE_DATA.get(run_config)

    def extract_data(self, run_config: RunConfiguration, db: DBInterface):
        logger.info("Extracting Prices and Fundamentals Indicators")
        df_prices_with_fund = self._extract_prices_and_fund(run_config, db)

        logger.info("Extracting Macro-Economical Indicators")
        df_macro = self._extract_macro(run_config)

        logger.info("Creating Data Dictionnary")
        data = self._run_extraction(run_config, df_prices_with_fund, df_macro)

        return data

    def _extract_prices_and_fund(self, run_config: RunConfiguration, db: DBInterface):
        logger.info(">> Extracting Prices")
        df_prices = self._extract_prices(run_config, db)

        logger.info(">> Extracting Fundamentals Indicators")
        df_fund = self._fundamental_indicators(run_config)

        logger.info(">> Merging Prices and Fundamentals Indicators")

        df_prices["year"] = df_prices.index.year

        df_merge = df_prices.merge(
            df_fund,
            how="left",
            on="year",
        ).drop("year", axis=1)

        df_merge.set_index(df_prices.index, inplace=True)

        multi = pd.MultiIndex.from_tuples(
            [tuple(col.split("_")) for col in df_merge.columns]
        )

        df_merge.columns = multi

        df_merge = df_merge.reorder_levels([0, 1], axis=1).sort_index(axis=1)

        df_merge["diff"] = (
            -(df_merge.reset_index()["quote_date"].shift(1) - df_merge.index)
            .dt.days.fillna(0)
            .values
        )

        return df_merge

    def _extract_prices(self, run_config: RunConfiguration, db: DBInterface):
        df = db.read_sql(query="SELECT * FROM stocks")

        df_symbol = df[lambda f: f["symbol"].isin(run_config.ingest.targets)]

        assert not df_symbol.empty, "The dataframe is empty. Ingest the data before"

        df_pivot = df_symbol.pivot(
            index="quote_date",
            columns="symbol",
            values=["close"],
        )
        df_pivot.columns = [
            stock.lower().replace(".sa", "") + "_price"
            for _, stock in df_pivot.columns.values
        ]
        # focus only on close price for now
        df_close = df_pivot.filter(like="price").sort_index(ascending=False)

        return df_close

    def _extract_fund(self, run_config: RunConfiguration):
        df_fund = pd.read_csv(self.data_path / "company_indicators_top5_banks.csv")

        TARGETS_LOW = [
            col.lower().replace(".sa", "") for col in run_config.ingest.targets
        ]
        COMP_IND = run_config.ingest.fundamental_indicators

        df_ind = df_fund.loc[
            lambda f: f["stock"].isin(TARGETS_LOW) & f["indicators"].isin(COMP_IND)
        ]

        df_ind["quote_date"] = pd.to_datetime(df_ind["quote_date"])
        df_ind["col"] = (
            df_ind["stock"]
            + "_"
            + df_ind["indicators"]
            .str.lower()
            .str.replace(".", "")
            .str.replace(" ", "-")
        )
        df_pivot = df_ind.pivot(index="quote_date", columns="col", values="valor")
        df_pivot["year"] = df_pivot.index.year

        return df_pivot

    def _extract_macro(self, run_config: RunConfiguration, db: DBInterface):
        df_macro = db.read_sql(query="SELECT * FROM macro")

        macro_to_keep = run_config.ingest.macro_indicators.keys()
        df_macro = df_macro[macro_to_keep]

        assert not df_macro.empty, "The macro is empty. Ingest the data before"

        df_macro_sorted = (
            df_macro.pivot(index="quote_date", columns="indicators", values="valor")
            .sort_index(ascending=False)
            .reset_index()
        )
        df_macro_sorted["quote_date"] = pd.to_datetime(df_macro_sorted["quote_date"])

        return df_macro_sorted

    def _run_extraction(
        self,
        run_config: RunConfiguration,
        df_prices_with_fund: pd.DataFrame,
        df_macro: pd.DataFrame,
    ):
        df_macro = (
            pd.DataFrame(df_prices_with_fund.reset_index()["quote_date"])
            .merge(df_macro, on="quote_date", how="left")
            .fillna(method="backfill")
            .fillna(0)
            .set_index("quote_date")
        )

        hyperparam = run_config.data_prep

        pe = PositionalEncoding(hyperparam["pe_t"])
        N = df_prices_with_fund.shape[0]

        data = {"train": {}, "test": {}, "pred": {}, "macro": {}}

        for t in tqdm(range(hyperparam["horizon_forecast"], N - hyperparam["history"])):
            # extract prices history, futures and companies name
            df_prices, y, companies = add_time_component(
                df=df_prices_with_fund,
                time=t,
                history=hyperparam["history"],
                horizon=hyperparam["horizon_forecast"],
            )

            data["pred"][t] = dict(zip(companies, y))

            macro_features = torch.tensor(df_macro.iloc[t].values, dtype=torch.float)

            if torch.isnan(macro_features).any():
                print("NaN Detected In Macro")

            data["macro"][t] = macro_features

            d_t = {}

            for col in companies:
                # get company and order component
                df_col = (
                    df_prices.loc[:, pd.IndexSlice[[col, "order"], :]]
                    .dropna(subset=[(col, "price")])
                    .fillna(0)
                )

                # tensor
                tensor_col = torch.tensor(df_col.values, dtype=torch.float)
                prices = tensor_col[hyperparam["horizon_forecast"] :, :-1]
                pos = tensor_col[hyperparam["horizon_forecast"] :, -1].unsqueeze(-1)

                # encode the position
                pos_enc = pe(pos)

                features = torch.concat((prices, pos_enc), dim=1)

                if torch.isnan(features).any():
                    print("NaN Detected In Features")

                # add features in a list to pad later
                d_t[col] = features

            if len(d_t) == 0:
                continue

            if t < hyperparam["test_days"]:
                data["test"][t] = d_t
            else:
                data["train"][t] = d_t

        return data


def get_data(run_config, data_prep):
    return data_prep.get_data(run_config)
