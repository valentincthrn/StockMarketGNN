import logging
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import streamlit as st
from typing import List, Union

from src.utils.db import DBInterface
from src.configs import RunConfiguration
from src.data_prep.utils import (
    add_time_component,
    get_pred_data_with_time_comp,
    normalize_and_diff,
)
from src.utils.common import PositionalEncoding

logger = logging.getLogger(__name__)


class DataPrep:
    CACHE_DATA: dict

    def __init__(
        self,
        config: Union[RunConfiguration, Path],
        db: DBInterface,
        device: str = "cuda",
        overwrite_params: dict = None,
    ) -> None:
        if isinstance(config, Path):
            self.config = RunConfiguration.from_yaml(config)
        elif isinstance(config, RunConfiguration):
            self.config = config
        else:
            raise ValueError("config must be a Path or a RunConfiguration object")
        self.db = db
        self.device = device

        if overwrite_params is not None:
            for k, v in overwrite_params.items():
                self.config.data_prep[k] = v

        project_path = Path(os.getcwd())
        self.data_path = project_path / "data"

    def get_data(self, st_progress=False):
        return self._extract_data(st_progress=st_progress)

    def _extract_data(self, st_progress):
        logger.info("Extracting Prices and Fundamentals Indicators")
        df_prices_with_fund, d_size = self._extract_prices_and_fund()

        df_macro = pd.DataFrame()
        if not self.config.ingest["macro_indicators"] is None:
            logger.info("Extracting Macro-Economical Indicators")
            df_macro = self._extract_macro()

        logger.info("Creating Data Dictionnary")
        (
            data,
            means_stds,
            quote_date_index_train,
            quote_date_index_test,
        ) = self._run_extraction(df_prices_with_fund, df_macro, st_progress)

        return data, means_stds, d_size, quote_date_index_train, quote_date_index_test

    def _extract_prices_and_fund(self):
        logger.info(">> Extracting Prices")
        df_prices = self._extract_prices()

        df_fund = pd.DataFrame()
        if not self.config.ingest["fundamental_indicators"] is None:
            logger.info(">> Extracting Fundamentals Indicators")
            df_fund = self._extract_fund()

            df_prices["year"] = df_prices.index.year
            df_merge = df_prices.merge(
                df_fund,
                how="left",
                on="year",
            ).drop("year", axis=1)

            df_merge.set_index(df_prices.index, inplace=True)

        else:
            df_merge = df_prices.copy()

        multi = pd.MultiIndex.from_tuples(
            [tuple(col.split("_")) for col in df_merge.columns]
        )

        df_merge.columns = multi

        df_merge = df_merge.reorder_levels([0, 1], axis=1).sort_index(axis=1)

        d_size = {}
        for comp in list(set(multi.get_level_values(0))):
            size = df_merge.loc[:, (comp, slice(None))].shape[1]
            d_size[comp] = size
            if not self.config.ingest["fundamental_indicators"] is None:
                logger.info(f">>> {comp} has {size-1} fundamentals indicators")

        df_merge["diff"] = (
            -(df_merge.reset_index()["quote_date"].shift(1) - df_merge.index)
            .dt.days.fillna(0)
            .values
        )

        return df_merge, d_size

    def _extract_prices(self):
        df = self.db.read_sql(query="SELECT * FROM stocks")

        df_symbol = df[lambda f: f["symbol"].isin(self.config.ingest["target_stocks"])]

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

    def _extract_fund(self):
        df_fund = pd.read_csv(self.data_path / "company_indicators_top5_banks.csv")

        TARGETS_LOW = [
            col.lower().replace(".sa", "")
            for col in self.config.ingest["target_stocks"]
        ]
        COMP_IND = self.config.ingest["fundamental_indicators"]

        if COMP_IND is None:
            return pd.DataFrame()

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

    def _extract_macro(self):
        df_macro = self.db.read_sql(query="SELECT * FROM macro")

        macro_to_keep = self.config.ingest["macro_indicators"]
        df_macro = df_macro.loc[lambda f: f["indicators"].isin(macro_to_keep)]

        df_macro = df_macro.loc[lambda f: abs(f["valor"]) != np.inf]

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
        df_prices_with_fund: pd.DataFrame,
        df_macro: pd.DataFrame,
        st_progress: bool,
    ):
        if not df_macro.empty:
            df_macro = (
                pd.DataFrame(df_prices_with_fund.reset_index()["quote_date"])
                .merge(df_macro, on="quote_date", how="left")
                .fillna(method="backfill")
                .fillna(0)
                .set_index("quote_date")
            )

        hyperparam = self.config.data_prep

        pe = PositionalEncoding(hyperparam["pe_t"])
        N = df_prices_with_fund.shape[0]

        data = {"train": {}, "test": {}, "pred": {}, "macro": {}, "last_raw_price": {}}

        quote_date_index_train = []
        quote_date_index_test = []

        if st_progress:
            progress_text = "Data Preparation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

        tot_loop = (
            N
            - (hyperparam["start"] + hyperparam["history"])
            - hyperparam["horizon_forecast"]
        )
        tot_loop = tot_loop // hyperparam["step_every"]
        loop = 0

        # First part of the range with step size of 1
        range_part1 = range(hyperparam["horizon_forecast"], hyperparam["test_days"])

        # Second part of the range with the desired step size
        range_part2 = range(
            hyperparam["test_days"],
            N - (hyperparam["start"] + hyperparam["history"]),
            self.config.data_prep["step_every"],
        )

        df_prices_norm, means_stds = normalize_and_diff(
            df_prices_with_fund, hyperparam["test_days"]
        )

        # Combine the two ranges
        combined_range = list(range_part1) + list(range_part2)

        for t in tqdm(combined_range):
            loop += 1
            if st_progress:
                pct = min(int(loop * 100 / tot_loop), 100)
                my_bar.progress(pct, text=progress_text)
            # extract prices history, futures and companies name
            df_prices, y, last_prices, companies = add_time_component(
                df=df_prices_norm,
                time=t,
                df_raw_prices=df_prices_with_fund,
                history=hyperparam["history"],
                horizon=hyperparam["horizon_forecast"],
                min_points=hyperparam["min_points_history"],
            )

            if df_prices.drop(["diff", "order"], axis=1).empty:
                continue

            if torch.isnan(torch.tensor(y)).any():
                print("NaN Detected In Target")

            data["pred"][t] = dict(zip(companies, y))

            data["last_raw_price"][t] = dict(zip(companies, last_prices))

            if not df_macro.empty:
                macro_features = torch.tensor(
                    df_macro.iloc[t].values, dtype=torch.float
                ).to(self.device)

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
                tensor_col = torch.tensor(df_col.values, dtype=torch.float).to(
                    self.device
                )
                prices = tensor_col[:, :-1]
                pos = tensor_col[:, -1].unsqueeze(-1)

                # encode the position
                pos_enc = pe(pos)

                features = torch.concat((prices, pos_enc), dim=1)

                if torch.isnan(features).any():
                    print("NaN Detected In Features")

                # add features in a list to pad later
                d_t[col] = features

            if len(d_t) == 0:
                continue

            if loop == 1:
                st.info("Shape of the data: {}".format(d_t[companies[0]].shape))

            if t < hyperparam["test_days"]:
                data["test"][t] = d_t
                quote_date_index_test.append(df_prices.index[0])

            else:
                data["train"][t] = d_t
                quote_date_index_train.append(df_prices.index[0])

        if st_progress:
            st.success("Data Preparation Completed")
        return data, means_stds, quote_date_index_train, quote_date_index_test

    def get_future_data(self, st_progress=False, snapshot=None):
        return self._extract_future_data(st_progress=st_progress, snapshot=snapshot)

    def _extract_future_data(self, st_progress, snapshot):
        logger.info("Extracting Prices and Fundamentals Indicators")
        df_prices_with_fund, d_size = self._extract_prices_and_fund()

        df_macro = pd.DataFrame()
        if not self.config.ingest["macro_indicators"] is None:
            logger.info("Extracting Macro-Economical Indicators")
            df_macro = self._extract_macro()

        logger.info("Creating Data Dictionnary")
        if not df_macro.empty:
            df_macro = (
                pd.DataFrame(df_prices_with_fund.reset_index()["quote_date"])
                .merge(df_macro, on="quote_date", how="left")
                .fillna(method="backfill")
                .fillna(0)
                .set_index("quote_date")
            )

        hyperparam = self.config.data_prep

        pe = PositionalEncoding(hyperparam["pe_t"])

        data_to_pred = {"train": {}, "macro": {}, "last_raw_price": {}}

        df_prices_norm, _ = normalize_and_diff(df_prices_with_fund)

        # extract prices history, futures and companies name
        df_prices, companies = get_pred_data_with_time_comp(
            df=df_prices_norm,
            history=hyperparam["history"],
            snapshot=snapshot,
        )

        first_lines = df_prices_with_fund.loc[lambda f: f.index <= snapshot]
        cols = [c[0] for c in first_lines.columns]
        first_lines.columns = cols
        data_to_pred["last_raw_price"] = first_lines.iloc[0].to_dict()

        if not df_macro.empty:
            macro_features = torch.tensor(
                df_macro.iloc[0].values, dtype=torch.float
            ).to(self.device)

            if torch.isnan(macro_features).any():
                print("NaN Detected In Macro")

            data_to_pred["macro"] = macro_features

        d_t = {}

        past_data = {}

        for col in companies:
            # get company and order component
            df_col = (
                df_prices.loc[:, pd.IndexSlice[[col, "order"], :]]
                .dropna(subset=[(col, "price")])
                .fillna(0)
            )

            past_data[col] = df_col[(col, "price")]

            # tensor
            tensor_col = torch.tensor(df_col.values, dtype=torch.float).to(self.device)

            prices = tensor_col[:, :-1]
            pos = tensor_col[:, -1].unsqueeze(-1)

            # encode the position
            pos_enc = pe(pos)

            features = torch.concat((prices, pos_enc), dim=1)

            if torch.isnan(features).any():
                print("NaN Detected In Features")

            # add features in a list to pad later
            d_t[col] = features

            data_to_pred["train"] = d_t

        if st_progress:
            st.success("Data Preparation Completed")

        return data_to_pred, d_size, past_data, companies, df_prices_with_fund
