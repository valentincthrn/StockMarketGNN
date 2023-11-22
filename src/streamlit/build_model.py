import streamlit as st
from pathlib import Path
import torch

from src.configs import RunConfiguration
from src.configs import GROUPS
from src.utils.db import DBInterface
from src.streamlit.ingest import extract_current_stocks_data
from src.data_prep.main import DataPrep
from src.model.main import run_gnn_model


# Function to display the model building page
def build_model_page():
    st.title("Build New Model")

    st.subheader("Modelr Selections")
    # Example: Parameters for building a model
    history = st.slider("History", 0, 2000, 730)
    horizon_forecast = st.slider("Horizon Forecast", 0, 200, 90)
    overwrite_params = {
        "history": history,
        "horizon_forecast": horizon_forecast,
    }
    use_gnn = st.checkbox("Use GNN")
    epochs = st.slider("Epochs", 0, 100, 5)

    st.subheader("Features Selections")
    overwrite_params_model = {
        "use_gnn": use_gnn,
        "epochs": epochs,
    }
    stocks_options = st.selectbox(
        "Which groups of stocks do you want to see?", ("Banks", "Distincts", "Others")
    )

    use_fundamental = False
    if stocks_options == "Others":
        # Call the function to get the stock status DataFrame
        stock_status_df, _ = extract_current_stocks_data()
        stocks_selected = st.multiselect(
            "Select the groups of stocks to use", stock_status_df["symbol"].tolist()
        )

        check_in = [x for x in stocks_selected if x in GROUPS["Banks"]]
        if len(check_in) == 0:
            st.info("Can't use Fundamental for tickers not in the groups")
        else:
            st.info(
                f"Can use Fundamental ONLY for tickers >> {check_in} << in the groups"
            )
            use_fundamental = st.checkbox("Use Fundamental")

    if stocks_options == "Banks":
        stocks_selected = GROUPS["Banks"]
        use_fundamental = st.checkbox("Use Fundamental")

    if stocks_options == "Distincts":
        stocks_selected = GROUPS["Distincts"]
        use_fundamental = st.checkbox("Use Fundamental")

    fund_indicators = None
    if use_fundamental:
        fund_indicators = st.multiselect(
            "Select the Fundamental Indicators to use",
            ["P/L", "PL/ATIVOS", "M. EBIT", "ROA", "CAGR LUCROS 5 ANOS"],
            default=["P/L", "PL/ATIVOS", "M. EBIT", "ROA", "CAGR LUCROS 5 ANOS"],
        )

    use_macro = st.checkbox("Use Macro")
    macros = None
    if use_macro:
        macros = st.multiselect(
            "Select the Fundamental Indicators to use",
            ["Risco-Brasil", "PIB", "Dolar", "Selic Over", "IPCA"],
            default=["Risco-Brasil", "PIB", "Dolar", "Selic Over", "IPCA"],
        )
    st.subheader("Model Training")
    with st.form("model_train"):
        query = st.text_input("Please enter the experiment name")
        submit_button = st.form_submit_button("Train the model")

    if submit_button:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Define config file
        config = RunConfiguration.from_yaml(Path("params/run_config.yml"))

        config.ingest["target_stocks"] = stocks_selected
        config.ingest["macro_indicators"] = macros
        config.ingest["fundamental_indicators"] = fund_indicators

        data_prep = DataPrep(
            config=config,
            db=DBInterface(),
            device=device,
            overwrite_params=overwrite_params,
        )

        (
            data,
            d_size,
            quote_date_index_train,
            quote_date_index_test,
        ) = data_prep.get_data(st_progress=True)
        st.info(f"Data Feature For Each Company: {d_size}")

        # get the data
        run_gnn_model(
            data=data,
            d_size=d_size,
            dt_index=(quote_date_index_train, quote_date_index_test),
            exp_name=query,
            device=device,
            config=data_prep.config,
            st_plot=True,
            overwrite_dataprep=overwrite_params,
            overwrite_hyperparams=overwrite_params_model,
        )
