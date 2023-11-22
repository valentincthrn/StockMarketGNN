import streamlit as st
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from typing import List
import pandas as pd

from src.configs import GROUPS
from src.utils.db import DBInterface
from src.streamlit.ingest import extract_current_stocks_data
from src.data_prep.main import DataPrep

# from src.model.main import run_gnn_model


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
    epochs = st.slider("Epochs", 0, 100, 50)

    st.subheader("Features Selections")
    overwrite_params_model = {
        "use_gnn": use_gnn,
        "epochs": epochs,
    }
    stocks_options = st.selectbox(
        "Which groups of stocks do you want to see?", ("Banks", "Distincts", "Others")
    )

    if stocks_options == "Others":
        # Call the function to get the stock status DataFrame
        stock_status_df, _ = extract_current_stocks_data()
        stocks_selected = st.multiselect(
            "Select the groups of stocks to use", stock_status_df["symbol"].tolist()
        )
        st.info("Can't use Fundamental for others groups!")

    use_fundamental = False
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

        data_prep = DataPrep(
            config_path=Path("params/run_config.yml"),
            db=DBInterface(),
            target_stocks=stocks_selected,
            fund_indicators=fund_indicators,
            macros=macros,
            device=device,
            overwrite_params=overwrite_params,
        )

        (
            data,
            d_size,
            quote_date_index_train,
            quote_date_index_test,
        ) = data_prep.get_data(st_progress=True)

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


def plot_training_pred(df_pred: pd.DataFrame, comps: List):
    # Set the style of matplotlib to 'ggplot' for better aesthetics
    plt.style.use("ggplot")

    fig, axs = plt.subplots(len(comps), 1, figsize=(10, 5 * len(comps)))

    # Loop through each subplot and plot the data
    for j, comp in enumerate(comps):
        # Setting the background color to transparent
        axs[j].set_facecolor("none")
        axs[j].plot(
            df_pred.index,
            df_pred[comp + "_pred"],
            label="pred",
            linewidth=2,
        )
        axs[j].plot(
            df_pred.index,
            df_pred[comp + "_true"],
            label="true",
            linewidth=2,
        )
        axs[j].set_title(comp)
        axs[j].legend()

        # Making the frame invisible
        for spine in axs[j].spines.values():
            spine.set_visible(False)

        # Reducing the number of ticks and labels for a cleaner look
        axs[j].xaxis.set_major_locator(plt.MaxNLocator(5))
        axs[j].yaxis.set_major_locator(plt.MaxNLocator(5))

        # Making the grid lighter and set it to be behind the plots
        axs[j].grid(
            True,
            color="gray",
            linestyle="--",
            linewidth=0.5,
            alpha=0.7,
        )
        axs[j].set_axisbelow(True)

    # Remove the space between the subplots
    plt.tight_layout(pad=2)

    st.pyplot(fig)
