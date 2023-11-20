import streamlit as st
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from src.utils.db import DBInterface
from src.streamlit.ingest import extract_current_stocks_data
from src.ingest.stock import ingest_data_local
from src.ingest.macroeco import ingest_macroeco_data
from src.model.main import run_gnn_model
from src.data_prep.main import DataPrep


db = DBInterface()


# Function to display the data ingestion page
def ingest_data_page():
    st.title("Data Ingestion")

    # Call the function to get the stock status DataFrame
    stock_status_df, macro_status_df = extract_current_stocks_data()

    # Display the DataFrame in Streamlit
    st.header("Data Status")

    # Use st.columns to create a layout with 2 columns
    col1, col2 = st.columns(2)

    # Display df1 in the first column
    with col1:
        st.subheader("Stocks Status")
        st.dataframe(stock_status_df, use_container_width=True)
    with col2:
        st.subheader("Macro Status")
        st.dataframe(macro_status_df, use_container_width=True)

    # Display the DataFrame in Streamlit
    st.header("Ingestion")

    # Use st.columns to create a layout with 2 columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stocks Ingestion")
        df_stocks_ingestion = stock_status_df[["symbol", "name"]]
        df_stocks_ingestion["to_ingest"] = True

        df_stocks_ingestion = st.data_editor(
            df_stocks_ingestion,
            column_config={
                "to_ingest": st.column_config.CheckboxColumn(
                    "To Ingest?",
                    help="Select the stocks to **ingest** the data",
                    default=True,
                )
            },
            disabled=["widgets"],
            hide_index=True,
        )
    with col2:
        st.subheader("Macro Ingestion")
        df_macro_ingestion = macro_status_df[["indicators"]]
        df_macro_ingestion["to_ingest"] = True

        df_macro_ingestion = st.data_editor(
            df_macro_ingestion,
            column_config={
                "to_ingest": st.column_config.CheckboxColumn(
                    "To Ingest?",
                    help="Select the macro to **ingest** the data",
                    default=True,
                )
            },
            disabled=["widgets"],
            hide_index=True,
        )

    # Persistent list for new stocks
    if "new_stock_data_list" not in st.session_state:
        st.session_state["new_stock_data_list"] = []

    # Place the reset button outside the form
    reset_button = st.button("Reset Form")
    if reset_button:
        st.session_state.new_stock_data_list = []

    with st.form("add_stock_form"):
        st.subheader("Add New Stock")
        new_stock_data = st.text_input("Stock Symbol")
        submit_button = st.form_submit_button("Add Stock")

        if submit_button:
            st.session_state.new_stock_data_list.append(new_stock_data)

        if reset_button:
            st.session_state.new_stock_data_list = []

    st.subheader("Stocks To Ingest")

    df_to_ingest = df_stocks_ingestion.loc[lambda f: f["to_ingest"], ["symbol"]]
    new_stock_to_ingest = pd.DataFrame(
        st.session_state.new_stock_data_list, columns=["symbol"]
    )

    if not new_stock_to_ingest.empty:
        df_to_ingest = df_to_ingest.append(new_stock_to_ingest, ignore_index=True)

    # complete by empty value
    df_to_ingest_grid = df_to_ingest.copy()
    st.table(df_to_ingest_grid)

    ingest_stock_btn = st.button("Ingesting Stocks Data", use_container_width=True)
    ingest_macro_btn = st.button("Ingesting Macro Data", use_container_width=True)
    remove_btn = st.button(
        "TEST > Removing 3 last quote dates",
        use_container_width=True,
        type="primary",
    )

    if remove_btn:
        success = db.remove_last_three_quotes(
            df_to_ingest["symbol"].tolist()
        )  # Modify as needed
        if success:
            st.success(
                "Successfully removed the last three quotes for the selected symbols."
            )
        else:
            st.error("Failed to remove quotes. Please check logs for more details.")

    # Button logic (as an example)
    if ingest_stock_btn:
        st.write("Ingesting stocks ....")
        ingest_data_local(df_to_ingest["symbol"].tolist(), db=db, st_res=True)
        st.success("Successfully ingested for the selected symbols.")

    if ingest_macro_btn:
        st.write("Ingesting Macro Indicators ...")
        ingest_macroeco_data(
            df_macro_ingestion["indicators"].tolist(), db=db, st_res=True
        )
        st.success("Successfully ingested for the selected symbols.")


# Function to display the model building page
def build_model_page():
    st.title("Build New Model")
    # Example: Parameters for building a model
    history = st.slider("History", 0, 2000, 730)
    horizon_forecast = st.slider("Horizon Forecast", 0, 200, 90)
    overwrite_params = {
        "history": history,
        "horizon_forecast": horizon_forecast,
    }

    if st.button("Build Model"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        data_prep = DataPrep(
            config_path=Path("params/run_config.yml"),
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

        # get the data
        run_gnn_model(
            data=data,
            d_size=d_size,
            dt_index=(quote_date_index_train, quote_date_index_test),
            config_path=Path("params/run_config.yml"),
            exp_name="Test",
            device=device,
            st_plot=True,
        )


# Function to display the prediction page
def prediction_page():
    st.title("Run Future Predictions")
    # Example: List of models with descriptions
    models = {
        "Model 1": "Description of Model 1",
        "Model 2": "Description of Model 2",
        "Model 3": "Description of Model 3",
    }
    selected_model = st.selectbox("Select a Model", list(models.keys()))
    st.write(models[selected_model])
    if st.button("Run Predictions with Selected Model"):
        st.write("Running predictions using:", selected_model)
        # Add your prediction logic here


# Main app logic
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a Page", ["Ingest Data", "Build Model", "Predictions"]
    )

    if page == "Ingest Data":
        ingest_data_page()
    elif page == "Build Model":
        build_model_page()
    elif page == "Predictions":
        prediction_page()


if __name__ == "__main__":
    main()
