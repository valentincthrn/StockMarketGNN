from pandas.tseries.offsets import BDay
import streamlit as st
import pandas as pd

from src.utils.db import DBInterface
from src.configs import DATE_FORMAT
from datetime import datetime

from src.configs import GROUPS
from src.ingest.stock import ingest_data_local
from src.ingest.macroeco import ingest_macroeco_data


def extract_current_stocks_data():
    # Create a database instance
    db = DBInterface()

    # Query to get the latest quote date for each stock
    query_stocks = """
            SELECT symbol, MAX(quote_date) as last_quote_date
            FROM stocks
            GROUP BY symbol
            """

    query_metadata = """
            SELECT symbol, name
            FROM stocks_metadata
            """

    query_macro = """
            SELECT indicators, MAX(quote_date) as last_quote_date
            FROM macro
            GROUP BY indicators
            """

    df_stocks = db.read_sql(query=query_stocks)
    df_metadata = db.read_sql(query=query_metadata)
    df_macro = db.read_sql(query=query_macro)

    # Calculate the last business day
    last_bday = (datetime.now() - BDay(1)).strftime(DATE_FORMAT)

    # Check if each stock is up to date
    df_stocks["is_prices_uptodate"] = df_stocks["last_quote_date"] >= last_bday
    df_macro["is_prices_uptodate"] = df_macro["last_quote_date"] >= last_bday

    # get company name
    df_stocks_name = df_stocks.merge(df_metadata, on="symbol", how="left")

    # Fill NaN
    df_stocks_name["name"] = df_stocks_name["name"].fillna("Not Available")

    # Change date format
    df_stocks_name["last_quote_date"] = pd.to_datetime(
        df_stocks_name["last_quote_date"]
    )
    df_stocks_name["last_quote_date"] = df_stocks_name["last_quote_date"].dt.strftime(
        "%d-%b-%y"
    )

    # Reorder columns
    df_stocks_name = df_stocks_name[
        ["symbol", "name", "last_quote_date", "is_prices_uptodate"]
    ]

    return df_stocks_name, df_macro


def interactive_df_st(df, to):
    df = st.data_editor(
        df,
        column_config={
            "to_ingest": st.column_config.CheckboxColumn(
                "To Ingest?",
                help=f"Select the {to} to **ingest** the data",
                default=True,
            )
        },
        disabled=["widgets"],
        hide_index=True,
    )
    return df


# Function to display the data ingestion page
def ingest_data_page():
    db = DBInterface()

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

    def initialize_state():
        if "df_stocks_ingestion" not in st.session_state:
            st.session_state.df_stocks_ingestion = stock_status_df[["symbol", "name"]]
            st.session_state.df_stocks_ingestion["to_ingest"] = True

    # Initialize the session state
    initialize_state()

    # Use st.columns to create a layout with 2 columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stocks Ingestion")
        df_stocks_ingestion = st.session_state.df_stocks_ingestion
        interactive_df_st(df_stocks_ingestion, "stocks")
    with col2:
        st.subheader("Macro Ingestion")
        df_macro_ingestion = macro_status_df[["indicators"]]
        df_macro_ingestion["to_ingest"] = True

        df_macro_ingestion = interactive_df_st(df_macro_ingestion, "macro")

    # Use st.columns to create a layout with 2 columns
    col1_bt, col2_bt = st.columns(2)
    banks_button = col1_bt.button("Preselect Only 'Banks'", use_container_width=True)
    distinct_button = col2_bt.button(
        "Preselect Only 'Distinct'", use_container_width=True
    )
    # Button logic
    if banks_button:
        st.session_state.df_stocks_ingestion["to_ingest"] = False
        st.session_state.df_stocks_ingestion.loc[
            st.session_state.df_stocks_ingestion["symbol"].isin(GROUPS["Banks"]),
            "to_ingest",
        ] = True
        st.experimental_rerun()

    if distinct_button:
        st.session_state.df_stocks_ingestion["to_ingest"] = False
        st.session_state.df_stocks_ingestion.loc[
            st.session_state.df_stocks_ingestion["symbol"].isin(GROUPS["Distinct"]),
            "to_ingest",
        ] = True
        st.experimental_rerun()

    st.subheader("Add New Stocks")

    # Persistent list for new stocks
    if "new_stock_data_list" not in st.session_state:
        st.session_state["new_stock_data_list"] = []

    # Place the reset button outside the form
    reset_button = st.button("Reset Form")
    if reset_button:
        st.session_state.new_stock_data_list = []

    with st.form("add_stock_form"):
        st.text("Write The New Stock")
        new_stock_data = st.text_input("Stock Symbol")
        submit_button = st.form_submit_button("Add Stock")

        if submit_button:
            st.session_state.new_stock_data_list.append(new_stock_data)

        if reset_button:
            st.session_state.new_stock_data_list = []

    st.subheader("Final List of Stocks To Ingest")
    df_to_ingest = st.session_state.df_stocks_ingestion.loc[
        lambda f: f["to_ingest"], ["symbol"]
    ]

    new_stock_to_ingest = pd.DataFrame(
        st.session_state.new_stock_data_list, columns=["symbol"]
    )

    if not new_stock_to_ingest.empty:
        df_to_ingest = df_to_ingest.append(new_stock_to_ingest, ignore_index=True)

    # complete by empty value
    st.table(df_to_ingest)

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
