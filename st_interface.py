import streamlit as st
import pandas as pd

from src.streamlit.ingest import extract_current_stocks_data

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

        st.data_editor(
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

        st.data_editor(
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
        
    # Custom CSS to inject for styling the buttons
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #4CAF50; 
            color: white;
        }
        div.stButton > button:last-child {
            background-color: #f44336; 
            color: white;
            width: 33%;
        }
        </style>""", unsafe_allow_html=True)

    # Layout with two buttons
    col1, col2 = st.columns([3,1])
    with col1:
        ingest_btn = st.button("Ingest")
    with col2:
        remove_btn = st.button("Remove 3 last dates")

    # Button logic (as an example)
    if ingest_btn:
        st.write("Ingest button clicked")

    if remove_btn:
        st.write("Remove button clicked")

# Function to display the model building page
def build_model_page():
    st.title("Build New Model")
    # Example: Parameters for building a model
    param1 = st.slider("Parameter 1", 0, 100, 50)
    param2 = st.selectbox("Parameter 2", ["Option 1", "Option 2", "Option 3"])
    if st.button("Build Model"):
        st.write("Model built with parameters:", param1, param2)
        # Add your model building logic here

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
    page = st.sidebar.radio("Choose a Page", ["Ingest Data", "Build Model", "Predictions"])

    if page == "Ingest Data":
        ingest_data_page()
    elif page == "Build Model":
        build_model_page()
    elif page == "Predictions":
        prediction_page()

if __name__ == "__main__":
    main()
