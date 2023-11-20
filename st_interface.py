import streamlit as st

from src.streamlit.ingest import ingest_data_page
from src.streamlit.build_model import build_model_page
from src.streamlit.prediction import prediction_page


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
