import os
import glob
import pandas as pd
import yaml
import streamlit as st


# Function to display the prediction page
def prediction_page():
    st.title("Run Future Predictions")

    st.header("Model Summary")
    df_model = extract_model_info(Path("models"))
    st.dataframe(df_model)

    st.header("Model Results")
    df_model_str = df_model.copy()
    df_model_str["select"] = (
        df_model_str["model_name"] + " - " + df_model_str["id"].astype("str")
    )
    with st.form("model_res"):
        query = st.multiselect("Please select model ids to run", df_model_str["select"])
        submit_button = st.form_submit_button("Run the predictions")

    if submit_button:
        st.write("Getting the data")

        st.write(
            f"{df_model_str.loc[lambda f: f['select'].isin(query), 'id'].tolist()}"
        )


def extract_model_info(models_dir):
    # Step 1: Find the correct directories
    model_dirs = [d for d in glob.glob(f"{models_dir}/*") if os.path.isdir(d)]
    valid_dirs = []
    for d in model_dirs:
        pt_files = glob.glob(f"{d}/*.pt")
        yml_files = glob.glob(f"{d}/*.yml")
        csv_files = glob.glob(f"{d}/*.csv")
        if len(pt_files) == 3 and len(yml_files) == 1 and len(csv_files) == 1:
            valid_dirs.append(d)

    # Step 2: Extract information from each directory
    model_info = []
    for d in valid_dirs:
        training_date, model_name = os.path.basename(d).split("_")

        # Read YAML file for stocks and parameters
        with open(glob.glob(f"{d}/*.yml")[0], "r") as yml_file:
            config = yaml.safe_load(yml_file)
            stocks = config["ingest"].get("target_stocks", [])
            use_gnn = config["hyperparams"].get("use_gnn")

        # Read CSV file for best loss
        csv_file = pd.read_csv(glob.glob(f"{d}/*.csv")[0])
        best_loss = csv_file["test_loss"].min()

        # Append to list
        model_info.append(
            {
                "id": training_date,
                "model_name": model_name,
                "training_date": training_date,
                "stocks": stocks,
                "use_gnn": use_gnn,
                "best_loss": best_loss,
            }
        )

    # Step 3: Create DataFrame
    df = pd.DataFrame(model_info)

    df["training_date"] = pd.to_datetime(df["training_date"])
    df["training_date"] = df["training_date"].dt.strftime("%d-%b-%y")

    return df
