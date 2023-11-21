import os
import glob
import pandas as pd
import yaml
import streamlit as st
from pathlib import Path
import torch

from src.prediction.main import (
    prepare_data_for_prediction,
    initialize_models,
    initialize_weights,
)
from src.prediction.plot import plot_stock_predictions

from src.model.utils import run_lstm_separatly, run_mlp_heads_separatly


# Function to display the prediction page
def prediction_page():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.title("Run Future Predictions")

    st.header("Model Summary")
    df_model = extract_model_info(Path("models"))

    st.header("Model Results")
    df_model_str = df_model.copy()
    df_model_str["select"] = (
        df_model_str["model_name"] + " - " + df_model_str["id"].astype("str")
    )
    with st.form("model_res"):
        query = st.multiselect("Please select model ids to run", df_model_str["select"])
        submit_button = st.form_submit_button("Run the predictions")

    models_to_test = df_model_str.loc[
        lambda f: f["select"].isin(query), "select"
    ].tolist()

    if submit_button:
        for selected_model in models_to_test:
            st.info(f"Getting the data for {selected_model}")

            df_sub = df_model_str.loc[
                lambda f: f["select"] == selected_model, ["id", "model_name"]
            ]
            use_gnn = df_model_str.loc[
                lambda f: f["select"] == selected_model, ["use_gnn"]
            ].values[0]

            subfolder_name = (
                df_sub["id"].values[0] + "_" + df_sub["model_name"].values[0]
            )

            run_config_path = Path(f"models/{subfolder_name}/run_config.yml")

            data, d_size, past_data = prepare_data_for_prediction(run_config_path)

            if len(data["macro"]) == 0:
                macro_size = 0
            else:
                macro_size = data["macro"].shape[0]

            models_trio_init = initialize_models(
                run_config_path, subfolder_name, device, d_size, macro_size
            )

            model_trio = initialize_weights(models_trio_init, subfolder_name)

            data_t = data["train"]
            pred_t = None
            macro = data["macro"]
            if len(macro) == 0:
                macro = None

            # PHASE 1: LSTM EXTRACTION
            features_extracted, comps = run_lstm_separatly(
                model_trio[0], data_t, device
            )

            # PHASE 2: GNN EXTRACTION
            if use_gnn:
                features_encoded = model_trio[1](features_extracted)
            else:
                features_encoded = features_extracted

            # PHASE 3: MLP HEAD EXTRACTION
            pred = run_mlp_heads_separatly(
                model_trio[2],
                features_encoded,
                comps,
                pred_t,
                macro,
                device,
                to_pred=True,
            )

            plot_stock_predictions(past_data, pred, 400)


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
