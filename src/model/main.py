from pathlib import Path
import logging
import mlflow
import pandas as pd
import numpy as np
from torch.nn import ModuleDict
import torch
from torch.optim import Adam, lr_scheduler
import random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import streamlit as st


from src.configs import RunConfiguration
from src.model.module import CompanyExtractor, MyGNN, MLPWithHiddenLayer
from src.model.utils import run_all
from src.utils.logs import log_errors

logger = logging.getLogger(__name__)


@log_errors
def run_gnn_model(
    data: pd.DataFrame,
    d_size: dict,
    dt_index: tuple,
    config_path: Path,
    exp_name: str,
    device: str,
    st_plot: bool = False,
    overwrite_dataprep: dict = None,
    overwrite_hyperparams: dict = None,
    base_path_save_csv: str = "data/",
):
    config = RunConfiguration.from_yaml(config_path)

    PROJECT_PATH = Path(os.getcwd())
    DATA_PATH = PROJECT_PATH / "data"

    if overwrite_dataprep is not None:
        for k, v in overwrite_dataprep.items():
            config.data_prep[k] = v
    if overwrite_hyperparams is not None:
        for k, v in overwrite_hyperparams.items():
            config.hyperparams[k] = v

    if len(data["macro"]) == 0:
        macro_size = 0
    else:
        macro_size = next(iter(data["macro"].values())).shape[0]

    try:
        exp_id = mlflow.create_experiment(exp_name)
    except:
        exp_id = mlflow.set_experiment(exp_name).experiment_id

    rid = pd.to_datetime("today").strftime("%Y%m%d%H%M%S")

    os.mkdir(DATA_PATH / rid)

    with mlflow.start_run(run_name=rid, experiment_id=exp_id) as run:
        logger.info("Initialize models, elements...")

        lstm_models = ModuleDict(
            {
                comp: CompanyExtractor(
                    size + config.data_prep["pe_t"],
                    config.hyperparams["out_lstm_size"],
                    device=device,
                )
                for comp, size in d_size.items()
            }
        )
        if config.hyperparams["use_gnn"]:
            in_channels_mlp = config.hyperparams["out_gnn_size"]
        else:
            in_channels_mlp = config.hyperparams["out_lstm_size"]

        mlp_heads = ModuleDict(
            {
                comp: MLPWithHiddenLayer(
                    in_channels_mlp + macro_size,
                    config.data_prep["horizon_forecast"],
                    device,
                )
                for comp in d_size.keys()
            }
        )

        my_gnn = MyGNN(
            in_channels=config.hyperparams["out_lstm_size"],
            out_channels=config.hyperparams["out_gnn_size"],
            device=device,
        )

        # Define a loss function and optimizer
        # criterion = torch.nn.MSELoss()  # Mean Squared Error Loss for regression tasks
        if config.hyperparams["use_gnn"]:
            list_gnn = list(my_gnn.parameters())
        else:
            list_gnn = []
        optimizer = Adam(
            list(lstm_models.parameters()) + list_gnn + list(mlp_heads.parameters()),
            lr=config.hyperparams["lr"],
        )

        # Learning rate scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            patience=config.hyperparams["patience"],
            factor=config.hyperparams["factor"],
            verbose=True,
        )

        logger.info("Training the model...")
        int_subset = int(config.hyperparams["pct_subset"] * len(data["train"]))

        mlflow.log_params(config.data_prep)
        mlflow.log_params(config.hyperparams)

        subset_stocks = "random"  # Top5 Random
        subset_vars = "prices_fund_macro"
        if config.hyperparams["use_gnn"]:
            model = "with_gnn"
        else:
            model = "without_gnn"

        best_loss = np.inf
        best_pred_df = pd.DataFrame()
        stop_count = 0
        list_test_loss = []

        if st_plot:
            progress_text = "Model Training in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

        for epoch in range(config.hyperparams["epochs"]):
            total_train_loss = 0.0
            total_test_loss = 0.0
            train_timesteps = list(data["train"].keys())[:int_subset]
            test_timesteps = list(data["test"].keys())
            random.shuffle(train_timesteps)

            if st_plot:
                pct = int(epoch * 100 / config.hyperparams["epochs"])
                my_bar.progress(pct, text=progress_text)

            for timestep in tqdm(train_timesteps):
                loss, _, _, _ = run_all(
                    data=data,
                    timestep=timestep,
                    train_or_test="train",
                    optimizer=optimizer,
                    lstms=lstm_models,
                    my_gnn=my_gnn,
                    mlp_heads=mlp_heads,
                    use_gnn=config.hyperparams["use_gnn"],
                    device=device,
                )

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_timesteps)

            pred_list = []
            my_gnn.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                for k, timestep in tqdm(enumerate(test_timesteps)):
                    loss, pred, true, comps = run_all(
                        data=data,
                        timestep=timestep,
                        train_or_test="test",
                        optimizer=optimizer,
                        lstms=lstm_models,
                        my_gnn=my_gnn,
                        mlp_heads=mlp_heads,
                        use_gnn=config.hyperparams["use_gnn"],
                        device=device,
                    )

                    prices = {
                        **dict(
                            zip(
                                [comp + "_pred" for comp in comps],
                                list(pred.cpu().numpy().squeeze()),
                            )
                        ),
                        **dict(
                            zip(
                                [comp + "_true" for comp in comps],
                                list(true.cpu().numpy().squeeze()),
                            )
                        ),
                    }

                    df_pred = pd.DataFrame(data=prices)
                    df_pred["last_history_date"] = [dt_index[1][k]] * config.data_prep[
                        "horizon_forecast"
                    ]

                    pred_list.append(df_pred)

                    if st_plot:
                        if timestep == config.data_prep["horizon_forecast"]:
                            fig, axs = plt.subplots(
                                len(comps), 1, figsize=(10, 5 * len(comps))
                            )

                            for j, comp in enumerate(comps):
                                axs[j].plot(
                                    df_pred.index, df_pred[comp + "_pred"], label="pred"
                                )
                                axs[j].plot(
                                    df_pred.index, df_pred[comp + "_true"], label="true"
                                )
                                axs[j].set_title(f"{comp} for epoch {epoch}")
                                axs[j].legend()

                            st.pyplot(fig)

                    total_test_loss += loss.item()

                df_pred = pd.concat(pred_list)
                avg_test_loss = total_test_loss / len(test_timesteps)
                if avg_test_loss < best_loss:
                    stop_count = 0
                    print("Best Loss! >> ", avg_test_loss)
                    best_loss = avg_test_loss
                    best_pred_df = df_pred
                    PATH_CSV = (
                        "pred_"
                        + str(round(best_loss, 2))
                        + "_"
                        + str(epoch)
                        + "_"
                        + subset_stocks
                        + "_"
                        + subset_vars
                        + "_"
                        + model
                        + ".csv"
                    )

                    base_path_save_csv = "data/" + str(rid) + "/"

                    df_pred.to_csv(base_path_save_csv + PATH_CSV)
                else:
                    stop_count += 1
                if stop_count == config.hyperparams["patience_stop"]:
                    logger.info(
                        f"Stop! {config.hyperparams['patience_stop']} epochs without improving test loss"
                    )
                    break

            # Update the learning rate
            scheduler.step(avg_test_loss)
            print(
                f"Epoch [{epoch+1}/{config.hyperparams['epochs']}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
            )

            mlflow.log_metric("Training Loss", avg_train_loss)
            mlflow.log_metric("Validation Loss", avg_test_loss)
            list_test_loss.append(avg_test_loss)

        mlflow.log_param("Stocks", subset_stocks)
        mlflow.log_param(
            "Variable", subset_vars
        )  # Prices & Fundamental , Prices & Fundamental & Macro
        mlflow.log_metric("Best Test Mape", min(list_test_loss))

        PATH_CSV = (
            "best_pred_"
            + str(round(min(list_test_loss), 2))
            + "_"
            + subset_stocks
            + "_"
            + subset_vars
            + "_"
            + model
            + ".csv"
        )

        base_path_save_csv = "data/" + str(rid) + "/"

        best_pred_df.to_csv(base_path_save_csv + PATH_CSV)
