from pathlib import Path
import logging
import pandas as pd
import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
import random
from tqdm import tqdm
import os
import streamlit as st

from src.configs import RunConfiguration
from src.model.utils import run_all
from src.utils.logs import log_errors
from src.streamlit.utils import plot_training_pred
from src.utils.common import save_yaml_config, save_pickle
from src.prediction.main import initialize_models
from src.prediction.plot import plot_uniplot

logger = logging.getLogger(__name__)


@log_errors
def run_gnn_model(
    data: pd.DataFrame,
    d_size: dict,
    means_stds: dict,
    exp_name: str,
    device: str,
    config: RunConfiguration,
    st_plot: bool = False,
    overwrite_dataprep: dict = None,
    overwrite_hyperparams: dict = None,
):
    PROJECT_PATH = Path(os.getcwd())
    MODEL_PATH = PROJECT_PATH / "models"

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

    rid = pd.to_datetime("today").strftime("%Y%m%d%H%M%S")
    rid = rid + "_" + exp_name

    os.mkdir(MODEL_PATH / rid)

    save_yaml_config(config=config, MODEL_PATH_RID=MODEL_PATH / rid)
    save_pickle(dictio=means_stds, MODEL_PATH_RID=MODEL_PATH / rid, file_name="normalization_config.pkl")

    logger.info("Initialize models, elements...")

    lstm_models, my_gnn, mlp_heads = initialize_models(
        config=config,
        device=device,
        d_size=d_size,
        macro_size=macro_size,
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

    best_loss = np.inf
    stop_count = 0
    list_test_loss = []
    list_train_loss = []

    if st_plot:
        progress_text = "Model Training in progress. Please wait."
        my_bar_epoch = st.progress(0, text=progress_text)

    for epoch in range(config.hyperparams["epochs"]):
        total_train_loss = 0.0
        total_test_loss = 0.0
        train_timesteps = list(data["train"].keys())[:int_subset]
        test_timesteps = list(data["test"].keys())
        random.shuffle(train_timesteps)

        if st_plot:
            pct = int(epoch * 100 / config.hyperparams["epochs"])
            my_bar_epoch.progress(pct, text=f"Training Epoch {epoch}")

            my_bar_inside = st.progress(0, text=progress_text)
            total_inside = len(train_timesteps)
            i = 0
            
            
        lstm_models.train()
        mlp_heads.train()
        my_gnn.train()
        for timestep in tqdm(train_timesteps):
            if st_plot:
                i += 1
                pct = int(i * 100 / total_inside)
                my_bar_inside.progress(pct, text=f"Timesteps {i} for epoch {epoch}")

            loss, _, _, _ = run_all(
                data=data,
                timestep=timestep,
                train_or_test="train",
                optimizer=optimizer,
                criterion=config.hyperparams["criterion"],
                lstms=lstm_models,
                my_gnn=my_gnn,
                mlp_heads=mlp_heads,
                use_gnn=config.hyperparams["use_gnn"],
                device=device,
            )

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_timesteps)

        pred_list = []
        lstm_models.eval()
        mlp_heads.eval()
        my_gnn.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for k, timestep in tqdm(enumerate(test_timesteps)):
                loss, pred, true, comps = run_all(
                    data=data,
                    timestep=timestep,
                    train_or_test="test",
                    optimizer=optimizer,
                    criterion=config.hyperparams["criterion"],
                    lstms=lstm_models,
                    my_gnn=my_gnn,
                    mlp_heads=mlp_heads,
                    use_gnn=config.hyperparams["use_gnn"],
                    device=device,
                )
                
                pred_list = pred.cpu().numpy()
                true_list = true.cpu().numpy()
                if len(comps) > 1:
                    pred_list = pred_list.squeeze()
                    true_list = true_list.squeeze()
                    
                last_price_t_list = np.array([data["last_raw_price"][timestep][comp] for comp in comps])[:, np.newaxis]
                
                
                prices = {
                    **dict(
                        zip(
                            [comp + "_pred" for comp in comps],
                            np.concatenate((last_price_t_list, pred_list), axis=1),
                        )
                    ),
                    **dict(
                        zip(
                            [comp + "_true" for comp in comps],
                            np.concatenate((last_price_t_list, true_list), axis=1),
                        )
                    ),
                }

                df_pred = pd.DataFrame(data=prices)
                
                df_pred.iloc[1:, :] = (df_pred.iloc[1:, :] + 1)
                df_pred_prices = df_pred.cumprod(axis=0)
     
                if k == 0:
                    if st_plot:
                        fig = plot_training_pred(df_pred_prices, comps)
                        st.pyplot(fig)
                    plot_uniplot(df_pred_prices, comps)

                total_test_loss += loss.item()

            avg_test_loss = total_test_loss / len(test_timesteps)
            if avg_test_loss < best_loss:
                stop_count = 0
                print("Best Loss! >> ", avg_test_loss)
                best_loss = avg_test_loss
                torch.save(
                    lstm_models.state_dict(), MODEL_PATH / rid / "lstm_models.pt"
                )
                torch.save(my_gnn.state_dict(), MODEL_PATH / rid / "my_gnn.pt")
                torch.save(mlp_heads.state_dict(), MODEL_PATH / rid / "mlp_heads.pt")

            else:
                stop_count += 1
            if stop_count == config.hyperparams["patience_stop"]:
                logger.info(
                    f"Stop! {config.hyperparams['patience_stop']} epochs without improving test loss"
                )
                break

            list_train_loss.append(avg_train_loss)
            list_test_loss.append(avg_test_loss)

            res_loss = pd.DataFrame(
                {
                    "train_loss": list_train_loss,
                    "test_loss": list_test_loss,
                }
            )

            res_loss.to_csv(MODEL_PATH / rid / "loss.csv", index=False)

            # Update the learning rate
            scheduler.step(avg_test_loss)
            print(
                f"Epoch [{epoch+1}/{config.hyperparams['epochs']}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
            )
            if st_plot:
                st.write(
                    f"Epoch [{epoch+1}/{config.hyperparams['epochs']}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
                )
