from pathlib import Path
import logging
import mlflow
import pandas as pd
from torch.nn import ModuleDict
import torch
from torch.optim import Adam, lr_scheduler
import random
from tqdm import tqdm


from src.configs import RunConfiguration
from src.model.module import CompanyExtractor, MyGNN
from src.model.utils import run_all
from src.utils.logs import log_errors

logger = logging.getLogger(__name__)


@log_errors
def run_gnn_model(data: pd.DataFrame, d_size: dict, config_path: Path, exp_name: str):
    config = RunConfiguration.from_yaml(config_path)

    macro_size = next(iter(data["macro"].values())).shape[0]

    try:
        exp_id = mlflow.create_experiment(exp_name)
    except:
        exp_id = mlflow.set_experiment(exp_name).experiment_id

    rid = pd.to_datetime("today").strftime("%Y%m%d%H%M%S")

    with mlflow.start_run(run_name=rid, experiment_id=exp_id) as run:
        logger.info("Initialize models, elements...")

        lstm_models = ModuleDict(
            {
                comp: CompanyExtractor(
                    size + config.data_prep["pe_t"],
                    config.hyperparams["out_lstm_size"],
                )
                for comp, size in d_size.items()
            }
        )
        mlp_heads = ModuleDict(
            {
                comp: torch.nn.Linear(
                    config.hyperparams["out_gnn_size"] + macro_size,
                    config.data_prep["horizon_forecast"],
                )
                for comp in d_size.keys()
            }
        )

        my_gnn = MyGNN(
            in_channels=config.hyperparams["out_lstm_size"],
            out_channels=config.hyperparams["out_gnn_size"],
        )

        # Define a loss function and optimizer
        # criterion = torch.nn.MSELoss()  # Mean Squared Error Loss for regression tasks
        optimizer = Adam(
            list(lstm_models.parameters())
            + list(my_gnn.parameters())
            + list(mlp_heads.parameters()),
            lr=0.01,
        )

        # Learning rate scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=8, factor=0.8, verbose=True
        )

        logger.info("Training the model...")

        for epoch in range(config.hyperparams["epochs"]):
            total_train_loss = 0.0
            total_test_loss = 0.0
            train_timesteps = list(data["train"].keys())
            test_timesteps = list(data["test"].keys())
            random.shuffle(train_timesteps)

            for timestep in tqdm(train_timesteps):
                loss = run_all(
                    data=data,
                    timestep=timestep,
                    train_or_test="train",
                    optimizer=optimizer,
                    lstms=lstm_models,
                    my_gnn=my_gnn,
                    mlp_heads=mlp_heads,
                    use_gnn=config.hyperparams["use_gnn"],
                )

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_timesteps)

            my_gnn.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                for timestep in tqdm(test_timesteps):
                    loss = run_all(
                        data=data,
                        timestep=timestep,
                        train_or_test="test",
                        optimizer=optimizer,
                        lstms=lstm_models,
                        my_gnn=my_gnn,
                        mlp_heads=mlp_heads,
                        use_gnn=config.hyperparams["use_gnn"],
                    )
                    total_test_loss += loss.item()

                avg_test_loss = total_test_loss / len(test_timesteps)

            # results_loss["train"].append(avg_train_loss)
            # results_loss["test"].append(avg_test_loss)
            # Update the learning rate
            scheduler.step(avg_test_loss)
            print(
                f"Epoch [{epoch+1}/{config.hyperparams['epochs']}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
            )
