from pathlib import Path
import torch
from torch.nn import ModuleDict
import streamlit as st
from typing import Union

from src.configs import RunConfiguration
from src.utils.db import DBInterface
from src.data_prep.main import DataPrep
from src.model.module import CompanyExtractor, MyGNN, MLPWithHiddenLayer


def initialize_models(
    config: Union[Path, RunConfiguration], device: str, d_size: dict, macro_size: int
):
    if isinstance(config, Path):
        config = RunConfiguration.from_yaml(config)
    elif isinstance(config, RunConfiguration):
        pass
    else:
        raise ValueError("config must be a Path or a RunConfiguration object")

    # Initialize the models
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

    return (lstm_models, my_gnn, mlp_heads)


def initialize_weights(trio_model, subfolder_name):
    names = ["lstm_models", "my_gnn", "mlp_heads"]

    for model, name in zip(trio_model, names):
        weights = torch.load(f"models/{subfolder_name}/{name}.pt")
        model.load_state_dict(weights)
        model.eval()

    return trio_model


def prepare_data_for_prediction(
    config_path: Path,
):
    config = RunConfiguration.from_yaml(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_prep = DataPrep(
        config=Path(config_path),
        db=DBInterface(),
        target_stocks=config.ingest["target_stocks"],
        fund_indicators=config.ingest["fundamental_indicators"],
        macros=config.ingest["macro_indicators"],
        device=device,
        overwrite_params=None,
    )

    data_to_pred, d_size, past_data = data_prep.get_future_data(st_progress=True)

    return data_to_pred, d_size, past_data
