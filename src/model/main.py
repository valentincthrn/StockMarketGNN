from pathlib import Path
import logging
import mlflow
import pandas as pd

from src.options import EXP, EXP_OBJECTS
from src.utils.db import DBInterface
from src.configs import RunConfiguration, DATE_FORMAT
from src.utils.logs import log_errors

logger = logging.getLogger(__name__)


@log_errors
def run_gnn_model(exp_name: str, config_path: Path):
    db = DBInterface()
    config = RunConfiguration.from_yaml(config_path)

    try:
        exp_id = mlflow.create_experiment(exp_name)
    except:
        exp_id = mlflow.set_experiment(exp_name).experiment_id

    rid = pd.to_datetime("today").strftime("%Y%m%d%H%M%S")
    with mlflow.start_run(run_name=rid, experiment_id=exp_id) as run:
        exp_enum = EXP(exp_name)
        exp_obj = EXP_OBJECTS[exp_enum](config=config, db=db)

        logger.info("Building the dataset...")
        # Calling the method to prepare the data
        exp_obj.data_prep()

        logger.info("Training the model...")
        exp_obj.train()

        logger.info("Evaluating the model...")
        exp_obj.evaluate()

        logger.info("Saving the model...")
        exp_obj.postprocess()
