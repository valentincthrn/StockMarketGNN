from pathlib import Path
import logging

from src.options import EXP, EXP_OBJECTS
from src.utils.db import DBInterface
from src.configs import RunConfiguration, DATE_FORMAT
from src.utils.logs import log_errors

logger = logging.getLogger(__name__)


@log_errors
def run_gnn_model(exp_name: str, config_path: Path):
    db = DBInterface()
    config = RunConfiguration.from_yaml(config_path)

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

    exp_obj.save()
