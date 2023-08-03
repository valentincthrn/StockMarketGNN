from pathlib import Path
import logging

from src.model.pricer import GraphStockPricer
from src.utils.db import DBInterface
from src.configs import RunConfiguration, DATE_FORMAT

logger = logging.getLogger(__name__)


def run_gnn_model(config_path: Path):
    db = DBInterface()
    config = RunConfiguration.from_yaml(config_path)

    # Instantiate the pricer object
    pricer = GraphStockPricer(config, db)

    logger.info("Building the dataset...")
    # Calling the method to prepare the data
    pricer.data_prep()

    logger.info("Training the model...")
    pricer.train()

    logger.info("Evaluating the model...")
    pricer.evaluate()

    logger.info("Saving the model...")
    pricer.postprocess()

    return
