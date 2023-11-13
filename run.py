import click
from pathlib import Path
import os, time

from src.utils.db import DBInterface
from src.ingest.main import ingest_data
from src.utils.logs import configure_logs
from src.data_prep.main import DataPrep
from src.model.main import run_gnn_model

# Setting same timezone
os.environ["TZ"] = "America/Sao_Paulo"
time.tzset()


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-c",
    "--config-path",
    default="params/run_config.yml",
    type=click.Path(exists=True),
    help="Path to run configuration",
)
@click.option(
    "-i",
    "--ignore-ingest",
    is_flag=True,
    type=click.BOOL,
    help="Whether to ignore the ingestion",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    type=click.BOOL,
    help="Whether to include debug logging",
)
@click.option(
    "--force",
    is_flag=True,
    type=click.BOOL,
    help="Force regenerating the SQL database",
)
def stock_predictions(config_path: Path, ignore_ingest: bool, debug: bool, force: bool):
    """Ingesting data to Big Query by getting last data (for prediction purpose)
    or loading past data (fill the database)

    :param ingestion_params: Path to ingestion configuration file
    :type ingestion_params: Path
    :param debug: Whether to include debug logging
    :type debug: bool
    """

    configure_logs(logs_folder="output/_data_update", debug=debug)

    if not ignore_ingest:
        # ingest locally the data in the db
        ingest_data(config_path=Path(config_path), force=force)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("SCRIPT RUNNING ON DEVICE: ", device)

    data_prep = DataPrep(
        config_path=Path(config_path),
        db=DBInterface(),
    )
    data, d_size, quote_date_index_train, quote_date_index_test = data_prep.get_data()

    # get the data
    run_gnn_model(
        data=data, 
        d_size=d_size, 
        dt_index = (quote_date_index_train, quote_date_index_test),
        config_path=Path(config_path), 
        exp_name="Test", 
        device=device
    )


if __name__ == "__main__":
    cli()
