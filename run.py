import click
from pathlib import Path
import os, time

from src.options import EXP
from src.ingest.main import ingest_data_local
from src.model.main import run_gnn_model
from src.utils.logs import configure_logs

# Setting same timezone
os.environ["TZ"] = "America/Sao_Paulo"
time.tzset()


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-e",
    "--exp",
    type=click.Choice([e.value for e in EXP]),
    help="The experiment to run",
)
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
def stock_predictions(
    exp: str, config_path: Path, ignore_ingest: bool, debug: bool, force: bool
):
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
        ingest_data_local(config_path=Path(config_path), force=force)

    # run the gnn model to get the predictions
    run_gnn_model(exp_name=exp, config_path=Path(config_path))


if __name__ == "__main__":
    cli()
