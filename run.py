import click
from pathlib import Path
import os, time
import torch

from src.configs import GROUPS
from src.utils.db import DBInterface
from src.ingest.main import ingest_data
from src.utils.logs import configure_logs
from src.data_prep.main import DataPrep
from src.model.main import run_gnn_model
from src.configs import RunConfiguration

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
    "-s",
    "--stocks-group",
    default="FromConfig",
    type=click.Choice(["Banks", "Distinct", "FromConfig"]),
    help="Groups of stocks to use (if others, then put the stocks in the run_config.yml)",
)
@click.option(
    "-m",
    "--macro",
    default="FromConfig",
    type=click.Choice(["All", "FromConfig", "Not"]),
    help="Include 'All' macro indicators, 'FromConfig' or None",
)
@click.option(
    "-f",
    "--fund",
    default="FromConfig",
    type=click.Choice(["All", "FromConfig", "Not"]),
    help="Include 'All' fundamentals indicators, 'FromConfig' or None",
)
@click.option(
    "-e",
    "--exp-name",
    default="Test",
    type=click.STRING,
    help="Name of the experiment",
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
    config_path: Path,
    ignore_ingest: bool,
    stocks_group: str,
    macro: str,
    fund: str,
    exp_name: str,
    debug: bool,
    force: bool,
):
    """Ingesting data to Big Query by getting last data (for prediction purpose)
    or loading past data (fill the database)

    :param ingestion_params: Path to ingestion configuration file
    :type ingestion_params: Path
    :param debug: Whether to include debug logging
    :type debug: bool
    """

    configure_logs(logs_folder="output/_data_update", debug=debug)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("SCRIPT RUNNING ON DEVICE: ", device)

    if stocks_group in ["Banks", "Distinct"]:
        target_stocks = GROUPS[stocks_group]
    else:
        target_stocks = None

    if macro == "All":
        macro = ["Risco-Brasil", "PIB", "Dolar", "Selic Over", "IPCA"]
    elif macro == "FromConfig":
        macro = None
    else:
        macro = []

    if fund == "All":
        fund = ["P/L", "PL/ATIVOS", "M. EBIT", "ROA", "CAGR LUCROS 5 ANOS"]
        if stocks_group not in ["Banks", "Distinct"]:
            print(
                "No fundamental indicators for stocks not in 'Banks' or 'Distinct' group"
            )
    elif fund == "FromConfig":
        fund = None
    else:
        fund = []

    # Define config file
    config = RunConfiguration.from_yaml(config_path)

    if not target_stocks is None:
        config.ingest["target_stocks"] = target_stocks
    if not macro is None:
        if len(macro) == 0:
            config.ingest["macro_indicators"] = None
        else:
            config.ingest["macro_indicators"] = macro
    if not fund is None:
        if len(fund) == 0:
            config.ingest["fundamental_indicators"] = None
        else:
            config.ingest["fundamental_indicators"] = fund

    if not ignore_ingest:
        # ingest locally the data in the db
        ingest_data(config=config, force=force)

    data_prep = DataPrep(
        config=config,
        db=DBInterface(),
        target_stocks=target_stocks,
        fund_indicators=fund,
        macros=macro,
        device=device,
    )
    data, d_size, quote_date_index_train, quote_date_index_test = data_prep.get_data()
    print("Features Size for Each Company", d_size)

    # get the data
    run_gnn_model(
        data=data,
        d_size=d_size,
        dt_index=(quote_date_index_train, quote_date_index_test),
        config=data_prep.config,
        exp_name=exp_name,
        device=device,
    )


if __name__ == "__main__":
    cli()
