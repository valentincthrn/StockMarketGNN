import logging
import traceback
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Tuple, Any, Callable


def setup_logger(output_dir: Optional[Path] = None, debug: bool = False):
    logger = logging.getLogger()
    level = logging.INFO
    if debug:
        level = logging.DEBUG

    logger.setLevel(level)

    if output_dir:
        fh = logging.FileHandler(str(output_dir / "optimization.log"), mode="w")
        fh.setLevel(level)
        fh_formatter = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    return logger


def configure_logs(
    str_format: str = "{asctime} {levelname} ({module}:{lineno:d}) {message}",
    log_to_file: bool = True,
    logs_folder: Union[str, Path, None] = None,
    key_as_dir: bool = False,
    log_fname: str = "execution.log",
    debug: bool = False,
) -> Tuple[str, Union[Path, None], logging.Logger]:
    level = logging.DEBUG if debug else logging.INFO

    # get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # get the execution key
    exec_key = datetime.now().strftime("%Y%m%d%H%M%S")

    # configure a handler to the console
    if len(root_logger.handlers) > 0:
        chandler = root_logger.handlers[0]
    else:
        chandler = logging.StreamHandler()
        root_logger.addHandler(chandler)
    chandler.setLevel(level)

    # setup the logging format
    formatter = logging.Formatter(str_format, style="{", datefmt="%Y-%m-%d %H:%M:%S")
    chandler.setFormatter(formatter)

    # configure a handler for a file
    if log_to_file:
        # create the logs folder
        if logs_folder is None:
            log_dir = Path(__file__).parent.parent.parent / "logs"
        else:
            log_dir = Path(logs_folder)

        if key_as_dir:
            log_dir = log_dir / exec_key
            f_path = log_dir / log_fname
        else:
            fpath = log_dir / f"{exec_key}.log"

        log_dir.mkdir(parents=True, exist_ok=True)

        # configure handler
        fhandler = logging.FileHandler(filename=fpath, mode="a")
        fhandler.setLevel(level)

        # change formatting
        fhandler.setFormatter(formatter)
        root_logger.addHandler(fhandler)

    else:
        log_dir = None

    # execute first log
    logger = logging.getLogger(__name__)
    logger.info(f"Starting execution {exec_key}")

    return exec_key, log_dir, logger


def log_errors(func: Callable) -> Callable:
    """log_errors Creates a decorator to be place in the main functions of the code in a way
    that allows you to transfer the traceback if there is an error in the log

    Parameters
    ----------
    func : Callable
        function to be decorated

    Returns
    -------
    Callable
        function that will export traceback to log
    """

    def func_log(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger(__name__)
        try:
            return func(*args, **kwargs)
        except BaseException:
            logger.error(traceback.format_exc())
            sys.exit(-1)

    return func_log
