import sqlite3
import logging
import os
from pathlib import Path
from typing import Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)


class DBInterface:
    """
    Lightweight database interface for SQLite database
    """

    @property
    def base_db_path(self) -> Path:
        return Path("db")

    @property
    def init_sql_path(self) -> Path:
        return self.base_db_path / "init_db.sql"

    def __init__(self, db_path: Optional[Path] = None) -> Path:
        self._target_db_location = (
            db_path if db_path else self.base_db_path / "stock_datastore.db"
        )

    def initialize_db(self, force: bool = False) -> None:
        """
        Runs the databse initialization script

        :param force: force the creation of a new file, defaults to False
        :type force: bool, optional
        """
        if force and self._target_db_location.exists():
            os.unlink(self._target_db_location)

        self.execute_scripts(self.init_sql_path)
        logger.info("Successfully initialized database.")

    def execute_scripts(self, *args: Any) -> None:
        """
        Helper method to connect and run a series of scripts
        """
        with sqlite3.connect(self._target_db_location) as con:
            for arg in args:
                with open(arg, "r") as fp:
                    sql_script = fp.read()
                    logger.info(f"Executing {arg}...")
                    con.executescript(sql_script)
        con.close()

    def execute_statement(self, statement: str, *args: Any) -> Any:
        """Execute a SQL query and parse the result

        :param statement: SQL query to be executed
        :type statement: str
        """
        logger.debug(f"Executing {statement} with {args}")
        with sqlite3.connect(self._target_db_location) as con:
            recordset = con.execute(statement, *args)
            result = recordset.fetchall()
        con.close()
        return result

    def commit_many(self, query: str, *args: Any, **kwargs: Any) -> None:
        """Call the commit many method given a query and some arguments

        :param query: query to be executed
        :type query: str
        """
        with sqlite3.connect(self._target_db_location) as con:
            con.executemany(query, *args, **kwargs)
            con.commit()
        con.close()

    def df_to_sql(self, pdf: pd.DataFrame, tablename: str, **kwargs) -> None:
        """Call the to_sql method on the dataframe with the given kwargs

        :param pdf: dataframe
        :type pdf: pd.DataFrame
        :param tablename: tablename to export the dataframe
        :type tablename: str
        """
        with sqlite3.connect(self._target_db_location) as con:
            pdf.to_sql(name=tablename, con=con, **kwargs)
        con.close()
