import sqlite3
import logging
import os
from pathlib import Path
from typing import Optional, Any
import pandas as pd

from src.configs import DATE_FORMAT


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
            db_path if db_path else self.base_db_path / "stock_datastore.sqlite"
        )
        self._target_db_location_str = str(self._target_db_location)

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
        with sqlite3.connect(self._target_db_location_str) as con:
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
        with sqlite3.connect(self._target_db_location_str) as con:
            recordset = con.execute(statement, *args)
            result = recordset.fetchall()
        con.close()
        return result

    def commit_many(self, query: str, *args: Any, **kwargs: Any) -> None:
        """Call the commit many method given a query and some arguments

        :param query: query to be executed
        :type query: str
        """
        with sqlite3.connect(self._target_db_location_str) as con:
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
        with sqlite3.connect(self._target_db_location_str) as con:
            pdf.to_sql(name=tablename, con=con, **kwargs)
        con.close()

    def read_sql(self, query: str, **kwargs) -> pd.DataFrame:
        with sqlite3.connect(self._target_db_location_str) as con:
            df = pd.read_sql(
                sql=query, con=con, parse_dates={"quote_date": {"format": DATE_FORMAT}}
            )
        con.close()
        return df

    def remove_last_three_quotes(self, symbol_list) -> bool:
        """
        Removes the three most recent quote dates for each symbol in the symbol_list
        from the 'stocks' table in the database.
        """
        try:
            with sqlite3.connect(self._target_db_location_str) as con:
                for symbol in symbol_list:
                    query = """
                    DELETE FROM stocks
                    WHERE rowid IN (
                        SELECT rowid FROM stocks
                        WHERE symbol = ?
                        ORDER BY quote_date DESC
                        LIMIT 3
                    )
                    """
                    con.execute(query, (symbol,))
                con.commit()
            logger.info("Successfully removed last three quotes for the symbols.")
            return True
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return False
        except Exception as e:
            logger.error(f"Exception in remove_last_three_quotes: {e}")
            return False
