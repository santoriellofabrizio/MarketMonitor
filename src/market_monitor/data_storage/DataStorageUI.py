import logging
from abc import ABC, abstractmethod

import aiosqlite
import pandas as pd


class DataStorageUI(ABC):
    """
    Abstract base class for data storage user interfaces. This class is meant to be inherited by
    specific data storage implementations.

    Attributes:
        db_name (str): The name of the database file.
        table_name (str): The name of the table where data will be stored.
    """
    def __init__(self, db_name: str | None = None, logger=None, **kwargs):
        """
        Initializes the data storage with the specified database name and columns.

        Args:
            db_name (str): The name of the database file. Defaults to DB_DEFAULT_PATH.
            columns (list): List of column names for the DataFrame. Defaults to an empty list if not provided.
            **kwargs: Additional keyword arguments.
        """
        self.db_name = db_name or db_name + __name__
        self.logger = logger or logging.getLogger()
        self.kwargs = kwargs

    async def commit_to_db(self, data_storage: pd.DataFrame, table_name: str):
        """
        Commits the data stored in the DataFrame to the SQLite database asynchronously.
        Creates the table dynamically with DataFrame columns if it does not exist.

        Args:
            data_storage (pd.DataFrame): DataFrame containing the data to store.
            table_name (str): table name.
        """

        # Dynamically create the table schema based on the DataFrame columns
        column_definitions = ", ".join(
            f"{col} {self._infer_sqlite_type(data_storage[col])}" for col in data_storage.columns
        )
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions})"

        try:
            self.logger.info("Started commit to database")
            async with aiosqlite.connect(self.db_name) as conn:
                # Create the table if it doesn't exist
                await conn.execute(create_table_query)

                # Insert the data into the table
                await conn.executemany(
                    f'INSERT INTO {table_name} ({", ".join(data_storage.columns)}) '
                    f'VALUES ({", ".join("?" * len(data_storage.columns))})',
                    data_storage.to_records(index=False)
                )
                await conn.commit()

            self.logger.info("Finished commit to database")
            data_storage.drop(data_storage.index, inplace=True)
        except Exception as e:
            self.logger.error(f"Error in commit to database: {e}")

    def _infer_sqlite_type(self, series: pd.Series) -> str:
        """
        Infers the SQLite data type for a Pandas Series.

        Args:
            series (pd.Series): The Pandas Series to analyze.

        Returns:
            str: The SQLite data type.
        """
        if pd.api.types.is_integer_dtype(series):
            return "INTEGER"
        elif pd.api.types.is_float_dtype(series):
            return "REAL"
        elif pd.api.types.is_bool_dtype(series):
            return "INTEGER"  # SQLite does not have a native BOOLEAN type
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "TIMESTAMP"
        else:
            return "TEXT"
