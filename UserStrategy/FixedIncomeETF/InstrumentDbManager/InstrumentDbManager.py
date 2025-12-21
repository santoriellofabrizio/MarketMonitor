import sqlite3
from typing import List, Tuple, Any, Dict, Optional, Union

import pandas as pd

from sfm_dbconnections.SQLiteConnection import SQLiteConnection


class InstrumentDbManager(SQLiteConnection):

    def __init__(self, db_path: str):
        super().__init__(db_path)

    def read_data(
            self,
            table: str,
            where_clause: Optional[str] = None,
            join_clause: Optional[str] = None,
            columns: Optional[List[str]] = None
    ) -> pd.DataFrame:

        # Sanity check
        self._sanity_check(table, columns)

        col_str = ", ".join(columns) if columns else "*"
        query_parts = [f"SELECT {col_str} FROM {table}"]

        if join_clause:
            query_parts.append(join_clause)  # full JOIN text, e.g. "LEFT JOIN B ON A.id=B.id"

        if where_clause:
            query_parts.append(f"{where_clause}")

        query = " ".join(query_parts)

        # Execute
        result = self.execute_query(query)
        if result is None:
            return pd.DataFrame()

        rows, colnames = result
        return pd.DataFrame(rows, columns=colnames)

    def insert(self, table: str, data: Dict[str, Any]):
        columns = ', '.join(data.keys())
        placeholders = ', '.join([self.get_next_placeholder()] * len(data))
        values = list(data.values())
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        self.execute_query(query, values)
        self.commit()

    def delete_where(self, table: str, where_clause: str, params: List[Any]):
        query = f"DELETE FROM {table} WHERE {where_clause}"
        self.execute_query(query, params)
        self.commit()

    def update_where(self, table: str, data: Dict[str, Any], where_clause: str, params: List[Any]):
        set_clause = ', '.join([f"{k} = {self.get_next_placeholder()}" for k in data.keys()])
        values = list(data.values())
        query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        self.execute_query(query, values + params)
        self.commit()

    def _sanity_check(self, table: str, columns: Union[List[str], None]):
        # Check table existence
        table_check_query = """
            SELECT name FROM sqlite_master WHERE type='table' AND name=?;
        """
        result = self.execute_query(table_check_query, [table])
        if not result or not result[0]:
            raise ValueError(f"Table '{table}' does not exist in the database.")

        # Check columns existence if specified
        # if columns:
        #     pragma_query = f"PRAGMA table_info({table});"
        #     result = self.execute_query(pragma_query)
        #     if not result:
        #         raise Exception(f"Failed to retrieve schema info for table '{table}'.")
        #     actual_columns = {row[1] for row in result[0]}  # Second column is column name
        #     missing = [col for col in columns if col not in actual_columns]
        #     if missing:
        #         raise ValueError(f"Columns {missing} do not exist in table '{table}'.")
