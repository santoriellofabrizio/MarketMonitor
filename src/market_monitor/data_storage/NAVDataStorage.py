import logging
from datetime import datetime
from typing import Union, Optional
import pandas as pd
import numpy as np
import sqlite3

from market_monitor.data_storage.DataStorageUI import DataStorageUI

# Default path for the SQLite database
DB_DEFAULT_PATH = "data_storage/data_storage.db"


class NAVDataStorage(DataStorageUI):
    """
    A class for storing and managing NAV (Net Asset Value) data in a database.
    Provides functionality for asynchronous database operations and outlier detection.
    """

    def __init__(self, db_name=DB_DEFAULT_PATH, outlier_detection: Union[bool, str] = True, logger=None, **kwargs):
        """
        Initializes the NAVDataStorage with a specified database name and optional outlier detection.

        Args:
            db_name (str): The name of the database file.
            outlier_detection (bool | str): Flag to enable or disable outlier detection.
            logger (logging.Logger): Optional logger instance.
            **kwargs: Additional arguments, such as `isin_to_ticker` and quantiles for outlier detection.
        """
        super().__init__(db_name, **kwargs)
        self.pivot_data_nav: pd.DataFrame | None = None
        self.logger = logger or logging.getLogger()
        self.kwargs = kwargs
        self._init_table("NAV_PRICE")
        self._init_outlier_detection("NAV_PRICE")

    def store_data(self, book_etfs: pd.Series, NAVs: pd.Series) -> Optional[dict[str, pd.DataFrame]]:
        """
        Stores NAV data into a DataFrame for further processing or database storage.

        Args:
            book_etfs (pd.Series): Series containing ETF prices.
            NAVs (pd.Series): Series containing NAV values.

        Returns:
            pd.DataFrame: DataFrame containing the processed data.
        """
        if self.pivot_data_nav is None:
            self._init_table("NAV_PRICE")
        try:
            timestamp = datetime.now()
            if timestamp.hour < 9 or timestamp.hour < 17:
                return
            df = []
            for isin, price in book_etfs.items():
                if isin not in NAVs:
                    continue
                NAV = NAVs.loc[isin]
                ticker = self.isin_to_ticker.get(isin, isin)
                if self.kwargs.get("outlier_detection", True):
                    if not self.is_outliers_IQR(isin, price, NAV):
                        return
                df.append({
                    'DATETIME': timestamp.strftime('%Y/%m/%d %H:%M:%S'),
                    'ETF': ticker,
                    'PRICE': np.round(price, 4),
                    'NAV': np.round(NAV, 4),
                })
            self.pivot_data_nav = pd.concat(self.pivot_data_nav, ignore_index=True)
            return {"NAV_PRICE": pd.DataFrame(df)}
        except Exception as e:
            self.logger.error(f"Error in storing data: {e}")
            return {"NAV_PRICE": pd.DataFrame()}

    def set_isin_to_ticker(self, isin_to_ticker: dict):
        """
        Sets the mapping between ISINs and tickers.

        Args:
            isin_to_ticker (dict): Dictionary mapping ISINs to tickers.
        """
        self.isin_to_ticker = isin_to_ticker

    def _init_table(self, table_name):
        if self.pivot_data_nav is not None:
            return
        try:
            query = f"""
            SELECT *
            FROM {table_name}
            WHERE DATETIME >= datetime('now', '-2 days', 'localtime')
            """
            df_data_storage = pd.read_sql_query(query, sqlite3.connect(self.db_name))
            df_data_storage["BASIS"] = df_data_storage["PRICE"] - df_data_storage["NAV"]
            self.pivot_data_nav = df_data_storage.pivot_table(index="DATETIME",
                                                              columns="ETF",
                                                              values="BASIS",
                                                              aggfunc="sum")
        except Exception as e:
            self.logger.error(f"Error in initializing table {table_name}: {e}")

    def _init_outlier_detection(self, table_name):
        """
        Initializes the outlier detection mechanism using historical data.
        """
        self._init_table(table_name)
        try:
            def find_outliers_IQR(df):
                q1 = df.quantile(float(self.kwargs.get("lower quantile", 0.25)))
                q3 = df.quantile(float(self.kwargs.get("upper quantile", 0.75)))
                IQR = q3 - q1
                return q1 - 1.5 * IQR, q3 + 1.5 * IQR

            self.lower, self.upper = find_outliers_IQR(self.pivot_data_nav)
        except Exception as e:
            self.logger.error(f"Error initializing outlier detection: {e}")
            self.outlier_detection = False

    def is_outliers_IQR(self, isin: str, price: float, NAV: float) -> bool:
        """
        Checks if the given price and NAV are outliers based on the IQR method.

        Args:
            isin (str): The ISIN of the security.
            price (float): The price of the security.
            NAV (float): The NAV of the security.

        Returns:
            bool: True if the data point is an outlier, False otherwise.
        """
        if pd.isna(price) or pd.isna(NAV):
            return True
        lower, upper = self.lower.get(isin, -1e4), self.upper.get(isin, 1e4)
        return not (lower < price - NAV < upper)

    def get_hist_NAV(self, isins: list | None = None) -> pd.DataFrame:
        if isins is None:
            return self.pivot_data_nav
        else:
            return self.pivot_data_nav.loc[self.pivot_data_nav["ISIN"].isin(isins)]
