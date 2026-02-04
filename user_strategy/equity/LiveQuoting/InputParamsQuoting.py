import logging
import sqlite3
from typing import List, Optional

import pandas as pd


from user_strategy.utils.pricing_models.AggregationFunctions import forecast_aggregation
from user_strategy.utils.InputParams import InputParams
from market_monitor.live_data_hub.real_time_data_hub import EUR_SYNONYM
from user_strategy.utils.enums import ISIN_TO_TICKER

logger = logging.getLogger()


class InputParamsQuoting(InputParams):

    def __init__(self, path_db, **kwargs):

        super().__init__()

        self.inactive_isin = []
        self._isin_inputs: pd.DataFrame | None = None
        self._isin_driver = []
        self._isin_cluster = []
        self._isin_nav = []
        self.isins = []
        self._isin_quoting = []
        # self._pcf_db_manager: Optional[PCFDBManager] = None
        self.forecast_aggregator_driver = None
        self.forecast_aggregator_cluster = None
        self.r2_cluster = None
        self.logger = logging.getLogger()
        self._currency_exposure = pd.DataFrame()
        self._currency_hedged = pd.DataFrame()
        self._beta_driver = pd.DataFrame()
        self._beta_cluster = pd.DataFrame()
        self._beta_cluster_index = pd.DataFrame()
        self._load_inputs_db(path_db)
        # self._load_inputs_excel(file_path)


    @property
    def beta_cluster(self):
        return self._beta_cluster

    @beta_cluster.setter
    def beta_cluster(self, value: pd.DataFrame):

        value = value.loc[value.sum(axis=1) != 0, value.sum() != 0]
        for etf in value.index:
            if etf not in self._isin_cluster:
                self._isin_cluster.append(etf)

        self._beta_cluster = value

    @beta_cluster.getter
    def beta_cluster(self):
        return self._beta_cluster

    @property
    def beta_cluster_index(self):
        return self._beta_cluster_index

    @beta_cluster_index.getter
    def beta_cluster_index(self):
        return self._beta_cluster_index

    def set_forecast_aggregation_func(self, kwargs):

        for key in ["cluster"]:
            try:
                params = kwargs[key]
                self.__setattr__(f"_forecast_aggregator_{key}",
                                 forecast_aggregation[params["forecast_aggregation"]](
                                     **params[params["forecast_aggregation"]]))

            except KeyError:
                self.logger.error(f"forecast aggregator for {key} not implemented.")

    @beta_cluster_index.setter
    def beta_cluster_index(self, value):
        value = value.loc[value.sum(axis=1) != 0, value.sum() != 0]
        for etf in value.index:
            if etf not in self._isin_cluster:
                self._isin_cluster.append(etf)

        self._beta_cluster_index = value

    def _load_inputs_db(self, db_path):

        # Connessione al database
        conn = sqlite3.connect(db_path)
        # Mapping: nome tabella DB -> nome attributo classe
        table_mapping = {
            "beta_large_cluster": "beta_cluster",
            "beta_index_cluster": "beta_cluster_index"
        }
        for table_name, attr_name in table_mapping.items():
            query = f"""
                            SELECT *
                            FROM {table_name}
                            WHERE (ISIN, DATETIME) IN (
                                SELECT ISIN, MAX(DATETIME)
                                FROM {table_name}
                                GROUP BY ISIN
                            )
                            """

            # Caricamento dei dati in un DataFrame
            df = pd.read_sql(query, conn)
            beta_matrix = df.pivot_table(index='ISIN',
                                         columns='DRIVER',
                                         values='BETA',
                                         aggfunc='sum').fillna(0)

            setattr(self, attr_name, beta_matrix)

        self.inactive_instrument_db = pd.read_sql(f"""
                        SELECT *
                        FROM market_status
                        """, conn)

        conn.close()
