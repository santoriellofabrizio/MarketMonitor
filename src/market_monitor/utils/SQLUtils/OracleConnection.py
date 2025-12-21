from datetime import datetime, timedelta, date
from typing import Tuple, List

import pandas as pd
from joblib import Memory
from sfm_dbconnections.OracleConnection import OracleConnection as sfm_OracleConnection

memoryPCF = Memory(".cache/cachePcf", verbose=False)
# Define the location for caching

class OracleConnection(sfm_OracleConnection):

    MAX_MB = 20
    MAX_CACHE_SIZE = MAX_MB * 1024 * 1024

    # DATABASE CONNECTION SETTINGS
    database = "ORACLE"
    user = "AF_PCF"
    password = "AAAbbb2022!!!"
    host = "dcdwboh-cli.sg.gbs.pro"
    port = 1521
    service = "OTH_ORABOH.bsella.it"
    tns = "ORABOH"

    def __init__(self, *args, **kwargs):

        for key in ["user", "password", "host", "port", "service", "tns"]:
            setattr(self, key, kwargs.get(key, getattr(self, key)))

        super().__init__(self.user, self.password, self.tns, is_encrypted=False)

    def _get_query_data(self, query) -> Tuple[List, List[str]]:
        return self.execute_query(query)

    @staticmethod
    def get_missing_pcf(isins, pcfs):
        etfs_available = pcfs["BSH_ID_ETF"].unique().tolist()
        missing_etfs = [isin for isin in isins if isin not in etfs_available]
        missing_etfs_str = '\n'.join(missing_etfs)
        print(f"PCFs missing:\n {missing_etfs_str}")
        return missing_etfs

    def get_nav_daily(self, isins: List[str], date_from: date):
        date_from = date_from - timedelta(days=1)
        query = f'''
                       SELECT REF_DATE, BSH_ID, NAV, NAV_CCY
                       FROM AF_PCF.PCF_DAILY_INFO 
                       WHERE BSH_ID IN ('{"', '".join(isins)}') AND
                 REF_DATE > TO_DATE('{date_from.strftime("%Y-%m-%d")}','yyyy-MM-dd')

                   '''
        # Eseguire la query per recuperare i dati
        db, columns = self._get_query_data(query)
        # Costruire un DataFrame Pandas con i risultati della query
        df = pd.DataFrame(db, columns=columns)
        df.index = [d.date() for d in df["REF_DATE"]]
        return df

    def get_instruments_details(self, isins):
        query = f'''SELECT * FROM AF_PCF.PCF_INSTRUMENT_DETAILS"
                                 f" WHERE BSH_ID IN ('{"', '".join(isins[:900])}')
                                 OR BSH_ID IN ('{"', '".join(isins[900:1800])}')
                                OR BSH_ID IN ('{"', '".join(isins[1800:2700])}')
                                 OR BSH_ID IN ('{"', '".join(isins[2700:3600])}')
                                 OR BSH_ID IN ('{"', '".join(isins[3600:4500])}')
                                 OR BSH_ID IN ('{"', '".join(isins[4500:5400])}')
                                 OR BSH_ID IN ('{"', '".join(isins[5400:6300])}')
                                 OR BSH_ID IN ('{"', '".join(isins[6300:7200])}')
                                 OR BSH_ID IN ('{"', '".join(isins[7200:8100])}')'''
        db, columns = self._get_query_data(query)
        instruments_details = pd.DataFrame(db, columns=columns)
        return instruments_details

    def get_instruments_details(self, isins):
        chunk_size = 1000
        query_conditions = []

        for i in range(0, len(isins), chunk_size):
            chunk = isins[i:i + chunk_size]
            condition = f"""BSH_ID IN ('{"', '".join(chunk)}')"""
            query_conditions.append(condition)

        query = f'''SELECT * FROM AF_PCF.PCF_INSTRUMENT_DETAILS WHERE {' OR '.join(query_conditions)}'''

        db, columns = self._get_query_data(query)
        instruments_details = pd.DataFrame(db, columns=columns)

        return instruments_details

    def get_last_pcf(self, isins):
        # Costruzione della query SQL utilizzando gli isin forniti
        day = datetime.now() - timedelta(days=20)
        query = f'''
             SELECT *
             FROM AF_PCF.PCF_COMPOSITION
             WHERE (BSH_ID_ETF, REF_DATE) IN (
                 SELECT BSH_ID_ETF, MAX(REF_DATE) AS MaxDate
                 FROM AF_PCF.PCF_COMPOSITION
                 WHERE BSH_ID_ETF IN ('{"', '".join(isins)}') AND
                 REF_DATE > TO_DATE('{day.strftime("%Y-%m-%d")}','yyyy-MM-dd')
                 GROUP BY BSH_ID_ETF
             )
         '''

        # Eseguire la query per recuperare i dati
        db, columns = self._get_query_data(query)
        # Costruire un DataFrame Pandas con i risultati della query
        pcf = pd.DataFrame(db, columns=columns)

        return pcf

    def get_pcf_from_date(self, isins, day):
        # Costruzione della query SQL utilizzando gli isin forniti

        query = f'''
             SELECT *
             FROM AF_PCF.PCF_COMPOSITION
             WHERE BSH_ID_ETF IN ('{"', '".join(isins)}') AND
                 REF_DATE > TO_DATE('{day.strftime("%Y-%m-%d")}','yyyy-MM-dd')

         '''

        # Eseguire la query per recuperare i dati
        db, columns = self._get_query_data(query)
        # Costruire un DataFrame Pandas con i risultati della query
        pcf = pd.DataFrame(db, columns=columns)

        return pcf

    def get_cash_from_date(self, isins, day):
        query = f'''

                       SELECT p.*
                       FROM AF_PCF.PCF_CASH 
                       WHERE BSH_ID IN ('{"', '".join(isins)}') AND
                 REF_DATE > TO_DATE('{day.strftime("%Y-%m-%d")}','yyyy-MM-dd')

                   '''
        # Eseguire la query per recuperare i dati
        db, columns = self._get_query_data(query)
        # Costruire un DataFrame Pandas con i risultati della query
        pcf = pd.DataFrame(db, columns=columns)
        return pcf

    def get_currency_exposure(self, isins):
        day = datetime.now() - timedelta(days=20)
        query = f"""
            SELECT REFERENCE_DATE, ISIN, CURRENCY, WEIGHT, WEIGHT_FX_FORWARD 
            FROM AF_PCF.ETF_FX_COMPOSITION fx
            JOIN INSTRUMENTS i ON fx.INSTRUMENT_ID_ETF = i.ID
            WHERE (fx.INSTRUMENT_ID_ETF, REFERENCE_DATE) IN (
                SELECT fx.INSTRUMENT_ID_ETF, MAX(REFERENCE_DATE) AS MaxDate
                FROM AF_PCF.ETF_FX_COMPOSITION fx
                JOIN INSTRUMENTS i ON fx.INSTRUMENT_ID_ETF = i.ID
                WHERE REFERENCE_DATE > TO_DATE('{day.strftime("%Y-%m-%d")}','yyyy-MM-dd')
                AND ISIN in  ('{"', '".join(isins)}')
                GROUP BY fx.INSTRUMENT_ID_ETF
            )

        """

        db, columns = self._get_query_data(query)
        # Costruire un DataFrame Pandas con i risultati della query
        currency_exposure = pd.DataFrame(db, columns=columns)
        return currency_exposure

# TODO Capire a cosa serve questo
# @memoryPCF.cache
# def execute_query_cached(_connection_params, query: str, query_variables: Optional[List[Any]] = None):
#
#     oracledb.init_oracle_client()
#     try:
#         _connection = oracledb.connect(params=oracledb.ConnectParams(**_connection_params))
#     except oracledb.DatabaseError:
#         raise Exception("Connection to Oracle DB failed!!")
#
#     cursor = _connection.cursor()
#     cursor.execute(query) if query_variables is None else cursor.execute(query, query_variables)
#     _placeholder_counter = 0
#     if cursor.description is not None:
#         names = [des[0] for des in cursor.description]
#         data = []
#         while True:
#             rows = cursor.fetchmany(numRows=100_000)
#             if not rows:
#                 break
#             data.extend(rows)
#         return data, names
#     else:
#         # No results to fetch
#         return None