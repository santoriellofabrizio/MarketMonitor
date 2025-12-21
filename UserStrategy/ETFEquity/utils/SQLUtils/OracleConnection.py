from datetime import datetime, timedelta, date
from typing import Tuple, List, Optional, Any

import oracledb
import pandas as pd


class OracleConnection:
    # Define the location for caching

    MAX_MB = 20
    MAX_CACHE_SIZE = MAX_MB * 1024 * 1024

    # DATABASE CONNECTION SETTINGS
    database = "ORACLE"
    user = "AF_PCF"
    password = "AAAbbb2022!!!"
    host = "dcdwboh-cli.sg.gbs.pro"
    port = 1521
    service = "OTH_ORABOH.bsella.it"

    def __init__(self, *args, **kwargs):

        self._connection = None
        for key in ["user", "password", "host", "port", "service"]:
            setattr(self, key, kwargs.get(key, getattr(self, key)))
        self._connection_params = oracledb.ConnectParams(user=self.user, password=self.password,
                                                         host=self.host, port=self.port, service_name=self.service)
        self._placeholder_template = ":var"
        self._placeholder_counter: int = 0
        self._max_number_params = 990

    def connect(self):
        oracledb.init_oracle_client()
        try:
            self._connection = oracledb.connect(params=self._connection_params)
        except oracledb.DatabaseError:
            raise Exception("Connection to Oracle DB failed!!")

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def commit(self):
        if self._connection is not None:
            self._connection.commit()

    def execute_query(self, query: str, query_variables: Optional[List[Any]] = None) \
            -> Optional[Tuple[List[Tuple[Any, ...]], List[str]]]:
        if self._connection is None:
            self.connect()
        cursor = self._connection.cursor()
        cursor.execute(query) if query_variables is None else cursor.execute(query, query_variables)
        self._placeholder_counter = 0
        if cursor.description is not None:
            names = [des[0] for des in cursor.description]
            data = []
            while True:
                rows = cursor.fetchmany(numRows=100_000)
                if not rows:
                    break
                data.extend(rows)
            return data, names
        else:
            # No results to fetch
            return None

    def get_next_placeholder(self) -> str:
        self._placeholder_counter += 1
        return self._placeholder_template + str(self._placeholder_counter)

    def reset_placeholder(self):
        self._placeholder_counter = 0

    def _get_query_data(self, query) -> Tuple[List, List[str]]:
        return self.execute_query(query)

    def get_today_composition(self, isins, date):
        query = f'''SELECT * FROM AF_PCF.PCF_COMPOSITION"
                 f" WHERE REF_DATE = TO_DATE('{date.strftime("%Y-%m-%d")}','yyyy-MM-dd')
                 AND BSH_ID_ETF IN ('{"', '".join(isins)}')'''
        db, columns = self._get_query_data(query)
        pcf_composition = pd.DataFrame(db, columns=columns)
        return pcf_composition

    @staticmethod
    def get_missing_pcf(isins, pcfs):
        etfs_available = pcfs["BSH_ID_ETF"].unique().tolist()
        missing_etfs = [isin for isin in isins if isin not in etfs_available]
        missing_etfs_str = '\n'.join(missing_etfs)
        print(f"PCFs missing:\n {missing_etfs_str}")
        return missing_etfs

    def get_cash_components(self, isins, date):
        query = f'''SELECT * FROM AF_PCF.PCF_CASH"
                          WHERE REF_DATE = TO_DATE('{date.strftime("%Y-%m-%d")}','yyyy-MM-dd')
                         AND BSH_ID IN ('{"', '".join(isins)}')'''
        db, columns = self._get_query_data(query)
        pcf_cash = pd.DataFrame(db, columns=columns)

        return pcf_cash

    def get_last_cash_components(self, isins):
        # Calcola la data di inizio
        day = datetime.now() - timedelta(days=20)

        # Costruzione della query SQL utilizzando gli ISIN forniti
        isins_str = "', '".join(isins)
        query = f'''
                WITH RecentDates AS (
                    SELECT BSH_ID, MAX(REF_DATE) AS MaxDate
                    FROM AF_PCF.PCF_CASH
                    WHERE BSH_ID IN ('{isins_str}') 
                      AND REF_DATE > TO_DATE('{day.strftime("%Y-%m-%d")}', 'yyyy-MM-dd')
                    GROUP BY BSH_ID
                )
                SELECT p.*
                FROM AF_PCF.PCF_CASH p
                JOIN RecentDates r
                ON p.BSH_ID = r.BSH_ID AND p.REF_DATE = r.MaxDate
                ORDER BY p.BSH_ID
            '''

        # Eseguire la query per recuperare i dati
        db, columns = self._get_query_data(query)
        # Costruire un DataFrame Pandas con i risultati della query
        pcf = pd.DataFrame(db, columns=columns)

        return pcf

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

    # def get_instruments_details(self, etfs):
    #     query = f'''SELECT * FROM AF_PCF.PCF_INSTRUMENT_DETAILS"
    #                              f" WHERE BSH_ID IN ('{"', '".join(etfs[:900])}')
    #                              OR BSH_ID IN ('{"', '".join(etfs[900:1800])}')
    #                             OR BSH_ID IN ('{"', '".join(etfs[1800:2700])}')
    #                              OR BSH_ID IN ('{"', '".join(etfs[2700:3600])}')
    #                              OR BSH_ID IN ('{"', '".join(etfs[3600:4500])}')
    #                              OR BSH_ID IN ('{"', '".join(etfs[4500:5400])}')
    #                              OR BSH_ID IN ('{"', '".join(etfs[5400:6300])}')
    #                              OR BSH_ID IN ('{"', '".join(etfs[6300:7200])}')
    #                              OR BSH_ID IN ('{"', '".join(etfs[7200:8100])}')'''
    #     db, columns = self._get_query_data(query)
    #     instruments_details = pd.DataFrame(db, columns=columns)
    #
    #     return instruments_details

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

        from datetime import datetime, timedelta

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
