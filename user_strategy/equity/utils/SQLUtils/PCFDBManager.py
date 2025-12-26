from datetime import datetime

import pandas as pd
from typing import Optional, List, Dict


from pandas import DataFrame
from pandas._libs.tslibs.offsets import BDay

from user_strategy.utils.bloomberg_subscription_utils.OracleConnection import OracleConnection


class PCFDBManager:
    database = "ORACLE"
    user = "AF_PCF"
    password = "AAAbbb2022!!!"
    host = "dcdwboh-entry.sg.gbs.pro"
    port = 1521
    service = "OTH_ORABOH.bsella.it"

    def __init__(self, **kwargs):

        for key in ["user", "password", "host", "port", "service"]:
            setattr(self, key, kwargs.get(key, getattr(self, key)))
        self._connection: OracleConnection = OracleConnection(self.host, self.port, self.service, self.user,
                                                              self.password)
        self._placeholder = "???"
        self._max_query_parameters = 900  # because sql can accept only 1000 parameters
        self._connection.connect()
        self._names_mapping: Dict[str, str] = {"REFERENCE_DATE": "REF_DATE", "ISIN_ETF": "BSH_ID_ETF", "ISIN": "BSH_ID",
                                               "ISIN_COMP": "BSH_ID_COMP", "INSTRUMENT_TYPE": "INSTR_TYPE"}
        self.cache_bool = kwargs.get("cache_bool", False)

    def _execute_query(self, query_list: List[str]) -> Optional[DataFrame]:
        """
        Standard method to execute query. It executes all the query belonging to the given list.
        PAY ATTENTION: this method may generate problems if your query use a group by on a variable
        different from the isin, consider checking the results!

        Param:
            query_list: list of query to execute.
        Return:
            DataFrame
        """

        return self.execute_query_static(self._connection, query_list, self.cache_bool)

    @staticmethod
    def execute_query_static(connection, query_list, cache_bool=False) -> DataFrame:
        df_final = None
        for query in query_list:
            matrix, columns = connection.execute_query(query, cache_bool=cache_bool)
            if matrix is not None:
                df = pd.DataFrame(matrix, columns=columns)
                df_final = pd.concat([df_final, df], ignore_index=True)
        return df_final

    def close(self):
        """
        Close DB connection.
        """
        self._connection.close()

    def _query_generator(self, query_template: str, isin_list: List[str]) -> List[str]:
        """
        Generate a list of query respecting the _max_query_parameters constraint. It replaces the _placeholder with
        the respective etfs on each query.

        Param:
            query_template: template where only the etfs are missing (all the others param
            such as dates shall be present)
            isin_list: list of isin
        Return:
            list of query
        """
        counter = 0
        number_of_isin = len(isin_list)
        query_list = []
        while counter < number_of_isin:
            if number_of_isin - counter > self._max_query_parameters:
                isin_string = "('" + "', '".join(isin_list[counter:counter + self._max_query_parameters]) + "')"
                counter += self._max_query_parameters
            else:
                isin_string = "('" + "', '".join(isin_list[counter:number_of_isin]) + "')"
                counter = number_of_isin
            query = query_template.replace(self._placeholder, isin_string)
            query_list.append(query)
        return query_list

    def get_etf_cash_components(self, start_date: datetime.date, end_date: datetime.date,
                                isin_list: List[str], use_old_names: bool = False) -> DataFrame:
        """
        Getter for the cash_no_CIL components of a list of ETFs on a requested date interval.

        Param:
            start_date:
            end_date:
            isin_list:
        Return:
            DataFrame
        """
        query_template = f"""SELECT c.reference_date, i.isin, c.currency, c.quantity,
                                    c.weight_nav, c.weight_risk, c.cash_in_lieu
                             FROM af_pcf.etf_cash_components c, af_pcf.instruments i
                             WHERE c.instrument_id_etf = i.id AND i.isin IN {self._placeholder} 
                                   AND c.reference_date >= to_date('{start_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD') 
                                   AND c.reference_date <= to_date('{end_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD')
                             ORDER BY reference_date, isin"""
        query_list = self._query_generator(query_template, isin_list)
        df = self._execute_query(query_list)

        if df is None:
            raise Exception(f"""No cash_no_CIL component found for etf {isin_list} in period {start_date.strftime("%d-%m-%Y")}, 
                                {end_date.strftime("%d-%m-%Y")}, isin: {isin_list}""")
        if use_old_names:
            df = df.rename(self._names_mapping, axis="columns")

        return df

    def get_components_prices(self, isin_list: List[str], n_days) -> pd.DataFrame:
        """
        Getter for the component prices of a list of ETFs by ISIN for the last 30 days.

        Params:
            isin_list: List of ISINs of the ETFs to retrieve component prices for.
            start_date: Start date to filter the data (last 30 days).
        Returns:
            DataFrame with component prices and related information for the specified date range.
        """

        start_date = datetime.today() - BDay(n_days)
        if not isin_list:
            raise ValueError("The ISIN list cannot be empty.")

        if not isinstance(start_date, datetime):
            raise ValueError("The start_date must be a datetime object.")

        placeholder = ', '.join(f"'{isin}'" for isin in isin_list)
        start_date_str = start_date.strftime("%Y-%m-%d")

        query_template = f"""
            SELECT c.reference_date, 
                   i_etf.isin AS isin_etf, 
                   i_comp.isin AS isin_comp, 
                   i_comp.description, 
                   c.price_fund_ccy
            FROM af_pcf.etf_composition c
            JOIN af_pcf.instruments i_etf ON c.instrument_id_etf = i_etf.id
            JOIN af_pcf.instruments i_comp ON c.instrument_id_component = i_comp.id
            WHERE i_etf.isin IN ({placeholder})
              AND c.reference_date >= TO_DATE('{start_date_str}', 'YYYY-MM-DD')
            ORDER BY c.reference_date, i_etf.isin, i_comp.isin
        """
        query_list = self._query_generator(query_template, isin_list)
        df = self._execute_query(query_list)

        if df is None or df.empty:
            raise Exception(f"No data found for ISINs {isin_list} in the last 30 days.")

        return df

    from typing import List
    import pandas as pd

    def get_issuers_by_isin(self, isin_list: List[str]) -> pd.DataFrame:
        """
        Retrieve the issuer names for a given list of ISINs.

        Params:
            isin_list: List of ISINs to query.

        Return:
            DataFrame containing ISINs and their corresponding issuer names.
        """
        query_template = f"""
            SELECT DISTINCT i.isin, s.short_name AS issuer_name
            FROM INSTRUMENTS i
            JOIN ISSUERS s ON i.issuer_id = s.issuer_id
            WHERE i.isin IN {self._placeholder}
            ORDER BY i.isin
        """
        query_list = self._query_generator(query_template, isin_list)
        df = self._execute_query(query_list)

        if df is None or df.empty:
            raise Exception(f"No issuers found for the provided ISINs: {isin_list}")

        return df

    def get_etf_composition(self, start_date: datetime.date, end_date: datetime.date, isin_list: List[str],
                            all_details: Optional[bool] = False, use_old_names: bool = False) -> DataFrame:
        """
        Getter for the composition of a list of ETFs on a requested date interval.

        Param:
            start_date:
            end_date:
            isin_list:
            all_details: boolean variable. If True, then returns a more detailed description of the components
        Return:
            DataFrame
        """
        if all_details:
            query_template = f"""SELECT c.reference_date, i_etf.isin as isin_etf, 
                                        i_comp.isin as isin_comp, i_comp.description, c.weight_nav, c.weight_risk, 
                                        c.price_local_ccy, c.local_ccy, c.price_fund_ccy, c.n_instruments, 
                                        i_comp.instrument_type
                                 FROM af_pcf.etf_composition c, af_pcf.instruments i_etf, af_pcf.instruments i_comp
                                 WHERE c.instrument_id_component = i_comp.id AND c.instrument_id_etf = i_etf.id 
                                       AND i_etf.isin IN {self._placeholder} AND 
                                       c.reference_date >= to_date('{start_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD') 
                                       AND c.reference_date <= to_date('{end_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD')
                                 ORDER BY reference_date, isin_etf
                                 """
        else:
            query_template = f"""SELECT c.reference_date, i_etf.isin as isin_etf, 
                                        i_comp.isin as isin_comp, c.weight_nav, c.weight_risk, c.price_local_ccy, 
                                        c.local_ccy, c.price_fund_ccy, c.n_instruments
                                 FROM af_pcf.etf_composition c, af_pcf.instruments i_etf, 
                                      af_pcf.instruments i_comp 
                                 WHERE c.instrument_id_component = i_comp.id AND c.instrument_id_etf = i_etf.id 
                                       AND i_etf.isin IN {self._placeholder} AND 
                                       reference_date >= to_date('{start_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD') 
                                       AND reference_date <= to_date('{end_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD')
                                 ORDER BY reference_date, isin_etf"""

        query_list = self._query_generator(query_template, isin_list)
        df = self._execute_query(query_list)

        if not df or len(df) == 0:
            return pd.DataFrame()
            # raise Exception(f"""No composition found for etf {isin_list} in period {start_date.strftime("%d-%m-%Y")},
            #                     {end_date.strftime("%d-%m-%Y")}, isin: {isin_list}""")
        if use_old_names:
            df = df.rename(self._names_mapping, axis="columns")

        return df

    def get_etf_fund_nav_ccy(self, start_date: datetime.date, end_date: datetime.date, isin_list: List[str],
                             use_old_names: bool = False) -> DataFrame:
        query_template = f"""SELECT d.reference_date, i.isin, d.nav_ccy
                             FROM af_pcf.etf_daily_info d, af_pcf.instruments i
                             WHERE d.instrument_id_etf = i.id AND i.isin IN {self._placeholder} 
                                   AND d.reference_date >= to_date('{start_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD')
                                   AND d.reference_date <= to_date('{end_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD')
                             UNION ALL
                             SELECT d.reference_date, i.isin, d.nav_ccy
                             FROM af_pcf.etf_daily_info_hs d, af_pcf.instruments i
                             WHERE d.instrument_id_etf = i.id AND i.isin IN {self._placeholder} 
                                   AND d.reference_date >= to_date('{start_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD')
                                   AND d.reference_date <= to_date('{end_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD')"""
        query_list = self._query_generator(query_template, isin_list)
        df = self._execute_query(query_list)

        if len(df) == 0:
            raise Exception(f"""Nessun dato trovato per isin list {isin_list}""")

        if use_old_names:
            df = df.rename(self._names_mapping, axis="columns")

        return df

    def get_etf_nav(self, start_date: datetime.date, end_date: datetime.date, isin_list: List[str],
                    all_details: Optional[bool] = False, use_old_names: bool = False) -> DataFrame:
        """
        Getter for the NAV of a list of ETFs on a requested date interval. (NAV= Net Asset Value)

        Param:
            start_date:
            end_date:
            isin_list:
            all_details: boolean variable. If True, then returns a more detailed description of the components
        Return:
            DataFrame
        """
        if all_details:
            query_template = f"""SELECT d.reference_date, i.isin, d.nav, d.nav_ccy, 
                                        d.expense_ratio, d.creation_unit, d.outstanding_shares, d.dividend_amount
                                 FROM af_pcf.etf_daily_info d, af_pcf.instruments i 
                                 WHERE d.instrument_id_etf = i.id AND i.isin IN {self._placeholder} AND 
                                       reference_date >= to_date('{start_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD') 
                                       AND reference_date <= to_date('{end_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD')
                                 UNION ALL
                                 SELECT d.reference_date, i.isin, d.nav, d.nav_ccy, 
                                        d.expense_ratio, d.creation_unit, d.outstanding_shares, d.dividend_amount
                                 FROM af_pcf.etf_daily_info_hs d, af_pcf.instruments i 
                                 WHERE d.instrument_id_etf = i.id AND i.isin IN {self._placeholder} AND 
                                       reference_date >= to_date('{start_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD') 
                                       AND reference_date <= to_date('{end_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD')
                                 ORDER BY reference_date, isin
                                 """
        else:
            query_template = f"""SELECT d.reference_date, i.isin, d.nav, d.nav_ccy
                                 FROM af_pcf.etf_daily_info d, af_pcf.instruments i 
                                 WHERE d.instrument_id_etf = i.id AND i.isin IN {self._placeholder} AND 
                                       d.reference_date >= to_date('{start_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD') 
                                       AND d.reference_date <= to_date('{end_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD')
                                 UNION ALL
                                 SELECT d.reference_date, i.isin, d.nav, d.nav_ccy
                                 FROM af_pcf.etf_daily_info_hs d, af_pcf.instruments i 
                                 WHERE d.instrument_id_etf = i.id AND i.isin IN {self._placeholder} AND 
                                       d.reference_date >= to_date('{start_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD') 
                                       AND d.reference_date <= to_date('{end_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD')
                                 ORDER BY reference_date, isin"""
        query_list = self._query_generator(query_template, isin_list)
        df = self._execute_query(query_list)

        if df is None:
            raise Exception(f"""No etf nav found for etf {isin_list} in period {start_date.strftime("%d-%m-%Y")}, 
                                {end_date.strftime("%d-%m-%Y")}, isin: {isin_list}""")
        if use_old_names:
            df = df.rename(self._names_mapping, axis="columns")

        return df

    def get_stock_info(self, isin_list):

        query_template = """
                SELECT id, ISIN, DESCRIPTION, INSTRUMENT_TYPE, LEI_CODE,
                       SHORT_NAME, COUNTRY, ISSUER_TYPE, CORPORATE_SECTOR, 
                       ISSUER_INDUSTRY, ECONOMY, GEOGRAPHICAL_AREA, 
                       ISSUER_RATING_SP, ISSUER_RATING_IG_HY, SHARES_OUTSTANDING
                FROM af_datamart_dba.instruments inst
                JOIN af_datamart_dba.issuers iss ON inst.issuer_id = iss.issuer_id
                JOIN af_datamart_dba.equities e ON e.instrument_id = inst.id
                JOIN af_datamart_dba.issuer_industry_mapping iim 
                    ON iim.CORPORATE_SECTOR = iss.CORPORATE_SECTOR
                """
        query_list = self._query_generator(query_template, isin_list)
        df = self._execute_query(query_list)

        if len(df) == 0:
            raise Exception(f"""Nessun dato trovato per isin list {isin_list}""")
        return df


    def get_etf_fx_composition(self, start_date: datetime.date, end_date: datetime.date, isin_list: List[str],
                               use_old_names: bool = False) -> DataFrame:
        query_template = f"""SELECT c.reference_date, i.isin, c.currency, c.weight, c.weight_fx_forward
                            FROM af_pcf.etf_fx_composition c, af_pcf.instruments i
                            WHERE c.instrument_id_etf = i.id AND i.isin IN {self._placeholder} 
                                  AND c.reference_date >= to_date('{start_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD')
                                  AND c.reference_date <= to_date('{end_date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD')"""
        query_list = self._query_generator(query_template, isin_list)
        df = self._execute_query(query_list)

        if len(df) == 0:
            raise Exception(f"""Nessun dato trovato per isin list {isin_list}""")

        if use_old_names:
            df = df.rename(self._names_mapping, axis="columns")

        return df

    def get_available_pcf(self, isin_list: List[str], date: Optional[datetime] = None) -> List[str]:
        """
           It returns a List of ETF with an available pcf.
           When passed a date, the method runs on the pcf_composition, otherwise it runs on the pcf_daily_info for a
           shorter execution time.

           Param:
                isin_list: a list of ETF's isin
                date: requested date
            Return:
                  a list of available ETF
        """
        if date is not None:
            query_template = f"""SELECT i.isin as ISIN
                                 FROM af_pcf.etf_composition c, af_pcf.instruments i
                                 WHERE i.id = c.instrument_id_etf AND i.isin IN {self._placeholder} 
                                       AND c.reference_date= to_date('{date.strftime("%Y-%m-%d")}', 'YYYY-MM-DD')
                                 GROUP BY i.isin"""
        else:
            query_template = f"""SELECT i.isin as ISIN
                                 FROM af_pcf.etf_daily_info d, af_pcf.instruments i
                                 WHERE i.id = d.instrument_id_etf AND i.isin IN {self._placeholder}
                                 GROUP BY i.isin"""

        query_list = self._query_generator(query_template, isin_list)
        df = self._execute_query(query_list)
        available_isin = [isin for isin in isin_list if isin in df['ISIN'].to_list()]

        return available_isin

    def get_missing_pcf(self, isin_list: List[str], date: Optional[datetime] = None) -> List[str]:
        """
           It returns a List of ETF with a not available pcf.
           When passed a date, the method runs on the pcf_composition, otherwise it runs on the pcf_daily_info for a shorter
           execution time.

           Param:
                isin_list: a list of ETF's isin
                date: requested date
            Return:
                  a list of missing ETF
        """
        available_isin = self.get_available_pcf(isin_list, date)
        missing_isin = [isin for isin in isin_list if isin not in available_isin]
        return missing_isin

    def get_instrument_details(self, isin_list: List[str]) -> DataFrame:
        query = f"""SELECT * 
                    FROM af_pcf.pcf_instrument_details 
                    WHERE bsh_id in {self._placeholder}"""
        return self._execute_query(self._query_generator(query, isin_list))

    def get_last_etf_cash_components(self, isin_list: List[str], use_old_names: bool = False) -> DataFrame:
        """
        Getter for the last available cash_no_CIL components of a list of ETFs.

        Param:
            isin_list:
        Return:
            DataFrame
        """
        query_template = f"""SELECT c.reference_date, i.isin, c.currency, c.quantity,
                                    c.weight_nav, c.weight_risk, c.cash_in_lieu
                             FROM af_pcf.etf_cash_components c, af_pcf.instruments i
                             WHERE i.id=c.instrument_id_etf
                                    AND (i.isin, c.reference_date) IN (SELECT i1.isin, MAX(c1.reference_date) AS MaxDate
                                                                        FROM af_pcf.etf_cash_components c1, af_pcf.instruments i1
                                                                        WHERE c1.instrument_id_etf=i1.id AND i1.isin IN {self._placeholder}
                                                                        GROUP BY i1.isin)
                                                                        
                             ORDER BY reference_date, isin"""
        query_list = self._query_generator(query_template, isin_list)
        df = self._execute_query(query_list)

        if df is None:
            raise Exception(f"""No cash_no_CIL component found for etfs {isin_list} """)

        if use_old_names:
            df = df.rename(self._names_mapping, axis="columns")

        return df

    def get_last_etf_fx_composition(self, isin_list: List[str], use_old_names: bool = False) -> DataFrame:
        """
        Getter for the last available cash_no_CIL components of a list of ETFs.

        Param:
            isin_list:
        Return:
            DataFrame
        """
        query_template = f"""SELECT c.reference_date, i.isin, c.currency, c.weight, c.weight_fx_forward
                             FROM af_pcf.etf_fx_composition c, af_pcf.instruments i
                             WHERE i.id=c.instrument_id_etf
                                    AND (i.isin, c.reference_date) IN (SELECT i1.isin, MAX(c1.reference_date) AS MaxDate
                                                                        FROM af_pcf.etf_fx_composition c1, af_pcf.instruments i1
                                                                        WHERE c1.instrument_id_etf=i1.id AND i1.isin IN {self._placeholder}
                                                                        GROUP BY i1.isin)

                             ORDER BY reference_date, isin"""
        query_list = self._query_generator(query_template, isin_list)
        df = self._execute_query(query_list)

        if df is None:
            raise Exception(f"""No cash_no_CIL component found for etfs {isin_list} """)

        if use_old_names:
            df = df.rename(self._names_mapping, axis="columns")

        return df

    def get_last_etf_composition(self, isin_list: List[str], use_old_names: bool = False) -> DataFrame:
        """
        Getter for the last available composition of a list of ETFs.

        Param:
            isin_list:
        Return:
            DataFrame
        """
        query_template = f"""SELECT c.reference_date, i_etf.isin as isin_etf, 
                                        i_comp.isin as isin_comp, i_comp.description, c.weight_nav, c.weight_risk, 
                                        c.price_local_ccy, c.local_ccy, c.price_fund_ccy, c.n_instruments, 
                                        i_comp.instrument_type
                                 FROM af_pcf.etf_composition c, af_pcf.instruments i_etf, af_pcf.instruments i_comp
                                 WHERE c.instrument_id_component = i_comp.id AND c.instrument_id_etf = i_etf.id 
                                    AND (i_etf.isin, c.reference_date) IN (SELECT i1.isin, MAX(c1.reference_date) AS MaxDate
                                                                        FROM af_pcf.etf_composition c1, af_pcf.instruments i1
                                                                        WHERE c1.instrument_id_etf=i1.id AND i1.isin IN {self._placeholder}
                                                                        GROUP BY i1.isin)

                             ORDER BY reference_date, i_etf.isin, i_comp.isin"""
        query_list = self._query_generator(query_template, isin_list)
        df = self._execute_query(query_list)
        if df is None:
            raise Exception(f"""No cash_no_CIL component found for etfs {isin_list} """)

        if use_old_names:
            df = df.rename(self._names_mapping, axis="columns")

        return df

    def get_weight_differences_from_previous_day(self, isin_list: Optional[List[str] | str] = None, underlying: str = "EQUITY",
                                                 first_day: datetime.date = (datetime.today() + BDay(-1)).date(),
                                                 second_day: datetime.date = (datetime.today() + BDay(-2)).date(),
                                                 delta_weight: float = 0) -> pd.DataFrame:
        """
        Param:
        - isins: list of isin to check
        - underlying: if no isins are specified, it checks all the etfs of the specified underlying
        - first_day: first day to check
        - second_day: second day to check
        - delta_weight: only shows differences in weight greater than this parameter
        Return:
        - pd.DataFrame with all the couples (BSH_ID_ETF, BSH_ID_COMP) with a weight difference of the two days greater
          than the threshold, sorted by the greatest one
        """
        if isin_list is None:
            query = f"""
                        SELECT ISIN
                        FROM ETPS_INSTRUMENTS
                        WHERE UNDERLYING_TYPE = '{underlying}'"""
            results, _ = self._get_query_data(query)
            isin_list = [row[0] for row in results]

        isin_list = [isin_list] if isinstance(isin_list, str) else isin_list

        query = f"""
                    SELECT REF_DATE, BSH_ID_ETF, BSH_ID_COMP, WEIGHT_RISK
                    FROM AF_PCF.PCF_COMPOSITION_ONLINE
                    WHERE BSH_ID_ETF IN ('{"', '".join(isin_list)}') AND (REF_DATE = TO_DATE('{first_day.strftime("%Y-%m-%d")}','yyyy-MM-dd')
                          OR REF_DATE = TO_DATE('{second_day.strftime("%Y-%m-%d")}','yyyy-MM-dd'))
                 """
        db, columns = self._get_query_data(query)
        df = pd.DataFrame(db, columns=columns)

        df_first_day = df[df['REF_DATE'] == first_day.strftime("%Y-%m-%d")][['BSH_ID_ETF', 'BSH_ID_COMP', 'WEIGHT_RISK']]
        df_second_day = df[df['REF_DATE'] == second_day.strftime("%Y-%m-%d")][['BSH_ID_ETF', 'BSH_ID_COMP', 'WEIGHT_RISK']]

        merged_df = pd.merge(df_first_day, df_second_day, on=['BSH_ID_ETF', 'BSH_ID_COMP'],
                             how='outer', suffixes=('_first_day', '_second_day'))

        merged_df.fillna(0, inplace=True)
        merged_df['WEIGHT_DIFF'] = abs(merged_df['WEIGHT_RISK_first_day'] - merged_df['WEIGHT_RISK_second_day'])

        filtered_df = merged_df[merged_df['WEIGHT_DIFF'] >= delta_weight]

        sorted_df = filtered_df.sort_values(by='WEIGHT_DIFF', ascending=False)

        return sorted_df[['BSH_ID_ETF', 'BSH_ID_COMP', 'WEIGHT_DIFF']].reset_index(drop=True)
