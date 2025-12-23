import logging
import datetime as dt
from typing import List, Optional
import pandas as pd
import os
import calendar
from dateutil.relativedelta import relativedelta

from user_strategy.FixedIncomeETF.InstrumentDbManager.InstrumentDbManager import \
    InstrumentDbManager
from user_strategy.utils import CustomBDay

from user_strategy.utils.pricing_models.AggregationFunctions import ForecastAggregator, forecast_aggregation
from user_strategy.utils.SvnDownloader import download_fxdincomedb_from_svn

from user_strategy.utils.InputParams import InputParams
from sfm_pcf_db_library.PCFDBManager import PCFDBManager
from sfm_dbconnections.DbConnectionParameters import DbConnectionParameters, OracleConnectionParameters
from sfm_dbconnections.OracleConnection import OracleConnection

logger = logging.getLogger()


class InputParamsFI(InputParams):
    """
    This class manages and configures input parameters for Fixed Income ETF analysis.

    Attributes:
        price_snipping_time_string (str | None): A string representing the cutoff time for prices.
        logger (logging.Logger): Logger to track the class execution.
        params (dict): Dictionary containing configuration parameters.
        _TER (pd.DataFrame): DataFrame that holds the Total Expense Ratio (TER) for various ISINs.
        use_cache_ts (bool): Flag indicating whether to use time-series data caching.
        outlier_percentage_NAV (None | float): Threshold percentage for identifying outliers in NAV calculations.
        _YTM_mapping (pd.DataFrame): DataFrame mapping yields to maturity (YTM) to ISINs.
        cluster_anagraphic (pd.DataFrame): Anagraphic data for ISIN clusters.
        _hedge_ratios_cluster (pd.DataFrame): Currency hedging ratios for each ISIN.
        _currency_exposure (pd.DataFrame): Represents the currency exposure for each ISIN.
        isins (List[str]): List of ISINs monitored by the strategy.
        book_storage_size (int | None): Maximum buffer size for historical prices being monitored.
        number_of_days (None | int): Number of days to be used for historical data analysis.
        today (pd.Timestamp): Today's date used as a reference for time-based processing.
        yesterday (datetime.date): Yesterday's date used for daily price change calculations.
        date_from (pd.Timestamp): Start date for historical data analysis.
        price_snipping_time (datetime.time): Cutoff time for acquiring prices.
        min_ctv_to_show_trades (float): Minimum trade value to display trades.
        trade_export_cell (str | None): Reference to the Excel cell for exporting trade data.
        trade_export_sheet (str | None): Name of the Excel sheet for exporting trade data.
        output_trade_columns (List[str] | None): Columns to include in the trade export output_NAV file.
        output_prices_cell (str | None): Excel cell for exporting price data.
        output_prices_sheet (str | None): Excel sheet for exporting price data.
    """

    def __init__(self, params, **kwargs):

        """Initializes the class with the given parameters and sets basic attributes like logger, parameters, and data variables."""

        self._pricing = None
        self.price_snipping_time_string: str | None = None  # Time cutoff for prices in string format
        self.logger = logging.getLogger()  # Logger to track the class activities
        self.params = params  # Configuration parameters passed to the constructor

        self._oracle_connection = None
        self.Oracle_DB_connection = None
        self.sql_db_fi_file = None
        self._sql_db_manager = None
        self._pcf_db_manager: Optional[PCFDBManager] = None

        # Attributes for storing data like TER, hedge ratios, currency exposure, etc.
        self.use_cache_ts: bool = True
        self.outlier_percentage_NAV: None | float = None
        self._YTM_mapping: pd.DataFrame = pd.DataFrame()
        self.cluster_anagraphic: pd.DataFrame = pd.DataFrame()
        self._hedge_ratios_cluster: pd.DataFrame = pd.DataFrame()
        self._hedge_ratios_drivers: pd.DataFrame = pd.DataFrame()
        self._hedge_ratios_brothers: pd.DataFrame = pd.DataFrame()
        self._hedge_ratios_credit_futures_cluster: pd.DataFrame = pd.DataFrame()
        self._hedge_ratios_credit_futures_brothers: pd.DataFrame = pd.DataFrame()

        self._currency_exposure: pd.DataFrame = pd.DataFrame()
        self._currency_weights: pd.DataFrame = pd.DataFrame()

        self._forecast_aggregator_driver: ForecastAggregator | None = None
        self._forecast_aggregator_cluster: ForecastAggregator | None = None
        self._forecast_aggregator_brother: ForecastAggregator | None = None
        self._forecast_aggregator_nav: ForecastAggregator | None = None
        self.etf_isins: List[str] = []  # List of ISINs being monitored
        self.all_instruments: List[str] = []
        self.drivers: pd.DataFrame = pd.DataFrame()
        self.book_storage_size: int | None = None  # Buffer size for storing historical data

        self.number_of_days: None | int = None  # Number of days for historical data analysis
        self.today: pd.Timestamp = pd.Timestamp.today()  # Today's date
        self.yesterday = (self.today - CustomBDay).date()  # Yesterday's date for daily price calculations
        self._set_config_parameters()  # Call method to set configuration parameters
        self.min_ctv_to_show_trades: float = 0  # Minimum trade value for displaying trades
        self.trade_export_cell: str | None = None  # Excel cell for exporting trades
        self.trade_export_sheet: str | None = None  # Excel sheet for exporting trades
        self.output_trade_columns: List[str] | None = None  # Columns to include in trade export
        self.halflife_ewma_cluster: float | None = None  # Placeholder for half-life parameter of EWMA
        self.halflife_ewma_nav: float | None = None
        self.halflife_ewma_driver: float | None = None
        self.output_prices_cell: str | None = None  # Excel cell for exporting prices
        self.output_prices_sheet: str | None = None  # Excel sheet for exporting prices

        self._set_config_parameters()  # Set configuration parameters from provided params
        self._load_inputs()  # Load input data from db
        self._elaborate_inputs()  # Process loaded inputs

    def _set_config_parameters(self) -> None:
        """
        Sets configuration parameters from keyword arguments.

        """
        for key, value in self.params.items():
            setattr(self, key, value)  # Set attributes dynamically from params

    def _load_inputs(self) -> None:
        """
        Load anagraphic data from an Excel file into class attributes.

        The data loaded includes:
            - ISINs
            - Hedge ratio
            - TER
            - Carry
            - FX forward composition

        Args:
            file_path (str): Path to the Excel file containing the data.
        """
        if self._sql_db_manager is None:
            self._initialize_sql_db_manager()

        self.etf_isins = self._sql_db_manager.read_data(
            table='InstrumentsAnagraphic', where_clause="WHERE INSTRUMENT_TYPE = 'ETF'",
            columns=['INSTRUMENT_ID'])['INSTRUMENT_ID'].tolist()
        self.currency_exposure, self.currency_weights = self._get_currency_data(self.etf_isins)

        self.drivers, self.credit_futures_data, self.index_data, self.irs_data, self.irp_data = self.get_drivers_data()
        self.YTM_mapping = self._sql_db_manager.read_data(
            table='YasMapping', columns=['INSTRUMENT_ID', 'MAPPING_INSTRUMENT_ID']).set_index("INSTRUMENT_ID")
        new_rows = pd.DataFrame(self.YTM_mapping.loc[self.credit_futures_data['INSTRUMENT']].values, index=self.credit_futures_data.index, columns=self.YTM_mapping.columns)
        self.YTM_mapping = pd.concat([self.YTM_mapping, new_rows])

        self.cluster_anagraphic = self._sql_db_manager.read_data(
            table='StatModelHyperparameters', columns=['INSTRUMENT_ID', 'CLUSTER_ID']).set_index('INSTRUMENT_ID')
        self.brothers = self._sql_db_manager.read_data(
            table='StatModelHyperparameters', columns=['INSTRUMENT_ID', 'BROTHER_ID']).set_index('INSTRUMENT_ID')

        self.hedge_ratios_drivers = self._sql_db_manager.read_data(
            table='BetaDriver', columns=['INSTRUMENT_ID', 'DRIVER', 'BETA'],
            where_clause="WHERE DATE = (SELECT MAX(DATE) FROM BetaDriver)").pivot(
            index="INSTRUMENT_ID", columns="DRIVER", values="BETA").fillna(0).reindex(
            columns=self.drivers.index, fill_value=0)

        self.hedge_ratios_cluster = self._sql_db_manager.read_data(
            table='BetaCluster', columns=['INSTRUMENT_ID', 'REFERENCE_INSTRUMENT_ID', 'BETA'],
            where_clause="WHERE DATE = (SELECT MAX(DATE) FROM BetaCluster)").pivot(
            index="INSTRUMENT_ID", columns="REFERENCE_INSTRUMENT_ID", values="BETA").fillna(0)
        self._check_hedge_ratios(self.hedge_ratios_cluster)

        self.hedge_ratios_credit_futures_cluster = self._sql_db_manager.read_data(
            table='CreditFuturesBetaCluster', columns=['INSTRUMENT_ID', 'REFERENCE_INSTRUMENT_ID', 'BETA'],
            where_clause="WHERE DATE = (SELECT MAX(DATE) FROM CreditFuturesBetaCluster)").pivot(
            index="INSTRUMENT_ID", columns="REFERENCE_INSTRUMENT_ID", values="BETA").fillna(0)
        new_rows = pd.DataFrame(self.hedge_ratios_credit_futures_cluster.loc[self.credit_futures_data['INSTRUMENT']].values,
                                index=self.credit_futures_data.index, columns=self.hedge_ratios_credit_futures_cluster.columns)
        self.hedge_ratios_credit_futures_cluster = pd.concat([self.hedge_ratios_credit_futures_cluster, new_rows])
        self._check_hedge_ratios(self.hedge_ratios_credit_futures_cluster)

        self.hedge_ratios_brothers = self._create_hedge_ratios_brothers(self.brothers)
        self.hedge_ratios_credit_futures_brothers = self._sql_db_manager.read_data(
            table='CreditFuturesFinancialConfig', columns=['INSTRUMENT_ID', 'DRIVER_INSTRUMENT_ID', 'WEIGHT']).pivot(
            index="INSTRUMENT_ID", columns="DRIVER_INSTRUMENT_ID", values="WEIGHT").fillna(0)
        new_rows = pd.DataFrame(self.hedge_ratios_credit_futures_brothers.loc[self.credit_futures_data['INSTRUMENT']].values,
                                index=self.credit_futures_data.index, columns=self.hedge_ratios_credit_futures_brothers.columns)
        self.hedge_ratios_credit_futures_brothers = pd.concat([self.hedge_ratios_credit_futures_brothers, new_rows])

        credit_futures_trading_currency = self._sql_db_manager.read_data(
            table='CreditFutures', columns=['INSTRUMENT_ID', 'UNDERLYING_INDEX', 'CURRENCY'])
        self.trading_currency = pd.concat([credit_futures_trading_currency.set_index('INSTRUMENT_ID')['CURRENCY'],
                                            credit_futures_trading_currency.set_index('UNDERLYING_INDEX')['CURRENCY'],
                                            pd.DataFrame('EUR', index=self.etf_isins, columns=['CURRENCY'])])

        price_multiplier = self._get_price_multipliers(self.drivers.index.tolist())
        self.price_multiplier = pd.concat([price_multiplier,
                                           self.credit_futures_data.merge(price_multiplier, how="left", left_on="INSTRUMENT", right_index=True)['CONTRACT_SIZE']
                                           ])

    def _elaborate_inputs(self) -> None:
        """Elaborates the inputs by processing the currency exposure and hedge information."""
        self._currency_exposure.columns = ["EUR" + c for c in
                                           self._currency_exposure.columns]  # Prefix columns with 'EUR'
        self.currencies_EUR_ccy: List[
            str] = self._currency_exposure.columns.tolist()  # Store column names as currency list
        self.all_instruments = self.etf_isins + self.drivers.index.tolist()

    def _get_currency_data(self, isins: List[str]) -> (pd.DataFrame, pd.DataFrame):
        fx_mapping = self._sql_db_manager.read_data(table='FxMapping', columns=['INSTRUMENT_ID', 'MAPPING_INSTRUMENT_ID'])
        fx_mapping_dict = fx_mapping.set_index('INSTRUMENT_ID')['MAPPING_INSTRUMENT_ID'].to_dict()

        currency_exposure_oracle, currency_weights_oracle = self._get_currency_data_oracle(isins)
        currency_exposure_manual, currency_weights_manual = self._get_currency_data_manual(isins)
        currency_exposure_manual.reindex(columns = currency_exposure_oracle.columns, fill_value = 0)

        missing_isins = [isin for isin in self.etf_isins if isin not in currency_exposure_oracle.index]

        for isin in reversed(missing_isins):
            if isin in currency_exposure_manual.index:
                currency_exposure_oracle.loc[isin] = currency_exposure_manual.loc[isin]

                rows_to_add = currency_weights_manual.loc[[isin]]
                rows_to_add.index = [isin] * len(rows_to_add)
                currency_weights_oracle = pd.concat([currency_weights_oracle, rows_to_add])

                missing_isins.remove(isin)
            elif isin in fx_mapping_dict:
                mapping_isin = fx_mapping_dict[isin]
                if mapping_isin not in currency_exposure_oracle.index:
                    raise Exception(f'Trying to map fx of instrument {isin} to fx of instrument {mapping_isin}'
                                    ' which is not on Oracle DB')
                else:
                    currency_exposure_oracle.loc[isin] = currency_exposure_oracle.loc[mapping_isin]

                    rows_to_copy = currency_weights_oracle.loc[[mapping_isin]]
                    rows_to_copy.index = [isin] * len(rows_to_copy)
                    currency_weights_oracle = pd.concat([currency_weights_oracle, rows_to_copy])

                    missing_isins.remove(isin)

        if missing_isins:
            raise Exception(f'Missing isins: {missing_isins}')

        return currency_exposure_oracle, currency_weights_oracle

    def _create_hedge_ratios_brothers(self, brothers_df: pd.DataFrame):
        isins = brothers_df.index
        hr_brothers = pd.DataFrame(0., index=isins, columns=isins, dtype=float)

        # Group by cluster
        for _, group in brothers_df.groupby('BROTHER_ID'):
            brothers = group.index.to_list()
            for isin in brothers:
                other_brothers = [bro for bro in brothers if bro != isin]
                if other_brothers:
                    weight = 1 / len(other_brothers)
                    hr_brothers.loc[isin, other_brothers] = weight
                else:
                    hr_brothers.loc[isin, isin] = 1

        return hr_brothers


    def _get_currency_data_oracle(self, isins: List[str]) -> (pd.DataFrame, pd.DataFrame):
        if self._pcf_db_manager is None:
            self._initialize_pcf_db_manager()
        fx_weights = self._pcf_db_manager.get_last_etf_fx_composition(isins, suppress_logging=True)[['ISIN', 'CURRENCY', 'WEIGHT', 'WEIGHT_FX_FORWARD']]
        fx_comp = fx_weights.pivot(index="ISIN", columns="CURRENCY", values="WEIGHT").fillna(0)
        fx_comp.drop('EUR', axis=1, inplace=True)
        return fx_comp, fx_weights.set_index("ISIN")

    def _get_currency_data_manual(self, isins: List[str]) -> (pd.DataFrame, pd.DataFrame):
        where_clause_isins= "', '".join(isins)
        fx_weights = self._sql_db_manager.read_data(
            table='FxCompositionManual', columns=['INSTRUMENT_ID', 'CURRENCY', 'WEIGHT', 'WEIGHT_FX_FORWARD'],
            where_clause=f"WHERE INSTRUMENT_ID in ('{where_clause_isins}')"
        )
        fx_comp = fx_weights.pivot(index='INSTRUMENT_ID', columns='CURRENCY', values='WEIGHT_FX_FORWARD').fillna(0)

        return fx_comp, fx_weights.set_index("INSTRUMENT_ID")

    def _get_price_multipliers(self, isin_list: List[str]):
        oracle_conn = self._get_oracle_connection()
        isin_str = "','".join(isin_list)
        query = f'''SELECT a.exch_symbol, a.contract_size from AF_DATAMART_DBA.FUTURES_ROOTS a WHERE a.exch_symbol in ('{isin_str}')'''
        data, names = oracle_conn.execute_query(query)
        return pd.DataFrame(data, columns=names).set_index('EXCH_SYMBOL')

    @property
    def pricing(self):
        return self._pricing

    @pricing.setter
    def pricing(self, kwargs):
        self._pricing = kwargs
        self.set_forecast_aggregation_func(kwargs)

    def set_forecast_aggregation_func(self, kwargs):

        for key in ["cluster", "driver", "nav"]:
            try:
                params = kwargs[key]
                self.__setattr__(f"_forecast_aggregator_{key}",
                                 forecast_aggregation[params["forecast_aggregation"]](
                                     **params[params["forecast_aggregation"]]))

            except KeyError:
                self.logger.critical(
                    f"forecast aggregator for {key} not implemented. available: {forecast_aggregation}")
                raise KeyboardInterrupt

    @property
    def forecast_aggregator_cluster(self):
        return self._forecast_aggregator_cluster

    @property
    def forecast_aggregator_nav(self):
        return self._forecast_aggregator_nav

    @forecast_aggregator_cluster.setter
    def forecast_aggregator_cluster(self, val):
        self._forecast_aggregator_cluster = val

    @forecast_aggregator_nav.setter
    def forecast_aggregator_nav(self, val):
        self._forecast_aggregator_nav = val

    @property
    def currency_exposure(self) -> pd.DataFrame:
        """Returns the processed currency exposure DataFrame."""
        if (missing_ccy_exposure := self._currency_exposure.index.symmetric_difference(self.etf_isins)).__len__():
            self.logger.critical(f"Missing currency exposure for {', '.join(missing_ccy_exposure)}")
            if input("Do you want to continue? [Y/N] ").lower() != "y": raise KeyError
        return self._currency_exposure

    @property
    def currency_weights(self) -> pd.DataFrame:
        """Returns the processed currency weights DataFrame."""
        return self._currency_weights

    @property
    def forecast_aggregator_driver(self):
        return self._forecast_aggregator_driver

    @forecast_aggregator_driver.setter
    def forecast_aggregator_driver(self, val):
        self._forecast_aggregator_driver = val

    @property
    def forecast_aggregator_brother(self):
        return self._forecast_aggregator_brother

    @forecast_aggregator_brother.setter
    def forecast_aggregator_brother(self, val):
        self._forecast_aggregator_brother = val

    @currency_exposure.setter
    def currency_exposure(self, value: pd.DataFrame) -> None:
        self._currency_exposure = value

    @currency_weights.setter
    def currency_weights(self, value: pd.DataFrame) -> None:
        self._currency_weights = value

    @property
    def YTM_mapping(self):
        return self._YTM_mapping

    @YTM_mapping.setter
    def YTM_mapping(self, value: pd.DataFrame) -> None:
        self._YTM_mapping = value

    @property
    def hedge_ratios_cluster(self):
        return self._hedge_ratios_cluster

    @hedge_ratios_cluster.setter
    def hedge_ratios_cluster(self, value: pd.DataFrame) -> None:
        self._hedge_ratios_cluster = value

    @property
    def hedge_ratios_drivers(self):
        return self._hedge_ratios_drivers

    @hedge_ratios_drivers.setter
    def hedge_ratios_drivers(self, value: pd.DataFrame) -> None:
        value = value.loc[self.etf_isins, self.drivers.index.tolist()]
        self._hedge_ratios_drivers = value

    @property
    def hedge_ratios_brothers(self):
        return self._hedge_ratios_brothers

    @hedge_ratios_brothers.setter
    def hedge_ratios_brothers(self, value: pd.DataFrame) -> None:
        self._hedge_ratios_brothers = value

    @property
    def hedge_ratios_credit_futures_brothers(self):
        return self._hedge_ratios_credit_futures_brothers

    @hedge_ratios_credit_futures_brothers.setter
    def hedge_ratios_credit_futures_brothers(self, value: pd.DataFrame) -> None:
        self._hedge_ratios_credit_futures_brothers = value

    @property
    def hedge_ratios_credit_futures_cluster(self):
        return self._hedge_ratios_credit_futures_cluster

    @hedge_ratios_credit_futures_cluster.setter
    def hedge_ratios_credit_futures_cluster(self, value: pd.DataFrame) -> None:
        self._hedge_ratios_credit_futures_cluster = value

    def _get_oracle_connection(self):
        if self.Oracle_DB_connection is None:
            db_connection_parameters = DbConnectionParameters
            user: str = db_connection_parameters.get_oracle_parameter(OracleConnectionParameters.USERNAME)
            password: str = db_connection_parameters.get_oracle_parameter(OracleConnectionParameters.PASSWORD)
            tns_name: str = db_connection_parameters.get_oracle_parameter(OracleConnectionParameters.TNS_NAME)
            schema: str = db_connection_parameters.get_oracle_parameter(OracleConnectionParameters.SCHEMA)
            self.Oracle_DB_connection: OracleConnection = OracleConnection(user, password, tns_name, schema)
            self.Oracle_DB_connection.connect()
        return self.Oracle_DB_connection

    def _initialize_sql_db_manager(self):
        if self.sql_db_fi_file is None:
            raise Exception('Database file should be passed as config parameter to InputParamsFI')

        if not os.path.exists(self.sql_db_fi_file ):
            self.logger.warning("Downloading default from svn.")
            download_fxdincomedb_from_svn(self.sql_db_fi_file)

        self._sql_db_manager = InstrumentDbManager(self.sql_db_fi_file)

    def _initialize_pcf_db_manager(self):
        self._pcf_db_manager = PCFDBManager()

    @staticmethod
    def _check_hedge_ratios(hedge_ratios):
        all_zero_indices = hedge_ratios.index[(hedge_ratios == 0).all(axis=1)].tolist()
        if all_zero_indices:
            raise Exception(f'These instruments have all hedge ratios equal to 0 in cluster model: {all_zero_indices}')

    def get_drivers_data(self):
        selected_drivers_types = self._sql_db_manager.read_data(
            table='DriverInstrumentTypes', columns=['INSTRUMENT_TYPE'], where_clause='WHERE INSTRUMENT_TYPE NOT IN ("IRP")')['INSTRUMENT_TYPE'].tolist()
        where_clause_drivers = "', '".join(selected_drivers_types)
        drivers_data = self._sql_db_manager.read_data(
            table='InstrumentsAnagraphic', where_clause=f"WHERE INSTRUMENT_TYPE in ('{where_clause_drivers}')",
            columns=['INSTRUMENT_ID', 'INSTRUMENT_TYPE', 'BLOOMBERG_CODE', 'REGION']).set_index('INSTRUMENT_ID')

        where_clause_data_source = "', '".join(drivers_data.index.tolist())
        selected_markets = self._sql_db_manager.read_data(
            table='Markets', columns=['MARKET'])['MARKET'].tolist()
        for market in reversed(selected_markets):
            if market in ['BBG']:
                selected_markets.remove(market)
        where_clause_markets = "', '".join(selected_markets)

        drivers_price_source = self._sql_db_manager.read_data(
            table='InstrumentsDataConfig',
            where_clause=f"WHERE INSTRUMENT_ID in ('{where_clause_data_source}') and PRICE_SOURCE_MARKET in ('{where_clause_markets}')",
            columns=['INSTRUMENT_ID', 'PRICE_SOURCE_MARKET']).set_index('INSTRUMENT_ID')

        drivers_data['PRICE_SOURCE_MARKET'] = drivers_price_source['PRICE_SOURCE_MARKET'].reindex(drivers_data.index, fill_value='')
        drivers_data['MARKET_CODE'] = ''
        price_source_markets = drivers_price_source['PRICE_SOURCE_MARKET'].unique()
        for price_source_market in price_source_markets:
            mask = drivers_data['PRICE_SOURCE_MARKET'] == price_source_market
            if price_source_market == 'EUREX':
                eurex_code = ''
                drivers_data.loc[mask, 'MARKET_CODE'] = drivers_data.index[mask].astype(str) + eurex_code
            elif price_source_market == 'EURONEXT':
                drivers_data.loc[mask, 'MARKET_CODE'] = drivers_data.index[mask].astype(str)
            else:
                raise Exception(f'Price source market {price_source_market} not implemented. Implemented types are'
                                f' EUREX and EURONEXT')

        for driver in drivers_data.index:
            if drivers_data.loc[driver, 'PRICE_SOURCE_MARKET'] == 'EUREX':
                drivers_data.loc[driver, 'MARKET_CODE'] = str(driver) + self._get_present_eurex_code(driver, self.today)

        credit_futures_data = self._sql_db_manager.read_data(
            table='CreditFutures', columns=['INSTRUMENT_ID', 'CURRENCY', 'UNDERLYING_INDEX']).set_index('INSTRUMENT_ID')
        credit_futures_contracts = {}
        for cf, row in credit_futures_data.iterrows():
            credit_futures_contracts[str(cf) + self._get_present_eurex_code(cf, self.today)] = cf
            credit_futures_contracts[str(cf) + self._get_present_eurex_code(cf, self.today + relativedelta(months=3))] = cf

        credit_futures_df = pd.DataFrame({'INSTRUMENT': credit_futures_contracts})
        credit_futures_df = credit_futures_df.merge(drivers_data, left_on='INSTRUMENT', right_index=True, how='left')
        credit_futures_df['MARKET_CODE'] = credit_futures_df.index
        credit_futures_df['BLOOMBERG_CODE'] = self._get_contract_specific_bbg_code(generic_bbg_codes=credit_futures_df[['BLOOMBERG_CODE']])
        credit_futures_df['PREVIOUS_CONTRACT'] = self._get_previous_contract_eurex(credit_futures_df.index.to_list())
        credit_futures_df['EXPIRY_DATE'] = self._get_contract_specific_expiry_date(eurex_codes=credit_futures_df.index.to_list())
        credit_futures_df['PREVIOUS_CONTRACT_EXPIRY_DATE'] = self._get_contract_specific_expiry_date(eurex_codes=credit_futures_df['PREVIOUS_CONTRACT'].tolist())

        irs_data = self._sql_db_manager.read_data(
            table="InstrumentsAnagraphic", columns=["INSTRUMENT_ID", "REGION"], where_clause="WHERE INSTRUMENT_TYPE = 'IRS'",
        ).set_index('INSTRUMENT_ID')
        irp_data = self._sql_db_manager.read_data(
            table="InstrumentsAnagraphic", columns=["INSTRUMENT_ID", "REGION"], where_clause="WHERE INSTRUMENT_TYPE = 'IRP'",
        )

        meeting_dates_rate_cuts = self._sql_db_manager.read_data(
            table="MeetingDatesRateCuts", columns=["MEETING_DATE", "REGION"]
        )

        irp_data = irp_data.merge(meeting_dates_rate_cuts, left_on="REGION", right_on="REGION", how="left")
        irp_data = irp_data.groupby("INSTRUMENT_ID").agg({
            "REGION": "first",
            "MEETING_DATE": list
        })

        irp_data["MEETING_DATE"] = irp_data["MEETING_DATE"].apply(
            lambda lst: [pd.to_datetime(x, format="%d/%m/%Y").date() for x in lst if pd.notna(x)]
        )

        index_data = drivers_data[drivers_data['INSTRUMENT_TYPE'] == 'INDEX'].drop(columns=['INSTRUMENT_TYPE'])
        generic_drivers_data = drivers_data.drop(columns=['INSTRUMENT_TYPE'])


        return generic_drivers_data, credit_futures_df, index_data, irs_data, irp_data

    def _get_present_eurex_code(self, driver: str, day: dt.datetime):
        day = day.date()
        rolling_months = [3, 6, 9, 12]
        year = day.year
        special_rolling = driver in ['FEHY', 'FECX', 'FUIG', 'FUHY', 'FUEM', 'FESX', 'FGBC']
        next_rolling_date = self._get_rolling_date_candidate(year=year + 1, month=3, special_rolling=special_rolling)
        for month in rolling_months:
            candidate = self._get_rolling_date_candidate(year=year, month=month, special_rolling=special_rolling)
            if day <= candidate:
                next_rolling_date = candidate
                break

        return next_rolling_date.strftime("%Y%m")

    @staticmethod
    def _get_rolling_date_candidate(year: int, month: int, special_rolling: bool = False) -> dt.date:
        if special_rolling:
            cal = calendar.monthcalendar(year, month)

            if cal[0][calendar.FRIDAY] != 0:
                day = cal[2][calendar.FRIDAY]  # terza settimana
            else:
                day = cal[3][calendar.FRIDAY]  # quarta settimana

            return dt.date(year, month, day)
        else:
            return dt.date(year, month, 4)

    def _get_contract_specific_bbg_code(self, generic_bbg_codes: pd.DataFrame):
        df = generic_bbg_codes.copy()
        df['ROOT BBG'] = df['BLOOMBERG_CODE'].apply(lambda x: x.split('A Index')[0])
        df['CONTRACT SPECIFIC SUFFIX'] = [self._convert_eurex_suffix_to_bbg(idx[4:]) for idx in df.index]
        df['BLOOMBERG_CODE'] = df['ROOT BBG'] + df['CONTRACT SPECIFIC SUFFIX'] + ' Index'
        return df['BLOOMBERG_CODE']

    @staticmethod
    def _get_contract_specific_expiry_date(eurex_codes: List[str]):
        expiry_months = [dt.datetime.strptime(x[4:], "%Y%m") for x in eurex_codes]
        expiry_dates = [(x + dt.timedelta(days=14 + (4 - x.weekday()) % 7)).date() for x in expiry_months]
        return expiry_dates

    @staticmethod
    def _convert_eurex_suffix_to_bbg(eurex_suffix: str):
        month_bbg_mapping = {'03': 'H', '06': 'M', '09': 'U', '12': 'Z'}
        if len(eurex_suffix) < 5:
            raise ValueError(f"Wrong Eurex suffix '{eurex_suffix}', check InputParamsFI")
        month_code = month_bbg_mapping.get(eurex_suffix[-2:])
        if not month_code:
            raise ValueError(f"Invalid month code in Eurex suffix '{eurex_suffix}'")
        year_code = eurex_suffix[-3]
        return month_code + year_code

    @staticmethod
    def _get_previous_contract_eurex(credit_futures_contracts: List[str]):
        previous_contracts_eurex = []
        for contract in credit_futures_contracts:
            contract_date = dt.datetime.strptime(contract[4:], "%Y%m")
            contract_date = contract_date - dt.timedelta(days=90)
            previous_contracts_eurex.append(contract[:4] + contract_date.strftime("%Y%m"))
        return previous_contracts_eurex
