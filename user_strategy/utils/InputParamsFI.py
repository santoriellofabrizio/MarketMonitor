import logging
import datetime as dt
from datetime import time
from typing import List, Optional, Tuple
import pandas as pd
import os
import calendar
from dateutil.relativedelta import relativedelta
from pandas._libs.tslibs.offsets import BDay

from user_strategy.fixed_income.InstrumentDbManager.InstrumentDbManager import InstrumentDbManager
from user_strategy.utils import CustomBDay
from user_strategy.utils.FIInputConfig import DataFetchingConfig, PricingConfig
from user_strategy.utils.pricing_models.AggregationFunctions import ForecastAggregator, forecast_aggregation
from user_strategy.utils.SvnDownloader import download_fxdincomedb_from_svn
from user_strategy.utils.InputParams import InputParams
from sfm_datalibrary.queries import OracleDynamicDataQuery
from sfm_datalibrary.connections.db_connections import DbConnectionParameters, OracleConnectionParameters, OracleConnection

logger = logging.getLogger()


class InputParamsFI(InputParams):
    """
    Manages and configures input parameters for Fixed Income ETF analysis.

    Data sources:
    - SQLite DB (path from config key `sql_db_fi_file`): instrument anagraphic, hedge ratios,
      cluster config, credit futures, FX composition, meeting dates.
    - Oracle DB (credentials from config key `oracle_connection`): FX composition (PCF),
      futures contract sizes.
    - YAML config: pricing aggregation, snipping time, number of days, export cells, etc.
    """

    def __init__(self, params, **kwargs):
        self._pricing = None
        self.price_snipping_time_string: str | None = None
        self.logger = logging.getLogger()
        self.params = params

        # DB connections (lazy-initialized)
        self.Oracle_DB_connection: OracleConnection | None = None
        self.sql_db_fi_file: str | None = None
        self._sql_db_manager: InstrumentDbManager | None = None
        self._pcf_db_manager: OracleDynamicDataQuery | None = None

        # Runtime config
        self.use_cache_ts: bool = True
        self.outlier_percentage_NAV: None | float = None
        self.book_storage_size: int | None = None
        self.number_of_days: int | None = None
        self.trade_export_cell: str | None = None
        self.trade_export_sheet: str | None = None
        self.output_prices_cell: str | None = None
        self.output_prices_sheet: str | None = None

        # Instrument data (populated by _load_inputs)
        self.etf_isins: List[str] = []
        self.drivers: pd.DataFrame = pd.DataFrame()
        self.credit_futures_data: pd.DataFrame = pd.DataFrame()
        self.index_data: pd.DataFrame = pd.DataFrame()
        self.irs_data: pd.DataFrame = pd.DataFrame()
        self.irp_data: pd.DataFrame = pd.DataFrame()
        self.brothers: pd.DataFrame = pd.DataFrame()
        self.cluster_anagraphic: pd.DataFrame = pd.DataFrame()
        self.trading_currency: pd.DataFrame = pd.DataFrame()
        self.price_multiplier: pd.DataFrame = pd.DataFrame()

        # Hedge ratios (populated by _load_inputs)
        self.hedge_ratios_cluster: pd.DataFrame = pd.DataFrame()
        self.hedge_ratios_drivers: pd.DataFrame = pd.DataFrame()
        self.hedge_ratios_brothers: pd.DataFrame = pd.DataFrame()
        self.hedge_ratios_credit_futures_cluster: pd.DataFrame = pd.DataFrame()
        self.hedge_ratios_credit_futures_brothers: pd.DataFrame = pd.DataFrame()

        # FX (populated by _load_inputs + _elaborate_inputs)
        self._currency_exposure: pd.DataFrame = pd.DataFrame()
        self.currency_weights: pd.DataFrame = pd.DataFrame()
        self.currencies_EUR_ccy: List[str] = []

        # Forecast aggregators (populated via pricing setter)
        self.forecast_aggregator_driver: ForecastAggregator | None = None
        self.forecast_aggregator_cluster: ForecastAggregator | None = None
        self.forecast_aggregator_brother: ForecastAggregator | None = None
        self.forecast_aggregator_nav: ForecastAggregator | None = None

        # YTM (populated by _load_inputs, used by PricesProvider)
        self.YTM_mapping: pd.DataFrame = pd.DataFrame()

        # Date/time (computed after _set_config_parameters)
        self.today: pd.Timestamp = pd.Timestamp.today()
        self.yesterday = (self.today - CustomBDay).date()
        self._set_config_parameters()

        if self.number_of_days is not None:
            self.date_from: pd.Timestamp = self.today - BDay(self.number_of_days)
        if self.price_snipping_time_string is not None:
            h, m, s = map(int, self.price_snipping_time_string.split(":"))
            self.price_snipping_time: time = time(hour=h, minute=m, second=s)

        self._load_inputs()
        self._elaborate_inputs()
        self._build_configs()

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def _set_config_parameters(self) -> None:
        """Apply all YAML config keys as instance attributes."""
        for key, value in self.params.items():
            setattr(self, key, value)

    # -------------------------------------------------------------------------
    # Data loading — orchestrator and sub-loaders
    # -------------------------------------------------------------------------

    def _load_inputs(self) -> None:
        """Load all instrument data from SQLite and Oracle into class attributes."""
        if self._sql_db_manager is None:
            self._initialize_sql_db_manager()
        self._load_etf_isins()
        self._load_fx_data()
        self._load_instrument_anagraphic()
        self._load_ytm_mapping()
        self._load_cluster_config()
        self._load_hedge_ratios()
        self._load_trading_config()

    def _load_etf_isins(self) -> None:
        self.etf_isins = self._sql_db_manager.read_data(
            table='InstrumentsAnagraphic',
            where_clause="WHERE INSTRUMENT_TYPE = 'ETF'",
            columns=['INSTRUMENT_ID']
        )['INSTRUMENT_ID'].tolist()

    def _load_fx_data(self) -> None:
        self.currency_exposure, self.currency_weights = self._get_currency_data(self.etf_isins)

    def _load_instrument_anagraphic(self) -> None:
        self.drivers, self.credit_futures_data, self.index_data, self.irs_data, self.irp_data = self.get_drivers_data()

    def _load_ytm_mapping(self) -> None:
        self.YTM_mapping = self._sql_db_manager.read_data(
            table='YasMapping', columns=['INSTRUMENT_ID', 'MAPPING_INSTRUMENT_ID']
        ).set_index("INSTRUMENT_ID")
        new_rows = pd.DataFrame(
            self.YTM_mapping.loc[self.credit_futures_data['INSTRUMENT']].values,
            index=self.credit_futures_data.index,
            columns=self.YTM_mapping.columns
        )
        self.YTM_mapping = pd.concat([self.YTM_mapping, new_rows])

    def _load_cluster_config(self) -> None:
        self.cluster_anagraphic = self._sql_db_manager.read_data(
            table='StatModelHyperparameters', columns=['INSTRUMENT_ID', 'CLUSTER_ID']
        ).set_index('INSTRUMENT_ID')
        self.brothers = self._sql_db_manager.read_data(
            table='StatModelHyperparameters', columns=['INSTRUMENT_ID', 'BROTHER_ID']
        ).set_index('INSTRUMENT_ID')

    def _load_hedge_ratios(self) -> None:
        hr_drivers_raw = self._sql_db_manager.read_data(
            table='BetaDriver', columns=['INSTRUMENT_ID', 'DRIVER', 'BETA'],
            where_clause="WHERE DATE = (SELECT MAX(DATE) FROM BetaDriver)"
        ).pivot(index="INSTRUMENT_ID", columns="DRIVER", values="BETA").fillna(0).reindex(
            columns=self.drivers.index, fill_value=0
        )
        self.hedge_ratios_drivers = hr_drivers_raw.loc[self.etf_isins]

        self.hedge_ratios_cluster = self._sql_db_manager.read_data(
            table='BetaCluster', columns=['INSTRUMENT_ID', 'REFERENCE_INSTRUMENT_ID', 'BETA'],
            where_clause="WHERE DATE = (SELECT MAX(DATE) FROM BetaCluster)"
        ).pivot(index="INSTRUMENT_ID", columns="REFERENCE_INSTRUMENT_ID", values="BETA").fillna(0)
        self._check_hedge_ratios(self.hedge_ratios_cluster)

        cf_cluster_raw = self._sql_db_manager.read_data(
            table='CreditFuturesBetaCluster', columns=['INSTRUMENT_ID', 'REFERENCE_INSTRUMENT_ID', 'BETA'],
            where_clause="WHERE DATE = (SELECT MAX(DATE) FROM CreditFuturesBetaCluster)"
        ).pivot(index="INSTRUMENT_ID", columns="REFERENCE_INSTRUMENT_ID", values="BETA").fillna(0)
        new_rows = pd.DataFrame(
            cf_cluster_raw.loc[self.credit_futures_data['INSTRUMENT']].values,
            index=self.credit_futures_data.index,
            columns=cf_cluster_raw.columns
        )
        self.hedge_ratios_credit_futures_cluster = pd.concat([cf_cluster_raw, new_rows])
        self._check_hedge_ratios(self.hedge_ratios_credit_futures_cluster)

        self.hedge_ratios_brothers = self._create_hedge_ratios_brothers(self.brothers)

        cf_brothers_raw = self._sql_db_manager.read_data(
            table='CreditFuturesFinancialConfig', columns=['INSTRUMENT_ID', 'DRIVER_INSTRUMENT_ID', 'WEIGHT']
        ).pivot(index="INSTRUMENT_ID", columns="DRIVER_INSTRUMENT_ID", values="WEIGHT").fillna(0)
        new_rows = pd.DataFrame(
            cf_brothers_raw.loc[self.credit_futures_data['INSTRUMENT']].values,
            index=self.credit_futures_data.index,
            columns=cf_brothers_raw.columns
        )
        self.hedge_ratios_credit_futures_brothers = pd.concat([cf_brothers_raw, new_rows])

    def _load_trading_config(self) -> None:
        cf_currencies = self._sql_db_manager.read_data(
            table='CreditFutures', columns=['INSTRUMENT_ID', 'UNDERLYING_INDEX', 'CURRENCY']
        )
        self.trading_currency = pd.concat([
            cf_currencies.set_index('INSTRUMENT_ID')['CURRENCY'],
            cf_currencies.set_index('UNDERLYING_INDEX')['CURRENCY'],
            pd.DataFrame('EUR', index=self.etf_isins, columns=['CURRENCY'])
        ])
        price_multiplier = self._get_price_multipliers(self.drivers.index.tolist())
        self.price_multiplier = pd.concat([
            price_multiplier,
            self.credit_futures_data.merge(
                price_multiplier, how="left", left_on="INSTRUMENT", right_index=True
            )['CONTRACT_SIZE']
        ])

    def _elaborate_inputs(self) -> None:
        """Add EUR prefix to FX columns and build currencies list."""
        self._currency_exposure.columns = ["EUR" + c for c in self._currency_exposure.columns]
        self.currencies_EUR_ccy: List[str] = self._currency_exposure.columns.tolist()

    def _build_configs(self) -> None:
        """Assemble the two typed config objects from loaded attributes."""
        self.data_config = DataFetchingConfig(
            etf_isins=self.etf_isins,
            drivers=self.drivers,
            index_data=self.index_data,
            credit_futures_data=self.credit_futures_data,
            irs_data=self.irs_data,
            irp_data=self.irp_data,
            YTM_mapping=self.YTM_mapping,
            currencies_EUR_ccy=self.currencies_EUR_ccy,
            currency_weights=self.currency_weights,
            currency_exposure=self.currency_exposure,  # property validates completeness
            trading_currency=self.trading_currency,
            price_snipping_time=self.price_snipping_time,
            number_of_days=self.number_of_days,
            use_cache_ts=self.use_cache_ts,
        )
        self.pricing_config = PricingConfig(
            hedge_ratios_cluster=self.hedge_ratios_cluster,
            hedge_ratios_drivers=self.hedge_ratios_drivers,
            hedge_ratios_brothers=self.hedge_ratios_brothers,
            hedge_ratios_credit_futures_cluster=self.hedge_ratios_credit_futures_cluster,
            hedge_ratios_credit_futures_brothers=self.hedge_ratios_credit_futures_brothers,
            cluster_anagraphic=self.cluster_anagraphic,
            brothers=self.brothers,
            forecast_aggregator_cluster=self.forecast_aggregator_cluster,
            forecast_aggregator_driver=self.forecast_aggregator_driver,
            forecast_aggregator_nav=self.forecast_aggregator_nav,
            forecast_aggregator_brother=self.forecast_aggregator_brother,
        )

    # -------------------------------------------------------------------------
    # Instrument drivers (split by type)
    # -------------------------------------------------------------------------

    def get_drivers_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all driver-type instruments and return (drivers, credit_futures, index, irs, irp)."""
        generic_drivers = self._load_generic_drivers()
        credit_futures = self._load_credit_futures(generic_drivers)
        irs_data, irp_data = self._load_irs_irp()
        index_data = generic_drivers[generic_drivers['INSTRUMENT_TYPE'] == 'INDEX'].drop(columns=['INSTRUMENT_TYPE'])
        generic_drivers = generic_drivers.drop(columns=['INSTRUMENT_TYPE'])
        return generic_drivers, credit_futures, index_data, irs_data, irp_data

    def _load_generic_drivers(self) -> pd.DataFrame:
        """Load driver and index instruments with market codes from SQLite."""
        selected_drivers_types = self._sql_db_manager.read_data(
            table='DriverInstrumentTypes', columns=['INSTRUMENT_TYPE'],
            where_clause='WHERE INSTRUMENT_TYPE NOT IN ("IRP")'
        )['INSTRUMENT_TYPE'].tolist()
        where_clause_drivers = "', '".join(selected_drivers_types)
        drivers_data = self._sql_db_manager.read_data(
            table='InstrumentsAnagraphic',
            where_clause=f"WHERE INSTRUMENT_TYPE in ('{where_clause_drivers}')",
            columns=['INSTRUMENT_ID', 'INSTRUMENT_TYPE', 'BLOOMBERG_CODE', 'REGION']
        ).set_index('INSTRUMENT_ID')

        selected_markets = [
            m for m in self._sql_db_manager.read_data(table='Markets', columns=['MARKET'])['MARKET'].tolist()
            if m != 'BBG'
        ]
        where_clause_markets = "', '".join(selected_markets)
        where_clause_data_source = "', '".join(drivers_data.index.tolist())
        drivers_price_source = self._sql_db_manager.read_data(
            table='InstrumentsDataConfig',
            where_clause=f"WHERE INSTRUMENT_ID in ('{where_clause_data_source}') and PRICE_SOURCE_MARKET in ('{where_clause_markets}')",
            columns=['INSTRUMENT_ID', 'PRICE_SOURCE_MARKET']
        ).set_index('INSTRUMENT_ID')

        drivers_data['PRICE_SOURCE_MARKET'] = drivers_price_source['PRICE_SOURCE_MARKET'].reindex(
            drivers_data.index, fill_value=''
        )
        drivers_data['MARKET_CODE'] = ''
        for price_source_market in drivers_price_source['PRICE_SOURCE_MARKET'].unique():
            mask = drivers_data['PRICE_SOURCE_MARKET'] == price_source_market
            if price_source_market in ('EUREX', 'EURONEXT'):
                drivers_data.loc[mask, 'MARKET_CODE'] = drivers_data.index[mask].astype(str)
            else:
                raise Exception(
                    f'Price source market {price_source_market} not implemented. '
                    f'Implemented types are EUREX and EURONEXT'
                )
        for driver in drivers_data.index:
            if drivers_data.loc[driver, 'PRICE_SOURCE_MARKET'] == 'EUREX':
                drivers_data.loc[driver, 'MARKET_CODE'] = str(driver) + self._get_present_eurex_code(driver, self.today)

        return drivers_data

    def _load_credit_futures(self, drivers_data: pd.DataFrame) -> pd.DataFrame:
        """Build the two rolling credit futures contracts for each root instrument."""
        credit_futures_data = self._sql_db_manager.read_data(
            table='CreditFutures', columns=['INSTRUMENT_ID', 'CURRENCY', 'UNDERLYING_INDEX']
        ).set_index('INSTRUMENT_ID')

        credit_futures_contracts = {}
        for cf in credit_futures_data.index:
            credit_futures_contracts[str(cf) + self._get_present_eurex_code(cf, self.today)] = cf
            credit_futures_contracts[str(cf) + self._get_present_eurex_code(cf, self.today + relativedelta(months=3))] = cf

        credit_futures_df = pd.DataFrame({'INSTRUMENT': credit_futures_contracts})
        credit_futures_df = credit_futures_df.merge(drivers_data, left_on='INSTRUMENT', right_index=True, how='left')
        credit_futures_df['MARKET_CODE'] = credit_futures_df.index
        credit_futures_df['BLOOMBERG_CODE'] = self._get_contract_specific_bbg_code(
            generic_bbg_codes=credit_futures_df[['BLOOMBERG_CODE']]
        )
        credit_futures_df['PREVIOUS_CONTRACT'] = self._get_previous_contract_eurex(
            credit_futures_df.index.to_list()
        )
        credit_futures_df['EXPIRY_DATE'] = self._get_contract_specific_expiry_date(
            eurex_codes=credit_futures_df.index.to_list()
        )
        credit_futures_df['PREVIOUS_CONTRACT_EXPIRY_DATE'] = self._get_contract_specific_expiry_date(
            eurex_codes=credit_futures_df['PREVIOUS_CONTRACT'].tolist()
        )
        return credit_futures_df

    def _load_irs_irp(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load IRS instruments and IRP instruments merged with central bank meeting dates."""
        irs_data = self._sql_db_manager.read_data(
            table="InstrumentsAnagraphic",
            columns=["INSTRUMENT_ID", "BLOOMBERG_CODE", "REGION"],
            where_clause="WHERE INSTRUMENT_TYPE = 'IRS'",
        ).set_index('INSTRUMENT_ID')

        irp_raw = self._sql_db_manager.read_data(
            table="InstrumentsAnagraphic",
            columns=["INSTRUMENT_ID", "REGION"],
            where_clause="WHERE INSTRUMENT_TYPE = 'IRP'",
        )
        meeting_dates = self._sql_db_manager.read_data(
            table="MeetingDatesRateCuts", columns=["MEETING_DATE", "REGION"]
        )
        irp_data = irp_raw.merge(meeting_dates, on="REGION", how="left")
        irp_data = irp_data.groupby("INSTRUMENT_ID").agg(
            REGION=("REGION", "first"), MEETING_DATE=("MEETING_DATE", list)
        )
        irp_data["MEETING_DATE"] = irp_data["MEETING_DATE"].apply(
            lambda lst: [pd.to_datetime(x, format="%d/%m/%Y").date() for x in lst if pd.notna(x)]
        )
        return irs_data, irp_data

    # -------------------------------------------------------------------------
    # FX / currency data
    # -------------------------------------------------------------------------

    def _get_currency_data(self, isins: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        fx_mapping_dict = self._sql_db_manager.read_data(
            table='FxMapping', columns=['INSTRUMENT_ID', 'MAPPING_INSTRUMENT_ID']
        ).set_index('INSTRUMENT_ID')['MAPPING_INSTRUMENT_ID'].to_dict()

        currency_exposure_oracle, currency_weights_oracle = self._get_currency_data_oracle(isins)
        currency_exposure_manual, currency_weights_manual = self._get_currency_data_manual(isins)
        currency_exposure_manual.reindex(columns=currency_exposure_oracle.columns, fill_value=0)

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
                    raise Exception(
                        f'Trying to map fx of instrument {isin} to fx of instrument {mapping_isin}'
                        ' which is not on Oracle DB'
                    )
                currency_exposure_oracle.loc[isin] = currency_exposure_oracle.loc[mapping_isin]
                rows_to_copy = currency_weights_oracle.loc[[mapping_isin]]
                rows_to_copy.index = [isin] * len(rows_to_copy)
                currency_weights_oracle = pd.concat([currency_weights_oracle, rows_to_copy])
                missing_isins.remove(isin)

        if missing_isins:
            raise Exception(f'Missing isins: {missing_isins}')
        return currency_exposure_oracle, currency_weights_oracle

    def _get_currency_data_oracle(self, isins: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self._pcf_db_manager is None:
            self._initialize_pcf_db_manager()
        fx_weights = self._pcf_db_manager.get_last_etf_fx_composition(
            isins, suppress_logging=True
        )[['ISIN', 'CURRENCY', 'WEIGHT', 'WEIGHT_FX_FORWARD']]
        fx_comp = fx_weights.pivot(index="ISIN", columns="CURRENCY", values="WEIGHT").fillna(0)
        fx_comp.drop('EUR', axis=1, inplace=True)
        return fx_comp, fx_weights.set_index("ISIN")

    def _get_currency_data_manual(self, isins: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        where_clause_isins = "', '".join(isins)
        fx_weights = self._sql_db_manager.read_data(
            table='FxCompositionManual',
            columns=['INSTRUMENT_ID', 'CURRENCY', 'WEIGHT', 'WEIGHT_FX_FORWARD'],
            where_clause=f"WHERE INSTRUMENT_ID in ('{where_clause_isins}')"
        )
        fx_comp = fx_weights.pivot(
            index='INSTRUMENT_ID', columns='CURRENCY', values='WEIGHT_FX_FORWARD'
        ).fillna(0)
        return fx_comp, fx_weights.set_index("INSTRUMENT_ID")

    # -------------------------------------------------------------------------
    # EUREX code utilities
    # -------------------------------------------------------------------------

    def _get_present_eurex_code(self, driver: str, day: dt.datetime) -> str:
        day = day.date()
        rolling_months = [3, 6, 9, 12]
        year = day.year
        special_rolling = driver in ['FEHY', 'FECX', 'FUIG', 'FUHY', 'FUEM', 'FESX', 'FGBC']
        next_rolling_date = self._get_rolling_date_candidate(year=year + 1, month=3, special_rolling=special_rolling)
        for month in rolling_months:
            candidate = self._get_rolling_date_candidate(year=year, month=month, special_rolling=special_rolling)
            if day < candidate:
                next_rolling_date = candidate
                break
        return next_rolling_date.strftime("%Y%m")

    @staticmethod
    def _get_rolling_date_candidate(year: int, month: int, special_rolling: bool = False) -> dt.date:
        if special_rolling:
            cal = calendar.monthcalendar(year, month)
            day = cal[3][calendar.MONDAY] if cal[0][calendar.FRIDAY] != 0 else cal[4][calendar.MONDAY]
            return dt.date(year, month, day)
        return dt.date(year, month, 4)

    def _get_contract_specific_bbg_code(self, generic_bbg_codes: pd.DataFrame) -> pd.Series:
        df = generic_bbg_codes.copy()
        df['ROOT BBG'] = df['BLOOMBERG_CODE'].apply(lambda x: x.split('A Index')[0])
        df['CONTRACT SPECIFIC SUFFIX'] = [self._convert_eurex_suffix_to_bbg(idx[4:]) for idx in df.index]
        df['BLOOMBERG_CODE'] = df['ROOT BBG'] + df['CONTRACT SPECIFIC SUFFIX'] + ' Index'
        return df['BLOOMBERG_CODE']

    @staticmethod
    def _get_contract_specific_expiry_date(eurex_codes: List[str]) -> List[dt.date]:
        expiry_months = [dt.datetime.strptime(x[4:], "%Y%m") for x in eurex_codes]
        return [(x + dt.timedelta(days=14 + (4 - x.weekday()) % 7)).date() for x in expiry_months]

    @staticmethod
    def _convert_eurex_suffix_to_bbg(eurex_suffix: str) -> str:
        month_bbg_mapping = {'03': 'H', '06': 'M', '09': 'U', '12': 'Z'}
        if len(eurex_suffix) < 5:
            raise ValueError(f"Wrong Eurex suffix '{eurex_suffix}', check InputParamsFI")
        month_code = month_bbg_mapping.get(eurex_suffix[-2:])
        if not month_code:
            raise ValueError(f"Invalid month code in Eurex suffix '{eurex_suffix}'")
        return month_code + eurex_suffix[-3]

    @staticmethod
    def _get_previous_contract_eurex(credit_futures_contracts: List[str]) -> List[str]:
        previous = []
        for contract in credit_futures_contracts:
            contract_date = dt.datetime.strptime(contract[4:], "%Y%m") - dt.timedelta(days=90)
            previous.append(contract[:4] + contract_date.strftime("%Y%m"))
        return previous

    # -------------------------------------------------------------------------
    # DB connection helpers
    # -------------------------------------------------------------------------

    def _get_oracle_connection(self) -> OracleConnection:
        if self.Oracle_DB_connection is None:
            db_params = DbConnectionParameters
            user = db_params.get_oracle_parameter(OracleConnectionParameters.USERNAME)
            password = db_params.get_oracle_parameter(OracleConnectionParameters.PASSWORD)
            tns_name = db_params.get_oracle_parameter(OracleConnectionParameters.TNS_NAME)
            schema = db_params.get_oracle_parameter(OracleConnectionParameters.SCHEMA)
            self.Oracle_DB_connection = OracleConnection(user, password, tns_name, schema)
            self.Oracle_DB_connection.connect()
        return self.Oracle_DB_connection

    def _initialize_sql_db_manager(self) -> None:
        if self.sql_db_fi_file is None:
            raise Exception('Database file should be passed as config parameter to InputParamsFI')
        if not os.path.exists(self.sql_db_fi_file):
            self.logger.warning("Downloading default from svn.")
            download_fxdincomedb_from_svn(self.sql_db_fi_file)
        self._sql_db_manager = InstrumentDbManager(self.sql_db_fi_file)

    def _initialize_pcf_db_manager(self) -> None:
        if self.Oracle_DB_connection is None:
            self.Oracle_DB_connection = self._get_oracle_connection()
        self._pcf_db_manager = OracleDynamicDataQuery(self.Oracle_DB_connection)

    @staticmethod
    def _check_hedge_ratios(hedge_ratios: pd.DataFrame) -> None:
        all_zero = hedge_ratios.index[(hedge_ratios == 0).all(axis=1)].tolist()
        if all_zero:
            raise Exception(
                f'These instruments have all hedge ratios equal to 0 in cluster model: {all_zero}'
            )

    # -------------------------------------------------------------------------
    # Hedge ratio helpers
    # -------------------------------------------------------------------------

    def _create_hedge_ratios_brothers(self, brothers_df: pd.DataFrame) -> pd.DataFrame:
        isins = brothers_df.index
        hr_brothers = pd.DataFrame(0., index=isins, columns=isins, dtype=float)
        for _, group in brothers_df.groupby('BROTHER_ID'):
            brothers = group.index.to_list()
            for isin in brothers:
                other_brothers = [bro for bro in brothers if bro != isin]
                if other_brothers:
                    hr_brothers.loc[isin, other_brothers] = 1 / len(other_brothers)
                else:
                    hr_brothers.loc[isin, isin] = 1
        return hr_brothers

    def _get_price_multipliers(self, isin_list: List[str]) -> pd.DataFrame:
        oracle_conn = self._get_oracle_connection()
        isin_str = "','".join(isin_list)
        query = (
            f"SELECT a.exch_symbol, a.contract_size FROM AF_DATAMART_DBA.FUTURES_ROOTS a "
            f"WHERE a.exch_symbol in ('{isin_str}')"
        )
        data, names = oracle_conn.execute_query(query)
        return pd.DataFrame(data, columns=names).set_index('EXCH_SYMBOL')

    # -------------------------------------------------------------------------
    # Properties (only those with actual logic)
    # -------------------------------------------------------------------------

    @property
    def pricing(self):
        return self._pricing

    @pricing.setter
    def pricing(self, kwargs):
        self._pricing = kwargs
        self.set_forecast_aggregation_func(kwargs)

    def set_forecast_aggregation_func(self, kwargs: dict) -> None:
        for key in ["cluster", "driver", "nav"]:
            try:
                params = kwargs[key]
                setattr(self, f"forecast_aggregator_{key}",
                        forecast_aggregation[params["forecast_aggregation"]](**params[params["forecast_aggregation"]]))
            except KeyError:
                self.logger.critical(
                    f"forecast aggregator for {key} not implemented. available: {forecast_aggregation}"
                )
                raise KeyboardInterrupt

    @property
    def currency_exposure(self) -> pd.DataFrame:
        if (missing := self._currency_exposure.index.symmetric_difference(self.etf_isins)).__len__():
            self.logger.critical(f"Missing currency exposure for {', '.join(missing)}")
            if input("Do you want to continue? [Y/N] ").lower() != "y":
                raise KeyError
        return self._currency_exposure

    @currency_exposure.setter
    def currency_exposure(self, value: pd.DataFrame) -> None:
        self._currency_exposure = value
