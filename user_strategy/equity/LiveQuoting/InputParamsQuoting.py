import logging
import sqlite3
from typing import List, Optional

import pandas as pd


from user_strategy.utils.pricing_models.AggregationFunctions import forecast_aggregation
from user_strategy.utils.InputParams import InputParams
from market_monitor.live_data_hub.real_time_data_hub import EUR_SYNONYM
from user_strategy.utils.enums import ISIN_TO_TICKER, TICKER_TO_ISIN

logger = logging.getLogger()

DISMISSED_ETFS = ("LU2037749822",
                  "LU1602145119",
                  "LU1437024992",
                  "LU1781540957",
                  "IE000M0ZXLY9",
                  "IE0009MG7KH8",
                  "IE0002MXIF34",
                  "LU1781541179",
                  "IE000QNJAOX1",
                  "IE000VTOHNZ0",
                  "IE00BJBYDR19",
                  "LU1931974692",
                  "IE00BMDX0K95",
                  "IE000TVPSRI1",
                  "IE00BF2B0N83",
                  'LU1377381980',
                  'LU1377382012',
                  'IE000Y61WD48')


class InputParamsQuoting(InputParams):

    def __init__(self, path_db, **kwargs):

        super().__init__()

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
        self.r2_cluster_index = None
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

        if value.columns[0].lower().startswith("r2"):
            self.r2_cluster = value.iloc[:, 0]
            value.drop(value.columns[0], axis=1, inplace=True)

        value.rename(index=TICKER_TO_ISIN, columns=TICKER_TO_ISIN, inplace=True)
        value = value.loc[value.sum(axis=1) != 0, value.sum() != 0]

        if dismissed := [m for m in DISMISSED_ETFS if m in value.columns]:
            logger.warning(f"Dropped dismissed ETFs in cols: {', '.join(dismissed)}")
            beta_affected = value[dismissed]
            etf_affected = beta_affected.loc[beta_affected.sum(axis=1) != 0]

            logger.warning(f"These ETFs are affected. Rebasing betas: \n" + etf_affected
                           .rename(index=ISIN_TO_TICKER,
                                   columns=ISIN_TO_TICKER)
                           .to_string())

            sum_before = value.sum(axis=1)[etf_affected.index]
            value.drop(dismissed, axis=1, inplace=True, errors='ignore')
            sum_after = value.sum(axis=1)[etf_affected.index]
            ratio = sum_before / sum_after
            for e in etf_affected.index:
                value.loc[e] *= ratio[e]

        if dismissed := [m for m in DISMISSED_ETFS if m in value.index]:
            logger.warning(f"Dropped dismissed ETFs in rows: {', '.join(dismissed)}")
            value.drop(dismissed, inplace=True, errors='ignore')

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

    @property
    def currency_exposure(self):
        return self._currency_exposure

    @currency_exposure.setter
    def currency_exposure(self, value: pd.DataFrame):
        if not isinstance(value, pd.DataFrame):
            logger.warning("currency_exposure must be a pandas DataFrame")
            return
        for isin in self.isins:
            if isin not in value.index:
                value.loc[isin] = 0
        for eur in EUR_SYNONYM: value.drop(eur, inplace=True, axis=1, errors='ignore')
        for c in ["currency_hedged", "CURRENCY_HEDGED", "CURRENCY_HEDGED_INDICATOR", "currency_hedged_indicator"]:
            if c in value.columns:
                self.currency_hedged = value[c]
                value.drop(c, inplace=True, axis=1, errors='ignore')
        invalid_rows_hed = value.loc[self.currency_hedged[self.currency_hedged == "Y"].index].sum(axis=1) == 0
        invalid_rows_unhed = value.loc[self.currency_hedged[self.currency_hedged == "N"].index].sum(axis=1) != 0
        if invalid_rows_hed.any():
            logger.warning(f"Hedged isin with no currency exposure:\n"
                           + '\n'.join(invalid_rows_hed[invalid_rows_hed].index))

        if invalid_rows_unhed.any():
            logger.warning(f"Unhedged isin with  currency exposure:\n"
                           + '\n'.join(invalid_rows_hed[invalid_rows_unhed].index))

        value.columns = [c.replace("EUR", "") for c in value.columns]
        value = value.loc[:, value.abs().sum() > 0]
        self._currency_exposure = value.loc[self.isins]

    @property
    def currency_hedged(self):
        return self._currency_hedged

    @currency_hedged.setter
    def currency_hedged(self, value: pd.Series):
        if not isinstance(value, pd.Series):
            value = value[value.columns[0]]
        for isin in self.isins:
            if isin not in value.index:
                value.loc[isin] = 0
        self._currency_hedged = value[self.isins]

    def set_forecast_aggregation_func(self, kwargs):

        for key in ["cluster", "driver", "nav"]:
            try:
                params = kwargs[key]
                self.__setattr__(f"_forecast_aggregator_{key}",
                                 forecast_aggregation[params["forecast_aggregation"]](
                                     **params[params["forecast_aggregation"]]))

            except KeyError:
                self.logger.error(f"forecast aggregator for {key} not implemented.")

    @property
    def isin_inputs(self):
        return self._isin_inputs

    @isin_inputs.setter
    def isin_inputs(self, value: dict | pd.DataFrame) -> None:
        if isinstance(value, dict): value = pd.DataFrame(dict).T
        value.columns = value.columns.str.lower()
        for group in ["driver", "nav", "cluster", "quoting"]:
            if group not in value.columns:
                logging.warning(f"isin_{group} columns not found in ISINS_INPUT")
            else:
                setattr(self, f"isin_{group.lower()}", value[group].dropna())

    @property
    def isin_quoting(self) -> list:
        return self._isin_quoting

    @isin_quoting.setter
    def isin_quoting(self, isins: list | pd.Series):
        if isinstance(isins, pd.Series):
            isins = isins[isins == True].index.tolist()
        for isin in isins:
            if isin in DISMISSED_ETFS:
                logging.warning(f"{isin} is Dismissed")
            else:
                if isin not in self.isins:
                    logging.warning(f"quoting {isin} but no pricing found for {isin}")
        self._isin_quoting = isins

    @property
    def isin_cluster(self):
        return self._isin_cluster

    @isin_cluster.setter
    def isin_cluster(self, isins: list | pd.Series):
        if isinstance(isins, pd.Series):
            isins = isins[isins == True].index.tolist()
        for isin in isins:
            if isin in DISMISSED_ETFS:
                logging.warning(f"{isin} is Dismissed")
            else:
                if isin not in self.isins: self.isins.append(isin)
        self._isin_cluster = isins

    @property
    def isin_nav(self):
        return self._isin_nav

    @isin_nav.setter
    def isin_nav(self, isins: list | pd.Series):
        if isinstance(isins, pd.Series):
            isins = isins[isins == True].index.tolist()
        for isin in isins:
            if isin not in self.isins: self.isins.append(isin)
        self._isin_nav = isins

    @beta_cluster_index.setter
    def beta_cluster_index(self, value):
        if value.columns[0].lower().startswith("r2"):
            self.r2_index_cluster = value.iloc[:, 0]
            value.drop(value.columns[0], axis=1, inplace=True)

        value.rename(index=TICKER_TO_ISIN, columns=TICKER_TO_ISIN, inplace=True)
        value = value.loc[value.sum(axis=1) != 0, value.sum() != 0]

        if dismissed := [m for m in DISMISSED_ETFS if m in value.columns]:
            logger.warning(f"Dropped dismissed ETFs in cols: {', '.join(dismissed)}")
            beta_affected = value[dismissed]
            etf_affected = beta_affected.loc[beta_affected.sum(axis=1) != 0]

            logger.warning(f"These ETFs are affected. Rebasing betas: \n" + etf_affected
                           .rename(index=ISIN_TO_TICKER,
                                   columns=ISIN_TO_TICKER)
                           .to_string())

            sum_before = value.sum(axis=1)[etf_affected.index]
            value.drop(dismissed, axis=1, inplace=True, errors='ignore')
            sum_after = value.sum(axis=1)[etf_affected.index]
            ratio = sum_before / sum_after
            for e in etf_affected.index:
                value.loc[e] *= ratio[e]

        if dismissed := [m for m in DISMISSED_ETFS if m in value.index]:
            logger.warning(f"Dropped dismissed ETFs in rows: {', '.join(dismissed)}")
            value.drop(dismissed, inplace=True, errors='ignore')

        for etf in value.index:
            if etf not in self._isin_cluster:
                self._isin_cluster.append(etf)

        self._beta_cluster_index = value

    def _load_inputs_db(self, db_path):

        # Connessione al database
        conn = sqlite3.connect(db_path)
        for table_name in ["BETA_CLUSTER","BETA_CLUSTER_INDEX"]:
            query = f"""
                            SELECT *
                            FROM {table_name}
                            WHERE (ETF, DATETIME) IN (
                                SELECT ETF, MAX(DATETIME)
                                FROM {table_name}
                                GROUP BY ETF
                            )
                            """

            # Caricamento dei dati in un DataFrame
            df = pd.read_sql(query, conn)
            beta_matrix = df.pivot_table(index='ETF', columns='DRIVER', values='BETA', aggfunc='sum').fillna(0)
            r2 = df[["ETF","R2"]].set_index("ETF").drop_duplicates()
            # beta_matrix["R2"] = r2["R2"]
            # beta_matrix.insert(0, 'R2', beta_matrix.pop('R2'))  #put R2 as first column
            setattr(self, table_name.lower(), beta_matrix)
        for table_name in ["TerManual"]:
            query = f"""
                            SELECT *
                            FROM {table_name}
                            """

            # Caricamento dei dati in un DataFrame
            ter_df = pd.read_sql(query, conn)
            ter_df = ter_df.set_index("INSTRUMENT_ID").drop(columns=['TICKER'])
            setattr(self, 'ter_manual', ter_df)

        for table_name in ["FxCompositionManual"]:
            query = f"""
                            SELECT *
                            FROM {table_name}
                            """

            # Caricamento dei dati in un DataFrame
            self._fx_manual_df = pd.read_sql(query, conn)
            self._fx_manual_df = self._fx_manual_df.drop(columns=['TICKER'])

        conn.close()

    # def _initialize_pcf_db_manager(self):
    #     self._pcf_db_manager = PCFDBManager()

    def _get_currency_data_oracle(self, isins: List[str]) -> (pd.DataFrame, pd.DataFrame):
        if self._pcf_db_manager is None:
            self._initialize_pcf_db_manager()
        fx_weights = self._pcf_db_manager.get_last_etf_fx_composition(isins, suppress_logging=True)[['ISIN', 'CURRENCY', 'WEIGHT', 'WEIGHT_FX_FORWARD']]
        fx_comp = fx_weights.pivot(index="ISIN", columns="CURRENCY", values="WEIGHT").fillna(0)
        fx_comp.drop('EUR', axis=1, inplace=True)
        return fx_comp, fx_weights.set_index("ISIN")

    def _get_currency_data_manual(self, isins: List[str]) -> (pd.DataFrame, pd.DataFrame):
        existing_isins = [i for i in isins if i in self._fx_manual_df['INSTRUMENT_ID'].to_list()]
        fx_weights = self._fx_manual_df[self._fx_manual_df['INSTRUMENT_ID'].isin(existing_isins)]
        fx_comp = fx_weights.pivot(index='INSTRUMENT_ID', columns='CURRENCY', values='WEIGHT_FX_FORWARD').fillna(0)

        return fx_comp, fx_weights.set_index("INSTRUMENT_ID")

    def get_currency_data(self, isins: List[str]) -> (pd.DataFrame, pd.DataFrame):
        fx_mapping_dict = {}

        currency_exposure_oracle, currency_weights_oracle = self._get_currency_data_oracle(isins)
        currency_exposure_manual, currency_weights_manual = self._get_currency_data_manual(isins)
        currency_exposure_manual.reindex(columns=currency_exposure_oracle.columns, fill_value=0)

        missing_isins = [isin for isin in isins if isin not in currency_exposure_oracle.index]

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