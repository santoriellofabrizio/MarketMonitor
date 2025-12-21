from typing import Optional, Tuple, Union, List, Type, Dict

import numpy as np
import pandas as pd
from dateutil.utils import today
import datetime as dt


class IRPManager:

    def __init__(self, cutoff_date: dt.date, irp_data: pd.DataFrame, irs_data: pd.DataFrame):

        """
                Interest Rates Probability manager
        """

        self._cutoff_date = cutoff_date   # cutting off meeting dates allows us to subscribe to only strictly relevant irp (up to 6 months from now)
        self._ir_data = irp_data.reset_index().merge(irs_data.reset_index(), left_on="REGION", right_on="REGION").set_index("INSTRUMENT_ID_x").rename(columns={"INSTRUMENT_ID_y": "IRS"})
        self._irs_contracts_list = irs_data.index.to_list()
        self._irp_contracts_dict, self._irp_contracts_list, self._irp_date_contract_mapping = self._generate_irp_contracts()

        self._irp_contracts_data = pd.DataFrame(index=self._irp_contracts_list)
        self._irp_contracts_data['BLOOMBERG_CODE'] = self._irp_contracts_list
        self._irp_contracts_data['PRICE_SOURCE_MARKET'] = ""
        self._irp_contracts_data['MARKET_CODE'] = self._irp_contracts_list
        self._historical_prices = pd.DataFrame()

    def _generate_irp_contracts(self):
        irp_contracts_dict = {}
        irp_contracts_list = []
        irp_date_contract_mapping = {}
        for id_irp, row in self._ir_data.iterrows():
            meeting_dates_list = pd.to_datetime(row['MEETING_DATE'], format='%d/%m/%Y')
            meeting_dates_list_refined = [d.date() for d in meeting_dates_list if d.date() < self._cutoff_date]
            irp_contracts_dict[id_irp] = []
            for date in meeting_dates_list_refined:
                contract = id_irp + f' {date.strftime("%b%Y").upper()} Index'
                irp_contracts_dict[id_irp].append(contract)
                irp_date_contract_mapping[(id_irp, date)] = contract
            irp_contracts_list += irp_contracts_dict[id_irp]

        return irp_contracts_dict, list(set(irp_contracts_list)), irp_date_contract_mapping

    def get_contracts_list_data(self):
        return self._irp_contracts_data

    def calculate_calendar_spread(self, irp_mapping: pd.Series, start: pd.Series, end: pd.Series, book: pd.Series):
        todays_date = dt.datetime.today().date()
        spread = pd.Series(0., index=irp_mapping.index)
        irs_mapping = pd.Series(self._ir_data.loc[irp_mapping.values, "IRS"].values, index=irp_mapping.index)

        meeting_dates = pd.Series(index=irp_mapping.index)
        relevant_contracts = pd.Series([[] for _ in irp_mapping.index], index=irp_mapping.index)
        relevant_times = pd.Series([[] for _ in irp_mapping.index], index=irp_mapping.index)
        index_start = pd.Series(0, index=irp_mapping.index, dtype=int)
        for instr, irp in irp_mapping.items():
            meeting_dates_for_instr = [d for d in self._ir_data.loc[irp, "MEETING_DATE"] if todays_date < d < end.loc[instr]]
            meeting_dates.loc[instr] = meeting_dates_for_instr if meeting_dates_for_instr else None
            if meeting_dates_for_instr:
                n_of_meeting_dates = len(meeting_dates.loc[instr])
                first_meeting_date = meeting_dates.loc[instr][0]
                if start.loc[instr] < first_meeting_date:
                    relevant_contracts.loc[instr].append(irs_mapping.loc[instr])
                    index_start.loc[instr] = 0
                else:
                    relevant_contracts.loc[instr].append(self._irp_date_contract_mapping[(irp_mapping.loc[instr], first_meeting_date)])
                    index_start.loc[instr] = next(i for i in range(len(meeting_dates.loc[instr])) if meeting_dates.loc[instr][i] > start.loc[instr])
            else:
                n_of_meeting_dates = 0
                relevant_contracts.loc[instr].append(irs_mapping.loc[instr])
                index_start.loc[instr] = 0

            relevant_times.loc[instr].append(start.loc[instr])
            for i in range(index_start.loc[instr], n_of_meeting_dates):
                d = meeting_dates.loc[instr][i]
                relevant_times.loc[instr].append(d)
                relevant_contracts.loc[instr].append(self._irp_date_contract_mapping[(irp_mapping.loc[instr], d)])
            relevant_times.loc[instr].append(end[instr])

        for instr, contracts in relevant_contracts.items():
            for i, contract in enumerate(contracts):
                time_interval = (relevant_times.loc[instr][i + 1] - relevant_times.loc[instr][i]).days
                spread.loc[instr] = (1 + spread.loc[instr]) * pow(1 + book.loc[contract] / 100, time_interval / 365) - 1

        return spread

    def calculate_average_rate_until_maturity(self, irp_mapping, end, book):
        todays_date = dt.datetime.today().date()
        average_rate = pd.Series(0., index=irp_mapping.index)
        irs_mapping = pd.Series(self._ir_data.loc[irp_mapping.values, "IRS"].values, index=irp_mapping.index)

        meeting_dates = pd.Series(index=irp_mapping.index)
        relevant_contracts = pd.Series([[] for _ in irp_mapping.index], index=irp_mapping.index)
        relevant_times = pd.Series([[] for _ in irp_mapping.index], index=irp_mapping.index)
        for instr, irp in irp_mapping.items():
            meeting_dates_for_instr = [d for d in self._ir_data.loc[irp, "MEETING_DATE"] if todays_date < d < end.loc[instr]]
            meeting_dates.loc[instr] = meeting_dates_for_instr if meeting_dates_for_instr else None
            if meeting_dates_for_instr:
                n_of_meeting_dates = len(meeting_dates.loc[instr])
            else:
                n_of_meeting_dates = 0

            relevant_contracts.loc[instr].append(irs_mapping.loc[instr])
            relevant_times.loc[instr].append(todays_date)
            for i in range(0, n_of_meeting_dates):
                d = meeting_dates.loc[instr][i]
                relevant_times.loc[instr].append(d)
                relevant_contracts.loc[instr].append(self._irp_date_contract_mapping[(irp_mapping.loc[instr], d)])
            relevant_times.loc[instr].append(end[instr])

        for instr, contracts in relevant_contracts.items():
            total_time = max((relevant_times.loc[instr][-1] - relevant_times.loc[instr][0]).days, 1)
            for i, contract in enumerate(contracts):
                time_interval = (relevant_times.loc[instr][i + 1] - relevant_times.loc[instr][i]).days
                average_rate.loc[instr] += book.loc[contract] * time_interval / total_time

        return average_rate / 100

    def save_historical_prices(self, historical_prices):
        missing = [c for c in self._irp_contracts_list + self._irs_contracts_list if c not in historical_prices]
        assert not missing, f"Missing prices for the following contracts: {missing}"

        self._historical_prices = historical_prices