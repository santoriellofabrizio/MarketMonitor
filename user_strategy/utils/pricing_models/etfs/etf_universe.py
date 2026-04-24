import sqlite3
from collections import defaultdict

import pandas as pd

from sfm_data_provider.core.enums.instrument_types import InstrumentType
from sfm_data_provider.core.instruments.instruments import Instrument
from sfm_data_provider.interface.bshdata import BshData

_ETF_MARKETS = ("NA", "FP", "IM")


class EtfUniverse:
    """
    Owns the full instrument catalog for the fixed-income strategy.

    Built once at startup; afterwards treated as read-only by all consumers.
    Exposes three logical groups:
      - ETF geometry   : etfs_by_market, markets_by_isin, currency_per_isin_market, all_etf_isin
      - Instrument set : all_instruments, instruments_by_type, factored_instruments
      - DB overrides   : load_credit_futures(), load_ytm_override(), load_fx_override()
    """

    def __init__(self, api: BshData, db_path: str, start_date, end) -> None:
        self.api        = api
        self.db_path    = db_path
        self.start_date = start_date
        self.end        = end
        self._build()

    # ── Public build entry-point ──────────────────────────────────────────────

    def _build(self) -> None:
        self._load_etf_universe_from_oracle()
        self.all_etf_isin  = sorted({isin for isins in self.etfs_by_market.values() for isin in isins})
        self.isin_ticker   = self.api.info.get_etp_fields("TICKER", isin=self.all_etf_isin)["TICKER"].to_dict()
        self.all_instruments: set[str] = self._db_query_set("SELECT INSTRUMENT_ID FROM InstrumentsAnagraphic")
        self._build_instruments()

    # ── ETF geometry ──────────────────────────────────────────────────────────

    def _load_etf_universe_from_oracle(self) -> None:
        self.etfs_by_market:           dict[str, list[str]] = {mkt: [] for mkt in _ETF_MARKETS}
        self.currency_per_isin_market: dict[tuple, str]     = {}
        self.markets_by_isin:          dict[str, list[str]] = defaultdict(list)

        for mkt in _ETF_MARKETS:
            universe = self.api.general.get(
                fields=["etp_isins"], segments=[mkt], currency="EUR",
                underlying=["FIXED INCOME", "MONEY MARKET"],
                extra_fields=["CURRENCY"], source="oracle",
            )["etp_isins"]
            for isin, fields in universe.items():
                self.etfs_by_market[mkt].append(isin)
                self.markets_by_isin[isin].append(mkt)
                if currency := fields.get("CURRENCY"):
                    self.currency_per_isin_market[(isin, mkt)] = currency

    # ── Instrument set ────────────────────────────────────────────────────────

    def _build_instruments(self) -> None:
        self.instruments_by_type: dict[InstrumentType, list[Instrument]] = defaultdict(list)
        self.factored_instruments: list[Instrument] = []

        raw   = self.api.market.build_instruments(list(self.all_instruments), autocomplete=True)
        total = len(raw)
        for i, inst in enumerate(raw):
            self.instruments_by_type[inst.type].append(inst)
            self.factored_instruments.append(inst)
            self.api.market.register(inst, )
            print(f"\rBuilding instruments {i}/{total} | {'█' * (i * 20 // total):20} | {i / total:>4.0%}",
                  end="", flush=True)
