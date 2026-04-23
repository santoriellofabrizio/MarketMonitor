import datetime as dt
import logging
from abc import abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Literal

import pandas as pd

from market_monitor.strategy.common.trade_manager.book_memory import FairvaluePrice
from market_monitor.utils.book_utils import SpreadEWMA, PriceEWMA
from user_strategy.utils.TradeAnalysisBase import TradeAnalysisBase

logger = logging.getLogger(__name__)


class CreditTradeAnalysis(TradeAnalysisBase):

    def __init__(self, **kwargs) -> None:
        book_filter_params = kwargs.pop("book_filter_params", {})
        super().__init__(**kwargs)

        path_str = kwargs.get("instruments_list_path")
        excel_path = Path(path_str) if path_str else None

        self.id_for_each_isin = defaultdict(list)
        self.mid: dict[str, FairvaluePrice] = {}

        if excel_path and excel_path.exists():
            self.instruments_df = pd.read_excel(excel_path).set_index("isin")
            self.all_isin = [i.upper().strip() for i in self.instruments_df.keys()]
        else:
            raise Exception("missing valid input")

        self.instruments = [f"{instr}_{ccy}" for instr in self.instruments_df.index for ccy in ['EUR', 'USD']]
        # -------------------------------------- ETF SECTION -----------------------------------------------------------
        self.etf_markets = ["ETFP", "XPAR", "XAMS"]
        self.etf_isins = list(self.instruments_df[self.instruments_df["type"] == "ETF"].index)
        # -------------------------------------- BOND SECTION ----------------------------------------------------------
        self.bond_markets = ["ETLX", "XMOT", "MOTX"]
        self.bond_isins = list(self.instruments_df[self.instruments_df["type"] == "BOND"].index)
        # -------------------------------------- FUTURE SECTION --------------------------------------------------------
        self.future_markets = ["XEUR"]
        self.future_isins = list(self.instruments_df[self.instruments_df["type"] == "FUTURE"].index)
        # -------------------------------------- TRADE SECTION ---------------------------------------------------------
        self.trade_isin_multiplier_mapping = self.instruments_df["multiplier"].to_dict()
        self.trade_isin_description_mapping = self.instruments_df["description"].to_dict()
        # -------------------------------------- BOOK & PRICE SECTION --------------------------------------------------
        self.book_filter = SpreadEWMA(**book_filter_params)
        # -------------------------------------- SETTING INSTRUMENTS ---------------------------------------------------
        self.columns_dashboard = kwargs.get("columns_dashboard")

    def wait_for_book_initialization(self):
        while datetime.today().time() < dt.time(8, 50):
            return False
        while True:
            data = self.market_data.get_data_field(field=["BID", "ASK"])
            if data is not None and not data.empty:
                break
            sleep(1)
        self.on_start_strategy()
        return True

    def _pre_start_setup(self) -> None:
        for market in self.etf_markets:
            for etf in self.etf_isins:
                self.id_for_each_isin[etf].append(f"{market}_{etf}")

        for market in self.bond_markets:
            for bond in self.bond_isins:
                currency = self.market_data.currency_information.get(bond, "EUR")
                self.id_for_each_isin[f"{bond}_EUR"].append(f"{market}_{bond}")

    def on_market_data_setting(self) -> None:
        if self.price_source == 'kafka':
            self.subscribe_kafka()
        else:
            self.subscribe_bloomberg()

    def subscribe_bloomberg(self):

        markets_mapping = {
            "ETFP": "IM",
            "XAMS": "NA",
            "XPAR": "FP",
            "MOTX": "MILA",
            "ETLX": "ETLX"}

        for markets, symbols in (
                (self.etf_markets, self.etf_isins),
                (self.bond_markets, self.bond_isins),
        ):
            for m in markets:
                for s in symbols:
                    if s in self.etf_isins:
                        sub_string = f"{s} {markets_mapping[m]} EQUITY"
                    else:
                        sub_string = f"{s} ISIN@{markets_mapping[m]} Corp"

                    self.global_subscription_service.subscribe_bloomberg(
                        id=f"{m}_{s}",
                        subscription_string=sub_string,
                    )

    def subscribe_kafka(self):
        fields = {
            "BID": "bidBestLevel.price",
            "ASK": "askBestLevel.price",
            "BID_SIZE": "bidBestLevel.quantity",
            "ASK_SIZE": "askBestLevel.quantity",
        }

        for markets, symbols in (
                (self.etf_markets, self.etf_isins),
                (self.bond_markets, self.bond_isins),
                (self.future_markets, self.future_isins),
        ):
            for m in markets:
                base = f"COALESCENT_DUMA.{m}"
                for s in symbols:
                    self.global_subscription_service.subscribe_kafka(
                        id=f"{m}_{s}",
                        symbol_filter=s,
                        topic=f"{base}.BookBest",
                        fields_mapping=fields,
                    )
                    for ev in ("PublicDeal", "Trade"):
                        self.global_subscription_service.subscribe_trades_kafka(
                            id=f"{m}_{s}:{ev}",
                            symbol_filter=s,
                            topic=f"{base}.{ev}"
                        )

    def _enrich_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        trades["price_multiplier"] = trades["isin"].map(self.trade_isin_multiplier_mapping)
        trades["description"] = trades["isin"].map(self.trade_isin_description_mapping)
        return trades

    def get_live_data(self):
        raw = self.market_data.get_data_field(field=["BID", "ASK"])
        if raw is None or raw.empty:
            return

        # Filtro rapido e validazione
        valid = self.book_filter.get_valid_book(raw.dropna(subset=["BID", "ASK"]))
        if valid.empty:
            return

        grouped = {}
        for inst_id, row in valid.iterrows():
            # Split ottimizzato: se sai che è sempre il secondo elemento
            isin = inst_id.split("_", 1)[1]
            ccy = self.market_data.currency_information.get(inst_id, "EUR")

            # Aggregazione compatta
            d = grouped.setdefault(isin, {}).setdefault(ccy, {"bids": [], "asks": []})
            d["bids"].append(row["BID"])
            d["asks"].append(row["ASK"])

        # Aggiornamento FairvaluePrice
        for isin, ccy_prices in grouped.items():
            tracker = self.mid.setdefault(isin, FairvaluePrice.by_currency(isin, {}))
            for ccy, book in ccy_prices.items():
                b, a = book["bids"], book["asks"]
                if b and a:
                    tracker.update_price((max(b) + min(a)) / 2, currency=ccy)

        # Append dello stato attuale
        self.save_mid(self.mid.copy())

    def _post_trade_processing(self, processed: pd.DataFrame) -> None:
        self.publish_trades_on_dashboard(self.trade_manager.get_trades_to_publish())

    @property
    def instruments(self) -> list:
        return self._instruments

    @instruments.setter
    def instruments(self, value: list):
        self._instruments = value

    @instruments.getter
    def instruments(self) -> list[str]:
        return self._instruments
