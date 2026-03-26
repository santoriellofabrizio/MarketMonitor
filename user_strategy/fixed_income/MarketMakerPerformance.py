from dataclasses import dataclass, field
from typing import Literal, Optional, List
import logging

from market_monitor.live_data_hub.order import Order
from market_monitor.live_data_hub.subscription_service import SubscriptionService
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI

logger = logging.getLogger(__name__)


@dataclass
class QuotePerformance:
    isin: str
    market: str
    at_best_bid: bool = False
    at_best_ask: bool = False
    bid_order_price: Optional[float] = None
    ask_order_price: Optional[float] = None
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None

    @property
    def market_spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def our_spread(self) -> Optional[float]:
        if self.bid_order_price and self.ask_order_price:
            return self.ask_order_price - self.bid_order_price
        return None

    @property
    def is_two_sided(self) -> bool:
        return self.bid_order_price is not None and self.ask_order_price is not None


class MarketMakerPerformance(StrategyUI):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._isins = ["DE000F21VFM0"]
        self._best_level = {
            'ETFP': {'BID': {}, 'ASK': {}},
            'XEUR': {'BID': {}, 'ASK': {}},
            'XAMS': {'BID': {}, 'ASK': {}},
            'XPAR': {'BID': {}, 'ASK': {}},
        }
        self._isin_market_mapping: dict[str, Literal['ETFP', 'XEUR', 'XAMS', 'XPAR']] = {"DE000F21VFM0": "XEUR"}
        self._performance: dict[str, QuotePerformance] = {}  # key: f"{isin}:{market}"
        self.global_subscription_service = SubscriptionService()

    def on_market_data_setting(self):
        self.subscribe_orders()
        self.subscribe_best_book()

    # --- properties ---

    @property
    def isins(self) -> List[str]:
        return self._isins

    @property
    def best_level(self):
        return self._best_level

    @property
    def performance(self) -> dict[str, QuotePerformance]:
        return self._performance

    # --- subscriptions ---

    def subscribe_best_book(self):
        for isin, market in self._isin_market_mapping.items():
            self.global_subscription_service.subscribe_kafka(
                id=f"{isin}:{market}:",
                symbol_filter=isin,
                topic=f"COALESCENT_DUMA.{market}.BookBest",
                fields_mapping={
                    "BID": "bidBestLevel.price",
                    "ASK": "askBestLevel.price",
                    "BID_SIZE": "bidBestLevel.quantity",
                    "ASK_SIZE": "askBestLevel.quantity",
                })

    def subscribe_orders(self):
        for isin, market in self._isin_market_mapping.items():
            self.global_subscription_service.subscribe_orders_kafka(
                id=f"{isin}:{market}:Order",
                symbol_filter=isin,
                topic=f"COALESCENT_DUMA.{market}.Order",
                fields_mapping={})

    # --- HF update ---

    def update_HF(self):
        best_level = self.market_data.get_data_field(["BID", "ASK"])
        for el, val in best_level.items():
            isin, market, _ = el.split(":")
            bid = val.get("BID")
            ask = val.get("ASK")
            if bid is not None:
                self._best_level[market]["BID"][isin] = bid
            if ask is not None:
                self._best_level[market]["ASK"][isin] = ask

        orders = self.get_orders()
        self.check_market_making_performance(orders)

    def get_orders(self) -> List[Order]:
        """Thin wrapper — override per filtrare/mockare in test."""
        return self.market_data.get_orders()

    # --- performance check ---

    def check_market_making_performance(self, orders: List[Order]):
        """
        Per ogni ISIN monitorato, verifica se i nostri ordini attivi
        sono in linea con il best book (top-of-book presence, spread).
        """
        # Raggruppa ordini attivi per (isin, market) -> {side: best_price}
        active_quotes: dict[str, dict[str, float]] = {}

        for order in orders:
            if not self._is_active(order):
                continue

            key = f"{order.isin}:{order.market}"
            side = order.side.upper()  # "BUY" / "SELL"

            if key not in active_quotes:
                active_quotes[key] = {}

            # Teniamo il prezzo più aggressivo per ciascun lato
            if side == "BUY":
                active_quotes[key]["BID"] = max(
                    active_quotes[key].get("BID", float('-inf')), order.price)
            elif side == "SELL":
                active_quotes[key]["ASK"] = min(
                    active_quotes[key].get("ASK", float('inf')), order.price)

        # Costruisci/aggiorna il report di performance
        for isin, market in self._isin_market_mapping.items():
            key = f"{isin}:{market}"
            best_bid = self._best_level.get(market, {}).get("BID", {}).get(isin)
            best_ask = self._best_level.get(market, {}).get("ASK", {}).get(isin)
            our_quotes = active_quotes.get(key, {})
            our_bid = our_quotes.get("BID")
            our_ask = our_quotes.get("ASK")

            perf = QuotePerformance(
                isin=isin,
                market=market,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_order_price=our_bid,
                ask_order_price=our_ask,
                at_best_bid=self._is_at_best(our_bid, best_bid, side="BID"),
                at_best_ask=self._is_at_best(our_ask, best_ask, side="ASK"),
            )

            self._performance[key] = perf
            self._log_performance(perf)

    # --- helpers ---

    @staticmethod
    def _is_active(order: Order) -> bool:
        """Filtra solo ordini in stato attivo (adatta a seconda dell'enum di Order)."""
        return getattr(order, "status", None) in ("ACTIVE", "PARTIALLY_FILLED", "NEW")

    @staticmethod
    def _is_at_best(our_price: Optional[float],
                    market_price: Optional[float],
                    side: Literal["BID", "ASK"]) -> bool:
        if our_price is None or market_price is None:
            return False
        if side == "BID":
            return our_price >= market_price
        return our_price <= market_price

    @staticmethod
    def _log_performance(perf: QuotePerformance):
        if not perf.is_two_sided:
            logger.warning(
                "[%s:%s] Quote unilaterale — BID=%s ASK=%s",
                perf.isin, perf.market, perf.bid_order_price, perf.ask_order_price)

        if not perf.at_best_bid:
            logger.debug(
                "[%s:%s] Non al best BID: nostro=%.4f market=%.4f",
                perf.isin, perf.market, perf.bid_order_price or 0, perf.best_bid or 0)

        if not perf.at_best_ask:
            logger.debug(
                "[%s:%s] Non al best ASK: nostro=%.4f market=%.4f",
                perf.isin, perf.market, perf.ask_order_price or 0, perf.best_ask or 0)