from typing import Literal, Optional, List
import logging

from market_monitor.live_data_hub.order import Order
from market_monitor.live_data_hub.subscription_service import SubscriptionService
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI

from user_strategy.fixed_income._mm_models import EurexMMRequirements, MMComplianceTracker, QuotePerformance
from user_strategy.fixed_income.BookLevelDisplay import BookLevelDisplay

logger = logging.getLogger(__name__)


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
        self._isin_market_mapping: dict[str, Literal['ETFP', 'XEUR', 'XAMS', 'XPAR']] =\
            kwargs.get('instrument_market_mapping', {})
        self._isins = list(self._isin_market_mapping.keys())
        self._performance: dict[str, QuotePerformance] = {}  # key: f"{isin}:{market}"
        self.global_subscription_service = SubscriptionService()

        # Carica requisiti Eurex dal config YAML
        requirements_cfg = kwargs.get('market_maker_requirements', {})
        self._requirements: dict[str, EurexMMRequirements] = {}
        for market in self._best_level:
            if market not in requirements_cfg:
                logger.warning(f"[MARKET] {market} market requirements not configured. skipping.")
                continue
            self._requirements[market] = EurexMMRequirements(**requirements_cfg[market])

        self._compliance: dict[str, MMComplianceTracker] = {}  # key: f"{isin}:{market}"
        self._display = BookLevelDisplay()

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

    @property
    def compliance(self) -> dict[str, MMComplianceTracker]:
        return self._compliance

    # --- subscriptions ---

    def subscribe_best_book(self):
        for isin, market in self._isin_market_mapping.items():
            self.global_subscription_service.subscribe_kafka(
                id=f"{isin}:{market}:Book",
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
        for el, (bid, ask) in best_level.iterrows():
            isin, market, _ = el.split(":")
            if bid is not None:
                self._best_level[market]["BID"][isin] = bid
            if ask is not None:
                self._best_level[market]["ASK"][isin] = ask

        orders = self.get_orders()
        self.check_market_making_performance(orders)
        self._display.render(self._performance, self._compliance)


    def get_orders(self) -> List[Order]:
        """Thin wrapper — override per filtrare/mockare in test."""
        return self.market_data.get_orders()

    # --- performance check ---

    def check_market_making_performance(self, orders: List[Order]):
        active_quotes: dict[str, dict[str, tuple[float, float]]] = {}
        for order in orders:
            if order.orderStatus != "ACTIVE":
                continue
            key = f"{order.isin}:{order.instrument['market']}"
            side = order.side.upper()
            quotes = active_quotes.setdefault(key, {})
            if side == "BID":
                prev_price, prev_qty = quotes.get("BID", (float('-inf'), 0.0))
                quotes["BID"] = (max(prev_price, order.price), prev_qty + order.quantity)
            elif side == "ASK":
                prev_price, prev_qty = quotes.get("ASK", (float('inf'), 0.0))
                quotes["ASK"] = (min(prev_price, order.price), prev_qty + order.quantity)

        for isin, market in self._isin_market_mapping.items():
            key = f"{isin}:{market}"
            best_bid, best_ask = self._best_level.get(f"{market}:{isin}", (None, None))
            our_quotes = active_quotes.get(key, {})

            bid_price, bid_qty = our_quotes.get("BID", (None, None))
            ask_price, ask_qty = our_quotes.get("ASK", (None, None))
            if bid_price == float('-inf'):
                bid_price = bid_qty = None
            if ask_price == float('inf'):
                ask_price = ask_qty = None

            perf = QuotePerformance(
                isin=isin, market=market,
                best_bid=best_bid, best_ask=best_ask,
                bid_order_price=bid_price, ask_order_price=ask_price,
                bid_order_quantity=bid_qty, ask_order_quantity=ask_qty,
            )
            self._performance[key] = perf

            tracker = self._compliance.setdefault(key, MMComplianceTracker())
            tracker.total_ticks += 1
            if perf.is_compliant:
                tracker.compliant_ticks += 1
    # --- helpers ---

    @staticmethod
    def _is_active(order: Order) -> bool:
        """Filtra solo ordini in stato attivo (adatta a seconda dell'enum di Order)."""
        return order.orderStatus == "ACTIVE"

    @staticmethod
    def _is_at_best(our_price: Optional[float],
                    market_price: Optional[float],
                    side: Literal["BID", "ASK"]) -> bool:
        if our_price is None or market_price is None:
            return False
        if side == "BID":
            return our_price >= market_price
        return our_price <= market_price

