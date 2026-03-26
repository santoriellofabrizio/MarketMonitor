from dataclasses import dataclass, field
from typing import Literal, Optional, List
import logging

from market_monitor.live_data_hub.order import Order
from market_monitor.live_data_hub.subscription_service import SubscriptionService
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI

logger = logging.getLogger(__name__)


@dataclass
class EurexMMRequirements:
    """Requisiti Eurex per il market making su credit futures."""
    max_spread_pct: float   # spread massimo come % del mid price
    min_quantity: float      # quantità minima per lato (bid e ask)
    min_time_fraction: float # frazione minima del tempo in cui i requisiti devono essere soddisfatti


@dataclass
class MMComplianceTracker:
    """Traccia il tempo cumulativo di compliance dall'inizio della sessione."""
    total_ticks: int = 0
    compliant_ticks: int = 0

    @property
    def compliance_ratio(self) -> float:
        if self.total_ticks == 0:
            return 0.0
        return self.compliant_ticks / self.total_ticks


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
    bid_order_quantity: Optional[float] = None
    ask_order_quantity: Optional[float] = None
    meets_spread_req: bool = False
    meets_quantity_req: bool = False

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
    def our_spread_pct(self) -> Optional[float]:
        if self.bid_order_price and self.ask_order_price:
            mid = (self.bid_order_price + self.ask_order_price) / 2
            if mid:
                return (self.ask_order_price - self.bid_order_price) / mid
        return None

    @property
    def is_two_sided(self) -> bool:
        return self.bid_order_price is not None and self.ask_order_price is not None

    @property
    def is_compliant(self) -> bool:
        return self.is_two_sided and self.meets_spread_req and self.meets_quantity_req


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

    def get_orders(self) -> List[Order]:
        """Thin wrapper — override per filtrare/mockare in test."""
        return self.market_data.get_orders()

    # --- performance check ---

    def check_market_making_performance(self, orders: List[Order]):
        """
        Per ogni ISIN monitorato, verifica se i nostri ordini attivi
        soddisfano i requisiti Eurex per il market making su credit futures:
        - spread bid-ask <= max_spread_pct del mid price
        - quantità >= min_quantity per entrambi i lati
        Aggiorna il tracker cumulativo di compliance.
        """
        # Raggruppa ordini attivi per (isin, market) -> {side: (best_price, total_qty)}
        active_quotes: dict[str, dict[str, tuple[float, float]]] = {}

        for order in orders:
            if not self._is_active(order):
                continue

            key = f"{order.isin}:{order.instrument['market']}"
            side = order.side.upper()

            if key not in active_quotes:
                active_quotes[key] = {}

            # Teniamo il prezzo più aggressivo e la quantità totale per ciascun lato
            if side == "BID":
                prev_price, prev_qty = active_quotes[key].get("BID", (float('-inf'), 0.0))
                new_price = max(prev_price, order.price)
                active_quotes[key]["BID"] = (new_price, prev_qty + order.quantity)
            elif side == "ASK":
                prev_price, prev_qty = active_quotes[key].get("ASK", (float('inf'), 0.0))
                new_price = min(prev_price, order.price)
                active_quotes[key]["ASK"] = (new_price, prev_qty + order.quantity)

        # Costruisci/aggiorna il report di performance
        for isin, market in self._isin_market_mapping.items():
            key = f"{isin}:{market}"
            best_bid = self._best_level.get(market, {}).get("BID", {}).get(isin)
            best_ask = self._best_level.get(market, {}).get("ASK", {}).get(isin)
            our_quotes = active_quotes.get(key, {})

            our_bid, our_bid_qty = our_quotes.get("BID", (None, None))
            our_ask, our_ask_qty = our_quotes.get("ASK", (None, None))
            # Converti i sentinel values (inf/-inf) a None se non ci sono ordini
            if our_bid == float('-inf'):
                our_bid, our_bid_qty = None, None
            if our_ask == float('inf'):
                our_ask, our_ask_qty = None, None

            req = self._requirements.get(market)
            # Verifica requisito spread (% del mid)
            mid = (our_bid + our_ask) / 2 if our_bid is not None and our_ask is not None else None
            our_spread_pct = (our_ask - our_bid) / mid if mid else None
            meets_spread = our_spread_pct is not None and our_spread_pct <= req.max_spread_pct

            # Verifica requisito quantità minima per lato
            meets_qty = (
                our_bid_qty is not None and our_ask_qty is not None
                and our_bid_qty >= req.min_quantity and our_ask_qty >= req.min_quantity
            )

            perf = QuotePerformance(
                isin=isin,
                market=market,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_order_price=our_bid,
                ask_order_price=our_ask,
                bid_order_quantity=our_bid_qty,
                ask_order_quantity=our_ask_qty,
                at_best_bid=self._is_at_best(our_bid, best_bid, side="BID"),
                at_best_ask=self._is_at_best(our_ask, best_ask, side="ASK"),
                meets_spread_req=meets_spread,
                meets_quantity_req=meets_qty,
            )

            self._performance[key] = perf

            # Aggiorna tracker cumulativo
            tracker = self._compliance.setdefault(key, MMComplianceTracker())
            tracker.total_ticks += 1
            if perf.is_compliant:
                tracker.compliant_ticks += 1

            self._log_performance(perf, tracker, req)

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

    @staticmethod
    def _log_performance(perf: QuotePerformance, tracker: MMComplianceTracker,
                         req: EurexMMRequirements):
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

        if not perf.meets_spread_req:
            logger.debug(
                "[%s:%s] Spread fuori requisito: %.4f%% > %.4f%% (max consentito)",
                perf.isin, perf.market,
                (perf.our_spread_pct or 0) * 100, req.max_spread_pct * 100)

        if not perf.meets_quantity_req:
            logger.debug(
                "[%s:%s] Quantità insufficiente: BID=%s ASK=%s (minimo %.0f)",
                perf.isin, perf.market,
                perf.bid_order_quantity, perf.ask_order_quantity, req.min_quantity)

        if tracker.compliance_ratio < req.min_time_fraction:
            logger.warning(
                "[%s:%s] Compliance cumulativa %.1f%% < soglia %.1f%% (%d/%d ticks)",
                perf.isin, perf.market,
                tracker.compliance_ratio * 100, req.min_time_fraction * 100,
                tracker.compliant_ticks, tracker.total_ticks)
