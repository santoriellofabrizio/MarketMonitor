"""
Dataclass condivisi tra MarketMakerPerformance e BookLevelDisplay.
Separati per evitare import circolare.
"""

from dataclasses import dataclass
from typing import Optional


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
