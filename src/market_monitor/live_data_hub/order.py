"""
Order dataclass for Kafka order stream data.

Fields match the Kafka order message format (COALESCENT_DUMA order topic).
Only ACTIVE orders are kept in memory; EXPIRED and CANCELLED orders are discarded.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Order:
    """
    Represents a single order received from the Kafka order stream.

    Attributes:
        clientOrderId: Client-assigned order identifier (may be empty string)
        dataSource: Source of the order data (e.g., "COALESCENT_DUMA")
        decisionMakerShortcode: Shortcode of the decision maker
        decisionMakerShortcodeType: Type of decision maker (e.g., "NATURAL_PERSON")
        eventTimestampUTC: Event timestamp in UTC nanoseconds
        executingTraderShortcode: Shortcode of the executing trader
        executingTraderShortcodeType: Type of executing trader (e.g., "ALGORITHM")
        expiryTimestampUTC: Order expiry timestamp in UTC (0 if no expiry)
        instrument: Nested dict with keys: currency, isin, market, symbol
        liquidityProvision: Liquidity role (e.g., "MARKET_MAKER")
        mktOrderId: Market-assigned unique order identifier
        orderStatus: Current status — "ACTIVE", "EXPIRED", or "CANCELLED"
        orderType: Order type (e.g., "LIMIT")
        price: Order price
        quantity: Order quantity
        quantityDisplayed: Displayed quantity (may be NaN for iceberg orders)
        quantityFilled: Quantity already filled
        selfMatchPreventionId: Self-match prevention identifier (may be empty string)
        side: Order side — "BID" or "ASK"
        timeInForce: Time-in-force policy (e.g., "DAY")
    """

    clientOrderId: str
    dataSource: str
    decisionMakerShortcode: str
    decisionMakerShortcodeType: str
    eventTimestampUTC: int
    executingTraderShortcode: str
    executingTraderShortcodeType: str
    expiryTimestampUTC: int
    instrument: Dict[str, str]
    liquidityProvision: str
    mktOrderId: str
    orderStatus: str
    orderType: str
    price: float
    quantity: float
    quantityDisplayed: float
    quantityFilled: float
    selfMatchPreventionId: str
    side: str
    timeInForce: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        """
        Create an Order from a raw Kafka message dict.

        Handles missing keys (defaults to sensible zero/empty values) and
        NaN float values (e.g., quantityDisplayed for iceberg orders).

        Args:
            data: Deserialized Kafka message dict

        Returns:
            Order instance
        """

        def _float(value: Any, default: float = 0.0) -> float:
            try:
                v = float(value)
                return v if not math.isnan(v) else default
            except (TypeError, ValueError):
                return default

        def _int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _str(value: Any, default: str = "") -> str:
            return str(value) if value is not None else default

        instrument = data.get("instrument") or {}

        return cls(
            clientOrderId=_str(data.get("clientOrderId")),
            dataSource=_str(data.get("dataSource")),
            decisionMakerShortcode=_str(data.get("decisionMakerShortcode")),
            decisionMakerShortcodeType=_str(data.get("decisionMakerShortcodeType")),
            eventTimestampUTC=_int(data.get("eventTimestampUTC")),
            executingTraderShortcode=_str(data.get("executingTraderShortcode")),
            executingTraderShortcodeType=_str(data.get("executingTraderShortcodeType")),
            expiryTimestampUTC=_int(data.get("expiryTimestampUTC")),
            instrument=dict(instrument),
            liquidityProvision=_str(data.get("liquidityProvision")),
            mktOrderId=_str(data.get("mktOrderId")),
            orderStatus=_str(data.get("orderStatus")),
            orderType=_str(data.get("orderType")),
            price=_float(data.get("price")),
            quantity=_float(data.get("quantity")),
            quantityDisplayed=_float(data.get("quantityDisplayed"), default=float("nan")),
            quantityFilled=_float(data.get("quantityFilled")),
            selfMatchPreventionId=_str(data.get("selfMatchPreventionId")),
            side=_str(data.get("side")),
            timeInForce=_str(data.get("timeInForce")),
        )

    @property
    def is_active(self) -> bool:
        """True if the order status is ACTIVE."""
        return self.orderStatus == "ACTIVE"

    @property
    def isin(self) -> Optional[str]:
        """ISIN from the nested instrument dict, or None."""
        return self.instrument.get("isin")

    @property
    def symbol(self) -> Optional[str]:
        """Symbol from the nested instrument dict, or None."""
        return self.instrument.get("symbol")

    def __repr__(self) -> str:
        return (
            f"Order(mktOrderId={self.mktOrderId!r}, side={self.side!r}, "
            f"price={self.price}, quantity={self.quantity}, "
            f"status={self.orderStatus!r}, isin={self.isin!r})"
        )
