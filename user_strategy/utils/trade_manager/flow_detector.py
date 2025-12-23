"""
FlowDetector - Identifies coordinated trading flows in market data.

## Overview
A "flow" is a series of coordinated trades characterized by:
- Similar quantities (within a tolerance)
- Consistent time intervals between trades
- Same side (buy/sell direction)
- Statistical consistency measures

## Use Cases
- Detect algorithmic trading patterns
- Identify institutional order execution
- Monitor coordinated market activity
- Alert on unusual trading behavior

## Example Usage

### Basic Flow Detection
```python
from flow_detector import FlowDetector

# Initialize detector
detector = FlowDetector(
    min_trades=3,           # Minimum trades to form a flow
    qty_tolerance=0.3,      # 30% quantity deviation allowed
    time_tolerance=2.0,     # 2x interval deviation allowed
    max_gap=10.0,           # Close flow after 10s of inactivity
    min_total_qty=100.0     # Minimum total quantity
)

# Process new trades
new_trades_df = pd.DataFrame([...])
detected_flows = detector.process_trades(new_trades_df)

# Get active flows
active = detector.get_active_flows()
for flow in active:
    print(f"Flow {flow.flow_id}: {flow.total_quantity} qty, "
          f"{len(flow.trades)} trades, consistency={flow.consistency_score:.2f}")
```

### Monitor for New Flows
```python
# Poll for newly detected flows
if detector.has_new_flows():
    new_flows = detector.get_new_flows()
    for flow in new_flows:
        print(f"🔔 New flow detected: {flow.flow_id}")
        # Send alert, log, or update UI
```

### Integration with Dashboard
```python
# In your market data processor
def on_new_trades(trades_df):
    # Detect flows
    flows = flow_detector.process_trades(trades_df)

    # Export to gui
    if flow_detector.has_new_flows():
        new_flows = flow_detector.get_new_flows()
        for flow in new_flows:
            gui.export_command('flow_detected', data=flow.to_dict())
```

## Flow Lifecycle

1. **Detection**: Seed trade matches pattern in recent history
2. **Growth**: Subsequent trades added if they match criteria
3. **Active**: Flow continues receiving matching trades
4. **Closure**: No new trades for `max_gap` seconds
5. **Completion**: Moved to completed_flows list

## Configuration Parameters

### Detection Sensitivity
- `min_trades` (default=3): Minimum trades to form valid flow
  - Lower = more sensitive, more false positives
  - Higher = stricter, fewer detections

### Quantity Matching
- `qty_tolerance` (default=0.3): Max quantity deviation (fraction)
  - 0.3 = allow ±30% from average
  - Lower = stricter quantity matching

### Timing Matching
- `time_tolerance` (default=2.0): Max interval deviation (multiplier)
  - 2.0 = allow up to 2x average interval
  - Lower = stricter timing consistency

### Flow Management
- `max_gap` (default=10.0): Seconds of inactivity before closing flow
- `min_total_qty` (default=100.0): Minimum total quantity to report
- `lookback_window` (default=100): Number of recent trades to analyze

## Flow Statistics

### Consistency Score (0-1)
Measures how regular the flow pattern is:
- **1.0**: Perfect consistency (identical quantities and intervals)
- **0.8-1.0**: Very consistent (high-frequency algo)
- **0.5-0.8**: Moderately consistent (typical flow)
- **<0.5**: Low consistency (may not be coordinated)

Formula: `1.0 / (1.0 + qty_cv + interval_cv)`
where CV = coefficient of variation (std/mean)

### Other Metrics
- `avg_quantity`: Average trade size
- `total_quantity`: Cumulative quantity
- `avg_interval`: Average time between trades (seconds)
- `duration`: Total time from first to last trade
- `quantity_std`: Standard deviation of quantities
- `interval_std`: Standard deviation of intervals

## Advanced Usage

### Custom Flow Filtering
```python
# Get only high-consistency flows
significant = detector.get_significant_flows(min_consistency=0.7)

# Filter by side
buy_flows = [f for f in active if f.side == 'buy']

# Filter by volume
large_flows = [f for f in active if f.total_quantity > 1000]
```

### Flow Summary Statistics
```python
summary = detector.get_flow_summary()
print(f"Active: {summary['active_flows']}, "
      f"Buy: {summary['buy_flows']}, "
      f"Sell: {summary['sell_flows']}, "
      f"Volume: {summary['total_volume']}")
```

### Access Flow Details
```python
for flow in detector.get_active_flows():
    print(f"\\n=== {flow.flow_id} ===")
    print(f"Instrument: {flow.instrument_id}")
    print(f"Side: {flow.side}")
    print(f"Trades: {len(flow.trades)}")
    print(f"Avg Qty: {flow.avg_quantity:.2f}")
    print(f"Avg Interval: {flow.avg_interval:.2f}s")
    print(f"Consistency: {flow.consistency_score:.2f}")
    print(f"Duration: {flow.duration:.1f}s")

    # Access individual trades
    for trade in flow.trades:
        print(f"  {trade['timestamp']}: {trade['quantity']} @ {trade['price']}")
```

## Thread Safety
FlowDetector is **NOT thread-safe**. Use in single-threaded context or add locking:

```python
import threading

class ThreadSafeFlowDetector(FlowDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()

    def process_trades(self, new_trades):
        with self._lock:
            return super().process_trades(new_trades)
```

## Performance Considerations
- Memory: Stores last `lookback_window` trades (default 100)
- CPU: O(n*m) where n=new trades, m=active flows (typically <10)
- Recommended update frequency: 100ms - 1s for real-time detection

## Troubleshooting

### No Flows Detected
- Check `min_trades`: too high?
- Check `qty_tolerance`: too strict?
- Check `time_tolerance`: too strict?
- Check `min_total_qty`: threshold too high?
- Verify trades have required fields: timestamp, side, quantity

### Too Many False Positives
- Increase `min_trades` (e.g., 5)
- Decrease `qty_tolerance` (e.g., 0.2)
- Decrease `time_tolerance` (e.g., 1.5)
- Increase `min_total_qty`
- Filter by `consistency_score` (e.g., >0.7)

### Flows Not Closing
- Check `max_gap`: may be too large
- Verify timestamps are monotonically increasing
- Ensure `_close_stale_flows()` is being called

## API Reference
See class and method docstrings for detailed parameter descriptions.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

def _json_safe(obj: Any):
    """Convert any object to a JSON-serializable structure."""
    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Fallback: convert unknown objects to string
    return str(obj)



@dataclass
class Flow:
    """
    Represents a detected trading flow.

    Attributes:
        flow_id: Unique identifier (e.g., "FLOW_00001")
        instrument_id: Instrument identifier (ticker/ISIN)
        side: Trade direction ('buy' or 'sell')
        trades: List of trade dicts in this flow
        avg_quantity: Average trade quantity
        total_quantity: Sum of all trade quantities
        avg_interval: Average time between trades (seconds)
        start_time: Time of first trade
        end_time: Time of last trade
        duration: Total duration (seconds)
        quantity_std: Standard deviation of quantities
        interval_std: Standard deviation of intervals
        consistency_score: Flow consistency (0-1, higher=more regular)
        is_active: Whether flow is still receiving trades
    """
    flow_id: str
    instrument_id: str
    side: str  # 'buy' or 'sell'

    # Trade characteristics
    trades: List[dict] = field(default_factory=list)
    avg_quantity: float = 0.0
    total_quantity: float = 0.0
    avg_interval: float = 0.0  # seconds

    # Time bounds
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0  # seconds

    # Statistical measures
    quantity_std: float = 0.0
    interval_std: float = 0.0
    consistency_score: float = 0.0  # 0-1, higher = more consistent

    # Status
    is_active: bool = True

    def to_dict(self):
        """Return a 100% JSON-safe dict."""
        raw = asdict(self)
        return _json_safe(raw)

    def __post_init__(self):
        if self.trades:
            self._update_statistics()

    def to_card_data(self) -> dict:
        """
        Restituisce un dizionario sicuro per JSON contenente solo i campi
        necessari per l'aggiornamento di FlowCard.
        """
        # I campi necessari per FlowCard sono:
        # 1. Identificativi/Stato: flow_id, instrument_id (o ticker), side, is_active
        # 2. Metriche: trades (lunghezza lista), ctv, avg_interval, duration, consistency_score, avg_quantity

        # Nota: FlowCard utilizza 'ticker' come fallback se 'instrument_id' non c'è.
        # FlowCard calcola anche CTV se non fornito, ma lo forniremo per precisione.

        # Calcoliamo i campi che FlowCard richiede
        data = {
            'flow_id': self.flow_id,
            'instrument_id': self.instrument_id,
            'ticker': self.instrument_id,  # Fornisci il ticker/instrument_id anche come 'ticker'
            'side': self.side,
            'is_active': self.is_active,

            # Metriche
            'ctv': getattr(self, '_total_ctv', 0.0),  # Usa il CTV calcolato (se esiste)
            'avg_interval': self.avg_interval,
            'duration': self.duration,
            'consistency_score': self.consistency_score,
            'avg_quantity': self.avg_quantity,
        }

        # Utilizza la funzione _json_safe per gestire eventuali tipi numpy/datetime
        return _json_safe(data)

    def _update_statistics(self):
        """
        Recalculate flow statistics from current trades.

        Computes:
        - Average and total quantities
        - Average intervals between trades
        - Standard deviations
        - Consistency score (CV-based metric)
        """
        if not self.trades:
            return

        quantities = [t['quantity'] for t in self.trades]
        self.avg_quantity = np.mean(quantities)
        self.total_quantity = np.sum(quantities)
        self.quantity_std = np.std(quantities)

        if len(self.trades) > 1:
            times = [t['timestamp'] for t in self.trades]
            intervals = [(times[i+1] - times[i]).total_seconds()
                        for i in range(len(times)-1)]
            self.avg_interval = np.mean(intervals)
            self.interval_std = np.std(intervals)

            self.start_time = times[0]
            self.end_time = times[-1]
            self.duration = (self.end_time - self.start_time).total_seconds()

            # Consistency score based on coefficient of variation
            qty_cv = self.quantity_std / self.avg_quantity if self.avg_quantity > 0 else 1.0
            interval_cv = self.interval_std / self.avg_interval if self.avg_interval > 0 else 1.0
            self.consistency_score = 1.0 / (1.0 + qty_cv + interval_cv)

    def add_trade(self, trade: dict):
        """
        Add a trade to this flow and update statistics.

        Args:
            trade: Trade dict with keys: timestamp, side, quantity, price
        """
        self.trades.append(trade)
        self._update_statistics()

    def matches(self, trade: dict, qty_tolerance: float = 0.3,
                time_tolerance: float = 2.0) -> bool:
        """
        Check if a trade matches this flow pattern.

        A trade matches if:
        1. Same side (buy/sell)
        2. Quantity within tolerance of average
        3. Time interval within tolerance of average

        Args:
            trade: Trade dict with keys: timestamp, side, quantity, price
            qty_tolerance: Max deviation from avg quantity (as fraction, e.g., 0.3 = ±30%)
            time_tolerance: Max deviation from avg interval (as multiplier, e.g., 2.0 = up to 2x)

        Returns:
            True if trade matches flow pattern
        """
        if not self.trades:
            return False

        # Must be same side
        if trade['side'] != self.side:
            return False

        # Check quantity similarity
        qty_diff = abs(trade['quantity'] - self.avg_quantity) / self.avg_quantity
        if qty_diff > qty_tolerance:
            return False

        # Check time interval
        if len(self.trades) > 1:
            last_trade_time = self.trades[-1]['timestamp']
            interval = (trade['timestamp'] - last_trade_time).total_seconds()

            expected_interval = self.avg_interval
            if interval < 0:  # Out of order
                return False

            # Allow some flexibility in timing
            if interval > expected_interval * time_tolerance:
                return False

        return True

    def close(self):
        """Mark this flow as complete (no longer accepting trades)."""
        self.is_active = False
        self._update_statistics()


class FlowDetector:
    """
    Detects coordinated trading flows in market data.

    Args:
        min_trades: Minimum number of trades to form a valid flow (default: 3)
        qty_tolerance: Maximum quantity deviation as fraction (default: 0.3 = ±30%)
        time_tolerance: Maximum interval deviation as multiplier (default: 2.0 = up to 2x)
        max_gap: Seconds of inactivity before closing flow (default: 10.0)
        min_total_qty: Minimum total quantity to report flow (default: 100.0)
        lookback_window: Number of recent trades to keep in memory (default: 100)

    Example:
        ```python
        detector = FlowDetector(min_trades=3, qty_tolerance=0.3)
        flows = detector.process_trades(new_trades_df)

        if detector.has_new_flows():
            for flow in detector.get_new_flows():
                print(f"New flow: {flow.flow_id}, qty={flow.total_quantity}")
        ```
    """

    def __init__(
            self,
            min_trades: int = 3,
            qty_tolerance: float = 0.3,
            time_tolerance: float = 30000.0,
            max_gap: float = 6000.0,
            min_total_qty: float = 100.0,
            lookback_window: int = 10000,
    ):
        self.min_trades = min_trades
        self.qty_tolerance = qty_tolerance
        self.time_tolerance = time_tolerance
        self.max_gap = max_gap
        self.min_total_qty = min_total_qty
        self.lookback_window = lookback_window

        self.trade_history: List[dict] = []

        # Active flows being tracked
        self.active_flows: List[Flow] = []
        self.completed_flows: List[Flow] = []
        self.max_history = lookback_window

        # Flow ID counter
        self._flow_counter = 0

        # State tracking for external polling
        self._notified_flows = set()
        self._pending_new_flows = []

    def process_trades(self, new_trades: pd.DataFrame) -> List[Flow]:
        """
        Process new trades and detect flows.

        Analyzes ENTIRE recent history (lookback_window), not just new trades.
        Attempts to match new trades to existing flows or create new flows.

        Args:
            new_trades: DataFrame with columns: timestamp, side, quantity, price
                       Optional columns: ticker, isin

        Returns:
            List of Flow objects that were updated or created

        Note:
            - Automatically closes stale flows (no trades for max_gap seconds)
            - New flows are tracked internally for get_new_flows() polling
        """
        if new_trades.empty:
            return []

        new_trades = new_trades.sort_values('timestamp')
        for _, trade in new_trades.iterrows():
            self.trade_history.append(trade.to_dict())

        if len(self.trade_history) > self.lookback_window:
            self.trade_history = self.trade_history[-self.lookback_window:]

        all_trades_df = pd.DataFrame(self.trade_history)

        detected_flows = []

        for _, trade in new_trades.iterrows():
            trade_dict = trade.to_dict()

            # Try to match with existing active flows
            matched = False
            for flow in self.active_flows:
                if flow.matches(trade_dict, self.qty_tolerance, self.time_tolerance):
                    flow.add_trade(trade_dict)
                    detected_flows.append(flow)
                    matched = True
                    break

            if not matched:
                candidate_flow = self._try_create_flow(trade_dict, all_trades_df)
                if candidate_flow:
                    self.active_flows.append(candidate_flow)
                    detected_flows.append(candidate_flow)

        # Close flows that have gone silent
        if not all_trades_df.empty:
            self._close_stale_flows(all_trades_df['timestamp'].max())

        # Track new flows for external polling
        self._track_new_flows()

        return detected_flows

    def _try_create_flow(self, seed_trade: dict, all_trades: pd.DataFrame) -> Optional[Flow]:
        """
        Try to create a new flow starting from a seed trade.

        Looks back in recent history for trades matching the pattern.

        Args:
            seed_trade: Trade that triggered flow detection
            all_trades: DataFrame of all recent trades

        Returns:
            Flow object if valid pattern found, None otherwise
        """
        side = seed_trade['side']
        qty = seed_trade['quantity']
        timestamp = seed_trade['timestamp']
        instrument_id = seed_trade.get('ticker') or seed_trade.get('isin')

        lookback = timestamp - timedelta(seconds=6000)
        candidates = all_trades[
            (all_trades['timestamp'] >= lookback) &
            (all_trades['timestamp'] <= timestamp) &
            (all_trades['ticker'] == instrument_id) &
            (all_trades['side'] == side)
            ].copy()

        if len(candidates) < self.min_trades:
            return None

        # Check if quantities are similar
        qty_mean = candidates['quantity'].mean()
        qty_std = candidates['quantity'].std()

        if qty_std == 0:  # All quantities identical
            qty_cv = 0
        else:
            qty_cv = qty_std / qty_mean

        if qty_cv > self.qty_tolerance:
            return None

        # Check if timing is regular
        if len(candidates) > 1:
            candidates = candidates.sort_values('timestamp')
            intervals = candidates['timestamp'].diff().dt.total_seconds().dropna()

            if intervals.empty:
                return None

            interval_mean = intervals.mean()
            interval_std = intervals.std()

            # Intervals should be relatively consistent
            if interval_mean > 0:
                interval_cv = interval_std / interval_mean
                if interval_cv > 1.0:  # CV > 100%
                    return None

        # Check for duplicates with existing flows
        for existing_flow in self.active_flows:
            first_candidate_time = candidates.iloc[0]['timestamp']
            if any(t['timestamp'] == first_candidate_time for t in existing_flow.trades):
                return None

        # Create the flow
        self._flow_counter += 1

        flow = Flow(
            flow_id=f"FLOW_{self._flow_counter:05d}",
            instrument_id=instrument_id,
            side=side,
            trades=[t.to_dict() for _, t in candidates.iterrows()]
        )

        # Only return if significant
        if flow.total_quantity >= self.min_total_qty:
            return flow

        return None

    def _close_stale_flows(self, current_time: datetime):
        """
        Close flows that haven't received trades recently.

        Args:
            current_time: Current timestamp for gap calculation
        """
        to_close = []

        for flow in self.active_flows:
            if not flow.trades:
                continue

            last_trade_time = flow.trades[-1]['timestamp']
            gap = (current_time - last_trade_time).total_seconds()

            if gap > self.max_gap:
                flow.close()
                to_close.append(flow)

        # Move to completed
        for flow in to_close:
            self.active_flows.remove(flow)
            self.completed_flows.append(flow)

        # Trim history
        if len(self.completed_flows) > self.max_history:
            self.completed_flows = self.completed_flows[-self.max_history:]

    def get_active_flows(self) -> List[Flow]:
        """
        Get all currently active flows.

        Returns:
            List of Flow objects that are still receiving trades
        """
        return [f for f in self.active_flows if f.is_active]

    def get_significant_flows(self, min_consistency: float = 0.5) -> List[Flow]:
        """
        Get flows with high consistency scores.

        Args:
            min_consistency: Minimum consistency score (0-1, default: 0.5)

        Returns:
            List of Flow objects meeting criteria
        """
        return [
            f for f in self.active_flows
            if f.consistency_score >= min_consistency and len(f.trades) >= self.min_trades
        ]

    def get_flow_summary(self) -> dict:
        """
        Get summary statistics of detected flows.

        Returns:
            Dict with keys: active_flows, completed_flows, buy_flows,
                          sell_flows, total_volume
        """
        active = self.get_active_flows()

        return {
            'active_flows': len(active),
            'completed_flows': len(self.completed_flows),
            'buy_flows': len([f for f in active if f.side == 'buy']),
            'sell_flows': len([f for f in active if f.side == 'sell']),
            'total_volume': sum(f.total_quantity for f in active),
        }

    # ========================================================================
    # EXTERNAL POLLING API
    # ========================================================================

    def get_new_flows(self) -> List[Flow]:
        """
        Get flows that have been newly detected since last call.

        Calling this method marks these flows as "retrieved", so subsequent
        calls will not return the same flows again.

        Returns:
            List of newly detected Flow objects

        Example:
            ```python
            if detector.has_new_flows():
                for flow in detector.get_new_flows():
                    print(f"Alert: New flow {flow.flow_id}")
            ```
        """
        new_flows = self._pending_new_flows.copy()
        self._pending_new_flows.clear()
        return new_flows

    def has_new_flows(self) -> bool:
        """
        Check if there are new flows to retrieve.

        Returns:
            True if there are flows pending in get_new_flows()
        """
        return len(self._pending_new_flows) > 0

    def _track_new_flows(self):
        """
        Internal method to track NEW flows only.
        Called after processing trades.
        """
        for flow in self.active_flows:
            flow_id = flow.flow_id

            # Is this a new flow we haven't exported yet?
            if flow_id not in self._notified_flows:
                # Only notify if it meets minimum criteria
                if len(flow.trades) >= self.min_trades:
                    self._pending_new_flows.append(flow)
                    self._notified_flows.add(flow_id)