"""
Mock Price Engine - Simulates EtfEquityPriceEngine Redis publishing without real data.

Isolates the Redis layer (both RedisMessaging pub/sub and TimeSeriesPublisher)
by generating synthetic prices, returns, and misalignment data.

Channels published (RedisMessaging, port 6379):
  - market:return_{0..8}         live + historical returns (Series as JSON)
  - market:theoretical_live_index_cluster_price
  - market:theoretical_live_cluster_price
  - market:theoretical_live_intraday_price
  - market:mid

Channels published (TimeSeriesPublisher, port 6380):
  - ts:{isin}:live_idx_mis       misalignment index cluster vs mid
  - ts:{isin}:live_clust_mis     misalignment cluster vs mid
  - ts:{isin}:intraday_mis       misalignment intraday vs mid

Usage:
    python mock_price_engine.py
    python mock_price_engine.py --hf-interval 1 --lf-interval 30
    python mock_price_engine.py --isin-file path/to/other_list.txt
"""

import sys
import os
import time
import logging
import argparse
import random

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.publishers.timeseries_publisher import TimeSeriesPublisher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("MockPriceEngine")


# ---------------------------------------------------------------------------
# ISIN loading
# ---------------------------------------------------------------------------

ISIN_LIST_DEFAULT = os.path.join(os.path.dirname(__file__), "etc", "input", "isin_list.txt")


def load_isins(path: str = ISIN_LIST_DEFAULT) -> list[str]:
    """Load unique ISINs from a text file (one per line), preserving order."""
    with open(path, "r") as f:
        seen = set()
        isins = []
        for line in f:
            isin = line.strip()
            if isin and isin not in seen:
                seen.add(isin)
                isins.append(isin)
    return isins


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def generate_base_prices(isins: list[str], low: float = 5.0, high: float = 200.0) -> pd.Series:
    """Random initial mid prices for each ISIN."""
    return pd.Series(
        np.round(np.random.uniform(low, high, len(isins)), 4),
        index=isins
    )


def walk_prices(prices: pd.Series, volatility: float = 0.0005) -> pd.Series:
    """Apply a small random walk to prices."""
    returns = np.random.normal(0, volatility, len(prices))
    return (prices * (1 + returns)).round(4)


def compute_theoretical(mid: pd.Series, bias: float, noise: float = 0.0003) -> pd.Series:
    """Simulate a theoretical price = mid * (1 + bias + noise)."""
    return (mid * (1 + bias + np.random.normal(0, noise, len(mid)))).round(4)


def compute_misalignment(theoretical: pd.Series, mid: pd.Series) -> pd.Series:
    """misalignment = theo / mid - 1  (same formula as EtfEquityPriceEngine)."""
    safe_mid = mid.replace(0, np.nan)
    return (theoretical / safe_mid - 1).round(6)


# ---------------------------------------------------------------------------
# Publishing helpers
# ---------------------------------------------------------------------------

def publish_static_returns(gui: RedisMessaging, isins: list[str], n_periods: int = 8):
    """Publish fake historical returns on market:return_1 .. return_8."""
    for i in range(1, n_periods + 1):
        fake_return = pd.Series(
            np.round(np.random.normal(0, 0.5, len(isins)), 4),
            index=isins
        )
        gui.export_static_data(**{f"market:return_{i}": fake_return})
    logger.info(f"Published static returns for {n_periods} periods")


def publish_hf_update(
    gui: RedisMessaging,
    ts_pub: TimeSeriesPublisher,
    mid: pd.Series,
    isins: list[str],
):
    """
    One high-frequency tick:
      - walk mid prices
      - compute 3 theoretical prices
      - publish to RedisMessaging (pub/sub GUI channels)
      - publish misalignments to TimeSeriesPublisher
    Returns updated mid.
    """
    mid = walk_prices(mid)

    theo_idx   = compute_theoretical(mid, bias=0.0002)
    theo_clust = compute_theoretical(mid, bias=-0.0001)
    theo_intra = compute_theoretical(mid, bias=0.00005)

    # --- RedisMessaging pub/sub (port 6379) ---
    live_return = pd.Series(
        np.round(np.random.normal(0, 0.3, len(isins)), 4),
        index=isins
    )
    gui.export_message("market:return_0", live_return)
    gui.export_message("market:intraday_return_0", live_return * 0.8)

    gui.export_message("market:theoretical_live_index_cluster_price", theo_idx,
                        skip_if_unchanged=True, flat_mode=True)
    gui.export_message("market:theoretical_live_cluster_price", theo_clust,
                        skip_if_unchanged=True, flat_mode=True)
    gui.export_message("market:theoretical_live_intraday_price", theo_intra,
                        skip_if_unchanged=True, flat_mode=True)
    gui.export_message("market:mid", mid,
                        skip_if_unchanged=True, flat_mode=True)

    # --- TimeSeriesPublisher (port 6380) ---
    mis_idx   = compute_misalignment(theo_idx, mid)
    mis_clust = compute_misalignment(theo_clust, mid)
    mis_intra = compute_misalignment(theo_intra, mid)

    with ts_pub.ts_batch() as batch:
        for isin in isins:
            v_idx = mis_idx.get(isin)
            v_clust = mis_clust.get(isin)
            v_intra = mis_intra.get(isin)
            if v_idx is not None and not np.isnan(v_idx):
                batch.add(isin, "live_idx_mis", float(v_idx))
            if v_clust is not None and not np.isnan(v_clust):
                batch.add(isin, "live_clust_mis", float(v_clust))
            if v_intra is not None and not np.isnan(v_intra):
                batch.add(isin, "intraday_mis", float(v_intra))

    return mid


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Mock Price Engine for Redis isolation testing")
    p.add_argument("--isin-file", type=str, default=ISIN_LIST_DEFAULT,
                   help="Path to ISIN list file (default: etc/input/isin_list.txt)")
    p.add_argument("--hf-interval", type=float, default=2.0, help="Seconds between HF ticks")
    p.add_argument("--lf-interval", type=float, default=60.0, help="Seconds between LF updates (static returns)")
    p.add_argument("--redis-host", type=str, default="localhost")
    p.add_argument("--redis-pubsub-port", type=int, default=6379, help="RedisMessaging port")
    p.add_argument("--redis-ts-port", type=int, default=6380, help="TimeSeriesPublisher port")
    return p.parse_args()


def main():
    args = parse_args()

    isins = load_isins(args.isin_file)
    mid = generate_base_prices(isins)

    logger.info(f"Loaded {len(isins)} unique ISINs from {args.isin_file}")
    logger.info(f"First 10: {isins[:10]}")

    # --- connect ---
    gui = RedisMessaging(redis_host=args.redis_host, redis_port=args.redis_pubsub_port)
    ts_pub = TimeSeriesPublisher(redis_host=args.redis_host, redis_port=args.redis_ts_port)

    logger.info(f"RedisMessaging  -> {args.redis_host}:{args.redis_pubsub_port}")
    logger.info(f"TimeSeriesPublisher -> {args.redis_host}:{args.redis_ts_port}")

    # --- initial static publish ---
    publish_static_returns(gui, isins)

    tick = 0
    last_lf = time.time()

    logger.info(f"Starting mock loop: HF every {args.hf_interval}s, LF every {args.lf_interval}s")

    try:
        while True:
            tick += 1
            mid = publish_hf_update(gui, ts_pub, mid, isins)

            # periodic LF refresh of static returns
            now = time.time()
            if now - last_lf >= args.lf_interval:
                publish_static_returns(gui, isins)
                last_lf = now

            stats = ts_pub.ts_get_stats()
            logger.info(
                f"tick={tick}  "
                f"ts_published={stats['total_published']}  "
                f"ts_skipped={stats['duplicates_skipped']}  "
                f"ts_created={stats['timeseries_created']}"
            )

            time.sleep(args.hf_interval)

    except KeyboardInterrupt:
        logger.info("Stopped.")
        stats = ts_pub.ts_get_stats()
        logger.info(f"Final stats: {stats}")


if __name__ == "__main__":
    main()
