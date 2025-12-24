import logging
from typing import Dict

import numpy as np
import pandas as pd

from market_monitor.strategy.StrategyUI.StrategyUI import StrategyUI
from user_strategy.StrategyRegister import register_strategy


class PriceSpreadAnalyzerStrategy(StrategyUI):
    """
    Strategia che analizza gli spread tra BID/ASK.

    Test: Verifica che il book completo (BID/ASK) sia disponibile
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger("PriceSpreadAnalyzerStrategy")

        self.isins = [
            "DE0007667107",
            "LU0048584102",
            "LU0072462426",
        ]

        self.spread_history: Dict[str, list] = {isin: [] for isin in self.isins}
        self.update_count = 0

        self.logger.info(f"PriceSpreadAnalyzerStrategy initialized")

    def on_market_data_setting(self):
        """Setup market data."""
        subscription_dict = {
            isin: f"{isin} EQUITY"
            for isin in self.isins
        }
        self.market_data.subscription_dict = subscription_dict
        self.market_data.securities = self.isins

    def wait_for_book_initialization(self) -> bool:
        """Attendi book initialization."""
        mid = self.market_data.get_mid_eur()
        return len(mid[~mid.isna()]) >= 1

    def on_book_initialized(self):
        """Callback initialization."""
        self.logger.info("Book initialized!")

    def update_HF(self):
        """
        Analizza gli spread BID/ASK.
        """
        try:
            self.update_count += 1

            # Ottieni il book completo
            book = self.market_data.get_book_eur()

            if book is not None and not book.empty:
                for isin in self.isins:
                    if isin in book.index:
                        bid = book.loc[isin, "BID"]
                        ask = book.loc[isin, "ASK"]

                        if not np.isnan(bid) and not np.isnan(ask) and bid > 0 and ask > 0:
                            spread = ask - bid
                            spread_bps = (spread / bid) * 10000  # basis points

                            self.spread_history[isin].append({
                                "bid": bid,
                                "ask": ask,
                                "spread": spread,
                                "spread_bps": spread_bps,
                            })

            # Log periodico
            if self.update_count % 20 == 0:
                self._log_spread_summary()

        except Exception as e:
            self.logger.error(f"Error in update_HF: {e}")

    def on_trade(self, trades: pd.DataFrame):
        """Log trades."""
        self.logger.info(f"Received {len(trades)} trades")

    def _log_spread_summary(self):
        """Log spread analysis."""
        self.logger.info("=" * 60)
        self.logger.info(f"Spread Analysis (update #{self.update_count})")
        self.logger.info("=" * 60)

        for isin in self.isins:
            history = self.spread_history[isin]
            if history:
                spreads_bps = [h["spread_bps"] for h in history]
                self.logger.info(
                    f"{isin}: "
                    f"Avg spread: {np.mean(spreads_bps):.2f} bps | "
                    f"Min: {min(spreads_bps):.2f} | "
                    f"Max: {max(spreads_bps):.2f}"
                )

    def on_stop(self):
        """Final report."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("SPREAD ANALYSIS FINAL REPORT")
        self.logger.info("=" * 60)

        for isin in self.isins:
            history = self.spread_history[isin]
            if history:
                spreads_bps = [h["spread_bps"] for h in history]
                self.logger.info(
                    f"{isin}: {len(history)} samples | "
                    f"Avg: {np.mean(spreads_bps):.2f} bps"
                )


register_strategy("PriceSpreadAnalyzerStrategy", PriceSpreadAnalyzerStrategy)