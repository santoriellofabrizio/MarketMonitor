import logging
from typing import Optional

import pandas as pd

from market_monitor.strategy.StrategyUI.StrategyUI import StrategyUI


class TradeAccumulatorStrategy(StrategyUI):
    """
    Strategia che accumula e analizza i trades ricevuti.

    Test: Verifica che i trades si accumulino correttamente
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger("TradeAccumulatorStrategy")

        # Strumenti da monitorare
        self.isins = [
            "DE0007667107",
            "LU0048584102",
            "LU0072462426",
            "IE0002271879",
            "LU0073263215",
        ]

        # Accumulator
        self.trades_df: Optional[pd.DataFrame] = None
        self.trade_stats = {
            "total_trades": 0,
            "total_quantity": 0,
            "total_ctv": 0.0,
            "by_ticker": {},
        }

        self.logger.info(f"TradeAccumulatorStrategy initialized")

    def on_market_data_setting(self):
        """Setup market data."""
        subscription_dict = {
            isin: f"{isin} EQUITY"
            for isin in self.isins
        }
        self.market_data.subscription_dict_bloomberg = subscription_dict
        self.logger.info(f"Market data configured")

    def wait_for_book_initialization(self) -> bool:
        """Attendi book initialization."""
        mid = self.market_data.get_mid_eur()
        return len(mid[~mid.isna()]) >= 1

    def on_book_initialized(self):
        """Callback initialization."""
        self.logger.info("Book initialized!")

    def update_HF(self):
        """Update HF - monitora prezzi."""
        try:
            mid_eur = self.market_data.get_mid_eur()
            # Potremmo aggiungere monitoring qui se necessario
        except Exception as e:
            self.logger.error(f"Error in update_HF: {e}")

    def on_trade(self, trades: pd.DataFrame):
        """
        Accumula i trades ricevuti e computa statistiche.
        """
        try:
            # Accumula
            if self.trades_df is None:
                self.trades_df = trades.copy()
            else:
                self.trades_df = pd.concat([self.trades_df, trades], ignore_index=True)

            # Update stats
            self.trade_stats["total_trades"] += len(trades)
            self.trade_stats["total_quantity"] += trades["quantity"].sum()
            self.trade_stats["total_ctv"] += trades["ctv"].sum()

            # By ticker
            for _, trade in trades.iterrows():
                ticker = trade.get("ticker", "UNKNOWN")
                if ticker not in self.trade_stats["by_ticker"]:
                    self.trade_stats["by_ticker"][ticker] = {
                        "count": 0,
                        "qty": 0,
                        "ctv": 0.0,
                    }

                self.trade_stats["by_ticker"][ticker]["count"] += 1
                self.trade_stats["by_ticker"][ticker]["qty"] += trade["quantity"]
                self.trade_stats["by_ticker"][ticker]["ctv"] += trade["ctv"]

            # Log
            self._log_stats()

        except Exception as e:
            self.logger.error(f"Error in on_trade: {e}")

    def _log_stats(self):
        """Log statistics."""
        stats = self.trade_stats
        self.logger.info(
            f"Trades: {stats['total_trades']} | "
            f"Qty: {stats['total_quantity']} | "
            f"CTV: {stats['total_ctv']:.2f}"
        )

    def on_stop(self):
        """Final report."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TRADE ACCUMULATION REPORT")
        self.logger.info("=" * 60)

        stats = self.trade_stats
        self.logger.info(f"Total trades: {stats['total_trades']}")
        self.logger.info(f"Total quantity: {stats['total_quantity']}")
        self.logger.info(f"Total CTV: {stats['total_ctv']:.2f}")

        self.logger.info("\nBy Ticker:")
        for ticker, data in sorted(stats["by_ticker"].items()):
            self.logger.info(
                f"  {ticker}: {data['count']} trades | "
                f"{data['qty']} qty | "
                f"{data['ctv']:.2f} ctv"
            )


