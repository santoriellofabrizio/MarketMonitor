from abc import ABC, abstractmethod
from collections import deque

import pandas as pd
from sfm_data_provider.core.holidays.holiday_manager import HolidayManager

from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI
from user_strategy.equity.LiveQuoting.price_publisher import PricePublisherHub
from user_strategy.utils.pricing_models.PricingModelRegistry import PricingModelRegistry


class BasePriceEngine(StrategyUI, ABC):
    """
    Template base for price engines.

    Init skeleton (fixed order, enforced):
        1. _setup_instrument_universe()   — populate instrument lists & isin_to_ticker
        2. PricePublisherHub.from_config  — uses _get_isin_to_ticker() hook
        3. _setup_historical_data()       — load prices, build adjuster
        4. _setup_pricing_models()        — register models in self.models
        5. _finalize_setup()              — optional post-init hook
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.holidays: HolidayManager = HolidayManager()
        self.corrected_returns: pd.DataFrame = pd.DataFrame()
        self.models: PricingModelRegistry | None = None

        self._setup_instrument_universe()
        self.publisher = PricePublisherHub.from_config(kwargs, self._get_isin_to_ticker())
        self._setup_historical_data()
        self._setup_pricing_models()
        self._finalize_setup()

    # ── Hooks ─────────────────────────────────────────────────────────────────

    def _get_isin_to_ticker(self) -> dict:
        """Return ISIN→ticker mapping for PricePublisherHub labels.
        Override in engines that populate self.isin_to_ticker in
        _setup_instrument_universe()."""
        return {}

    def _finalize_setup(self) -> None:
        """Called after all setup phases. Override for engine-specific
        post-init work (e.g. publishing static returns)."""
        pass

    # ── Abstract setup phases ─────────────────────────────────────────────────

    @abstractmethod
    def _setup_instrument_universe(self) -> None: ...

    @abstractmethod
    def _setup_historical_data(self) -> None: ...

    @abstractmethod
    def _setup_pricing_models(self) -> None: ...

    # ── Concrete shared behaviour ─────────────────────────────────────────────

    def stop(self) -> None:
        pass
