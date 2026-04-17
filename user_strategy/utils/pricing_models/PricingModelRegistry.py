from typing import Optional, List, Type, Dict

import pandas as pd

from user_strategy.utils.pricing_models.PricingModel import PricingModel


class PricingModelRegistry:
    """Registry that holds and runs all pricing models.

    Models are registered by name before historical data is available;
    prices are computed via calculate_theoretical_prices on each tick.
    """

    def __init__(self):
        self._prices: Dict[str, pd.Series] = {}
        self._models: Dict[str, PricingModel] = {}

    def register(self, name: str, instruments: List[str], model: PricingModel, dtype: Type = float) -> None:
        if name in self._prices:
            raise ValueError(f"Model '{name}' is already registered.")
        self._prices[name] = pd.Series(dtype=dtype, name=name, index=instruments)
        self._models[name] = model

    def get_prices(self, name: Optional[str] = None) -> pd.DataFrame | pd.Series:
        if name is None:
            return pd.DataFrame(self._prices)
        if name not in self._prices:
            raise ValueError(f"Model '{name}' not found. Registered: {list(self._prices)}")
        return self._prices[name]

    def calculate_theoretical_prices(self, book: pd.Series, all_returns: pd.DataFrame) -> None:
        if not set(book.index).issubset(all_returns.index):
            missing = sorted(set(book.index) - set(all_returns.index))
            raise ValueError(
                f"calculate_theoretical_prices: instruments in book missing from all_returns: {missing}"
            )
        all_ret_t = all_returns.T
        for name, model in self._models.items():
            try:
                self._prices[name] = model.get_price_prediction(prices=book, all_returns=all_ret_t)
            except Exception as e:
                print(f"exception in model '{name}': {e}")
