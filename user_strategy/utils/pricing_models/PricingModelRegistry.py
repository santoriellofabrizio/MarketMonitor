import logging
from typing import Dict, List, Optional, Type

import pandas as pd

from user_strategy.utils.pricing_models.PricingModel import PricingModel

logger = logging.getLogger(__name__)


class _ModelEntry:
    __slots__ = ('model', 'returns_source')

    def __init__(self, model: PricingModel, returns_source: Optional[pd.DataFrame]):
        self.model = model
        self.returns_source = returns_source if returns_source is not None else pd.DataFrame()


class PricingModelRegistry:
    """Registry that holds and runs all pricing models.

    Model names are case-insensitive: "Cluster" and "cluster" refer to the same model.

    Supports two usage patterns:

    *Credit pattern* — returns passed at call time, shared across all models::

        registry.register("my model", instruments=etf_list, model=my_model)
        registry.predict_all(book_mid, all_returns=corrected_returns)
        prices = registry.get_prices("my model")

    *Equity pattern* — returns stored per-model, updated each tick::

        registry.register("my model", model=my_model, returns_source=corrected_return_df)
        registry.set_returns_source("my model", updated_returns)
        predictions = registry.predict_all(mid_series)   # uses stored returns_source
        predictions.get("my model")
    """

    def __init__(self):
        self._entries: Dict[str, _ModelEntry] = {}
        self._predictions: Dict[str, Optional[pd.Series]] = {}

    @staticmethod
    def _key(name: str) -> str:
        return name.lower()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
            self,
            name: str,
            model: PricingModel,
            instruments: Optional[List[str]] = None,
            returns_source: Optional[pd.DataFrame] = None,
            dtype: Type = float,
    ) -> "PricingModelRegistry":
        """Register a pricing model.

        Args:
            name: Unique model key (case-insensitive).
            model: The pricing model instance.
            instruments: Optional list of instrument IDs; pre-allocates the output Series.
            returns_source: Optional DataFrame stored per-model for the equity pattern.
            dtype: dtype for the pre-allocated output Series (only used when instruments is set).

        Returns self for chaining.
        """
        key = self._key(name)
        if key in self._entries:
            raise ValueError(f"Model '{name}' is already registered.")
        self._entries[key] = _ModelEntry(model, returns_source)
        self._predictions[key] = (
            pd.Series(dtype=dtype, name=key, index=instruments)
            if instruments is not None
            else None
        )
        return self

    def unregister(self, name: str) -> None:
        key = self._key(name)
        self._entries.pop(key, None)
        self._predictions.pop(key, None)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_all(
            self, mid: pd.Series, all_returns: Optional[pd.DataFrame] = None
    ) -> Dict[str, Optional[pd.Series]]:
        """Run all registered models and return {name: prediction}.

        Args:
            mid: Current mid prices (first positional arg to get_price_prediction).
            all_returns: If provided, all models use this DataFrame (credit pattern).
                         If None, each model uses its own stored returns_source (equity pattern).
        """
        for name, entry in self._entries.items():
            try:
                returns = (
                    all_returns.T
                    if all_returns is not None
                    else entry.returns_source.T
                )
                self._predictions[name] = entry.model.get_price_prediction(mid, returns)
            except Exception as e:
                logger.error(f"Error in model '{name}': {e}")
        return dict(self._predictions)

    def calculate_theoretical_prices(
            self, book: pd.Series, all_returns: pd.DataFrame
    ) -> None:
        """Alias of predict_all for backward compatibility with the credit engine."""
        self.predict_all(book, all_returns)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_prediction(self, name: str) -> Optional[pd.Series]:
        return self._predictions.get(self._key(name))

    def get_prices(self, name: str) -> pd.Series:
        """Alias of get_prediction (credit-pattern accessor)."""
        result = self._predictions.get(self._key(name))
        if result is None:
            raise ValueError(
                f"Model '{name}' not found or not yet predicted. "
                f"Registered: {list(self._predictions)}"
            )
        return result

    # ── Runtime updates (equity pattern) ─────────────────────────────────────

    def set_returns_source(self, name: str, returns_source: pd.DataFrame) -> None:
        """Update the returns DataFrame for a model."""
        entry = self._entries.get(self._key(name))
        if entry is not None:
            entry.returns_source = returns_source

    def update_beta(self, name: str, beta: pd.DataFrame, correction: pd.Series) -> None:
        """Update a model's beta matrix and cluster correction."""
        entry = self._entries.get(self._key(name))
        if entry is None:
            logger.error(f"Model '{name}' not found")
            return
        if beta.isna().any().any():
            nan_count = beta.isna().sum().sum()
            logger.warning(f"Beta matrix contains {nan_count} NaN values, replacing with 0")
            beta = beta.fillna(0)
        if correction.isna().any():
            logger.warning(
                f"Cluster correction contains NaN values: "
                f"{correction[correction.isna()].index.tolist()}"
            )
            correction = correction.fillna(1.0)
        entry.model.set_beta(beta)
        entry.model.set_cluster_correction(correction)

    def update_forecaster(self, name: str, forecaster) -> None:
        """Replace the forecast aggregator of a model at runtime."""
        entry = self._entries.get(self._key(name))
        if entry is None:
            logger.error(f"Model '{name}' not found in registry")
            return
        if not hasattr(entry.model, "forecast_aggregator"):
            logger.warning(f"Model '{name}' does not support forecast_aggregator")
            return
        entry.model.forecast_aggregator = forecaster
        logger.info(f"Model '{name}' forecaster updated to {type(forecaster).__name__}")

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def model_names(self) -> list:
        return list(self._entries.keys())

    @property
    def model_instruments(self) -> set[str]:
        needed_instruments = set()
        for entry in self._entries.values():
            needed_instruments.add(entry.model.declare_instruments())
        return needed_instruments

    def __len__(self):
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return self._key(name) in self._entries
