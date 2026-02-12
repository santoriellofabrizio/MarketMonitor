import logging
from typing import Dict, Optional, Protocol, runtime_checkable

import pandas as pd

logger = logging.getLogger(__name__)


@runtime_checkable
class PricePredictor(Protocol):
    """Any object that can predict prices given mid and returns."""
    name: str

    def get_price_prediction(self, mid: pd.Series, returns: pd.DataFrame) -> pd.Series: ...
    def set_beta(self, beta: pd.DataFrame) -> None: ...
    def set_cluster_correction(self, correction: pd.Series) -> None: ...


class _ModelEntry:
    __slots__ = ('model', 'returns_source')

    def __init__(self, model: PricePredictor, returns_source: pd.DataFrame):
        self.model = model
        self.returns_source = returns_source


class PricingModelRegistry:
    """Manages multiple named pricing models."""

    def __init__(self):
        self._entries: Dict[str, _ModelEntry] = {}
        self._predictions: Dict[str, Optional[pd.Series]] = {}

    def register(self, name: str, model: PricePredictor,
                 returns_source: pd.DataFrame) -> "PricingModelRegistry":
        """Register a pricing model with its returns DataFrame. Returns self for chaining."""
        self._entries[name] = _ModelEntry(model, returns_source)
        self._predictions[name] = None
        return self

    def unregister(self, name: str) -> None:
        self._entries.pop(name, None)
        self._predictions.pop(name, None)

    def predict_all(self, mid: pd.Series) -> Dict[str, Optional[pd.Series]]:
        """Run all registered models. Returns {name: predictions}."""
        for name, entry in self._entries.items():
            try:
                self._predictions[name] = entry.model.get_price_prediction(
                    mid, entry.returns_source.T)
            except Exception as e:
                logger.error(f"Error in model '{name}': {e}")
        return dict(self._predictions)

    def get_prediction(self, name: str) -> Optional[pd.Series]:
        return self._predictions.get(name)

    def update_beta(self, name: str, beta: pd.DataFrame, correction: pd.Series) -> None:
        """Update a model's beta matrix and cluster correction."""
        entry = self._entries.get(name)
        if entry is None:
            logger.error(f"Model '{name}' not found")
            return

        if beta.isna().any().any():
            nan_count = beta.isna().sum().sum()
            logger.warning(f"Beta matrix contains {nan_count} NaN values, replacing with 0")
            beta = beta.fillna(0)

        if correction.isna().any():
            logger.warning(f"Cluster correction contains NaN values: "
                           f"{correction[correction.isna()].index.tolist()}")
            correction = correction.fillna(1.0)

        entry.model.set_beta(beta)
        entry.model.set_cluster_correction(correction)

    def set_returns_source(self, name: str, returns_source: pd.DataFrame) -> None:
        """Update the returns DataFrame for a model."""
        entry = self._entries.get(name)
        if entry is not None:
            entry.returns_source = returns_source

    @property
    def model_names(self) -> list:
        return list(self._entries.keys())

    def __len__(self):
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries
