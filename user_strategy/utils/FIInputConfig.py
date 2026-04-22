from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from user_strategy.utils.pricing_models.AggregationFunctions import ForecastAggregator


@dataclass
class DataFetchingConfig:
    """Parameters consumed by CreditPriceEngine / EtfFiPriceEngine for historical data loading."""
    etf_isins: List[str]
    drivers: pd.DataFrame
    index_data: pd.DataFrame
    credit_futures_data: pd.DataFrame
    irs_data: pd.DataFrame
    irp_data: pd.DataFrame
    trading_currency: pd.DataFrame
    currency_exposure: pd.DataFrame


@dataclass
class PricingConfig:
    """Parameters consumed by PricingModelRegistry and pricing models."""
    hedge_ratios_cluster: pd.DataFrame
    hedge_ratios_drivers: pd.DataFrame
    hedge_ratios_brothers: pd.DataFrame
    hedge_ratios_credit_futures_cluster: pd.DataFrame
    hedge_ratios_credit_futures_brothers: pd.DataFrame
    forecast_aggregator_cluster: ForecastAggregator
    forecast_aggregator_driver: ForecastAggregator
    forecast_aggregator_brother: Optional[ForecastAggregator] = None
