from dataclasses import dataclass
from datetime import time
from typing import List, Optional

import pandas as pd

from user_strategy.utils.pricing_models.AggregationFunctions import ForecastAggregator


@dataclass
class DataFetchingConfig:
    """Parameters consumed by PricesProvider / PricesProviderFI.

    Captures everything needed to download historical prices: which
    instruments to fetch, from which source, over what horizon, and the
    FX / weight mappings needed for return adjustment.
    """
    etf_isins: List[str]
    drivers: pd.DataFrame
    index_data: pd.DataFrame
    credit_futures_data: pd.DataFrame
    irs_data: pd.DataFrame
    irp_data: pd.DataFrame
    YTM_mapping: pd.DataFrame
    currencies_EUR_ccy: List[str]
    currency_weights: pd.DataFrame
    currency_exposure: pd.DataFrame
    trading_currency: pd.DataFrame
    price_snipping_time: time
    number_of_days: int


@dataclass
class PricingConfig:
    """Parameters consumed by PricingModelRegistry and pricing models.

    Captures the calibrated model inputs: hedge ratios, cluster / brothers
    topology, and forecast aggregation functions for each model type.
    """
    hedge_ratios_cluster: pd.DataFrame
    hedge_ratios_drivers: pd.DataFrame
    hedge_ratios_brothers: pd.DataFrame
    hedge_ratios_credit_futures_cluster: pd.DataFrame
    hedge_ratios_credit_futures_brothers: pd.DataFrame
    cluster_anagraphic: pd.DataFrame
    brothers: pd.DataFrame
    forecast_aggregator_cluster: ForecastAggregator
    forecast_aggregator_driver: ForecastAggregator
    forecast_aggregator_nav: ForecastAggregator
    forecast_aggregator_brother: Optional[ForecastAggregator] = None
