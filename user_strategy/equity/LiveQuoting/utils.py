import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def round_series_to_tick(series, tick_dict, default_tick=0.001):
    """Arrotonda una Series ai tick specificati per ciascun strumento."""
    if series is None:
        return series
    if isinstance(tick_dict, pd.Series):
        tick_dict = tick_dict.to_dict()
    ticks = np.array([tick_dict.get(idx, default_tick) for idx in series.index]) / 2
    values = series.fillna(0).values.astype(float)
    rounded_values = np.round(np.round(values / ticks) * ticks, 10)
    return pd.Series(rounded_values, index=series.index).fillna(0)


def filter_outliers(df: pd.DataFrame, name: str = "data") -> pd.DataFrame:
    Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (df < Q1 - 1.5 * IQR) | (df > Q3 + 1.5 * IQR)
    if logger.isEnabledFor(logging.DEBUG) and outliers.any().any():
        out_dir = os.path.join(os.path.dirname(__file__), "debug_logs")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(os.path.join(out_dir, f"{name}_raw_{ts}.parquet"))
        outliers.to_parquet(os.path.join(out_dir, f"{name}_outliers_{ts}.parquet"))
        logger.debug(f"Outliers in {name}: {outliers.sum().sum()} values, saved to {out_dir}")
    df[outliers] = np.nan
    return df
