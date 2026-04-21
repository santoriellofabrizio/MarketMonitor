import numpy as np
import pandas as pd


def round_series_to_tick(
    series: pd.Series,
    tick_dict: dict | pd.Series,
    default_tick: float = 0.001,
) -> pd.Series:
    """Round a Series to the nearest full tick for each instrument.

    Args:
        series: Prices to round.
        tick_dict: Mapping of index label -> tick size (full tick).
        default_tick: Fallback tick size when an instrument is not in tick_dict.

    Returns:
        pd.Series with values rounded to the nearest tick.
    """
    if series is None:
        return series
    if isinstance(tick_dict, pd.Series):
        tick_dict = tick_dict.to_dict()
    ticks = np.array([tick_dict.get(idx, default_tick) for idx in series.index])
    values = series.fillna(0).values.astype(float)
    rounded_values = np.round(np.round(values / ticks) * ticks, 10)
    return pd.Series(rounded_values, index=series.index).fillna(0)
