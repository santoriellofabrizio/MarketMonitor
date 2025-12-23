from datetime import date
from typing import Union

import pandas as pd
from dateutil.utils import today
from scipy.stats import trim_mean

from user_strategy.utils import CustomBDay


class ForecastAggregator:
    pass


class TrimmedMean(ForecastAggregator):

    def __init__(self, perc_outlier: float):
        self.perc_outlier = perc_outlier

    def __call__(self, all_predictions: pd.DataFrame):
        return pd.Series(trim_mean(all_predictions, proportiontocut=self.perc_outlier), index=all_predictions.columns)


class Ewma(ForecastAggregator):

    def __init__(self, halflife: int | float, *args, **kwargs):
        self.halflife = halflife
        self.yesterday: date = (today() - CustomBDay).date()

    def __call__(self, all_predictions: pd.DataFrame):
        return (all_predictions
                .sort_index(ascending=True)
                .ewm(halflife=self.halflife, ignore_na=True)
                .mean()
                .ffill()
                .bill()
                .loc[-1])


class EwmaOutlier(ForecastAggregator):
    def __init__(self, halflife: Union[int, float], outlier_std: float = None):
        """
        :param halflife: Half-life for EWMA.
        :param outlier_std: Number of standard deviations to identify outliers.
        """
        self.halflife = halflife
        self.outlier_threshold = outlier_std
        self.yesterday: date = (pd.Timestamp.today() - CustomBDay).date()

    def remove_outliers(self, data: pd.DataFrame):
        if self.outlier_threshold is not None:
            # Calcola media e deviazione standard
            mean = data.mean()
            std_dev = data.std()
            # Filtra i dati per rimuovere gli outlier
            return data[((data - mean).abs() <= self.outlier_threshold * std_dev) | (std_dev == 0)]

        return data

    def __call__(self, all_predictions: pd.DataFrame):

        cleaned_data = self.remove_outliers(all_predictions).sort_index(ascending=True)
        return (cleaned_data.
                        ewm(halflife=self.halflife, ignore_na=True).
                        mean().
                        ffill().
                        bfill().
                        iloc[-1])


# Dizionario per gestire le classi di aggregazione previste
forecast_aggregation = {
    "ewma": Ewma,
    "trimmed_mean": TrimmedMean,
    "ewma_outlier": EwmaOutlier
}
