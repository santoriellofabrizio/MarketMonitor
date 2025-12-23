import logging
from abc import ABC
from typing import List

import pandas as pd
from dateutil.utils import today
from scipy.sparse import csr_matrix
import datetime as dt

from user_strategy.utils import CustomBDay
from user_strategy.utils.pricing_models.AggregationFunctions import ForecastAggregator, EwmaOutlier
from user_strategy.utils.pricing_models.IRPManager import IRPManager


class PricingModel:

    def __init__(self, returns: pd.DataFrame | None = None, *args, **kwargs):
        self.returns: pd.DataFrame = returns

        if returns is not None:  self.dates = returns.index

        self.yesterday: dt.date = (today() - CustomBDay).date()

    def predict_price(self, predicted_returns: pd.DataFrame, *args, **kwargs):
        pass

    def predict_returns(self, all_returns: pd.DataFrame) -> pd.DataFrame:
        pass


class LinearPricingModel(PricingModel, ABC):
    def __init__(self,
                 beta: pd.DataFrame,
                 returns: pd.DataFrame | None = None,
                 *args, **kwargs):
        super().__init__(returns, *args, **kwargs)
        self.beta = beta
        self.beta_sparse = csr_matrix(
            self.beta)
        self.target_variables = beta.index.tolist()
        self.regressor = beta.columns.tolist()


class MultiPeriodLinearPricingModel(LinearPricingModel, ABC):

    def __init__(self,
                 beta: pd.DataFrame,
                 returns: pd.DataFrame | None = None,
                 forecast_aggregator: ForecastAggregator | None = None,
                 *args, **kwargs):

        super().__init__(returns=returns, beta=beta)

        self.theoretical_returns_matrix: pd.DataFrame = pd.DataFrame(index=self.target_variables, columns=self.dates)
        self.theoretical_prices_matrix: pd.DataFrame = pd.DataFrame(index=self.target_variables, columns=self.dates)
        self.theoretical_price: pd.Series = pd.Series(dtype=float, index=self.target_variables)

        self.forecast_aggregator = forecast_aggregator or EwmaOutlier(5, 3)

    def make_matrix_mult(self, returns: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:

        missing_regressors = set(self.regressor) - set(returns.columns)
        if missing_regressors:
            logging.warning(f"missing regressor: {missing_regressors}")

        missing_dates = set(self.dates) - set(returns.index)
        if missing_dates:
            logging.warning(f"missing dates: {missing_dates}")

        missing_etfs = set(self.target_variables) - set(returns.columns)
        if missing_etfs:
            logging.warning(f"missing etfs: {missing_etfs}")
            for m in missing_etfs:
                self.target_variables.remove(m)
                self.beta.drop(m, inplace=True)

        returns = returns.loc[self.dates, self.regressor].T
        self.beta_sparse = csr_matrix(self.beta[self.regressor])
        predictions = pd.DataFrame(
            self.beta_sparse.dot(returns.values.astype(float)),
            columns=self.dates,
            index=self.beta.index
        )
        return predictions.T

    def predict_returns(self, all_returns: pd.DataFrame) -> pd.DataFrame:

        theoretical_live_return = self.make_matrix_mult(all_returns)
        return theoretical_live_return

    def get_price_prediction(self,
                             book: pd.DataFrame,
                             all_returns: pd.DataFrame) -> pd.Series:
        """
             Args:
                 book:
                 all_returns: pd.Dataframe: on rows you have dates, on cols instruments
                 forecast into a single one

             Returns: pd.Series of price prediction

             """
        self.dates = all_returns.index.tolist()
        predictions = self.predict_prices(book, all_returns)
        prediction = self.forecast_aggregator(predictions)
        self.theoretical_price.update(prediction)
        return self.theoretical_price

    def predict_prices(self, book, all_returns, *args) -> pd.DataFrame:
        pass


class ClusterPricingModel(MultiPeriodLinearPricingModel):

    def __init__(self,
                 name: str,
                 beta: pd.DataFrame,
                 returns: pd.DataFrame,
                 forecast_aggregator: ForecastAggregator | None = None,
                 cluster_correction: pd.Series | None = None,
                 disable_warning: bool = False,
                 *args, **kwargs):
        super().__init__(beta, returns, forecast_aggregator, *args, **kwargs)

        self.yesterday_misalignment_cluster: None | pd.Series = None
        rows, cols = beta.shape
        if not disable_warning:
            if rows != cols:
                logging.warning("Beta of clusters row != cols\n")

        self.cluster_correction = cluster_correction
        self.theoretical_price.name = name

    def predict_prices(self,
                       book: pd.DataFrame,
                       all_returns: pd.DataFrame | None = None,
                       *args) -> pd.DataFrame:

        correction = self.cluster_correction if self.cluster_correction is not None else 1

        theoretical_live_return = self.predict_returns(all_returns)

        misalignment = (theoretical_live_return - all_returns[theoretical_live_return.columns]) * correction
        self.yesterday_misalignment_cluster = misalignment.iloc[-1]
        all_predictions = (1 + misalignment) * book[self.target_variables]

        return all_predictions

    def calculate_cluster_correction(self):
        n_el_clusters = self.beta.apply(lambda row: (row != 0).sum(), axis=1)
        self.cluster_correction = n_el_clusters.apply(lambda x: max(x - 1, 1)) / n_el_clusters


class DriverPricingModel(MultiPeriodLinearPricingModel):

    def __init__(self,
                 beta: pd.DataFrame,
                 returns: pd.DataFrame,
                 forecast_aggregator: ForecastAggregator | None = None,
                 *args, **kwargs):
        super().__init__(beta, returns, forecast_aggregator, *args, **kwargs)

        self.yesterday_misalignment: None | pd.Series = None
        self.theoretical_price.name = "TH PRICE DRIVER"

    def predict_prices(self,
                       book: pd.DataFrame,
                       all_returns: pd.DataFrame | None = None, *args) -> pd.DataFrame:
        all_returns = self.returns if all_returns is None else all_returns
        theoretical_live_return = self.predict_returns(all_returns)
        misalignment = (theoretical_live_return - all_returns[self.target_variables])
        all_predictions = (1 + misalignment) * book[self.target_variables]
        return all_predictions


#------------------------------------------------------------------------------------#


class RatePricingModel:

    def __init__(self, name: str, target_variables: List[str], variables_proxy: pd.DataFrame):
        self.name = name
        self._target_variables = target_variables
        self._variables_proxy = variables_proxy
        self._rate: pd.DataFrame = None
        self._dates: pd.DataFrame = None
        self._theoretical_prices: pd.Series = pd.Series(dtype=float, index=self._target_variables)

    def _calculate_rate(self, book: pd.DataFrame):
        pass

    def _predict_prices(self, book: pd.DataFrame, all_returns: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_price_prediction(self,
                             book: pd.DataFrame,
                             all_returns: pd.DataFrame) -> pd.Series:

        self._dates = all_returns.index.tolist()
        prediction = self._predict_prices(book, all_returns)
        self._theoretical_prices.update(prediction)
        return self._theoretical_prices


class CreditFuturesCalendarSpreadPricingModel(RatePricingModel):

    def __init__(self, name: str, target_variables: List[str], variables_proxy: pd.DataFrame,
                 irp_manager: IRPManager):
        super().__init__(name, target_variables, variables_proxy)
        self._irp_manager = irp_manager

    def _calculate_rate(self, book: pd.DataFrame):
        self._rate = self._irp_manager.calculate_calendar_spread(irp_mapping=self._variables_proxy['IRP'],
                                                                 start=self._variables_proxy['Proxy Expiry'],
                                                                 end=self._variables_proxy['Expiry'],
                                                                 book=book)

    def _predict_prices(self, book: pd.DataFrame, all_returns: pd.DataFrame) -> pd.DataFrame:
        self._calculate_rate(book)
        book.name = 'Value'
        proxy_prices = self._variables_proxy.loc[self._target_variables].merge(book, left_on='Future Proxy', right_index=True)[book.name]
        all_predictions = (1 + self._rate.loc[self._target_variables]).mul(proxy_prices, axis=0)
        return all_predictions.T


class CreditFuturesInterestRatePricingModel(RatePricingModel):

    def __init__(self, name: str, target_variables: List[str], variables_proxy: pd.DataFrame,
                 irp_manager: IRPManager):
        super().__init__(name, target_variables, variables_proxy)
        self._irp_manager = irp_manager

    def _calculate_rate(self, book: pd.DataFrame):
        self._rate = self._irp_manager.calculate_average_rate_until_maturity(irp_mapping=self._variables_proxy['IRP'],
                                                                             end=self._variables_proxy['Expiry'],
                                                                             book=book)

    def _predict_prices(self, book: pd.DataFrame, all_returns: pd.DataFrame) -> pd.DataFrame:
        self._calculate_rate(book)
        return self._rate.T