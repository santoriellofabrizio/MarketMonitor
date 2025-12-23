from typing import Optional, List, Type, Dict

import pandas as pd

from user_strategy.utils.pricing_models.PricingModel import PricingModel


class TheoreticalPriceManager:
    def __init__(self):
        self._theoretical_pricings: Dict[str, pd.Series] = {}
        self._models: Dict[str, pd.DataFrame] = {}

    def get_theoretical_prices(self, name: Optional[str]=None) -> pd.DataFrame:
        if name is None:
            return pd.DataFrame(self._theoretical_pricings)
        else:
            if name not in self._theoretical_pricings:
                raise ValueError(f"Pricing '{name}' doesn't exist. Existing pricing names are : {self._theoretical_pricings.keys()}")
            return self._theoretical_pricings[name]

    def add_pricing(self, dtype: Type, name: str, instruments: List[str], model: PricingModel):
        if name in self._theoretical_pricings:
            raise ValueError(f"Pricing '{name}' already exists.")

        self._theoretical_pricings[name] = pd.Series(dtype=dtype, name=name, index=instruments)
        self._models[name] = model

    def calculate_theorical_prices(self, book: pd.DataFrame, all_returns: pd.DataFrame | None = None):
        if not set(book.index).issubset(all_returns.index):
            missing = set(book.index) - set(all_returns.index)
            raise Exception('Error when calculating theorical prices. The following instruments books were received but are not contained in all_returns:'
                            f'{sorted(missing)}')
        for name, model in self._models.items():
            all_ret_t = all_returns.T
            try:
                self._theoretical_pricings[name] = model.get_price_prediction(book=book, all_returns=all_ret_t)
            except Exception as e:
                print(f"exception in model {name}: {e}")


        # all_predictions_NAV = (self.book_mid[self.etf_isins]
        #                        * (1 + self.theoretical_misalignment_basis.mul(self.cluster_correction))
        #                        * (1 + self.cluster_model.yesterday_misalignment_cluster))
        # theoretical_nav_prices = self.input_params.forecast_aggregator_nav(all_predictions_NAV)
        #
        # self.theoretical_live_nav_price.update(theoretical_nav_prices)

