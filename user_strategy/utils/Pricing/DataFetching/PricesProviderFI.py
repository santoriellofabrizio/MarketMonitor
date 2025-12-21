from user_strategy.utils.Pricing.DataFetching.PricesProvider import PricesProvider
from sfm_return_adjustments_lib.ReturnAdjuster import ReturnAdjuster


class PricesProviderFI(PricesProvider):

    def _instantiate_return_adjuster(self):
        return_adjuster = ReturnAdjuster(self.etfs + self.drivers_anagraphic.index.to_list() +
                                         self.additional_contracts.index.to_list(), self.date_range,
                                         backdating=True, allow_logging=True)
        return_adjuster.set_ytm_mapping(self.ytm_mapping)
        return_adjuster.set_instrument_fx_weights(self.currency_weights)
        trading_currency_series = self.trading_currency.squeeze()
        return_adjuster.set_trading_currency(trading_currency_series)
        return return_adjuster