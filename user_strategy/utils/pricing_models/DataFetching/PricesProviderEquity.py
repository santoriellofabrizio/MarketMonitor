import logging

import pandas as pd

from user_strategy.utils.pricing_models.DataFetching.PricesProvider import PricesProvider
from user_strategy.utils.pricing_models.DataFetching.download_functions import process_downloaded_prices, \
    download_daily_prices
from user_strategy.utils.pricing_models.ExcelStoringDecorator import save_to_excel
from user_strategy.utils.enums import CURRENCY
from sfm_return_adjustments_lib.ReturnAdjuster import ReturnAdjuster



logger = logging.getLogger()


class PricesProviderEquity(PricesProvider):

    def _instantiate_return_adjuster(self):
        _, self.currency_weights = self._input_params.get_currency_data(self.etfs)
        return_adjuster = ReturnAdjuster(self.etfs + self.drivers_anagraphic.index.to_list(), self.date_range,
                                         backdating=True, allow_logging=True)
        self.ter_manual = self.ter_manual.loc[self.ter_manual.index.isin(self.etfs)]
        if not self.ter_manual.empty:
            return_adjuster.set_ter_hard_coding(self.ter_manual)
        return_adjuster.set_instrument_fx_weights(self.currency_weights)
        return return_adjuster

    @save_to_excel("prices drivers")
    def get_hist_generic_instr_prices_from_oracle(
            self,
            market_isins_dict: dict | None = None,
            drivers_anagraphic: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Fetch historical instrument prices from Oracle.

        Args:
            market_isins_dict (dict, optional): A dictionary mapping markets to lists of instrument ISINs.
            drivers_anagraphic (pd.DataFrame, optional): A dataframe with instrument details, including:
                - index: Instrument ID.
                - "subscription_TS": Subscription ID.
                - "market_TS": Associated market.
                - "subscription_BBG": Subscription of bloomberg. Optional, will be used in case of N/A

        Returns:
            pd.DataFrame: Historical prices for instruments.
        """
        if not market_isins_dict:
            if drivers_anagraphic is None or drivers_anagraphic.empty:
                return pd.DataFrame()
            market_isins_dict = {
                market: group["subscription_TS"].dropna().tolist()
                for market, group in drivers_anagraphic.groupby("market_TS") if market
            }

        prices = [
            download_daily_prices(self.date_range, isins, self.price_snipping_time, market=market,
                                  desc=f"Downloading driver prices from Oracle {market}")
            for market, isins in market_isins_dict.items()
        ]
        prices = process_downloaded_prices(pd.concat(prices, ignore_index=True), self.date_range, col_name="isin")

        if drivers_anagraphic is not None:
            prices.rename(
                columns={val: key for key, val in drivers_anagraphic["subscription_TS"].items() if
                         val in prices.columns},
                inplace=True
            )
            for instr in drivers_anagraphic.index:
                if instr not in prices.columns:
                    prices[instr] = None

        return prices

    def get_bbg_subscription(self, ticker: str) -> str:
        """
        Get Bloomberg subscription string for the given ticker.

        Args:
            ticker: Ticker symbol.

        Returns:
            str: Subscription string for the ticker.
        """
        if (self.drivers_anagraphic is not None) and (ticker in self.drivers_anagraphic.index):
            return self.drivers_anagraphic.loc[ticker, "subscription_BBG"]
        elif ticker in self.etfs:
            return f"{ticker} IM EQUITY"
        elif ticker in self.currencies_EUR_ccy:
            return f"{ticker} curncy"
        else:
            raise ValueError(f"Unknown ticker subscription for {ticker}")


if __name__ == "__main__":
    price_provider = PricesProvider(["IE00B02KXL92"])
    a = price_provider.get_hist_prices()
    b   = 0


if __name__ == '__main__':
    instruments = [
        "IE0007ULOZS8",
        "LU0274210672",
        "IE00BM9TV208",
        "LU1598689153",
        "IE00BMDPBZ72",
        "FR0013041530",
        "IE00BJXT3C94",
        "IE00BQZJBX31",
        "IE0004MFRED4",
        "IE000R85HL30",
        "IE00BHZPJ015",
        "IE00BQN1KC32",
        "FR0010261198",
        "IE00BKWQ0Q14",
        "LU1861137484",
        "IE00BK5BQY34",
        "LU1646360971",
        "IE00B14X4N27",
        "LU1291099718",
        "IE00BDGN9Z19"]


    input_params = {"currencies_EUR_ccy": [f"EUR{ccy}" for ccy in CURRENCY]}
    prices_provider = PricesProviderEquity(instruments, input_params)
    prices = prices_provider.get_hist_prices()
    adjustments = prices_provider.get_adjustments()
