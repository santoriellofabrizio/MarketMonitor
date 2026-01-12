import logging

import pandas as pd
from ruamel.yaml import YAML

from user_strategy.utils.bloomberg_subscription_utils.InstrumentMarketCurrency import \
    InstrumentMarketCurrency
from user_strategy.utils.bloomberg_subscription_utils.Classifier import Classifier
from user_strategy.utils.pricing_models.ExcelStoringDecorator import save_to_excel
from user_strategy.utils.enums import FIXED_INCOME_FUTURE, Instrument

CACHE_FILE = "input_threads/bloomberg/BbgSubscriptionManager/cachestocks/instrument_status.pkl"
LIVE_MARKETS = ("GB", "GD", "GZ", "IM", "BW", "GH", "GI", "IB", "GB" "S1", "S4", "TH", "UB")
CURRENCY_TO_IGNORE = ("EUR", "GBp", "EUREUR", "EUREUR CURCNY")
STOCK_DB_PATH = "strategy/user_strategy/equity/stocks.db"


GERMAN_COMPOSITE = {local: "GR" for local in ["GD", "GF", "GS","GH","GI","GB","GS","GM"]}

class SubscriptionManager:

    def __init__(self, securities: list[str],
                 config_bloomberg_subscription_path: str | None = None,
                 **kwargs):

        self.logger = logging.getLogger()
        self.securities = securities

        self.config_bloomberg_subscription_path = config_bloomberg_subscription_path

        self.currency_information = {}
        self.subscription_dict = {}
        self.instruments_information = {}
        self.subscription_status = {}
        self.inactive_instruments = []

        self.classifier = Classifier()
        self.anagraphic_manager = InstrumentMarketCurrency()

        self._set_hard_coded_subscriptions(config_bloomberg_subscription_path)
        self._set_default_for_ETFPLUS()
        self._generate_subscription_dict()

    def handle_missing_subscription(self, isin: str, description: str, msg):
        if description == "BAD_SEC": return
        self.instruments_information[isin] = {**self.instruments_information.get(isin, {}),
                                              "market_status": description}

    def get_subscription_string(self, instrument: str, ref_market: str = ""):
        """
        Generates a subscription string for a given instrument based on its classification. es: IHYG -> IHYG IM EQUIYY

        Args:
            ref_market: reference market (es: IM)
            instrument (str): The name or ticker of the instrument.

        Returns:
            str: A Bloomberg subscription string for the given instrument.
        """
        if instrument in self.subscription_dict: return self.subscription_dict[instrument]
        if instrument.upper().endswith(("INDEX", "EQUITY", "CURNCY", "COMDTY", "CORP")): return instrument

        match self.classifier.classify(instrument):
            case Instrument.ETF:
                return f'{instrument} IM Equity'
            case Instrument.FX:
                if instrument.upper() in ["EUR", "EUREUR", "EUREUR CURNCY"]: return "EUR"
                if len(instrument) == 6:
                    return f'{instrument} CURNCY'
                elif len(instrument) == 3:
                    return f'EUR{instrument} CURNCY'
            case Instrument.FUTURE:
                if instrument in FIXED_INCOME_FUTURE:
                    return f'{instrument} COMDTY'
                else:
                    return f'{instrument} INDEX'
            case Instrument.EQUITY:
                return f'{instrument} {self.get_ref_market(instrument)} EQUITY'

    def get_subscription_dict(self):
        return self.subscription_dict

    def get_instruments_information(self):
        self._generate_instrument_informations()
        return self.instruments_information

    def _generate_subscription_dict(self):
        self.subscription_dict.update({security: self.get_subscription_string(security)
                                       for security in self.securities if security not in self.subscription_dict})

    @save_to_excel("instrument_information")
    def _generate_instrument_informations(self):
        for instrument in self.securities:
            if instrument not in self.instruments_information:
                match self.classifier.classify(instrument):
                    case Instrument.ETF:
                        self.instruments_information[instrument] = {"subscription": f'{instrument} IM Equity',
                                                                    "currency": "EUR",
                                                                    "market_status": "ACTV"}
                    case Instrument.FX:
                        sub = self.subscription_dict[instrument]
                        self.instruments_information[instrument] = {"subscription": sub,
                                                                    "currency": "EUR",
                                                                    "market_status": "ACTV"}
                    case Instrument.FUTURE:
                        self.instruments_information[instrument] = {"subscription": self.subscription_dict[instrument],
                                                                    "currency":
                                                                        self.anagraphic_manager.get_crncy(instrument,
                                                                                                    "Future"),
                                                                    "market_status": "ACTV"}
                    case Instrument.EQUITY:
                        market = self.get_ref_market(instrument)
                        self.instruments_information[instrument] = {"subscription": f'{instrument} {market} EQUITY',
                                                                    "currency":
                                                                        self.anagraphic_manager.get_crncy(instrument,
                                                                                                          market),
                                                                    "market_status": "ACTV"}

        return pd.DataFrame(self.instruments_information).T

    def get_ref_market(self, isin) -> str:
        ref_markets = self.anagraphic_manager.get_markets(isin)
        if ref_markets and ref_markets != [None]:
            return ref_markets[0]
        else:
            self.inactive_instruments.append(isin)
            temp = self.instruments_information.get(isin, {})
            temp.update({"market_status": "UNLST"})
            self.instruments_information["isin"] = temp

    def get_currency_informations(self):

        for instrument in self.securities:
            if instrument not in self.currency_information:
                match self.classifier.classify(instrument):
                    case Instrument.ETF:
                        self.currency_information[instrument] = "EUR"
                    case Instrument.FX:
                        self.currency_information[instrument] = "EUR"
                    case Instrument.IR:
                        self.currency_information[instrument] = "EUR"
                    case Instrument.FUTURE:
                        if self.anagraphic_manager is None:
                            self.logger.error(f"No anagraphic manager is set, please set anagraphic in init."
                                              f" Cannot set currency of {instrument}")
                            continue

                        self.currency_information[instrument] = self.anagraphic_manager.get_crncy(instrument, "Future")

                    case Instrument.EQUITY:
                        market = self.get_ref_market(instrument)
                        self.currency_information[instrument] = self.anagraphic_manager.get_crncy(instrument, market)
        return self.currency_information

    def get_inactive_securities(self) -> list[str]:
        return self.inactive_instruments

    def get_future_crncy(self, isin: str) -> str:
        return self.anagraphic_manager.get_crncy(isin)

    def _set_default_for_ETFPLUS(self):
        for isin in self.securities:
            if self.classifier.is_etf(isin):
                if not isin in self.instruments_information:
                    self.instruments_information[isin] = {"subscription": f"{isin} IM EQUITY",
                                                          "currency": "EUR"}
                if isin not in self.subscription_dict:
                    self.subscription_dict[isin] = f"{isin} IM EQUITY"
                if isin not in self.subscription_status:
                    self.subscription_status[isin] = "ACTV"
                if isin not in self.currency_information:
                    self.currency_information[isin] = "EUR"

    def _set_hard_coded_subscriptions(self, path_config: str | None):
        if path_config is None: return
        try:
            with open(path_config, 'r') as file:
                yaml = YAML(typ='safe', pure=True)
                config = yaml.load(file)
                for isin, data in config.items():
                    if isin in self.securities:
                        if "market_status" not in data: data["market_status"] = "ACTV"
                        for flds in ["currency", "subscription"]:
                            if flds not in data:
                                self.logger.error(
                                    f"please specify {flds} for {isin} in config: {self.config_bloomberg_subscription_path}")

                        self.instruments_information[isin] = data
                        self.subscription_dict[isin] = data["subscription"]
                        self.currency_information[isin] = data["currency"]

        except Exception as e:
            self.logger.error(f"Error while loading {path_config}. Hard coded subscriptions has not been loaded."
                              f" Error: {e}")


if __name__ == "__main__":
    securities = ["IE000E4BATC9", "CDXHY", "VGA INDEX"]
    bloomberg_config = r"C:\AFMachineLearning\Projects\Trading\MarketMonitorFI\utils\SubscriptionManager\config_bloomberg_subscription.yaml"
    db_stock_config_path = r"/\UserStrategy\ETFEquity\stocks.db"
    sub_manager = SubscriptionManager(securities=securities,
                                      db_stock_config_path=db_stock_config_path,
                                      config_bloomberg_subscription_path=bloomberg_config)

    sub_dict = sub_manager.get_subscription_dict()
    ccy = sub_manager.get_currency_informations()

    a = 0
