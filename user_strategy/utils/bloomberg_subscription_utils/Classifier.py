from user_strategy.utils.enums import Instrument, ALL_ETF_ISINS, CURRENCY, ALL_INTEREST_RATES


class Classifier:
    """
    Classifies financial instruments based on their format (e.g., ISIN, FX, Futures).

    Attributes:
        classification (dict): A dictionary mapping instruments to their classified type.
    """

    def __init__(self):
        """Initializes the Classifier with empty classification and FX instrument dictionaries."""

        # Accedi alle configurazioni
        self.classification = {}  # Stores the classification of instruments

    def classify(self, instruments) -> Instrument:
        """
        Classifies a given instrument based on its format and caches the result.

        Args:
            instrument (str): The instrument to classify (e.g., ISIN, FX, ticker).

        Returns:
            Instrument: The classification type (e.g., FX, ETF, Future, FX Forward).
        """
        if isinstance(instruments, str): instruments = [instruments]
        for instrument in instruments:
            if instrument in self.classification:
                return self.classification[instrument]

            # Determine classification based on instrument type
            classification_type = (
                Instrument.ETF if self.is_etf(instrument) else
                Instrument.FXFWD if self.is_fx_frwd(instrument) else
                Instrument.FX if self.is_ccy(instrument) else
                Instrument.IR if self.is_interest_rate(instrument) else
                Instrument.FUTURE if self.is_future(instrument) else
                Instrument.BOND if self.is_bond(instrument) else
                Instrument.EQUITY
            )

            # Cache and return the classification type
            self.classification[instrument] = classification_type
            return classification_type

    @staticmethod
    def is_etf(isin: str):
        """
        Validates if a string matches the ISIN format.
        """
        return isin in ALL_ETF_ISINS

    @staticmethod
    def is_future(ticker):
        """
        Validates if a string matches the futures ticker format.
        """
        return ticker.upper().endswith('INDEX') or ticker.endswith('COMDTY')

    @staticmethod
    def is_fx_frwd(ticker: str):
        return ticker.upper().startswith('FX')

    @staticmethod
    def is_bond(ticker: str):
        return ticker.upper().startswith('CORP')

    @staticmethod
    def is_interest_rate(ticker: str):
        return ticker.split(' ')[0] in ALL_INTEREST_RATES

    def get_class(self, instrument_type):
        """
        Retrieves a list of instruments for a specific type.

        Args:
            instrument_type (Instrument): The type of instrument to retrieve.

        Returns:
            list: A list of instruments that match the specified type.
        """
        return [isin for isin, _type in self.classification.items() if _type == instrument_type]

    @staticmethod
    def is_ccy(security):
        """
        Checks if a security is a currency (FX) by matching against a list of currencies.

        Args:
            security (str): The security string to check.

        Returns:
            bool: True if the security is a currency, False otherwise.
        """
        return (security in CURRENCY or
                (security[:3] in CURRENCY and security[3:6] in CURRENCY))


    def is_equity(self, security):
        """
        Checks if a security is a currency (FX) by matching against a list of currencies.

        Args:
            security (str): The security string to check.

        Returns:
            bool: True if the security is a currency, False otherwise.
        """

        return self.classify(security) == Instrument.EQUITY
