from typing import Optional, List, Dict, Tuple

from enum import Enum

from user_strategy.utils.bloomberg_subscription_utils.OracleConnection import OracleConnection


class FutureType(Enum):
    """
    Possible futures underlying type
    """
    COMDTY = "COMDTY"  # Fixed Income
    INDEX = "INDEX"  # Equity


class InstrumentMarketCurrency:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._market_priority: List[str] = ["GB","GD", "GZ", "IM", "BW",
                                                "GH","GI", "IB","GB" "S1", "S4", "TH",  "UB"]
            self._instr_market_ccy: Dict[str, Dict[str, Optional[str]]] = {}
            self._futures_ccy: Dict[Tuple[str, str], str] = {}
            self._bond_ccy: Dict[str, str] = {}

            self._etf_ticker: Dict[str, str] = {}
            # self._load_data()

    def _load_data(self):
        oracle_conn = OracleConnection()
        oracle_conn.connect()
        self._load_futures(oracle_conn)
        self._load_bond(oracle_conn)
        self._load_etf_ticker(oracle_conn)
        self._load_equity_etf(oracle_conn)
        oracle_conn.close()

    def _load_futures(self, oracle_conn: OracleConnection):
        """
        Load data from the Oracle Connection to initialize the dict future - ccy
        """
        query = f"""SELECT TICKER, BBG_TYPE, CURRENCY FROM AF_PCF.FUTURES_ROOTS"""
        data, _ = oracle_conn.execute_query(query)
        self._futures_ccy = {(row[0], row[1]): row[2] for row in data}

    def _load_equity_etf(self, oracle_conn: OracleConnection):
        """
        Load data from the Oracle Connection to initialize the instruments market ccy.
        """
        query = f"""SELECT i.ISIN, ei.EXCHANGE_CODE, ei.CURRENCY
                    FROM AF_PCF.EXCHANGE_INSTRUMENTS ei, AF_PCF.INSTRUMENTS i
                    WHERE ei.INSTRUMENT_ID = i.ID and i.INSTRUMENT_TYPE IN ('EQUITY', 'ETP', 'UNDEFINED')"""
        data, _ = oracle_conn.execute_query(query)
        for row in data:
            if row[0] not in self._instr_market_ccy:
                self._instr_market_ccy[row[0]] = {}
            self._instr_market_ccy[row[0]][row[1]] = row[2]

    def _load_bond(self, oracle_conn: OracleConnection):
        """
        Load data from the Oracle Connection to initialize the dict bond - ccy
        """
        query = f"""SELECT ISIN, ISSUE_CURRENCY FROM AF_PCF.BONDS_INSTRUMENTS"""
        data, _ = oracle_conn.execute_query(query)
        self._bond_ccy = {row[0]: row[1] for row in data}

    def _load_etf_ticker(self, oracle_conn: OracleConnection):
        """
        Load data from the SQLite database to initialize the dict etf - ccy
        """
        query = f"""SELECT ISIN, TICKER FROM AF_PCF.ETPS_INSTRUMENTS"""
        data, _ = oracle_conn.execute_query(query)
        self._etf_ticker = {row[1]: row[0] for row in data}


    def get_crncy(self, isin: str, market: Optional[str] = None) -> Optional[str]:
        """
        Determines the currency associated with a given (isin, market) pair.

        The method follows these rules:
        1. If the ISIN matches the format of a future (identified using a regex),
           it returns the currency from the `_futures_ccy` mapping.
        2. If the ISIN is found between the bond list, it returns the currency of the bond
        3. If the `market` parameter is provided, it looks up the currency directly
           in the `_instruments_market_ccy` dictionary using the (isin, market) pair.
        4. If `market` is not provided, it uses the `_market_priority` list iterates over the prioritized markets and
           returns the first non-None currency found.
        5. If no match is found in the prioritized markets, it returns the first
           non-None currency available for the ISIN.
        6. If no currency is available, it returns `None`.

        Parameters:
        - isin (str): The identifier for the financial instrument.
        - market (Optional[str]): The name of the market. If not provided, the method applies the priority list logic.

        Returns:
        - Optional[str]: The currency found, or `None` if no match is available.
        """
        isin = isin.upper()

        if isin.endswith((FutureType.COMDTY.value, FutureType.INDEX.value)): # Future
            index_comdty = FutureType.INDEX if isin.endswith(FutureType.INDEX.value) else FutureType.COMDTY
            root = isin.replace(" COMDTY", "").replace(" INDEX", "")[:-2]
            return self._futures_ccy.get((root, index_comdty.value))

        if isin in self._bond_ccy: # Bond
            return self._bond_ccy.get(isin)

        isin = self._etf_ticker.get(isin) if isin in self._etf_ticker else isin

        if market:
            return self._instr_market_ccy.get(isin, {}).get(market)

        market_ccy = self._instr_market_ccy.get(isin, {})
        for priority_market in self._market_priority:
            if priority_market in market_ccy:
                return market_ccy.get(priority_market)

        for ccy in market_ccy.values():
            if ccy is not None:
                return ccy

        return None

    def get_markets(self, isin: str, n_markets: int = 1) -> List[str | None]:
        """
        Get the markets associated with the given ISIN, sorted based on the priority list.

        Parameters:
        - isin (str): The identifier for the financial instrument.
        - n_market (int): The number of markets to return.

        Returns:
        - Optional[str]: The list of markets in which the instrument is quoted.
        """
        isin = isin.upper()
        isin = self._etf_ticker.get(isin) if isin in self._etf_ticker else isin

        market_ccy = self._instr_market_ccy.get(isin, {})
        markets = list(market_ccy)

        prioritized_markets = [market for market in self._market_priority if market in markets]
        remaining_markets = sorted([market for market in markets if market not in self._market_priority])
        if not remaining_markets and not prioritized_markets:
            return [None]
        return (prioritized_markets + remaining_markets)[:n_markets]
