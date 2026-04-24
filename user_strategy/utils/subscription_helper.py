from time import sleep as sleep_time
from typing import Literal

from sfm_data_provider.core.enums.instrument_types import InstrumentType
from sfm_data_provider.core.instruments.instruments import Instrument
from sfm_data_provider.core.requests.subscriptions import BloombergSubscriptionBuilder

from user_strategy.utils.EtfUniverse import EtfUniverse

_RETRY_MARKETS    = ("NA", "FP")
_KAFKA_MAPPING    = {"IM": "ETFP", "NA": "XAMS", "FP": "XPAR"}
_STALE_TYPES      = {InstrumentType.SWAP, InstrumentType.INDEX, InstrumentType.CDXINDEX}
_MAX_FAILED_RATIO = 1 / 100


class SubscriptionManager:
    """
    Manages all Bloomberg/Kafka subscriptions for the fixed-income strategy.

    Responsibilities:
      - subscribe_all()         : dispatches the right subscription per instrument type
      - wait_for_initialization(): blocks until BBG subscriptions are ready, retries failures
    """

    def __init__(self, universe: EtfUniverse,
                 global_subscription_service, market_data,
                 live_book, live_book_etf) -> None:
        self.universe  = universe
        self.svc       = global_subscription_service
        self.mkt_data  = market_data
        self.live_book     = live_book
        self.live_book_etf = live_book_etf

    # ── Public API ────────────────────────────────────────────────────────────

    def subscribe_all(self, price_source: Literal['kafka', 'bloomberg'] = 'bloomberg') -> None:
        """Dispatches subscription for every instrument type by iterating instruments_by_type."""
        for instr_type, instruments in self.universe.instruments_by_type.items():
            for inst in instruments:
                match instr_type:
                    case InstrumentType.ETP:         self._subscribe_single_etf(inst, price_source)
                    case InstrumentType.FUTURE:       self._subscribe_single_future(inst, price_source)
                    case InstrumentType.CURRENCYPAIR: self._subscribe_single_fx(inst)
                    case t if t in _STALE_TYPES:      self._subscribe_single_stale(inst)

    def wait_for_initialization(self, instruments_list: list) -> bool:
        """Blocks until BBG subscriptions settle; returns False if too many BAD_SEC failures."""
        while self.mkt_data.get_pending_subscriptions("bloomberg"):
            sleep_time(1)
        self._retry_failed_subscriptions()
        bad = [s.get("id") for s in self.svc.get_failed_subscriptions()
               if s.get("last_error") == "BAD_SEC"]
        return bool(instruments_list) and len(bad) / len(instruments_list) < _MAX_FAILED_RATIO

    # ── Single-instrument subscription ───────────────────────────────────────

    def _subscribe_single_etf(self, inst: Instrument, price_source: Literal['kafka', 'bloomberg']) -> None:
        for mkt in self.universe.markets_by_isin.get(inst.id, []):
            sub_id   = f"{mkt}:{inst.id}"
            currency = self.universe.currency_per_isin_market.get((inst.id, mkt), "EUR")
            if price_source == 'bloomberg':
                self._subscribe_bloomberg(sub_id, f"{inst.id} {mkt} EQUITY", ["BID", "ASK"],
                                          live_book=self.live_book_etf, instr_id=inst.id,
                                          market=mkt, currency=currency, options={"interval": 1})
            else:
                self.svc.subscribe_kafka(
                    id=inst.id, symbol_filter=inst.id,
                    topic=f"COALESCENT_DUMA.{_KAFKA_MAPPING[mkt]}.BookBest",
                    fields_mapping={"BID": "bidBestLevel.price", "ASK": "askBestLevel.price"},
                )
                self.live_book_etf.register(sub_id=sub_id, instr_id=inst.id, market=mkt, currency=currency)

    def _subscribe_single_future(self, inst: Instrument, price_source: Literal['kafka', 'bloomberg']) -> None:
        if price_source != 'bloomberg': raise NotImplementedError
        self._subscribe_bloomberg(inst.id, f"{inst.root}A {inst.suffix}", ["BID", "ASK"],
                                  live_book=self.live_book, instr_id=inst.id,
                                  market=inst.market, currency=inst.currency, options={"interval": 1})

    def _subscribe_single_fx(self, inst: Instrument) -> None:
        self._subscribe_bloomberg(inst.id, f"{inst.id} Curncy", ["BID", "ASK"], options={"interval": 1})

    def _subscribe_single_stale(self, inst: Instrument) -> None:
        self._subscribe_bloomberg(inst.id, BloombergSubscriptionBuilder.build_subscription(inst), ["LAST_PRICE"],
                                  live_book=self.live_book, instr_id=inst.id,
                                  market=inst.market, currency=inst.currency)

    # ── Bloomberg wrapper ─────────────────────────────────────────────────────

    def _subscribe_bloomberg(self, sub_id: str, security: str, fields: list[str],
                             live_book=None, instr_id: str | None = None,
                             market: str | None = None, currency: str | None = None,
                             options: dict | None = None) -> None:
        if options is None:
            self.svc.subscribe_bloomberg(sub_id, security, fields)
        else:
            self.svc.subscribe_bloomberg(sub_id, security, fields, options)
        if live_book is not None:
            live_book.register(sub_id=sub_id, instr_id=instr_id, market=market, currency=currency)

    # ── Retry logic ───────────────────────────────────────────────────────────

    def _retry_failed_subscriptions(self) -> None:
        failed = {s.get("id") for s in self.svc.get_failed_subscriptions()
                  if s.get("id") in self.universe.all_etf_isin}
        for isin in failed:
            self.svc.unsubscribe(isin, 'bloomberg')
        for mkt in _RETRY_MARKETS:
            for isin in failed:
                self.svc.subscribe_bloomberg(isin, f"{isin} {mkt} EQUITY", ["BID", "ASK"])
            sleep_time(5)