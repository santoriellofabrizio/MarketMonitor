from typing import List, Optional, Union, Literal, TypeVar, Dict, Any

from attr.validators import is_callable
from dateutil.utils import today
from sfm_data_provider.core.enums.instrument_types import InstrumentType
from sfm_data_provider.core.instruments.instruments import Instrument
from sfm_data_provider.core.requests.subscriptions import BloombergSubscriptionBuilder
from sfm_data_provider.interface.bshdata import BshData
from sfm_return_adjustments_lib import Instruments

from market_monitor.live_data_hub.subscription_service import SubscriptionService


_SOURCES = {'kafka', 'bloomberg', 'redis'}

_DEFAULT_TYPE_MARKET_SOURCES = {
    InstrumentType.ETP: {"ETFP": 'kafka', "XAMS": "kafka", "XPAR": "kafka"},
    InstrumentType.FUTURE: {"XEUR": 'kafka', 'XCBT': 'kafka'}
}

_KAFKA_MARKET_TOPICS = {
    InstrumentType.ETP: {"ETFP": 'BookBest', "XAMS": "BookBest", "XPAR": "BookBest"},
    InstrumentType.FUTURE: {"XEUR": 'BookBest', 'XCBT': 'BookBest'}
}

AvailableSources = Literal['kafka', 'bloomberg', 'redis']
T = TypeVar('T', bound=AvailableSources)


class SubscriptionHelper:

    def __init__(self, api: BshData, subscription_service: SubscriptionService) -> None:
        self.api = api
        self.subscription_service = subscription_service
        self._rules = _DEFAULT_TYPE_MARKET_SOURCES
        self._bbg_builder = BloombergSubscriptionBuilder.build_subscription

    def subscribe_by_market(self, market: str,
                            instruments: List[Instruments],
                            fields: Union[List[str], Dict[str, str]],
                            preferable_source: Optional[T] = 'kafka',
                            **kwargs) -> None:

        for instrument in instruments:
            if preferable_source == 'kafka':
                if _KAFKA_MARKET_TOPICS.get(instrument.type, {}).get(market):
                    fields = kwargs.pop('fields', fields) or {"BID": "bidBestLevel.price", "ASK": "askBestLevel.price"}
                    self.subscription_service.subscribe_kafka(id=f"{market}:{instrument.id}",
                                                              symbol_filter=kwargs.get(
                                                                  'symbol_filter') or instrument.isin,
                                                              topic=f"COALESCENT_DUMA.{market}.{kwargs.get('topic') or 'BookBest'}",
                                                              fields_mapping=fields)
            else:
                inst = instrument
                inst.market = market
                self.subscription_service.subscribe_bloomberg(id=f"{market}:{instrument.id}",
                                                              subscription_string=self._bbg_builder(inst),
                                                              fields=fields,
                                                              params=kwargs.get('params', {}))

    def subscribe_by_isin(self, isin: str, fields: List[str], markets: Optional[list[str]], **kwargs) -> None:
        pass

    def set_rule(self, instrument_type: Union[InstrumentType, str],
                 market: str,
                 source: T) -> None:
        self._rules[instrument_type][market] = source

    def subscribe_kafka(
            self,
            id: str,
            topic: str,
            symbol_filter: Optional[str] = None,
            symbol_field: str = "instrument.isin",
            store: str = "market",
            fields_mapping: Optional[Dict[str, str]] = None,
            group: Optional[str] = None,
    ):

        self.subscription_service.subscribe_kafka(id=id,
                                                  topic=topic,
                                                  symbol_filter=symbol_filter,
                                                  symbol_field=symbol_field,
                                                  store=store,
                                                  fields_mapping=fields_mapping,
                                                  group=group)

    def subscribe_instrument(
            self,
            instrument: Instrument,
            source: AvailableSources,
            fields: Optional[List[str]] = None,
            params: Optional[Dict[str, Any]] = None,
            subscription_string: Optional[str] = None,
            market: str = 'GenericMarket',
            currency: str = 'GenericCurrency',
            **kwargs
    ) -> str:

        market = market or instrument.market
        currency = currency or instrument.currency

        instrument.currency = currency
        instrument.market = market

        id_sub = f"{market}:{instrument.id}:{currency}"
        if source == "bloomberg":
            subscription_string = subscription_string or BloombergSubscriptionBuilder.build_subscription(instrument)
            if callable(subscription_string):
                subscription_string = subscription_string(current_date=today())
            self.subscription_service.subscribe_bloomberg(id=id_sub,
                                                          subscription_string=subscription_string,
                                                          fields=fields,
                                                          params=params)
        elif source == "kafka":
            topic = f"COALESCENT_DUMA.{instrument.market}.BookBest"
            self.subscription_service.subscribe_kafka(id=id_sub,
                                                      topic=topic,
                                                      symbol_filter=kwargs.get('symbol_filter') or instrument.isin,
                                                      symbol_field=kwargs.get('symbol_field') or "instrument.isin",
                                                      fields_mapping=kwargs.get('fields_mapping')
                                                                     or {"BID": "bidBestLevel.price",
                                                                         "ASK": "askBestLevel.price"})

        return id_sub
