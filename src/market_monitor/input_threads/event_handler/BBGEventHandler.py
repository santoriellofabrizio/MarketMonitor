import logging

import blpapi

from market_monitor.input_threads.event_handler.EventHandlerUI import EventHandler
from market_monitor.live_data_hub.real_time_data_hub import RTData
from user_strategy.utils.enums import ISIN_TO_TICKER

logging.getLogger()


def logCallback(threadId, severity, timeStamp, category, msg):
    pass


blpapi.Logger.registerCallback(logCallback, blpapi.Logger.SEVERITY_ERROR)


class BBGEventHandler(EventHandler):
    """
    Handles Bloomberg market data events and routes to appropriate RTData store.
    Bloomberg data always goes to MarketStore (DataFrame).
    """

    def __init__(self, real_time_data: RTData, **kwargs):
        """
        Initialize Bloomberg event handler.

        Args:
            real_time_data: RTData instance
            **kwargs: Additional arguments for base event_handler
        """
        super().__init__(**kwargs)
        self.update_subs_arrived = 0
        self.real_time_data = real_time_data
        self.fields_by_id = {}

        # Load subscription fields
        pending = real_time_data.get_pending_subscriptions("bloomberg")
        active = real_time_data.get_bloomberg_subscription() or {}

        all_subs = {**pending, **active}

        for id, sub in all_subs.items():
            if hasattr(sub, 'fields'):
                self.fields_by_id[id] = sub.fields

    def on_market_data_update(self, isin, data):
        """
        Route market data to MarketStore.

        Args:
            isin: Instrument identifier
            data: Dict of field → value (e.g., {"BID": 100.0, "ASK": 102.0})
        """
        # Bloomberg always routes to MarketStore
        self.real_time_data.update(isin, data, store="market")

    def on_initial_summary(self, msg: blpapi.Message):
        """
        Handle initial summary message with subscription metadata.
        """
        _id = msg.correlationIds()[0].value()
        
        for field in ["IS_DELAYED_STREAM", "RT_TRADING_PERIOD", "RT_EXCH_TRADE_STATUS"]:
            if msg.hasElement(blpapi.Name(field), excludeNullElements=True):
                try:
                    self.real_time_data.subscription_status.get(_id, {})
                except KeyError:
                    logging.error(f"No subscription status for {_id}")

    def handle_subscription_started(self, msg: blpapi.Message):
        """
        Handle successful subscription start.
        Activates pending subscription.
        """
        self.update_subs_arrived += 1
        _id = msg.correlationIds()[0].value()

        # Mark as received (activates if pending)
        self.real_time_data.mark_subscription_received(_id, "bloomberg")
        self.real_time_data.instrument_status[_id] = "ACTV"

        logging.info(f"Subscription started and activated: {_id}")

    def handle_subscription_failed(self, msg: blpapi.Message):
        """
        Handle subscription failure.
        Moves subscription to failed state.
        """
        _id = msg.correlationIds()[0].value()
        description = msg.getElement("reason").getElementAsString("category")

        logging.error(f"{_id} ({ISIN_TO_TICKER.get(_id, '')}): Subscription Failed, {description}")

        # Mark as failed
        self.real_time_data.mark_subscription_failed(_id, "bloomberg", description)

    def process_market_data_events(self, msg: blpapi.Message):
        """
        Extract market data from Bloomberg message.

        Args:
            msg: Bloomberg API message

        Returns:
            tuple: (instrument_id, data_dict)
        """
        data = {}
        _id = msg.correlationIds()[0].value()

        if self.fields_by_id.get(_id):
            # Extract specified fields
            for field in self.fields_by_id.get(_id):
                if msg.hasElement(blpapi.Name(field), excludeNullElements=True):
                    try:
                        el = msg.getElementAsFloat(blpapi.Name(field))
                    except:
                        el = msg.getElementAsString(blpapi.Name(field))
                    data[field] = el
        else:
            # Extract all elements
            for element in msg.asElement().elements():
                if not element.isNull():
                    data[str(element.name())] = element.getValue()

        return _id, data
