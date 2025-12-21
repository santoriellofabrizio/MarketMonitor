import logging
import threading
import time
from typing import Union

from blpapi import SessionOptions, Session, SubscriptionList, CorrelationId
from market_monitor.input_threads.EventHandler.BBGEventHandler import BBGEventHandler
from market_monitor.live_data_hub.RTData import RTData

logging.getLogger()


class BloombergStreamingThread(threading.Thread):
    """
   A thread dedicated to managing and storing Bloomberg updates in a shared, thread-safe object.

   Attributes:
       _host (str): Default host for the Bloomberg API connection.
       _port (int): Default port for the Bloomberg API connection.
       _service (str): Bloomberg service name for market data.

       terminate (bool): Flag to signal termination of the thread's execution.
       instruments_subscription_string (dict): Subscription details for instruments, mapping instrument names to their subscription strings.
       currency_dict (dict): Dictionary mapping instruments to their respective currencies.
       instruments (Union[None, str]): Placeholder for storing instrument data.
       event_handler (BBGEventHandler): Handler for processing Bloomberg events. Inherits from EventHandelrUI, or callable.
       options (str | None): Additional options for Bloomberg subscriptions, such as "interval=1" or "delayed"
       fields (tuple): Data fields requested for Bloomberg instruments, defaulting to ("BID", "ASK").
       BloombergClient (BloombergClient): Client instance for interacting with Bloomberg's real-time data API.
       subscription_dict (dict): Dictionary with subscription details obtained from the event handler.

   Methods:
       __init__(event_processor, bloomberg_client=None, kwargs=None): Initializes the thread and its configuration.
       subscribe_data(instruments_sub_string, fields=("BID", "ASK"), **kwargs): Sets the subscription details for data retrieval.
       run(): Starts the Bloomberg data subscription and processes updates.
       stop(): Signals termination of the thread.
   """

    _host = 'localhost'
    _port = 8194
    _service = "//blp/mktdata"

    def __init__(self, event_processor: BBGEventHandler, **kwargs):
        """
        Initializes the bloomberg.

        Args:
            event_processor (BBGEventHandler): An instance of BBGEventHandler to process incoming Bloomberg events.
            bloomberg_client (Union[BloombergClient, None], optional): Bloomberg client instance for API interactions. If None, a new client is created.
            kwargs (dict | None, optional): Additional configuration options, including 'options' for the subscription.
        """

        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self.running = False
        self.session = None
        self.terminate = False
        self.instruments_subscription_string: list | dict = {}
        self.currency_dict = {}
        self.instruments = None
        self.daemon = True
        self._host = kwargs.get("host", 'localhost')
        self._server_port = kwargs.get("port", 8194)
        self.name = "bloomberg"
        self.event_handler = event_processor
        self.real_time_data: RTData = event_processor.real_time_data
        self.subscription_service = self.real_time_data.get_subscription_manager()
        self.options = kwargs.get("options", None)

    def run(self):

        pending = self.subscription_service.get_pending_subscriptions("bloomberg")
        active = self.subscription_service.get_bloomberg_subscription() or {}

        if not pending and not active:
            logging.error("No subscriptions required in Bloomberg, shutting thread down")
            return

        session_options = SessionOptions()
        session_options.setServerHost(self._host)
        session_options.setServerPort(self._server_port)

        self.session = (
            Session(session_options, self.event_handler)
            if self.event_handler
            else Session(session_options)
        )

        if not self.session.start():
            raise RuntimeError("Failed to start Bloomberg session")

        if not self.session.openService("//blp/mktdata"):
            raise RuntimeError("Failed to open //blp/mktdata")

        self._subscribe_pending()
        self.running = True

        while self.running:
            if self._stop_event.wait(timeout=5):
                break
            self._subscribe_pending()
            self._process_unsubscribe()

    def stop(self):
        """Stop the thread gracefully"""
        self.running = False
        self._stop_event.set()

    def _subscribe_pending(self):
        """Subscribe to all pending subscriptions"""
        from market_monitor.live_data_hub.RTData import BloombergSubscription

        pending = self.subscription_service.get_pending_subscriptions(source="bloomberg")

        if not pending:
            return

        logging.info(f"Processing {len(pending)} pending subscriptions")

        subscription_list = SubscriptionList()

        for id, sub in pending.items():
            if isinstance(sub, BloombergSubscription):
                subscription_list.add(
                    sub.subscription_string,
                    sub.fields,
                    sub.params,
                    CorrelationId(sub.id),
                )
                logging.debug(f"Adding pending subscription: {id} ({sub.subscription_string})")

        if subscription_list.size() > 0:
            try:
                self.session.subscribe(subscription_list)
                logging.info(f"Subscribed to {subscription_list.size()} pending subscriptions")
            except Exception as e:
                logging.error(f"Failed to subscribe pending: {e}")
                # Mark all as failed
                for id in pending.keys():
                    self.real_time_data.mark_subscription_failed(id, "bloomberg", str(e))

    def _process_unsubscribe(self):
        """Process subscriptions marked for removal"""
        from market_monitor.live_data_hub.RTData import BloombergSubscription

        to_unsub = self.subscription_service.get_to_unsubscribe(source="bloomberg")

        if not to_unsub:
            return

        logging.info(f"Processing {len(to_unsub)} unsubscriptions")

        subscription_list = SubscriptionList()

        for id, sub in to_unsub.items():
            if isinstance(sub, BloombergSubscription):
                subscription_list.add(
                    sub.subscription_string,
                    sub.fields,
                    sub.params,
                    CorrelationId(sub.id),
                )
                logging.debug(f"Unsubscribing: {id} ({sub.subscription_string})")

        if subscription_list.size() > 0:
            try:
                self.session.unsubscribe(subscription_list)

                # Clear from unsubscribe queue
                for id in to_unsub.keys():
                    self.subscription_service.clear_unsubscribed(id, "bloomberg")

                logging.info(f"Unsubscribed from {subscription_list.size()} subscriptions")
            except Exception as e:
                logging.error(f"Failed to unsubscribe: {e}")


if __name__ == "__main__":
    securities = ["LU1437024992"]
    fields = ["MID"]
    lock = threading.Lock()
    book = RTData(locker=lock, fields=fields)
    sub_dict = {"IHYG": "IHYG IM EQUITY"}
    book.subscribe_bloomberg("IHYG", "IHYG IM EQUITY", fields=["MID"])
    book.subscribe_bloomberg("IHYU", "IHYU IM EQUITY", fields=["BID","ASK"])
    event_handler = BBGEventHandler(book)
    bloombergstreamingthread = BloombergStreamingThread(event_handler)
    bloombergstreamingthread.start()
    try:
        while True:
            time.sleep(1)
            print(book.get_data_field())
    except KeyboardInterrupt:
        bloombergstreamingthread.stop()
