import json
import logging
from abc import ABC

import blpapi


class EventHandler(ABC):
    SESSION_STARTED = blpapi.Name("SessionStarted")
    SESSION_TERMINATED = blpapi.Name("SessionTerminated")
    SESSION_STARTUP_FAILURE = blpapi.Name("SessionStartupFailure")
    SESSION_CONNECTION_UP = blpapi.Name("SessionConnectionUp")
    SESSION_CONNECTION_DOWN = blpapi.Name("SessionConnectionDown")
    SESSION_CLUSTER_INFO = blpapi.Name("SessionClusterInfo")
    SESSION_CLUSTER_UPDATE = blpapi.Name("SessionClusterUpdate")
    SERVICE_OPENED = blpapi.Name("ServiceOpened")
    SERVICE_OPENED_FAILURE = blpapi.Name("ServiceOpenedFailure")
    MARKET_DATA_EVENTS = blpapi.Name("MarketDataEvents")
    SUMMARY = blpapi.Name("Summary")
    SUBSCRIPTION_STARTED = blpapi.Name("SubscriptionStarted")
    SUBSCRIPTION_FAILURE = blpapi.Name("SubscriptionFailure")
    SUBSCRIPTION_TERMINATED = blpapi.Name("SubscriptionTerminated")
    SUBSCRIPTION_STREAMS_ACTIVATED = blpapi.Name("SubscriptionStreamsActivated")
    SUBSCRIPTION_STREAMS_DEACTIVATED = blpapi.Name("SubscriptionStreamsDeactivated")
    REFERENCE_DATA_RESPONSE = blpapi.Name("ReferenceDataResponse")
    REQUEST_FAILURE = blpapi.Name("RequestFailure")
    AUTHORIZATION_SUCCESS = blpapi.Name("AuthorizationSuccess")
    AUTHORIZATION_FAILURE = blpapi.Name("AuthorizationFailure")
    RESPONSE_ERROR = blpapi.Name("ResponseError")
    ENTITLEMENT_CHANGED = blpapi.Name("EntitlementChanged")
    AUTHORIZATION_REVOKED = blpapi.Name("AuthorizationRevoked")
    SLOW_CONSUMER_WARNING = blpapi.Name("SlowConsumerWarning")
    IS_DELAYED_STREAM = blpapi.Name("IsDelayedStream")

    def __init__(self, **kwargs):
        self.parser = None
        self.EventHandler = {
            blpapi.Event.SESSION_STATUS: {
                self.SESSION_STARTED: self.handle_session_started,
                self.SESSION_TERMINATED: self.handle_session_terminated,
                self.SESSION_STARTUP_FAILURE: self.handle_session_startup_failure,
                self.SESSION_CONNECTION_UP: self.handle_session_connection_up,
                self.SESSION_CONNECTION_DOWN: self.handle_session_connection_down,
                self.SESSION_CLUSTER_INFO: self.handle_session_cluster_info,
                self.SESSION_CLUSTER_UPDATE: self.handle_session_cluster_update,
            },
            blpapi.Event.SERVICE_STATUS: {
                self.SERVICE_OPENED: self.handle_service_opened,
                self.SERVICE_OPENED_FAILURE: self.handle_service_opened_failure,
            },
            blpapi.Event.SUBSCRIPTION_DATA: {
                self.MARKET_DATA_EVENTS: self.handle_market_data_events,
            },
            blpapi.Event.SUBSCRIPTION_STATUS: {
                self.SUBSCRIPTION_STARTED: self.handle_subscription_started,
                self.SUBSCRIPTION_FAILURE: self.handle_subscription_failed,
                self.SUBSCRIPTION_TERMINATED: self.handle_subscription_terminated,
                self.SUBSCRIPTION_STREAMS_ACTIVATED: self.handle_subscription_stream_activated,
                self.SUBSCRIPTION_STREAMS_DEACTIVATED: self.handle_subscription_stream_deactivated,
            },
            blpapi.Event.PARTIAL_RESPONSE: {
                self.REFERENCE_DATA_RESPONSE: self.handle_reference_data_response,
            },
            blpapi.Event.RESPONSE: {
                self.REFERENCE_DATA_RESPONSE: self.handle_reference_data_response,
            },
            blpapi.Event.REQUEST_STATUS: {
                self.REQUEST_FAILURE: self.handle_request_failure,
            },
            blpapi.Event.AUTHORIZATION_STATUS: {
                self.AUTHORIZATION_SUCCESS: self.handle_authorization_success,
                self.AUTHORIZATION_FAILURE: self.handle_authorization_failure,
                self.RESPONSE_ERROR: self.handle_response_error,
                self.ENTITLEMENT_CHANGED: self.handle_entitlement_changed,
                self.AUTHORIZATION_REVOKED: self.handle_authorization_revoked,
            },
            blpapi.Event.ADMIN: {
                self.SLOW_CONSUMER_WARNING: self.handle_slow_consumer_warning,
            },
            # Aggiungi qui ulteriori eventi e MessageType...
        }

        self.subscription_status = {}

    def __call__(self, *args, **kwargs):
        self.processEvent(*args, **kwargs)

    def processEvent(self, event: blpapi.Event, session: blpapi.Session):
        for msg in event:
            func = self.EventHandler.get(event.eventType(), {}).get(msg.messageType(), None)
            if func:
                func(msg)
            else:
                logging.debug(f"Unhandled message type: {msg.messageType()}")

    def handle_market_data_events(self, msg: blpapi.Message):
        if msg.hasElement("MKTDATA_EVENT_SUBTYPE"):
            subtype = msg.getElementAsString("MKTDATA_EVENT_SUBTYPE")
            if subtype == "INITPAINT":
                self.on_initial_summary(msg)
        _id, data = self.process_market_data_events(msg)
        if msg.hasElement(self.IS_DELAYED_STREAM, excludeNullElements=True):
            self.subscription_status[_id].update({"delayed": msg.getElement(self.IS_DELAYED_STREAM)}, )
        logging.debug(f"Market data event processed for ID: {_id}")
        self.on_market_data_update(_id, data)

    def on_initial_summary(self, msg: blpapi.Message):
        """
        processes the summary that arrives when subscription is made, contains more than 40 fields.
        Args:
            msg:

        Returns:

        """
        logging.info(f"initial summary for {msg.correlationIds()[0].value()}")
        logging.warning(f"summary containts: {msg}")

    def process_market_data_events(self, msg: blpapi.Message):
        """
        Processes market data events and extracts relevant fields from the message.

        Args:
            msg (blpapi.Message): Bloomberg API message containing market data.

        Returns:
            tuple: A tuple containing the security ID and the extracted data as a dictionary.
        """
        data = {}
        _id = msg.correlationIds()[0].value()
        fields = msg.asElement().elements()  # Get the correlation ID for the message
        if fields:  # Check if fields are specified for monitoring
            for field in fields:
                if msg.hasElement(blpapi.Name(field), excludeNullElements=True):  # Check if field exists in the message
                    data[field] = msg.getElementAsFloat(blpapi.Name(field))  # Extract the field's value
        else:  # If no fields are specified, process all elements
            for element in msg.asElement().elements():
                if not element.isNull():
                    data[str(element.name())] = element.getValue()  # Extract the element's value
        return _id, data

    def on_market_data_update(self, isin, data):
        pass

    @staticmethod
    def handle_request_failure(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.error(f"Request Failure: {msg}")

    @staticmethod
    def handle_authorization_success(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.info("Authorization Success")

    @staticmethod
    def handle_authorization_failure(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.error(f"Authorization Failure {_id}: {msg}")

    @staticmethod
    def handle_response_error(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.critical(f"Response Error {_id}: {msg}")

    @staticmethod
    def handle_entitlement_changed(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.info(f"Entitlement Changed {_id}: {msg}")

    @staticmethod
    def handle_authorization_revoked(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.info(f"Authorization Revoked {_id}: {msg}")

    @staticmethod
    def handle_slow_consumer_warning(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.warning(f"Slow Consumer Warning {_id}: {msg}")

    @staticmethod
    def handle_session_started(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.info(f"Session Started {_id}")

    @staticmethod
    def handle_session_terminated(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.info(f"Session Terminated {_id}")

    @staticmethod
    def handle_session_startup_failure(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.error(f"Session Startup Failure {_id}: {msg}")

    @staticmethod
    def handle_session_connection_up(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.info(f"Session Connection Up {_id}")

    @staticmethod
    def handle_session_connection_down(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.error(f"Session Connection Down {_id}: {msg}")

    @staticmethod
    def handle_session_cluster_info(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.info(f"Session Cluster Info {_id}: {msg}")

    @staticmethod
    def handle_session_cluster_update(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.info(f"Session Cluster Update {_id}: {msg}")

    @staticmethod
    def handle_service_opened(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.info(f"Service Opened {_id}")

    @staticmethod
    def handle_service_opened_failure(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.error(f"Service Opened Failure {_id}: {msg}")

    def handle_subscription_started(self, msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        self.subscription_status[_id] = {"SubscriptionStarted": True}
        logging.info(f"Subscription Started {_id}")

    def handle_subscription_failed(self, msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        self.subscription_status[_id] = {"SubscriptionStarted": False}
        failure_info = msg.getElement("reason")
        source = failure_info.getElementAsString("source")
        category = failure_info.getElementAsString("category")
        error_code = failure_info.getElementAsInteger("errorCode")
        description = failure_info.getElementAsString("description")
        logging.error(f"Subscription Failed: {_id} {source} {msg} {category} {error_code} {description}")

    @staticmethod
    def handle_subscription_terminated(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.info(f"Subscription Terminated {_id}: {msg}")

    def handle_subscription_stream_activated(self, msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        self.subscription_status[_id] = {"SubscriptionActived": True}
        logging.info(f"Subscription Stream Activated {_id}")

    @staticmethod
    def handle_subscription_stream_deactivated(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.error(f"Subscription Stream Deactivated {_id}: {msg}")

    @staticmethod
    def handle_reference_data_response(msg: blpapi.Message):
        _id = msg.correlationIds()[0].value()
        logging.info(f"Reference Data Response {_id}: {msg}")

    def get_number_of_subs(self):
        return len(self.subscription_status)
