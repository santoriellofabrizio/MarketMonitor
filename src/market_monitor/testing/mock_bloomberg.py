import logging
import threading
import time
from typing import Dict
import random

from market_monitor.live_data_hub.real_time_data_hub import RTData

logging.getLogger()


class MockBloombergStreamingThread(threading.Thread):
    """
    Mock di BloombergStreamingThread che simula dati di mercato realistici.

    Funziona esattamente come BloombergStreamingThread ma genera dati simulati
    invece di connettersi all'API Bloomberg reale.

    Attributes:
        event_handler: Handler per processare gli eventi simulati
        real_time_data (RTData): Oggetto condiviso per gestire le subscription
        update_interval (float): Intervallo tra aggiornamenti in secondi
        price_volatility (float): Volatilità dei prezzi (spread casuale)
        running (bool): Flag per controllare il loop principale
    """

    def __init__(self, event_processor, **kwargs):
        """
        Inizializza il mock di bloomberg.

        Args:
            event_processor (BBGEventHandler): Handler che processa gli aggiornamenti
            kwargs: Parametri aggiuntivi (update_interval, price_volatility)
        """
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self.running = False
        self.terminate = False
        self.daemon = True
        self.name = "MockBloomberg"

        self.event_handler = event_processor
        self.real_time_data: RTData = event_processor.real_time_data

        # Parametri per la simulazione
        self.update_interval = kwargs.get("update_interval", 0.5)
        self.price_volatility = kwargs.get("price_volatility", 0.001)

        # Stato dei prezzi per ogni security
        self._price_state: Dict[str, Dict[str, float]] = {}

        logging.info("MockBloombergStreamingThread initialized")

    def run(self):
        """Esegue la simulazione continua di aggiornamenti di mercato."""
        pending = self.real_time_data.get_pending_subscriptions("bloomberg")
        self.active = self.real_time_data.get_bloomberg_subscription() or {}

        if not pending and not self.active:
            logging.error("No subscriptions required in Mock Bloomberg, shutting thread down")
            return

        # Inizializza i prezzi per le subscription pendenti
        self._subscribe_pending()
        self.running = True

        logging.info("MockBloombergStreamingThread started")

        try:
            while self.running:
                if self._stop_event.wait(timeout=self.update_interval):
                    break

                # Gestisci nuove subscription
                self._subscribe_pending()

                # Gestisci unsubscribe
                self._process_unsubscribe()

                # Aggiorna i prezzi per le subscription attive
                self._update_all_prices()

        except Exception as e:
            logging.error(f"Error in MockBloombergStreamingThread: {e}")
        finally:
            logging.info("MockBloombergStreamingThread stopped")

    def stop(self):
        """Stop the thread gracefully"""
        self.running = False
        self._stop_event.set()

    def _subscribe_pending(self):
        """Subscribe to all pending subscriptions"""
        from market_monitor.live_data_hub.real_time_data_hub import BloombergSubscription

        pending = self.real_time_data.get_pending_subscriptions(source="bloomberg")

        if not pending:
            return

        logging.info(f"Processing {len(pending)} pending subscriptions (MOCK)")

        for id, sub in pending.items():
            if isinstance(sub, BloombergSubscription):
                # Inizializza i prezzi per questa security
                self._initialize_price_for_security(id, sub.subscription_string)
                self.real_time_data.mark_subscription_received(id, "bloomberg")

                # Simula una subscription riuscita
                # (nella classe reale questo viene fatto dall'event handler quando arriva SUBSCRIPTION_STATUS)
                logging.debug(f"Mock subscription added: {id} ({sub.subscription_string})")

        logging.info(f"Mock subscribed to {len(pending)} pending subscriptions")

    def _process_unsubscribe(self):
        """Process subscriptions marked for removal"""
        from market_monitor.live_data_hub.real_time_data_hub import BloombergSubscription

        to_unsub = self.real_time_data.get_to_unsubscribe(source="bloomberg")

        if not to_unsub:
            return

        logging.info(f"Processing {len(to_unsub)} unsubscriptions (MOCK)")

        for id, sub in to_unsub.items():
            if isinstance(sub, BloombergSubscription):
                # Rimuovi lo stato del prezzo
                if id in self._price_state:
                    del self._price_state[id]

                # Clear from unsubscribe queue
                self.real_time_data.clear_unsubscribed(id, "bloomberg")
                logging.debug(f"Mock unsubscribed: {id} ({sub.subscription_string})")

        logging.info(f"Mock unsubscribed from {len(to_unsub)} subscriptions")

    def _initialize_price_for_security(self, isin: str, subscription_str: str):
        """Inizializza prezzi di partenza realistici per un singolo strumento."""
        if isin in self._price_state:
            return  # Già inizializzato

        # Genera un prezzo base pseudo-casuale ma deterministico basato sull'ISIN
        # In questo modo lo stesso ISIN avrà sempre lo stesso prezzo iniziale
        random.seed(hash(isin) % 2 ** 32)
        base_price = 50.0 + random.uniform(-30, 50)

        # Reset del seed per mantenere randomness nelle altre operazioni
        random.seed()

        self._price_state[isin] = {
            "mid": base_price,
            "spread": base_price * 0.0002,  # 2 bps default spread
            "trend": random.uniform(-0.0005, 0.0005),  # Piccolo trend casuale
        }

        logging.debug(f"Initialized mock price for {isin}: {base_price:.2f}")

    def _update_all_prices(self):
        """Aggiorna i prezzi di tutti i securities sottoscritti attivi."""

        current_time = time.time()

        for sub in self.real_time_data.get_active_subscriptions("bloomberg"):
            id = sub.id
            if sub.id not in self._price_state:
                continue

            state = self._price_state[id]

            # Random walk del mid price
            noise = random.gauss(0, self.price_volatility * state["mid"])
            trend = state["trend"] * state["mid"]
            state["mid"] += trend + noise

            # Spread dinamico (aumenta in mercati volatili)
            volatility_factor = 1.0 + abs(noise) * 100
            state["spread"] = state["mid"] * 0.0002 * volatility_factor

            # Calcola BID/ASK/MID in base ai fields richiesti
            mid = state["mid"]
            spread = state["spread"]
            bid = mid - spread / 2
            ask = mid + spread / 2

            # Prepara i dati in base ai fields della subscription
            data_dict = {}
            if "BID" in sub.fields:
                data_dict["BID"] = bid
            if "ASK" in sub.fields:
                data_dict["ASK"] = ask
            if "MID" in sub.fields:
                data_dict["MID"] = mid

            # Aggiorna tramite RTData
            try:
                self.real_time_data.update(id, data_dict)
            except Exception as e:
                logging.warning(f"Error updating mock price for {id}: {e}")


if __name__ == "__main__":
    # Test del mock
    fields = ["BID", "ASK", "MID"]
    lock = threading.Lock()
    book = RTData(locker=lock, fields=fields)

    # Subscribe a due strumenti
    book.subscribe_bloomberg("IHYG", "IHYG IM EQUITY", fields=["MID"])
    book.subscribe_bloomberg("IHYU", "IHYU IM EQUITY", fields=["BID", "ASK"])


    # Usa un mock event handler minimale
    class MockEventHandler:
        def __init__(self, rtdata):
            self.real_time_data = rtdata

        def update(self, id, data):

            self.real_time_data.update(id, data)


    event_handler = MockEventHandler(book)
    mock_thread = MockBloombergStreamingThread(
        event_handler,
    )
    mock_thread.start()

    try:
        while True:
            time.sleep(2)
            data = book.get_data_field()
            print(f"\nCurrent data: {data}")
    except KeyboardInterrupt:
        mock_thread.stop()
        print("\nStopped")