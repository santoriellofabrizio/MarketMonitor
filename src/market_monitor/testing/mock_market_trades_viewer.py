import logging
import sqlite3
import threading
import time
import random
import os
import pickle
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')


class MockMarketTradesViewer(threading.Thread):
    """
    Mock di trade che carica gli strumenti da un file pickle.
    """

    def __init__(self,
                 db_path: str,
                 trades_per_second: float = 2.5,
                 pickle_path: Optional[str] = None,
                 quantity_range: Tuple[int, int] = (100, 10000),
                 market: str = "ETFP",
                 price_initial_range: Tuple[float, float] = (30.0, 120.0),
                 price_step_max: float = 0.5):

        super().__init__(daemon=True)
        self.name = "MockMarketTradesViewer"
        self.running = False
        self.db_path = db_path

        # Percorso predefinito del pickle se non fornito
        if pickle_path is None:
            pickle_path = r'C:\AFMachineLearning\Libraries\MarketMonitor\src\market_monitor\testing\isin_to_ticker.pkl'

        # Caricamento strumenti dal pickle
        self.etf_instruments = self._load_instruments_from_pickle(pickle_path)

        self.trades_per_second = trades_per_second
        self.quantity_range = quantity_range
        self.market = market
        self.price_initial_range = price_initial_range
        self.price_step_max = price_step_max

        self._trade_count = 0
        self._last_update = time.time()
        self._last_prices: Dict[str, float] = {}

        self._initialize_prices()

        logging.info(f"MockMarketTradesViewer initialized: {trades_per_second} trades/sec, "
                     f"{len(self.etf_instruments)} ETFs loaded from pickle.")

    def _load_instruments_from_pickle(self, path: str) -> List[Dict]:
        """Carica il dict isin:ticker e lo trasforma nel formato richiesto."""
        if not os.path.exists(path):
            logging.error(f"Pickle file non trovato in {path}! Caricamento lista vuota.")
            return []

        try:
            with open(path, 'rb') as f:
                isin_map = pickle.load(f)  # Atteso dict {isin: ticker}

            instruments = []
            for isin, ticker in isin_map.items():
                instruments.append({
                    "ticker": ticker,
                    "isin": isin,
                    "name": ticker  # Uso ticker come name come richiesto
                })
            return instruments
        except Exception as e:
            logging.error(f"Errore nel caricamento del pickle: {e}")
            return []

    def _initialize_prices(self):
        """Inizializza il prezzo casuale per ogni ETF."""
        for etf in self.etf_instruments:
            self._last_prices[etf["ticker"]] = random.uniform(
                self.price_initial_range[0], self.price_initial_range[1]
            )

    def run(self):
        """Esegue la simulazione continua di trades in arrivo."""
        logging.info("MockMarketTradesViewer started")
        # Assicuriamoci che la directory del DB esista
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        try:
            while not self.running:
                trades_to_generate = self._calculate_trades_to_generate()

                if trades_to_generate > 0:
                    trades_list = [self._generate_trade_dict() for _ in range(trades_to_generate)]
                    trade_df = pd.DataFrame(trades_list)

                    with sqlite3.connect(self.db_path) as conn:
                        trade_df.to_sql(
                            name="TRADES",
                            con=conn,
                            index=False,
                            if_exists="append"
                        )

                base_sleep_time = 1.0 / self.trades_per_second
                jitter = random.uniform(-0.1 * base_sleep_time, 0.1 * base_sleep_time)
                time.sleep(max(0.001, base_sleep_time + jitter))

        except Exception as e:
            logging.error(f"Error in MockMarketTradesViewer: {e}", exc_info=True)
        finally:
            logging.info("MockMarketTradesViewer stopped")

    def _calculate_trades_to_generate(self) -> int:
        now = time.time()
        time_elapsed = now - self._last_update
        self._last_update = now
        return int(time_elapsed * self.trades_per_second)

    def _generate_trade_dict(self) -> Dict:
        """Genera un singolo trade come dizionario per performance migliori nel loop."""
        etf = random.choice(self.etf_instruments)
        ticker = etf["ticker"]

        last_price = self._last_prices.get(ticker, 100.0)
        step = random.uniform(-self.price_step_max, self.price_step_max)
        new_price = max(0.01, last_price + step)
        self._last_prices[ticker] = new_price

        price = round(new_price, 4)
        quantity = random.randint(self.quantity_range[0], self.quantity_range[1])
        own_trade_val = random.choices([0, 1, -1], weights=[60, 20, 20], k=1)[0]

        self._trade_count += 1

        return {
            "ticker": ticker,
            "isin": etf["isin"],
            "quantity": quantity,
            "last_update": datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f"),
            "price": price,
            "ctv": round(price * quantity, 4),
            "market": self.market,
            "currency": "EUR",
            "own_trade": own_trade_val,
        }

    def stop(self):
        self.running = True

    def get_stats(self) -> Dict:
        return {
            "total_trades_generated": self._trade_count,
            "etf_instruments_count": len(self.etf_instruments)
        }


if __name__ == "__main__":
    # Test rapido
    db_test = "mock_trades.db"
    viewer = MockMarketTradesViewer(db_path=db_test, trades_per_second=2.0)
    viewer.start()
    time.sleep(5)
    viewer.stop()
    viewer.join()
    print(viewer.get_stats())