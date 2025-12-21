import logging
import sqlite3
import threading
import time
from queue import Queue
from typing import Dict, List, Optional, Tuple
import random
from datetime import datetime
import pandas as pd
import os

# --- ASSUNZIONI: Rimuovi o Adatta se non necessario nel tuo ambiente ---
# Se TradeType non è strettamente necessario per il mock, lo commento
# from market_monitor_fi_fi.utils.enums import TradeType
# ----------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')


class MockMarketTradesViewer(threading.Thread):
    """
    Mock di trade che simula trades in arrivo con frequenza configurabile.

    Genera trades con la struttura:
    - ticker: str
    - isin: str
    - quantity: int
    - last_update: datetime
    - price: float
    - ctv: float (price * quantity)
    - market: Optional[str]
    - currency: Optional[str]
    - own_trade: int | None

    La simulazione include:
    1. Prezzi con fluttuazione realistica (Random Walk).
    2. Frequenza di trade irregolare (Jitter).
    3. Scrittura diretta nel DB SQLite.
    """

    # Strumenti ETF FI di default (circa 20)
    DEFAULT_ETFS = [
        {"ticker": "XESC", "isin": "DE0007667107", "name": "Allianz Global Investors"},
        {"ticker": "VEUR", "isin": "LU0048584102", "name": "Vanguard FTSE Developed Europe"},
        {"ticker": "CSMM", "isin": "LU0072462426", "name": "iShares MSCI Emerging Markets"},
        {"ticker": "AMND", "isin": "IE0002271879", "name": "Amundi CAC 40"},
        {"ticker": "LYXE", "isin": "LU0073263215", "name": "Lyxor Core MSCI World"},
        {"ticker": "XTRC", "isin": "IE0003400068", "name": "Xtrackers MSCI World"},
        {"ticker": "UBSE", "isin": "LU0240957833", "name": "UBS MSCI Europe"},
        {"ticker": "IUNA", "isin": "IE0009470246", "name": "iShares Global Clean Energy"},
        {"ticker": "IHYG", "isin": "LU0274211480", "name": "iShares High Yield Corporate Bond"},
        {"ticker": "CSIB", "isin": "IE0032174605", "name": "iShares Global Corp Bond"},
        {"ticker": "LUAG", "isin": "LU0496736636", "name": "Lyxor Core MSCI Emerging Markets"},
        {"ticker": "DBXD", "isin": "DE0005327218", "name": "iShares Core German Equities"},
        {"ticker": "IBCI", "isin": "IE0005042670", "name": "iShares Corporate Bond EUR"},
        {"ticker": "LUSB", "isin": "LU0055732411", "name": "Lyxor US Treasury Bond"},
        {"ticker": "IEMB", "isin": "IE0009505545", "name": "iShares Emerging Markets Corp Bond"},
        {"ticker": "LUEU", "isin": "LU0388261046", "name": "Lyxor MSCI Europe"},
        {"ticker": "IBTE", "isin": "IE0007266328", "name": "iShares Global Corp Bond All Maturities"},
        {"ticker": "VEEM", "isin": "LU0048584419", "name": "Vanguard FTSE Emerging Markets"},
        {"ticker": "XMUE", "isin": "DE0005933956", "name": "iShares MSCI World ESG Select"},
        {"ticker": "APAC", "isin": "IE0008830286", "name": "iShares Asia Pacific Dividend ETF"},
    ]

    def __init__(self,
                 db_path: str = "mock_trades.db",
                 trades_per_second: float = 2.5,
                 etf_instruments: Optional[List[Dict]] = None,
                 quantity_range: Tuple[int, int] = (100, 10000),
                 market: str = "ETFP",
                 price_initial_range: Tuple[float, float] = (30.0, 120.0),
                 price_step_max: float = 0.5):
        """
        Inizializza il mock di trade.

        Args:
            db_path: Percorso del file SQLite dove salvare i trades.
            trades_per_second: Frequenza approssimativa di trades al secondo.
            etf_instruments: Lista custom di ETF. Se None, usa DEFAULT_ETFS.
            quantity_range: Range di quantità per ogni trade (min, max).
            market: Mercato di default (ETFPLUS).
            price_initial_range: Range iniziale per la generazione del prezzo.
            price_step_max: Variazione massima del prezzo tra un trade e l'altro.
        """
        super().__init__(daemon=True)
        self.name = "MockMarketTradesViewer"
        self._stop = False

        # Connessione DB
        self.db_path = db_path

        # Parametri di simulazione
        self.trades_per_second = trades_per_second
        self.etf_instruments = etf_instruments or self.DEFAULT_ETFS
        self.quantity_range = quantity_range
        self.market = market

        # Stato interno per la simulazione del flusso
        self._trade_count = 0
        self._last_update = time.time()
        self.price_initial_range = price_initial_range
        self.price_step_max = price_step_max
        # Dizionario per tenere traccia dell'ultimo prezzo di ogni ticker
        self._last_prices: Dict[str, float] = {}

        self._initialize_prices()

        logging.info(f"MockMarketTradesViewer initialized: {trades_per_second} trades/sec, "
                     f"{len(self.etf_instruments)} ETFs. DB path: {os.path.abspath(self.db_path)}")

    def _initialize_prices(self):
        """Inizializza il prezzo casuale per ogni ETF."""
        for etf in self.etf_instruments:
            # Prezzo iniziale casuale nel range specificato
            self._last_prices[etf["ticker"]] = random.uniform(
                self.price_initial_range[0], self.price_initial_range[1]
            )

    def run(self):
        """Esegue la simulazione continua di trades in arrivo."""
        logging.info("MockMarketTradesViewer started")

        try:
            while not self._stop:
                # 1. Calcola quanti trade generare
                trades_to_generate = self._calculate_trades_to_generate()

                # 2. Genera e scrivi i trades
                for _ in range(trades_to_generate):
                    trade_df = self._generate_trade()

                    # Connessione e scrittura nel DB (crea il file e la tabella se non esistono)
                    with sqlite3.connect(self.db_path) as conn:
                        trade_df.to_sql(
                            name="TRADES",
                            con=conn,
                            index=False,
                            if_exists="append"
                        )

                # 3. Tempo di attesa con Jitter per irregolarità
                base_sleep_time = 1.0 / self.trades_per_second
                # Jitter (variazione casuale) fino al 10% del tempo base
                jitter = random.uniform(-0.1 * base_sleep_time, 0.1 * base_sleep_time)

                sleep_time = max(0.001, base_sleep_time + jitter)  # Assicura un tempo positivo minimo

                time.sleep(sleep_time)

        except Exception as e:
            logging.error(f"Error in MockMarketTradesViewer: {e}", exc_info=True)
        finally:
            logging.info("MockMarketTradesViewer stopped")

    def _calculate_trades_to_generate(self) -> int:
        """
        Calcola quanti trades generare in questo ciclo in base al tempo trascorso.
        """
        now = time.time()
        time_elapsed = now - self._last_update
        self._last_update = now

        expected_trades = time_elapsed * self.trades_per_second
        trades_to_generate = int(expected_trades)

        return max(0, trades_to_generate)

    def _generate_trade(self) -> pd.DataFrame:
        """
        Genera un singolo trade casuale con la struttura corretta.
        Applica il Random Walk per il prezzo.
        """
        etf = random.choice(self.etf_instruments)
        ticker = etf["ticker"]

        # 1. GENERAZIONE DEL PREZZO (Random Walk)
        last_price = self._last_prices.get(ticker)
        if last_price is None:
            last_price = random.uniform(self.price_initial_range[0], self.price_initial_range[1])

        # Calcola la variazione casuale (passo)
        step = random.uniform(-self.price_step_max, self.price_step_max)

        new_price = max(0.01, last_price + step)
        self._last_prices[ticker] = new_price
        price = round(new_price, 4)

        # 2. ALTRI PARAMETRI
        quantity = random.randint(self.quantity_range[0], self.quantity_range[1])

        # Alternanza casuale (60% trade standard, 20% buy proprietario, 20% sell proprietario)
        # 1 = buy (bid), -1 = sell (ask), 0 = no own trade (market)
        own_trade_val = random.choices([0, 1, -1], weights=[60, 20, 20], k=1)[0]

        trade_data = {
            "ticker": [ticker],
            "isin": [etf["isin"]],
            "quantity": [quantity],
            "last_update": [datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")],
            "price": [price],
            "ctv": [round(price * quantity, 4)],
            "market": [self.market],
            "currency": ["EUR"],
            "own_trade": [own_trade_val],  # Usa il valore numerico
        }

        self._trade_count += 1

        df = pd.DataFrame(trade_data)
        return df

    def stop(self):
        """Segnala la terminazione del thread."""
        self._stop = True
        logging.info("MockMarketTradesViewer stop signal received")

    def get_stats(self) -> Dict:
        """Ritorna statistiche sulla simulazione."""
        return {
            "total_trades_generated": self._trade_count,
            "trades_per_second": self.trades_per_second,
            "etf_instruments_count": len(self.etf_instruments),
            "current_prices_sample": dict(list(self._last_prices.items())[:3])  # Mostra solo i primi 3
        }


if __name__ == "__main__":
    # --- Esempio di utilizzo e test ---

    print("Avvio del test di MockMarketTradesViewer...")

    # 1. Inizializza il simulatore (creerà mock_trades.db nella CWD)
    viewer = MockMarketTradesViewer(trades_per_second=5.0)

    # 2. Avvia il thread
    viewer.start()

    print("Simulatore avviato. Generazione di trade per 10 secondi...")

    # 3. Lascia che la simulazione giri per un po'
    time.sleep(10)

    # 4. Ferma il thread
    viewer.stop()
    viewer.join()  # Attende che il thread termini

    # 5. Stampa statistiche
    print("\n--- STATISTICHE SIMULAZIONE ---")
    print(viewer.get_stats())

    # 6. Verifica il contenuto del DB
    print("\n--- VERIFICA DB ---")
    try:
        with sqlite3.connect("mock_trades.db") as conn:
            df_trades = pd.read_sql_query("SELECT * FROM TRADES LIMIT 5", conn)
            print(f"File DB generato: {os.path.abspath('mock_trades.db')}")
            print(
                f"Righe totali nel DB (approssimazione): {pd.read_sql_query('SELECT COUNT(*) FROM TRADES', conn).iloc[0, 0]}")
            print("\nEsempio di trades salvati:")
            print(df_trades)

    except Exception as e:
        print(f"Errore durante la lettura del DB: {e}")