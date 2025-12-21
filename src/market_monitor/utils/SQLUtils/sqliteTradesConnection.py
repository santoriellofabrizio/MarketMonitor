import logging
import sqlite3
from typing import Tuple
import pandas as pd

# Crea un logger dedicato per i warning, separato da quello globale
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Imposta il livello del logger

# Gestione del warning: il filtro è ora globale per tutte le istanze di questa classe
class SqliteTradesConnection:

    def __init__(self, path: str):
        self.path = path
        self.n_market_trades, self.n_own_trades = 0, 0
        self.conn: sqlite3.Connection = sqlite3.connect(self.path)

        # Messaggio di warning
        self.warning_message = "No OWN_TRADES table found. Please update market trades viewer to the latest one."

        # Se il logger non ha già un handler, aggiungi uno per stampare i messaggi di warning nella console
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Filtro per evitare che lo stesso warning venga loggato più di una volta
        if not hasattr(self, '_warning_logged'):
            self._warning_logged = False

    def _log_warning_once(self):
        if not self._warning_logged:
            logger.warning(self.warning_message)
            self._warning_logged = True

    def get_last_trade(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        conn: sqlite3.Connection = sqlite3.connect(self.path)
        market_trades = pd.read_sql_query(f"""SELECT * FROM TRADES LIMIT -1 OFFSET {self.n_market_trades}""", conn)
        try:
            own_trades = pd.read_sql_query(f"""SELECT * from OWN_TRADES LIMIT -1 OFFSET {self.n_own_trades}""", conn)
        except Exception as e:  # Uso di Exception per catturare qualsiasi tipo di errore
            self._log_warning_once()  # Logga il warning solo la prima volta
            own_trades = pd.DataFrame()  # Assegna un DataFrame vuoto in caso di errore
        finally:
            conn.close()  # Assicurati che la connessione venga sempre chiusa
        return self._elaborate_trade(market_trades, own_trades)

    def get_all_trades(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        conn: sqlite3.Connection = sqlite3.connect(self.path)
        market_trades = pd.read_sql_query(f"""SELECT * FROM TRADES""", conn)
        try:
            own_trades = pd.read_sql_query(f"""SELECT * from OWN_TRADES""", conn)
        except Exception as e:  # Uso di Exception per catturare qualsiasi tipo di errore
            self._log_warning_once()  # Logga il warning solo la prima volta
            own_trades = pd.DataFrame()  # Assegna un DataFrame vuoto in caso di errore
        finally:
            conn.close()  # Assicurati che la connessione venga sempre chiusa
        return self._elaborate_trade(market_trades, own_trades)

    def _elaborate_trade(self, market_trades: pd.DataFrame,
                         own_trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.n_market_trades += len(market_trades)
        self.n_own_trades += len(own_trades)

        # Gestisci "last_update" solo se own_trades non è vuoto
        market_trades["last_update"] = pd.to_datetime(market_trades["last_update"], dayfirst=True)
        if not own_trades.empty:  # Controlla che own_trades non sia vuoto
            own_trades["last_update"] = pd.to_datetime(own_trades["last_update"], dayfirst=True)
            own_trades = own_trades.rename(columns={"side": "own_trade"})

        return market_trades, own_trades
