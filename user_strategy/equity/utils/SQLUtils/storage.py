import sqlite3
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PriceDatabaseManager:
    def __init__(self, db_path: str, table_name: str = "price_history"):
        self.db_path = db_path
        self.table_name = table_name
        self._conn = None  # Connessione persistente
        self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")  # Write-Ahead Logging
        self._conn.execute("PRAGMA synchronous=NORMAL;")  # Bilancio sicurezza/velocità
        self._conn.execute("PRAGMA cache_size=-64000;")  # 64MB cache

        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                TIME_STAMP TEXT,
                ISIN TEXT,
                VALUE_NAME TEXT,
                VALUE REAL
            );
        """)
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_ts ON {self.table_name} (TIME_STAMP);")
        self._conn.commit()

    def store_data(self, data_to_store: dict):
        if not data_to_store:
            return

        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        try:
            # Costruzione diretta della lista di tuple (evita pd.concat)
            rows = []
            for name, series in data_to_store.items():
                if series is not None and len(series) > 0:
                    for isin, value in series.items():
                        rows.append((ts, isin, name, value))

            if not rows:
                return

            # executemany è più veloce di to_sql per inserimenti bulk
            self._conn.executemany(
                f"INSERT INTO {self.table_name} (TIME_STAMP, ISIN, VALUE_NAME, VALUE) VALUES (?, ?, ?, ?)",
                rows
            )
            self._conn.commit()

        except Exception as e:
            logger.error(f"Errore durante lo storage dei dati su DB: {e}")

    def close(self):
        if self._conn:
            self._conn.close()