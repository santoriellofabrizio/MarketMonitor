import logging
import os
import time  # Importa il modulo time per misurare il tempo
from configparser import ConfigParser
from datetime import datetime
from typing import Optional

import aiosqlite
import pandas as pd


class SqliteTradesConnectionAsync:

    def __init__(self, config: ConfigParser, **kwargs):
        self.path = os.path.join(config.get("PARAMETERS", "path"), "markettrades.db")
        self.num_trades = 0
        self.df = None
        self.anagraphic = pd.read_excel(kwargs.get("path_anagraphic", "Anagraphic/Anagrafica.xlsx"), index_col=0)
        self.isin_to_cluster = lambda x: self.anagraphic["Cluster"].to_dict().get(x["isin"], "none")
        self.isin_to_region = lambda x: self.anagraphic["Region"].to_dict().get(x["isin"], "none")

        # Configurazione del logger
        self.logger = logging.getLogger()

    async def get_last_trade(self) -> Optional[pd.DataFrame]:
        start_time = datetime.now()  # Misura il tempo di inizio dell'operazione
        async with aiosqlite.connect(self.path) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT trades.isin, ticker, last_update, price, quantity FROM trades LIMIT -1 OFFSET ?",
                    (self.num_trades,))
                rows = await cur.fetchall()
                columns = [col[0] for col in cur.description]
                new_trades = pd.DataFrame(rows, columns=columns)
            if new_trades.empty: return
            self.num_trades += len(new_trades)
            new_trades["ctv"] = new_trades["price"] * new_trades["quantity"]
            new_trades["last_update"] = pd.to_datetime(new_trades["last_update"], dayfirst=True)
            new_trades["side"] = "no_side"
            new_trades["cluster"] = "none" if new_trades.empty else new_trades.apply(self.isin_to_cluster, axis=1)
            new_trades["region"] = "none" if new_trades.empty else new_trades.apply(self.isin_to_region, axis=1)

        elapsed_time = (datetime.now() - start_time).seconds  # Calcola il tempo trascorso
        lag_time = (new_trades["last_update"].max() - start_time).seconds
        if self.logger is not None: self.logger.info(
            f'Retrieved last trade data successfully in {elapsed_time:.2f} seconds. lag: {lag_time:.4f}')
        return new_trades

    async def get_all_trades(self) -> pd.DataFrame:
        start_time = time.time()  # Misura il tempo di inizio dell'operazione
        async with aiosqlite.connect(self.path) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT isin, ticker, last_update, price, quantity FROM trades")
                rows = await cur.fetchall()
                columns = [col[0] for col in cur.description]
                trades = pd.DataFrame(rows, columns=columns)
            trades["last_update"] = pd.to_datetime(trades["last_update"], dayfirst=True)
            trades["ctv"] = trades["price"] * trades["quantity"]
            trades["side"] = "no_side"
            trades["cluster"] = trades.apply(self.isin_to_cluster, axis=1)
            trades["region"] = trades.apply(self.isin_to_region, axis=1)
            self.num_trades = len(trades)

        elapsed_time = time.time() - start_time  # Calcola il tempo trascorso
        if self.logger is not None:
            self.logger.info(f'Retrieved {self.num_trades} trades successfully in {elapsed_time:.2f} seconds.')

        return trades
