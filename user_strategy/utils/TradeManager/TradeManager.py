import datetime
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from user_strategy.utils.TradeManager.TimeZeroPLManager import TimeZeroPLManager
from user_strategy.utils.TradeManager.TradeClassTemplate import TradeFactory, TradeStorage, \
    MyTrade, Trade

logger = logging.getLogger(__name__)


class TradeManager:
    """
    Trade manager con persistenza append-only e cache intelligente.
    """

    # Configurabile via kwargs
    DEFAULT_MAX_TIME_TO_MATCH_SECONDS = 10.0
    DEFAULT_AUTO_SAVE_INTERVAL = 500

    def __init__(self, book_storage, model_prices: pd.Series | None = None,
                 time_zero_lag: float | None = 10.,
                 trade_folder: Path | None = None,
                 **kwargs):

        self.trade_factory = TradeFactory()
        self.trade_storage = TradeStorage()

        self._my_trades_index: list = []
        self.model_price = model_prices
        self.book_storage = book_storage

        # Config
        self.max_time_to_match = kwargs.get('max_time_to_match_side', self.DEFAULT_MAX_TIME_TO_MATCH_SECONDS)

        # Persistenza append-only (path configurabile)
        self.enable_persistence = kwargs.get("enable_persistence", True)
        self.auto_save_interval = kwargs.get("auto_save_interval", self.DEFAULT_AUTO_SAVE_INTERVAL)
        self.trade_folder = trade_folder or Path.cwd() / "data" / "trades"

        self._trades_since_last_save = 0
        self._last_saved_index = 0

        # ✅ Cache DataFrame (invalida su update)
        self._trades_df_cache: pd.DataFrame | None = None
        self._cache_valid = False

        # Time zero PL manager
        self.time_zero_pl_manager = TimeZeroPLManager(
            model_price=model_prices,
            mid_price_storage=book_storage,
            time_zero_lag=time_zero_lag,
            trade_storage=self.trade_storage
        )
        if time_zero_lag is not None:
            self.time_zero_pl_manager.start()

        # Load trades del giorno
        if self.enable_persistence:
            self._load_today_trades()

    # ========================================================================
    # CORE LOGIC
    # ========================================================================

    def on_trade(self, new_trades: pd.DataFrame):
        """
        Processa nuovi trade.

        Args:
            new_trades: DataFrame con nuovi trade da processare

        Returns:
            DataFrame con trade processati
        """
        # Validation input
        if new_trades is None or new_trades.empty:
            logger.debug("No new trades to process")
            return pd.DataFrame()

        processed = []

        for trade in self.trade_factory.build_trades_obj(new_trades):
            # Match side se recente
            if trade.time_since_trade() < self.max_time_to_match:
                if trade.is_my_trade():
                    self._my_trades_index.append(trade.trade_index)
                else:
                    self.match_side(trade)

                # Calcola spread PL
                mid = self._get_book_mid(trade.isin, old=True)
                if mid is not None:
                    trade.spread_pl = self.time_zero_pl_manager.calculate_time_zero_pl(trade, mid)

                if self.model_price is not None:
                    model = self.model_price.get(trade.isin)
                    if model is not None:
                        trade.spread_pl_model = self.time_zero_pl_manager.calculate_time_zero_pl(trade, model)

                trade.is_elaborated = True

            # Store trade
            self.trade_storage.add_trade(trade)
            processed.append(trade)

        # Invalida cache
        self._invalidate_cache()

        # Auto-save (append-only)
        if self.enable_persistence:
            self._trades_since_last_save += len(processed)
            if self._trades_since_last_save >= self.auto_save_interval:
                try:
                    self._append_new_trades()
                    self._trades_since_last_save = 0
                except Exception as e:
                    logger.error(f"Auto-save failed, will retry on next interval: {e}")

        # Ritorna solo nuovi trade processati (no conversione completa)
        return self._convert_trades_obj_to_df(processed)

    def match_side(self, trade):
        """Determina side (bid/ask) confrontando con mid."""
        mid = self._get_book_mid(trade.isin, old=True)
        if mid is not None:
            trade.side = 'bid' if trade.price < mid else 'ask'

    # ========================================================================
    # BOOK ACCESS
    # ========================================================================

    def _get_book_mid(self, isin: str, old: bool = False) -> float | None:
        """
        Get mid price dal book storage.

        Args:
            isin: Security identifier
            old: If True, get oldest book (index 0), else newest (index -1)

        Returns:
            Mid price o None se non trovato
        """
        try:
            if not self.book_storage:
                logger.debug("Book storage is empty")
                return None

            index = 0 if old else -1
            time_snip, book_value = self.book_storage[index]

            age = (datetime.datetime.now() - time_snip).total_seconds()
            logger.debug(f"Book age: {age:.1f}s ({'old' if old else 'new'})")

            mid_price = book_value.get(isin)

            if mid_price is None:
                logger.debug(f"ISIN {isin} not found in book")

            return mid_price

        except IndexError:
            logger.debug(f"Book storage index {index} not available")
            return None
        except (KeyError, TypeError) as e:
            logger.warning(f"Unexpected error getting book for {isin}: {e}")
            return None

    # ========================================================================
    # QUERY METHODS (con cache)
    # ========================================================================

    def get_trades(self, n_of_trades: int | None = None, use_cache: bool = True) -> pd.DataFrame:
        """
        Get trades come DataFrame.

        ✅ Usa cache se possibile (molto più veloce per query ripetute).

        Args:
            n_of_trades: Numero di trade da recuperare (None = tutti)
            use_cache: Se True usa la cache quando disponibile

        Returns:
            DataFrame con trade richiesti
        """
        # Se richiesti tutti i trade e cache valida
        if n_of_trades is None and use_cache and self._cache_valid and self._trades_df_cache is not None:
            return self._trades_df_cache.copy()

        # Altrimenti rigenera
        trades_obj_list = self.trade_storage.get_last_trades(n_of_trades)
        trades_df = self._convert_trades_obj_to_df(trades_obj_list)

        # Aggiorna cache se query completa
        if n_of_trades is None:
            self._trades_df_cache = trades_df.copy()
            self._cache_valid = True

        return trades_df.sort_values('timestamp', ascending=False) if not trades_df.empty else trades_df

    def get_my_trades(self, n_of_trades: int | None = None) -> pd.DataFrame:
        """
        Get solo my trades.

        Args:
            n_of_trades: Numero massimo di trade da recuperare

        Returns:
            DataFrame con solo trade propri
        """
        all_trades = self.get_trades(n_of_trades, use_cache=True)
        if all_trades.empty:
            return pd.DataFrame()
        return self._filter_and_sort(all_trades, 'own_trade', [True])

    def get_trades_from_isin(self, isins: list[str], n_of_trades: int | None = None) -> pd.DataFrame:
        """
        Get trades per ISIN list.

        Args:
            isins: Lista di ISIN da filtrare
            n_of_trades: Numero massimo di trade da recuperare

        Returns:
            DataFrame con trade filtrati per ISIN
        """
        if not isins:
            return pd.DataFrame()
        all_trades = self.get_trades(n_of_trades, use_cache=True)
        return self._filter_by_column(all_trades, 'isin', isins)

    def get_trades_from_ticker(self, tickers: list[str], n_of_trades: int | None = None) -> pd.DataFrame:
        """
        Get trades per ticker list.

        Args:
            tickers: Lista di ticker da filtrare
            n_of_trades: Numero massimo di trade da recuperare

        Returns:
            DataFrame con trade filtrati per ticker
        """
        if not tickers:
            return pd.DataFrame()
        all_trades = self.get_trades(n_of_trades, use_cache=True)
        return self._filter_by_column(all_trades, 'ticker', tickers)

    def get_filtered_trades(self,
                            n: Optional[int] = None,
                            min_ctv: float | None = None,
                            columns: list[str] | None = None,
                            only_my_trades: Optional[bool] = None,
                            start_time: Optional[datetime.datetime] = None) -> pd.DataFrame:
        """
        Get trades filtrati per output (gui, export, etc).

        Args:
            n: Numero massimo di trade da ritornare
            min_ctv: Valore minimo di CTV per filtrare (None = usa default)
            columns: Lista di colonne da includere (None = tutte)
            only_my_trades: show only my trades

        Returns:
            DataFrame con trade filtrati
        """

        # Get all trades (cached)
        trades = self.get_trades()

        if trades.empty:
            return pd.DataFrame()

        if min_ctv:
            trades = trades[trades['ctv'] > min_ctv]
        # Filtra per CTV e my_trades
        if only_my_trades:
            trades = trades[(trades['own_trade'] == True)]


        # Seleziona colonne
        if columns:
            available_cols = trades.columns.intersection(columns)
            filtered = trades[available_cols]

        if start_time:
            trades = trades[trades['timestamp'] >= start_time]

        # Sort e limit
        if n:
            return trades.sort_values('timestamp', ascending=False).head(n)
        return trades.sort_values('timestamp', ascending=False)

    # ========================================================================
    # PERSISTENZA APPEND-ONLY
    # ========================================================================

    def _get_today_filename(self) -> Path:
        """
        Path file per trade di oggi.

        Returns:
            Path completo del file parquet per i trade odierni
        """
        today = datetime.datetime.now().strftime("%Y%m%d")
        self.trade_folder.mkdir(parents=True, exist_ok=True)
        return self.trade_folder / f"trades_{today}.parquet"

    def _append_new_trades(self):
        """
        Append solo nuovi trade al file.
        Ottimizzato per evitare letture complete quando possibile.
        """
        filepath = self._get_today_filename()

        try:
            # Get solo trade non ancora salvati
            all_trades = self.trade_storage.get_last_trades()
            new_trades = all_trades[self._last_saved_index:]

            if not new_trades:
                logger.debug("No new trades to append")
                return

            # Converti a DataFrame
            new_df = self._convert_trades_obj_to_df(new_trades)

            # Append al file esistente
            if filepath.exists():
                # Leggi esistente e concatena
                existing_df = pd.read_parquet(filepath, engine='pyarrow')
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df

            # Salva
            combined_df.to_parquet(filepath, engine='pyarrow', compression='snappy')

            # Aggiorna indice
            self._last_saved_index = len(all_trades)

            logger.info(f"Appended {len(new_trades)} trades to {filepath}")

        except Exception as e:
            logger.error(f"Error appending trades: {e}", exc_info=True)
            raise

    def save_trades(self, filepath: Path | str | None = None):
        """
        Salva tutti i trade (per chiusura giornata).
        Usa cache se disponibile.

        Args:
            filepath: Path custom per salvare (None = usa path default giornaliero)
        """
        if filepath is None:
            filepath = self._get_today_filename()
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Usa cache se valida
            trades_df = self.get_trades(use_cache=True)

            if trades_df.empty:
                logger.info("No trades to save")
                return

            trades_df.to_parquet(filepath, engine='pyarrow', compression='snappy')
            self._last_saved_index = len(self.trade_storage.get_last_trades())

            logger.info(f"Saved {len(trades_df)} trades to {filepath}")

        except Exception as e:
            logger.error(f"Error saving trades to {filepath}: {e}", exc_info=True)
            raise

    def _load_today_trades(self):
        """
        Load trade del giorno con validazione migliorata.
        """
        filepath = self._get_today_filename()

        if not filepath.exists():
            logger.info(f"No previous trades found for today at {filepath}")
            return

        try:
            trades_df = pd.read_parquet(filepath, engine='pyarrow')

            if trades_df.empty:
                logger.info("Trade file exists but is empty")
                return

            # Filtra solo trade elaborati
            elaborated_trades = trades_df[trades_df.get('is_elaborated', False) == True]

            if elaborated_trades.empty:
                logger.info("No elaborated trades to load")
                return

            loaded_count = 0
            for _, row in elaborated_trades.iterrows():
                trade = self._reconstruct_trade_from_row(row)

                if trade:
                    # Add to storage (thread-safe)
                    with self.trade_storage.lock:
                        self.trade_storage._storage.append(trade)
                        if trade.is_my_trade():
                            self._my_trades_index.append(trade.trade_index)

                    loaded_count += 1

            self._last_saved_index = loaded_count
            self._invalidate_cache()  # Invalida cache dopo caricamento

            logger.info(f"Loaded {loaded_count} trades from {filepath}")

        except Exception as e:
            logger.error(f"Error loading trades from {filepath}: {e}", exc_info=True)

    @staticmethod
    def _reconstruct_trade_from_row(row: pd.Series) -> Trade | MyTrade | None:
        """
        Helper per ricostruire Trade object da DataFrame row.

        Args:
            row: Serie pandas rappresentante un trade

        Returns:
            Trade object ricostruito o None se errore
        """
        try:
            trade_dict = row.to_dict()

            # Determina tipo trade
            TradeClass = MyTrade if trade_dict.get('own_trade', False) else Trade

            # Common params
            params = {
                'ticker': trade_dict['ticker'],
                'isin': trade_dict['isin'],
                'timestamp': pd.to_datetime(trade_dict['timestamp']),
                'quantity': trade_dict['quantity'],
                'price': trade_dict['price'],
                'market': trade_dict.get('market'),
                'currency': trade_dict.get('currency'),
                'price_multiplier': trade_dict.get('price_multiplier', 1)
            }

            # MyTrade ha side nel constructor
            if TradeClass == MyTrade:
                params['side'] = trade_dict.get('side')

            trade = TradeClass(**params)

            # Ripristina valori calcolati
            if TradeClass == Trade:
                trade.side = trade_dict.get('side')
            trade.spread_pl = trade_dict.get('spread_pl')
            trade.spread_pl_model = trade_dict.get('spread_pl_model')
            trade.is_elaborated = True

            return trade

        except Exception as e:
            logger.error(f"Error reconstructing trade: {e}")
            return None

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _invalidate_cache(self):
        """Invalida la cache del DataFrame."""
        self._cache_valid = False
        self._trades_df_cache = None

    def _filter_by_column(self, trades_df: pd.DataFrame, column: str, values: list) -> pd.DataFrame:
        """
        Helper per filtrare DataFrame per colonna.

        Args:
            trades_df: DataFrame da filtrare
            column: Nome della colonna
            values: Lista di valori da cercare

        Returns:
            DataFrame filtrato e ordinato
        """
        if trades_df.empty:
            return pd.DataFrame()

        if column not in trades_df.columns:
            logger.warning(f"Column '{column}' not found in trades DataFrame")
            return pd.DataFrame()

        filtered = trades_df[trades_df[column].isin(values)]
        return filtered.sort_values('timestamp', ascending=False) if not filtered.empty else pd.DataFrame()

    def _filter_and_sort(self, trades_df: pd.DataFrame, filter_col: str,
                         filter_values: list) -> pd.DataFrame:
        """
        Helper generico per filtrare e ordinare trade.

        Args:
            trades_df: DataFrame da filtrare
            filter_col: Colonna su cui filtrare
            filter_values: Valori da cercare

        Returns:
            DataFrame filtrato e ordinato per timestamp
        """
        return self._filter_by_column(trades_df, filter_col, filter_values)

    @staticmethod
    def _convert_trades_obj_to_df(trades: list) -> pd.DataFrame:
        """
        Convert trade objects to DataFrame.

        Args:
            trades: Lista di Trade objects

        Returns:
            DataFrame con trade convertiti
        """
        if not trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in trades])

    def close(self):
        """
        Cleanup e salvataggio finale.
        Da chiamare alla chiusura dell'applicazione.
        """
        try:
            if self.enable_persistence:
                self.save_trades()
                logger.info("TradeManager closed, trades saved")
        except Exception as e:
            logger.error(f"Error during final save: {e}")
        finally:
            # Stop time zero PL manager
            if hasattr(self.time_zero_pl_manager, 'stop'):
                try:
                    self.time_zero_pl_manager.stop()
                except Exception as e:
                    logger.error(f"Error stopping TimeZeroPLManager: {e}")

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
        return False  # Non sopprime eccezioni