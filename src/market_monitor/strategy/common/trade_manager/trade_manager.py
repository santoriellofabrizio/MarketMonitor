"""
trade_manager ottimizzato con miglioramenti chiave.
"""

import datetime
import logging
import queue as queue_mod
from pathlib import Path
from typing import Optional, Tuple, Union
from threading import RLock

import numpy as np
import pandas as pd

from market_monitor.strategy.common.trade_manager.book_memory import FairvaluePrice
from market_monitor.strategy.common.trade_manager.time_zero_pl import TimeZeroPLManager
from market_monitor.strategy.common.trade_manager.trade_templates import TradeStorage, TradeFactory, Trade, MyTrade

logger = logging.getLogger(__name__)


class TradeManager:
    """Trade manager con persistenza append-only e cache thread-safe."""

    # Configurabili via kwargs
    DEFAULT_MAX_TIME_TO_MATCH_SECONDS = 10.0
    DEFAULT_AUTO_SAVE_INTERVAL = 500
    DEFAULT_MAX_CACHE_SIZE = 1e9

    def __init__(
            self,
            book_storage,
            model_prices: pd.Series | None = None,
            time_zero_lags: list[float] | None = None,
            trade_folder: Path | None = None,
            **kwargs
    ):

        # Validazione book_storage
        if not hasattr(book_storage, 'get_last_before'):
            raise TypeError(
                "book_storage must have 'get_last_before' method. "
                "Use BookStorage class from user_strategy.utils.BookStorage"
            )

        self.trade_factory = TradeFactory()
        self.trade_storage = TradeStorage()
        self._my_trades_index: list = []

        self.model_price = model_prices
        self.book_storage = book_storage

        # Config
        self.max_time_to_match = kwargs.get('max_time_to_match_side', self.DEFAULT_MAX_TIME_TO_MATCH_SECONDS)
        self.max_cache_size = kwargs.get("max_cache_size", self.DEFAULT_MAX_CACHE_SIZE)
        self.trade_folder = trade_folder or Path.cwd() / "etc" / "data" / "trades"
        self.use_timezone_aware = kwargs.get("use_timezone_aware", True)


        # Cache thread-safe
        self._cache_lock = RLock()
        self._trades_df_cache: pd.DataFrame | None = None
        self._cache_valid = False

        # Tracking dei trades inviati parzialmente (is_elaborated=False)
        self._pending_partial_indexes: set[int] = set()
        # Thread-safe queue: PL thread pushes trade indexes here after each horizon
        self._horizon_publish_queue: queue_mod.Queue = queue_mod.Queue()

        # Persistenza
        self.engine = kwargs.get("engine", "pyarrow")
        self.compression = kwargs.get("compression", "snappy")
        self.enable_persistence = kwargs.get("enable_persistence", True)
        self.auto_save_interval = kwargs.get("auto_save_interval", self.DEFAULT_AUTO_SAVE_INTERVAL)
        self._trades_since_last_save = 0
        self._last_saved_index = 0

        # Time zero PL manager (supports multiple horizons, e.g. [10, 20, 30, 40] seconds)
        _effective_lags = time_zero_lags if time_zero_lags is not None else [10., 20., 30., 40.]
        self.time_zero_pl_manager = TimeZeroPLManager(
            mid_price_storage=book_storage,
            time_zero_lags=_effective_lags,
            trade_storage=self.trade_storage,
            on_horizon_computed=self._on_horizon_computed
        )
        if _effective_lags:  # only start if there are horizons to compute
            self.time_zero_pl_manager.start()

        # Load trades
        if self.enable_persistence:
            self._load_today_trades()

    # ========================================================================
    # CORE LOGIC
    # ========================================================================

    def on_trade(self, new_trades: pd.DataFrame):
        """Processa nuovi trade con gestione errori migliorata."""
        if new_trades is None or new_trades.empty:
            return pd.DataFrame()

        processed = []

        try:
            for trade in self.trade_factory.build_trades_obj(new_trades):
                # Match side se recente
                if trade.time_since_trade() < self.max_time_to_match:
                    if trade.is_my_trade():
                        self._my_trades_index.append(trade.trade_index)
                    else:
                        self.match_side(trade)

                    # Calcola spread PL usando book al momento del trade
                    snapshot = self.book_storage.get_last_before(trade.timestamp)
                    if snapshot:
                        mid = self._get_mid_by_snapshot(snapshot, trade)
                        if mid is not None:
                            trade.spread_pl = self.time_zero_pl_manager.calculate_time_zero_pl(
                                trade, mid
                            )

                # Store trade
                self.trade_storage.add_trade(trade)
                processed.append(trade)

            # Invalida cache (thread-safe)
            self._invalidate_cache()

            # Auto-save
            if self.enable_persistence:
                self._trades_since_last_save += len(processed)
                if self._trades_since_last_save >= self.auto_save_interval:
                    self._auto_save()

        except Exception as e:
            logger.error(f"Error processing trades: {e}", exc_info=True)
            raise

        return self._convert_trades_obj_to_df(processed)

    def _auto_save(self):
        """Auto-save con retry e gestione errori."""
        try:
            self._append_new_trades()
            self._trades_since_last_save = 0
        except Exception as e:
            logger.warning(
                f"Auto-save failed (will retry): {e}. "
                f"Unsaved trades: {self._trades_since_last_save}"
            )
            # Non resetta counter per retry al prossimo intervallo

    def match_side(self, trade):
        """
        Determina side confrontando con mid al momento del trade.
        Usa get_last_before per precision matching temporale.
        """
        snapshot = self.book_storage.get_last_before(trade.timestamp)

        if snapshot:
            snapshot_time, mid_prices = snapshot
            mid = self._get_mid_by_snapshot(snapshot, trade)
            logger.info(f"matched side: snapshot time: {snapshot_time}, trade_time: {trade.timestamp}")
            if mid is not None and not np.isnan(mid):
                trade.side = "bid" if trade.price < mid else "ask"

    # ========================================================================
    # QUERY METHODS (thread-safe cache)
    # ========================================================================

    @staticmethod
    def _get_mid_by_snapshot(snapshot: tuple, trade: Trade) -> float | None:
        snapshot_time, mid_prices = snapshot
        entry = mid_prices.get(trade.isin)
        if entry is None:
            return None
        if isinstance(entry, FairvaluePrice):
            return entry.get(currency=trade.currency, market=trade.market)
        return entry  # backward compat: float grezzo

    def get_trades(
            self, n_seconds: int | None = None, n_of_trades: int | None = None, use_cache: bool = True
    ) -> pd.DataFrame:
        """Get trades con cache thread-safe e filtro temporale opzionale."""
        with self._cache_lock:
            # 1. Gestione Cache: Se chiedi n_seconds, di solito conviene bypassare la cache
            # o filtrarla dopo. Qui filtriamo il risultato finale per semplicità.

            # Cache hit (solo se non stiamo filtrando per secondi o numero specifico)
            if (
                    n_of_trades is None
                    and n_seconds is None
                    and use_cache
                    and self._cache_valid
                    and self._trades_df_cache is not None
            ):
                return self._trades_df_cache.copy()

            # Cache miss - rigenera
            trades_obj_list = self.trade_storage.get_last_trades(n_of_trades)
            trades_df = self._convert_trades_obj_to_df(trades_obj_list)

            # Aggiorna cache (solo se è il dump completo)
            if n_of_trades is None and n_seconds is None and len(trades_df) <= self.max_cache_size:
                self._trades_df_cache = trades_df.copy()
                self._cache_valid = True

            # --- FILTRO PER N_SECONDS ---
            if n_seconds is not None and not trades_df.empty:
                # Assicurati che la colonna 'timestamp' sia in formato datetime
                if not pd.api.types.is_datetime64_any_dtype(trades_df['timestamp']):
                    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

                cutoff_time = datetime.datetime.now() - datetime.timedelta(seconds=n_seconds)
                trades_df = trades_df.loc[trades_df['timestamp'] >= cutoff_time]

            # Ordinamento finale
            return (
                trades_df.sort_values("timestamp", ascending=False)
                if not trades_df.empty
                else trades_df
            )

    def get_my_trades(self, n_of_trades: int | None = None) -> pd.DataFrame:
        """Get solo my trades (ottimizzato a livello storage)."""
        trades_obj = self.trade_storage.get_last_trades(n_of_trades)
        if isinstance(trades_obj, dict):
            trades_obj = list(trades_obj.values())
        my_trades_obj = [t for t in trades_obj if t.is_my_trade()]
        return self._convert_trades_obj_to_df(my_trades_obj)

    def get_trades_from_isin(
            self, isins: list[str], n_of_trades: int | None = None
    ) -> pd.DataFrame:
        """Get trades per ISIN (ottimizzato)."""
        if not isins:
            return pd.DataFrame()

        trades_obj = self.trade_storage.get_last_trades(n_of_trades)
        if isinstance(trades_obj, dict):
            trades_obj = list(trades_obj.values())
        filtered_obj = [t for t in trades_obj if t.isin in isins]

        df = self._convert_trades_obj_to_df(filtered_obj)
        return df.sort_values("timestamp", ascending=False) if not df.empty else df

    def get_trades_from_ticker(
            self, tickers: list[str], n_of_trades: int | None = None
    ) -> pd.DataFrame:

        if not tickers:
            return pd.DataFrame()

        trades_obj = self.trade_storage.get_last_trades(n_of_trades)
        if isinstance(trades_obj, dict):
            trades_obj = list(trades_obj.values())
        filtered_obj = [t for t in trades_obj if t.ticker in tickers]

        df = self._convert_trades_obj_to_df(filtered_obj)
        return df.sort_values("timestamp", ascending=False) if not df.empty else df

    def get_filtered_trades(
            self,
            n: Optional[int] = None,
            min_ctv: float | None = None,
            columns: list[str] | None = None,
            only_my_trades: Optional[bool] = None,
            start_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Get trades filtrati."""
        trades = self.get_trades()

        if trades.empty:
            return pd.DataFrame()

        # Applica filtri
        if min_ctv:
            trades = trades[trades["ctv"] > min_ctv]

        if only_my_trades:
            trades = trades[trades["own_trade"]]

        if start_time:
            trades = trades[trades["timestamp"] >= start_time]

        # Seleziona colonne
        if columns:
            available_cols = trades.columns.intersection(columns)
            trades = trades[available_cols]

        # Sort e limit
        trades = trades.sort_values("timestamp", ascending=False)
        if n:
            trades = trades.head(n)

        return trades

    def _on_horizon_computed(self, trade) -> None:
        """Callback invocato da TimeZeroPLManager dopo ogni orizzonte calcolato.

        Inserisce l'indice del trade nella coda thread-safe _horizon_publish_queue
        e invalida la cache. Il prossimo drenaggio della coda (da get_trades_to_publish
        o get_horizon_updates) pubblicherà il trade con i dati aggiornati.
        """
        self._horizon_publish_queue.put(trade.trade_index)
        self._invalidate_cache()

    def get_horizon_updates(self) -> pd.DataFrame:
        """Drena la coda degli orizzonti e ritorna i trades aggiornati come DataFrame.

        Thread-safe. Può essere chiamato dalla strategy in update_LF(), on_trade(),
        o da qualsiasi contesto. Ritorna un DataFrame vuoto se non ci sono aggiornamenti.
        """
        indexes: set[int] = set()
        while True:
            try:
                idx = self._horizon_publish_queue.get_nowait()
                indexes.add(idx)
            except queue_mod.Empty:
                break

        if not indexes:
            return pd.DataFrame()

        trades = []
        for idx in indexes:
            trade = self.trade_storage.get_trades_by_index(idx)
            if trade:
                trades.append(trade)

        return self._convert_trades_obj_to_df(trades)

    def get_trades_to_publish(self, processed_trades: pd.DataFrame) -> pd.DataFrame:
        """
        Ritorna i trades da pubblicare alla GUI.

        Logica:
        1. I trades appena processati vengono sempre inviati
        2. Se un trade è parziale (is_elaborated=False), viene tracciato in _pending_partial_indexes
        3. I pending completamente elaborati (is_elaborated=True) vengono re-inviati e rimossi
        4. Drena la coda _horizon_publish_queue per includere aggiornamenti intermedi
           degli orizzonti (10s, 20s, 30s, 40s) senza aspettare l'ultimo
        5. La GUI si occupa della deduplicazione tramite trade_index (keep='last')

        Args:
            processed_trades: DataFrame dei trades appena processati da on_trade()

        Returns:
            DataFrame con i trades da pubblicare (nuovi + elaborati + aggiornamenti orizzonte)
        """
        with self._cache_lock:
            trades_to_publish = []
            fully_elaborated = set()

            # 1. Controlla i pending che sono diventati completamente elaborati
            for idx in list(self._pending_partial_indexes):
                trade = self.trade_storage.get_trades_by_index(idx)
                if trade and trade.is_elaborated:
                    trades_to_publish.append(trade)
                    fully_elaborated.add(idx)

            self._pending_partial_indexes -= fully_elaborated

            # 2. Aggiungi i nuovi trades processati e traccia i parziali
            if not processed_trades.empty:
                for _, row in processed_trades.iterrows():
                    trade_idx = row.get('trade_index')
                    if trade_idx is not None:
                        trade = self.trade_storage.get_trades_by_index(int(trade_idx))
                        if trade:
                            trades_to_publish.append(trade)
                            if not trade.is_elaborated:
                                self._pending_partial_indexes.add(int(trade_idx))

            # 3. Drena la coda degli aggiornamenti orizzonte (thread-safe, non-blocking)
            horizon_df = self.get_horizon_updates()
            base_df = self._convert_trades_obj_to_df(trades_to_publish)

            if horizon_df.empty:
                return base_df
            if base_df.empty:
                return horizon_df

            combined = pd.concat([base_df, horizon_df], ignore_index=True)
            if 'trade_index' in combined.columns:
                combined = combined.drop_duplicates(subset=['trade_index'], keep='last')
            return combined

    # ========================================================================
    # PERSISTENZA (ottimizzata)
    # ========================================================================

    def _get_today_filename(self) -> Path:
        """Path file per trade di oggi (timezone-aware)."""
        if self.use_timezone_aware:
            today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
        else:
            today = datetime.datetime.now().strftime("%Y%m%d")

        self.trade_folder.mkdir(parents=True, exist_ok=True)
        return self.trade_folder / f"trades_{today}.parquet"

    def _append_new_trades(self):
        """Append semplificato che funziona sempre."""
        filepath = self._get_today_filename()

        all_trades = [*self.trade_storage.get_last_trades().values()]
        new_trades = all_trades[self._last_saved_index:]

        if not new_trades:
            return

        new_df = self._convert_trades_obj_to_df(new_trades)

        try:
            if filepath.exists():
                try:
                    existing_df = pd.read_parquet(filepath, engine="pyarrow")
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                except Exception as read_error:
                    logger.warning(
                        f"Parquet file corrupted ({read_error}). "
                        f"Backing up and creating new file."
                    )
                    backup_path = filepath.with_stem(filepath.stem + "_corrupted_backup")
                    try:
                        filepath.rename(backup_path)
                        logger.info(f"Corrupted file backed up to: {backup_path}")
                    except Exception as backup_error:
                        logger.error(f"Could not backup corrupted file: {backup_error}")
                    combined_df = new_df
            else:
                combined_df = new_df

            combined_df.to_parquet(
                filepath,
                engine="pyarrow",  # ✅ USA PYARROW
                compression=self.compression,
                index=False
            )

            self._last_saved_index = len(all_trades)
            logger.info(f"Appended {len(new_trades)} trades to {filepath}")

        except Exception as e:
            logger.error(f"Error appending trades: {e}", exc_info=True)
            raise

    def save_trades(self, filepath: Path | str | None = None):
        """Salva tutti i trade (usa cache se disponibile)."""
        if filepath is None:
            filepath = self._get_today_filename()
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            trades_df = self.get_trades(use_cache=True)

            if trades_df.empty:
                logger.info("No trades to save")
                return

            trades_df.to_parquet(
                filepath,
                engine=self.engine,
                compression=self.compression,
            )

            self._last_saved_index = len(self.trade_storage.get_last_trades())
            logger.info(f"Saved {len(trades_df)} trades to {filepath}")

        except Exception as e:
            logger.error(f"Error saving trades to {filepath}: {e}", exc_info=True)
            raise

    def _load_today_trades(self):
        """Load trade del giorno."""
        filepath = self._get_today_filename()

        if not filepath.exists():
            logger.info(f"No previous trades found at {filepath}")
            return

        try:
            trades_df = pd.read_parquet(filepath, engine="pyarrow")

            if trades_df.empty:
                return

            if "is_elaborated" not in trades_df.columns:
                logger.info("No 'is_elaborated' column found, skipping load")
                return

            elaborated_trades = trades_df[trades_df["is_elaborated"] == True]

            if elaborated_trades.empty:
                return

            loaded_count = 0
            for _, row in elaborated_trades.iterrows():
                trade = self._reconstruct_trade_from_row(row)

                if trade:
                    # add_trade is already thread-safe; no outer lock needed here
                    self.trade_storage.add_trade(trade)
                    if trade.is_my_trade():
                        self._my_trades_index.append(trade.trade_index)
                    loaded_count += 1

            self._last_saved_index = loaded_count
            self._invalidate_cache()

            logger.info(f"Loaded {loaded_count} trades from {filepath}")

        except Exception as e:
            logger.error(f"Error loading trades: {e}", exc_info=True)

    @staticmethod
    def _reconstruct_trade_from_row(row: pd.Series):
        """Ricostruisce Trade object da row."""

        try:
            known_params = {'ticker', 'isin', 'timestamp', 'quantity', 'price', 'market',
                            'currency', 'price_multiplier', 'side', 'own_trade',
                            'spread_pl', 'is_elaborated', 'trade_index', ...}
            trade_dict = row.to_dict()
            extra = {k: v for k, v in trade_dict.items() if k not in known_params}

            TradeClass = MyTrade if trade_dict.get("own_trade", False) else Trade

            params = {
                "ticker": trade_dict["ticker"],
                "isin": trade_dict["isin"],
                "timestamp": pd.to_datetime(trade_dict["timestamp"]),
                "quantity": trade_dict["quantity"],
                "price": trade_dict["price"],
                "market": trade_dict.get("market"),
                "currency": trade_dict.get("currency"),
                "price_multiplier": trade_dict.get("price_multiplier", 1),
                **extra   # unpack extra fields directly so Trade(**extra) captures them flat
            }

            if TradeClass == MyTrade:
                params["side"] = trade_dict.get("side")

            trade = TradeClass(**params)

            if TradeClass == Trade:
                trade.side = trade_dict.get("side")

            trade.spread_pl = trade_dict.get("spread_pl")

            # Restore backward-compat lagged P&L alias (first horizon)
            trade.lagged_spread_pl = trade_dict.get("lagged_spread_pl")

            # Restore all horizon-specific lagged P&L fields (e.g. lagged_spread_pl_10s)
            for key, val in trade_dict.items():
                if key.startswith("lagged_spread_pl_") and val is not None:
                    setattr(trade, key, val)

            trade.is_elaborated = True

            return trade

        except Exception as e:
            logger.error(f"Error reconstructing trade: {e}")
            return None

    # ========================================================================
    # UTILITY
    # ========================================================================

    def _invalidate_cache(self):
        """Invalida cache (thread-safe)."""
        with self._cache_lock:
            self._cache_valid = False
            self._trades_df_cache = None

    @staticmethod
    def _convert_trades_obj_to_df(trades: Union[dict, list]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame()
        rows = []
        if isinstance(trades, dict):
            trades = [*trades.values()]
        for t in trades:
            d = {k: v for k, v in t.__dict__.items() if k != 'extra'}
            d.update(t.extra)  # flatten
            rows.append(d)
        return pd.DataFrame(rows)
    # ========================================================================
    # CLEANUP
    # ========================================================================

    def close(self):
        """Cleanup e salvataggio finale."""
        try:
            if self.enable_persistence:
                self.save_trades()
                logger.info("trade_manager closed, trades saved")
        except Exception as e:
            logger.error(f"Error during final save: {e}")
        finally:
            if self.time_zero_pl_manager and self.time_zero_pl_manager.is_alive():
                try:
                    # 1. Segnala lo stop
                    logger.info("Stopping TimeZeroPLManager...")
                    self.time_zero_pl_manager.stop()

                    # 2. ASPETTA che finisca (fondamentale)
                    # Mettiamo un timeout per evitare che il programma resti appeso se il thread è bloccato
                    self.time_zero_pl_manager.join(timeout=3.0)

                    if self.time_zero_pl_manager.is_alive():
                        logger.warning("TimeZeroPLManager did not stop within timeout")
                    else:
                        logger.info("TimeZeroPLManager stopped gracefully")

                except Exception as e:
                    logger.error(f"Error stopping TimeZeroPLManager: {e}")