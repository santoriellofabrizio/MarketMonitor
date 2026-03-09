import datetime
import logging
import threading

import pandas as pd
from market_monitor.strategy.common.trade_manager.trade_templates import AbstractTrade, TradeStorage, Trade

logger = logging.getLogger(__name__)

_DEFAULT_HORIZONS = [10., 20., 30., 40.]


class TimeZeroPLManager(threading.Thread):

    def __init__(self,
                 trade_storage: TradeStorage,
                 mid_price_storage: pd.Series,
                 time_zero_lags: list[float] | None = None,
                 on_horizon_computed: callable | None = None):
        # IMPORTANTE: Cambiato nome per evitare conflitti interni di threading
        super().__init__(name="TimeZeroPLThread", daemon=True)
        self.trade_storage = trade_storage
        self.mid_price_storage = mid_price_storage
        self.on_horizon_computed = on_horizon_computed

        # Support multiple horizons; keep backward-compat alias for the first one
        self.time_zero_lags: list[float] = sorted(time_zero_lags or _DEFAULT_HORIZONS)
        self.time_zero_lag: float = self.time_zero_lags[0]   # backward-compat alias
        self._max_lag: float = self.time_zero_lags[-1]

        # Flag per la chiusura
        self._is_running = True
        self._stop_event = threading.Event()

        logger.info(
            f"TimeZeroPLManager inizializzato | "
            f"time_zero_lags={self.time_zero_lags}s"
        )

    def stop(self):
        """Metodo chiamato esternamente per chiudere il thread."""
        logger.warning(f"[STOP] Segnale di terminazione ricevuto per TimeZeroPLThread")
        self._is_running = False
        self._stop_event.set()  # Sveglia il thread se sta dormendo

    def run(self) -> None:
        logger.info(f"[START] TimeZeroPLThread avviato | horizons={self.time_zero_lags}s")
        # pending: trade_index → (trade_object, remaining_lags_list)
        pending: dict[int, tuple[AbstractTrade, list[float]]] = {}
        trades_processed = 0

        try:
            while self._is_running:

                # 1. Drain the queue non-blocking — pick up all newly added trades
                while True:
                    idx = self.trade_storage.get_trade_index_to_elaborate(timeout=0)
                    if idx is None:
                        break
                    trade = self.trade_storage.get_trades_by_index(idx)
                    if trade:
                        pending[idx] = (trade, list(self.time_zero_lags))
                        logger.info(
                            f"[ENQUEUE] trade_id={idx} | isin={trade.isin} | "
                            f"horizons={self.time_zero_lags}s"
                        )

                # 2. Check all in-flight trades against current time
                now = datetime.datetime.now()
                to_remove: list[int] = []

                for idx, (trade, remaining) in list(pending.items()):
                    diff = (now - trade.timestamp).total_seconds()

                    # Hard timeout: 4× the longest horizon
                    if diff > self._max_lag * 4:
                        logger.info(
                            f"[TIMEOUT] trade_id={idx} | elapsed={diff:.1f}s | "
                            f"remaining={remaining} — marked elaborated"
                        )
                        self.trade_storage.set_trade_as_elaborated(trade)
                        to_remove.append(idx)
                        continue

                    # Fire all horizons whose threshold has been reached
                    newly_done = [lag for lag in remaining if diff >= lag]
                    for lag in newly_done:
                        self._calculate_pl_at_lag(trade, lag)
                        remaining.remove(lag)

                    # All horizons done
                    if not remaining:
                        self.trade_storage.set_trade_as_elaborated(trade)
                        to_remove.append(idx)
                        trades_processed += 1
                        logger.info(f"[DONE] trade_id={idx} | all horizons computed")

                for idx in to_remove:
                    del pending[idx]

                # 3. Sleep until next tick (wakes immediately if stop() is called)
                self._stop_event.wait(timeout=0.5)

        except Exception as e:
            logger.exception(f"[CRITICAL] Eccezione nel thread principale", exc_info=e)
        finally:
            logger.info(
                f"[SHUTDOWN] TimeZeroPLThread chiuso | "
                f"trades_processed={trades_processed} | pending_at_shutdown={len(pending)}"
            )

    def _calculate_pl_at_lag(self, trade: AbstractTrade, lag: float):
        """Calcola il PL a un orizzonte specifico e lo salva sull'oggetto trade.

        Attributi scritti:
          - ``lagged_spread_pl_{int(lag)}s``  (es. lagged_spread_pl_10s)

        Per il primo orizzonte (backward compat) aggiorna anche ``trade.lagged_spread_pl``.
        Chiama ``on_horizon_computed(trade)`` al termine per notificare il TradeManager.
        """
        col = f"lagged_spread_pl_{int(lag)}s"
        is_first_horizon = (lag == self.time_zero_lags[0])

        mid_price, _ = self.get_mid(trade)
        if mid_price is not None:
            pl = self.calculate_time_zero_pl(trade, mid_price)
            setattr(trade, col, pl)
            if is_first_horizon:
                trade.lagged_spread_pl = pl
            logger.info(
                f"[LAG_{int(lag)}s] trade_id={trade.trade_index} | "
                f"isin={trade.isin} | mid={mid_price} | {col}={pl}"
            )
        else:
            setattr(trade, col, None)
            if is_first_horizon:
                trade.lagged_spread_pl = None
            logger.warning(
                f"[LAG_{int(lag)}s] Mid price non disponibile | "
                f"trade_id={trade.trade_index} | isin={trade.isin}"
            )

        if self.on_horizon_computed is not None:
            self.on_horizon_computed(trade)

    def get_mid(self, trade: Trade):
        try:
            if not self.mid_price_storage:
                logger.debug(f"[GET_MID] Storage vuoto per ISIN {trade.isin}")
                return None, None

            time_snip, snapshot = self.mid_price_storage[-1]

            mid_entry = snapshot.get(trade.isin)
            if mid_entry is None:
                logger.debug(f"[GET_MID] ISIN {trade.isin} non presente nel book corrente")
                return None, None

            mid_price = mid_entry.get(currency=trade.currency, market=trade.market)

            if mid_price is None or mid_price <= 0:
                logger.warning(
                    f"[GET_MID] Mid price non valido | isin={trade.isin} | price={mid_price} | time={time_snip}")
                return None, None

            logger.debug(f"[GET_MID] isin={trade.isin} | mid={mid_price} | time={time_snip}")
            return mid_price, time_snip

        except Exception as e:
            logger.error(f"[GET_MID] Eccezione inaspettata | isin={trade.isin} | error={str(e)}", exc_info=e)
            return None, None

    @staticmethod
    def calculate_time_zero_pl(trade: AbstractTrade, price: float):
        """
        Calcola il P&L al time zero usando il prezzo fornito.

        Formula: PL = (price - trade_price) * quantity * side_multiplier * price_multiplier
        - side_multiplier: 1 per "bid" (long), -1 per "ask" (short)

        Restituisce:
        - pl: valore calcolato (float)
        - None: input non validi
        """
        if price is None or price <= 0:
            logger.debug(
                f"[CALC_PL] Price non valido, PL non calcolato | "
                f"trade_id={trade.trade_index} | "
                f"price={price}"
            )
            return None

        trade_price = trade.price
        qty = trade.quantity
        side = trade.side
        multiplier = getattr(trade, "price_multiplier", 1)

        if trade_price is None or trade_price <= 0:
            logger.warning(
                f"[CALC_PL] Trade price non valido | "
                f"trade_id={trade.trade_index} | "
                f"trade_price={trade_price}"
            )
            return None

        if qty is None or qty == 0:
            logger.warning(
                f"[CALC_PL] Quantity non valida | "
                f"trade_id={trade.trade_index} | "
                f"qty={qty}"
            )
            return None

        side_map = {"bid": 1, "ask": -1}
        side_multiplier = side_map.get(side, None)

        if side_multiplier is None:
            logger.info(
                f"[CALC_PL] Side non riconosciuto | "
                f"trade_id={trade.trade_index} | "
                f"side='{side}' (expected 'bid' or 'ask')"
            )
            return None

        pl = (price - trade_price) * qty * side_multiplier * multiplier

        logger.debug(
            f"[CALC_PL] Calcolo completato | "
            f"trade_id={trade.trade_index} | "
            f"side={side} (mult={side_multiplier}) | "
            f"trade_price={trade_price} | "
            f"mid_price={price} | "
            f"qty={qty} | "
            f"pl={pl:.2f}"
        )

        return pl
