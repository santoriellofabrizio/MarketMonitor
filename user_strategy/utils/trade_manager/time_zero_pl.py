import datetime
import logging
import threading
import pandas as pd
from user_strategy.utils.trade_manager.trade_templates import AbstractTrade, TradeStorage

logger = logging.getLogger(__name__)


class TimeZeroPLManager(threading.Thread):

    def __init__(self, trade_storage: TradeStorage,
                 mid_price_storage: pd.Series,
                 model_price: pd.Series | None = None,
                 time_zero_lag: float = 10.):
        # IMPORTANTE: Cambiato nome per evitare conflitti interni di threading
        super().__init__(name="TimeZeroPLThread", daemon=True)
        self.trade_storage = trade_storage
        self.mid_price_storage = mid_price_storage
        self.model_price = model_price
        self.time_zero_lag = time_zero_lag

        # Flag per la chiusura
        self._is_running = True
        self._stop_event = threading.Event()

        logger.info(
            f"TimeZeroPLManager inizializzato | "
            f"time_zero_lag={self.time_zero_lag}s | "
            f"model_price_enabled={'Yes' if model_price is not None else 'No'}"
        )

    def stop(self):
        """Metodo chiamato esternamente per chiudere il thread."""
        logger.warning(f"[STOP] Segnale di terminazione ricevuto per TimeZeroPLThread")
        self._is_running = False
        self._stop_event.set()  # Sveglia il thread se sta dormendo

    def run(self) -> None:
        logger.info(f"[START] TimeZeroPLThread avviato")
        trades_processed = 0

        try:
            while self._is_running:
                # Se get_trade_index_to_elaborate è bloccante, assicurati che abbia un timeout
                trade_index = self.trade_storage.get_trade_index_to_elaborate()

                # Se il manager dello storage restituisce None o un segnale di stop
                if trade_index is None:
                    if not self._is_running:
                        logger.info(f"[EXIT] Stop signal ricevuto, chiusura thread")
                        break
                    logger.debug(f"[WAIT] Nessun trade disponibile, riprovo tra 1s")
                    self._stop_event.wait(1.0)
                    continue

                trade = self.trade_storage.get_trades_by_index(trade_index)
                if trade:
                    logger.debug(f"[FETCH] Trade {trade_index} recuperato dal storage")
                    # Se process_trade restituisce False, significa che dobbiamo chiudere
                    if not self.process_trade(trade):
                        logger.info(f"[EXIT] Stop signal durante process_trade")
                        break
                    trades_processed += 1
                else:
                    logger.warning(f"[ERROR] Trade index {trade_index} non trovato nel storage")
                    self._stop_event.wait(0.5)
        except Exception as e:
            logger.exception(f"[CRITICAL] Eccezione non gestita nel thread principale", exc_info=e)
        finally:
            logger.info(
                f"[SHUTDOWN] TimeZeroPLThread chiuso | "
                f"trades_processed={trades_processed}"
            )

    def process_trade(self, trade: AbstractTrade) -> bool:
        """
        Elabora un singolo trade per il calcolo PL con time lag.

        Restituisce:
        - True: elaborazione completata normalmente
        - False: interrotto da segnale di stop
        """
        trade_timestamp = trade.timestamp
        logger.info(
            f"[PROCESS] Inizio elaborazione trade | "
            f"trade_id={trade.trade_index} | "
            f"isin={trade.isin} | "
            f"timestamp={trade_timestamp} | "
            f"qty={trade.quantity} | "
            f"price={trade.price} | "
            f"side={trade.side}"
        )

        while self._is_running:
            diff_seconds = (datetime.datetime.now() - trade_timestamp).total_seconds()

            if diff_seconds > self.time_zero_lag * 4:
                logger.info(
                    f"[TIMEOUT] Trade {trade.trade_index} timeout (elapsed={diff_seconds:.2f}s > "
                    f"max={self.time_zero_lag * 4}s), marcato come elaborato senza PL"
                )
                self.trade_storage.set_trade_as_elaborated(trade)
                return True

            elif diff_seconds > self.time_zero_lag:
                logger.info(
                    f"[CALC] Tempo di lag raggiunto | "
                    f"trade_id={trade.trade_index} | "
                    f"elapsed={diff_seconds:.2f}s > lag={self.time_zero_lag}s"
                )
                self._calculate_time_zero_pl(trade)
                self.trade_storage.set_trade_as_elaborated(trade)
                logger.info(
                    f"[DONE] Elaborazione completata | "
                    f"trade_id={trade.trade_index} | "
                    f"lagged_pl={trade.lagged_spread_pl}"
                    + (f" | model_pl={trade.lagged_spread_pl_model}" if self.model_price is not None else "")
                )
                return True

            else:
                wait_time = self.time_zero_lag - diff_seconds
                logger.debug(
                    f"[WAIT] Trade in attesa | "
                    f"trade_id={trade.trade_index} | "
                    f"elapsed={diff_seconds:.2f}s | "
                    f"wait={wait_time:.2f}s"
                )

                # Invece di time.sleep(max(wait_time, 1))
                # Aspetta wait_time, ma se stop() viene chiamato, si sveglia subito
                interrupted = self._stop_event.wait(timeout=max(wait_time, 0.1))
                if interrupted or not self._is_running:
                    logger.warning(
                        f"[INTERRUPT] Elaborazione interrotta | "
                        f"trade_id={trade.trade_index} | "
                        f"elapsed={diff_seconds:.2f}s | interrupted={interrupted}"
                    )
                    return False

        logger.warning(f"[LOOP_EXIT] Uscita dal loop while con _is_running=False")
        return False

    def _calculate_time_zero_pl(self, trade: AbstractTrade):
        """Calcola il PL al time zero per mid price e opzionalmente model price."""
        logger.debug(f"[CALC_START] Inizio calcolo PL per trade {trade.trade_index}")

        # Recupero mid price
        mid_price, time_snip = self.get_mid(trade.isin)

        if mid_price is not None:
            logger.info(
                f"[MID_PRICE] trade_id={trade.trade_index} | "
                f"isin={trade.isin} | "
                f"mid={mid_price} | "
                f"time_snip={time_snip}"
            )
            trade.lagged_spread_pl = self.calculate_time_zero_pl(trade, mid_price)
            logger.debug(
                f"[MID_PL] trade_id={trade.trade_index} | "
                f"pl={trade.lagged_spread_pl}"
            )
        else:
            logger.warning(
                f"[MID_PRICE_MISSING] trade_id={trade.trade_index} | "
                f"isin={trade.isin} | "
                f"impossibile calcolare PL"
            )

        # Recupero model price se disponibile
        if self.model_price is not None:
            model_price = self.get_model_price(trade.isin)
            if model_price is not None:
                logger.info(
                    f"[MODEL_PRICE] trade_id={trade.trade_index} | "
                    f"isin={trade.isin} | "
                    f"model_price={model_price}"
                )
                trade.lagged_spread_pl_model = self.calculate_time_zero_pl(trade, model_price)
                logger.debug(
                    f"[MODEL_PL] trade_id={trade.trade_index} | "
                    f"model_pl={trade.lagged_spread_pl_model}"
                )
            else:
                logger.info(
                    f"[MODEL_PRICE_MISSING] trade_id={trade.trade_index} | "
                    f"isin={trade.isin} | "
                    f"impossibile calcolare model PL"
                )

    def get_mid(self, isin: str):
        """
        Recupera il prezzo di riferimento (mid price) per l'ISIN specificato.

        Restituisce:
        - (mid_price, time_snip): valori trovati
        - (None, None): ISIN non trovato o storage vuoto
        """
        try:
            if not len(self.mid_price_storage):
                logger.debug(f"[GET_MID] Storage vuoto per ISIN {isin}")
                return None, None

            time_snip, storage = self.mid_price_storage[-1]

            if isin not in storage:
                logger.debug(f"[GET_MID] ISIN {isin} non presente nel book corrente")
                return None, None

            mid_price = storage[isin]

            if mid_price is None or mid_price <= 0:
                logger.warning(
                    f"[GET_MID] Mid price non valido | "
                    f"isin={isin} | "
                    f"price={mid_price} | "
                    f"time={time_snip}"
                )
                return None, None

            logger.debug(f"[GET_MID] isin={isin} | mid={mid_price} | time={time_snip}")
            return mid_price, time_snip

        except KeyError as e:
            logger.debug(f"[GET_MID] KeyError per ISIN {isin}: {str(e)}")
            return None, None
        except Exception as e:
            logger.error(
                f"[GET_MID] Eccezione inaspettata | "
                f"isin={isin} | "
                f"error={str(e)}",
                exc_info=e
            )
            return None, None

    def get_model_price(self, isin: str):
        """
        Recupera il model price per l'ISIN specificato.

        Restituisce:
        - price: valore trovato (float)
        - None: ISIN non trovato o errore
        """
        try:
            mid_price = self.model_price.get(isin, None)

            if mid_price is None:
                logger.debug(f"[GET_MODEL] ISIN {isin} non trovato nel model price storage")
                return None

            if mid_price <= 0:
                logger.info(
                    f"[GET_MODEL] Model price non valido | "
                    f"isin={isin} | "
                    f"price={mid_price}"
                )
                return None

            logger.debug(f"[GET_MODEL] isin={isin} | model_price={mid_price}")
            return mid_price

        except KeyError as e:
            logger.debug(f"[GET_MODEL] KeyError per ISIN {isin}: {str(e)}")
            return None
        except Exception as e:
            logger.error(
                f"[GET_MODEL] Eccezione inaspettata | "
                f"isin={isin} | "
                f"error={str(e)}",
                exc_info=e
            )
            return None

    @staticmethod
    def calculate_time_zero_pl(trade: AbstractTrade, price: float):
        """
        Calcola il P&L al time zero usando il prezzo fornito.

        Formula: PL = (price - trade_price) * quantity * side_multiplier
        - side_multiplier: 1 per "bid" (long), -1 per "ask" (short)

        Restituisce:
        - pl: valore calcolato (float)
        - None: input non validi
        """
        # Validazione input
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

        # Validazione trade
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

        # Mapping side
        side_map = {"bid": 1, "ask": -1}
        side_multiplier = side_map.get(side, None)

        if side_multiplier is None:
            logger.info(
                f"[CALC_PL] Side non riconosciuto | "
                f"trade_id={trade.trade_index} | "
                f"side='{side}' (expected 'bid' or 'ask')"
            )
            return None

        # Calcolo PL
        pl = (price - trade_price) * qty * side_multiplier

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