"""
BacktestRunner
==============
Esegue il backtest di una StrategyUI contro dati storici.

Il runner:
1. Crea RTData (identico alla modalità live)
2. Istanzia la strategia e chiama i callback di inizializzazione
   (set_market_data → on_market_data_setting → on_book_initialized)
   esattamente come fa il Builder in live
3. Riproduce i tick storici iniettandoli in RTData.update()
4. Chiama update_HF / update_LF sincronamente in base al tempo simulato
5. Raccoglie i risultati in un BacktestResult

NON viene usato asyncio: i metodi della strategia sono chiamati in modo sincrono.
La strategia non ha bisogno di modifiche rispetto alla versione live.

Utilizzo
--------
    from datetime import timedelta
    from market_monitor_backtesting import BacktestRunner
    from user_strategy.equity.MyStrategy import MyStrategy

    data = {
        "IE00B4L5Y983": pd.read_parquet("data/iwda.parquet"),
        "LU0048584102": pd.read_csv("data/fund.csv", index_col=0, parse_dates=True),
    }

    runner = BacktestRunner(
        strategy_class=MyStrategy,
        data=data,
        fields=["BID", "ASK"],
        mid_key=["BID", "ASK"],
        hf_frequency=timedelta(seconds=30),
        lf_frequency=timedelta(minutes=60),
        strategy_kwargs={"my_param": 42},
    )

    result = runner.run()
    print(result.summary())
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Type

import pandas as pd

from market_monitor.live_data_hub.real_time_data_hub import RTData
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI

from backtesting.data_feed import HistoricalDataFeed
from backtesting.result import BacktestResult

logger = logging.getLogger(__name__)

# Numero massimo di tick iniettati per soddisfare wait_for_book_initialization()
# prima di procedere comunque.
_MAX_SEED_TICKS = 10_000


class BacktestRunner:
    """
    Orchestratore del backtest.

    Riutilizza la stessa StrategyUI della modalità live senza alcuna modifica:
    - on_market_data_setting() viene chiamato (la strategia può registrare
      subscription bloomberg/redis – rimarranno pending, non è un problema)
    - wait_for_book_initialization() viene rispettato: il runner inietta tick
      finché il metodo non ritorna True
    - on_book_initialized() viene chiamato dopo l'inizializzazione
    - update_HF() e update_LF() vengono chiamati sincronamente al ritmo
      definito da hf_frequency / lf_frequency in tempo simulato
    """

    def __init__(
        self,
        strategy_class: Type[StrategyUI],
        data: Dict[str, pd.DataFrame],
        fields: Optional[List[str]] = None,
        mid_key: Optional[List[str]] = None,
        hf_frequency: timedelta = timedelta(seconds=0.5),
        lf_frequency: timedelta = timedelta(seconds=60),
        strategy_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            strategy_class:   Sottoclasse di StrategyUI da eseguire.
            data:             Dict ticker -> DataFrame (DatetimeIndex + colonne field).
            fields:           Campi da tracciare in RTData (es. ["BID", "ASK"]).
                              Default: tutte le colonne presenti nei DataFrame.
            mid_key:          Campi usati per calcolare il mid price.
                              Default: uguale a fields.
            hf_frequency:     Intervallo di tempo simulato tra chiamate a update_HF.
            lf_frequency:     Intervallo di tempo simulato tra chiamate a update_LF.
            strategy_kwargs:  Kwargs aggiuntivi passati al costruttore della strategia.
        """
        self.strategy_class = strategy_class
        self.data = data
        self.fields = fields
        self.mid_key = mid_key or fields
        self.hf_frequency = hf_frequency
        self.lf_frequency = lf_frequency
        self.strategy_kwargs = strategy_kwargs or {}

    # ------------------------------------------------------------------
    # Entry point principale
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """
        Esegue il backtest completo.

        Returns:
            BacktestResult con tutti gli output della strategia e le metriche.
        """
        feed = HistoricalDataFeed(data=self.data, fields=self.fields)

        if feed.total_ticks == 0:
            logger.warning("Nessun tick trovato nel dataset. Restituisco BacktestResult vuoto.")
            return BacktestResult()

        logger.info(
            f"Backtest: {len(feed.tickers)} ticker, {feed.total_ticks} tick totali, "
            f"{feed.start_time} → {feed.end_time}"
        )

        # 1. Crea RTData (identico al Builder in live)
        lock = threading.Lock()
        rtdata = RTData(
            locker=lock,
            fields=self.fields,
            mid_key=self.mid_key,
        )

        # 2. Istanzia la strategia e collega RTData
        #    set_market_data() chiama internamente on_market_data_setting()
        strategy: StrategyUI = self.strategy_class(**self.strategy_kwargs)
        strategy.set_market_data(rtdata)

        # 3. Pre-seed: inietta tick finché wait_for_book_initialization() è True
        self._preseed(strategy, rtdata, feed)

        # 4. Segnala inizializzazione completata
        strategy.on_book_initialized()

        # 5. Loop principale
        result = BacktestResult(
            start_time=feed.start_time,
            end_time=feed.end_time,
        )
        next_hf: datetime = feed.start_time
        next_lf: datetime = feed.start_time

        for ts, ticker, data_dict in feed.events():
            rtdata.update(ticker, data_dict)
            result.tick_count += 1

            if ts >= next_hf:
                hf_out = self._call_update_hf(strategy, ts)
                result.hf_outputs.append((ts, hf_out))
                result.hf_call_count += 1
                next_hf = ts + self.hf_frequency

            if ts >= next_lf:
                self._call_update_lf(strategy, ts)
                result.lf_call_times.append(ts)
                result.lf_call_count += 1
                next_lf = ts + self.lf_frequency

        logger.info(
            f"Backtest completato: {result.tick_count} tick, "
            f"{result.hf_call_count} chiamate HF, {result.lf_call_count} chiamate LF"
        )
        return result

    # ------------------------------------------------------------------
    # Helpers privati
    # ------------------------------------------------------------------

    def _preseed(
        self,
        strategy: StrategyUI,
        rtdata: RTData,
        feed: HistoricalDataFeed,
    ) -> None:
        """
        Inietta tick fino a quando wait_for_book_initialization() è soddisfatto
        (o fino a _MAX_SEED_TICKS come limite di sicurezza).
        """
        injected = 0
        for ts, ticker, data_dict in feed.events():
            rtdata.update(ticker, data_dict)
            injected += 1

            if strategy.wait_for_book_initialization():
                logger.debug(
                    f"wait_for_book_initialization() soddisfatto dopo {injected} tick."
                )
                return

            if injected >= _MAX_SEED_TICKS:
                logger.warning(
                    f"wait_for_book_initialization() non soddisfatto dopo {injected} "
                    f"tick. Procedo comunque."
                )
                return

        logger.warning(
            "Dataset esaurito durante il pre-seed. "
            "wait_for_book_initialization() non è mai diventato True."
        )

    @staticmethod
    def _call_update_hf(strategy: StrategyUI, ts: datetime):
        try:
            return strategy.update_HF()
        except Exception:
            logger.error(f"Errore in update_HF al timestamp {ts}", exc_info=True)
            return None

    @staticmethod
    def _call_update_lf(strategy: StrategyUI, ts: datetime) -> None:
        try:
            strategy.update_LF()
        except Exception:
            logger.error(f"Errore in update_LF al timestamp {ts}", exc_info=True)
