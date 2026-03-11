"""
BacktestResult
==============
Raccoglie e presenta i risultati di un'esecuzione di backtest.

Struttura
---------
- hf_outputs     : lista di (timestamp_simulato, return_value_di_update_HF)
- lf_call_times  : timestamp simulati di ogni chiamata a update_LF
- contatori      : tick_count, hf_call_count, lf_call_count
- intervallo     : start_time, end_time del dataset storico

Metodi principali
-----------------
- summary()              → dict con metriche aggregate
- hf_outputs_as_dataframe() → DataFrame indicizzato per timestamp simulato
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class BacktestResult:
    """
    Risultato completo di un'esecuzione di BacktestRunner.

    Attributes:
        hf_outputs:      Lista di (ts_simulato, output_di_update_HF).
                         L'output può essere None se update_HF non restituisce nulla,
                         oppure un DataFrame/tuple (come in live).
        lf_call_times:   Timestamp simulati in cui è stato chiamato update_LF.
        tick_count:      Numero totale di tick iniettati in RTData.
        hf_call_count:   Numero di volte che update_HF è stato chiamato.
        lf_call_count:   Numero di volte che update_LF è stato chiamato.
        start_time:      Inizio del dataset storico.
        end_time:        Fine del dataset storico.
    """

    hf_outputs: List[Tuple[datetime, Any]] = field(default_factory=list)
    lf_call_times: List[datetime] = field(default_factory=list)

    tick_count: int = 0
    hf_call_count: int = 0
    lf_call_count: int = 0

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Metriche aggregate
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """
        Restituisce un dizionario con le metriche principali del backtest.

        Returns:
            Dict con le seguenti chiavi:
                - start_time / end_time
                - simulated_duration_seconds
                - tick_count
                - hf_calls, lf_calls
                - hf_outputs_non_none  (quante volte update_HF ha restituito qualcosa)
                - ticks_per_hf_call    (media tick tra una chiamata HF e l'altra)
        """
        elapsed = (
            (self.end_time - self.start_time).total_seconds()
            if self.start_time and self.end_time
            else 0.0
        )
        non_none = sum(1 for _, v in self.hf_outputs if v is not None)
        avg_ticks_per_hf = (
            self.tick_count / self.hf_call_count if self.hf_call_count > 0 else 0.0
        )

        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "simulated_duration_seconds": elapsed,
            "tick_count": self.tick_count,
            "hf_calls": self.hf_call_count,
            "lf_calls": self.lf_call_count,
            "hf_outputs_non_none": non_none,
            "avg_ticks_per_hf_call": round(avg_ticks_per_hf, 2),
        }

    # ------------------------------------------------------------------
    # Conversione output HF
    # ------------------------------------------------------------------

    def hf_outputs_as_dataframe(self) -> pd.DataFrame:
        """
        Converte hf_outputs in un DataFrame con DatetimeIndex.

        Ogni riga corrisponde a una chiamata HF.
        La colonna "output" contiene il valore restituito da update_HF
        (può essere None, un DataFrame, una tupla, ecc.).

        Returns:
            DataFrame con colonne ["output"], index=timestamp_simulato.
            Vuoto se nessun output HF è stato raccolto.
        """
        if not self.hf_outputs:
            return pd.DataFrame()

        timestamps, outputs = zip(*self.hf_outputs)
        return pd.DataFrame({"output": list(outputs)}, index=list(timestamps))

    # ------------------------------------------------------------------
    # Rappresentazione
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        s = self.summary()
        duration_h = s["simulated_duration_seconds"] / 3600
        return (
            f"BacktestResult("
            f"ticks={self.tick_count}, "
            f"hf_calls={self.hf_call_count}, "
            f"lf_calls={self.lf_call_count}, "
            f"duration={duration_h:.1f}h)"
        )
