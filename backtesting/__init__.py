"""
backtesting
===========
Framework per il backtest del MarketMonitor con dati storici.

Questo modulo è progettato per vivere in una repo esterna (EtfEquityLab)
e importa da `market_monitor` come dipendenza.

Componenti principali
---------------------
BacktestRunner
    Orchestratore: prende una StrategyUI e un dict di DataFrame storici,
    li riproduce in RTData e chiama update_HF / update_LF in tempo simulato.

HistoricalDataFeed
    Fonde N stream di dati per-ticker (DataFrame con DatetimeIndex) in un
    unico stream di eventi ordinati cronologicamente.

BacktestResult
    Raccoglie gli output della strategia e produce metriche aggregate.

Esempio rapido
--------------
    from datetime import timedelta
    import pandas as pd
    from backtesting import BacktestRunner
    from user_strategy.equity.MyStrategy import MyStrategy

    data = {
        "IE00B4L5Y983": pd.read_parquet("data/iwda.parquet"),
        "LU0048584102": pd.read_csv("data/fund.csv", index_col=0, parse_dates=True),
    }

    runner = BacktestRunner(
        strategy_class=MyStrategy,
        data=data,
        fields=["BID", "ASK"],
        hf_frequency=timedelta(seconds=30),
        lf_frequency=timedelta(hours=1),
    )

    result = runner.run()
    print(result.summary())
    print(result.hf_outputs_as_dataframe())
"""

from backtesting.data_feed import HistoricalDataFeed
from backtesting.result import BacktestResult
from backtesting.runner import BacktestRunner

__all__ = [
    "BacktestRunner",
    "HistoricalDataFeed",
    "BacktestResult",
]
