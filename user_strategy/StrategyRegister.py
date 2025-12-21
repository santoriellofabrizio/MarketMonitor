from typing import Type, Dict

from market_monitor.strategy.StrategyUI.StrategyUIAsync import StrategyUIAsync
from user_strategy.ETFEquity.LiveAnalysis.EtfEquityLiveAnalysis import EtfEquityLiveAnalysis
from user_strategy.ETFEquity.LiveQuoting.EtfEquityPriceEngine import EtfEquityPriceEngine
from user_strategy.FixedIncomeETF.EtfFiAnalysis import FIAnalysis
from user_strategy.FixedIncomeETF.EtfFiPriceEngine import EtfFiPriceEngine

# Registro globale
strategy_register: Dict[str, Type[StrategyUIAsync]] = {}


def register_strategy(name: str, cls: Type[StrategyUIAsync]) -> None:
    """Registra una strategia nel registro."""
    strategy_register[name] = cls


def get_strategy(name: str) -> Type[StrategyUIAsync]:
    """Restituisce la classe della strategia, solleva KeyError se non trovata."""
    return strategy_register[name]


def create_strategy(name: str, *args, **kwargs) -> StrategyUIAsync:
    """Istanzia la strategia con i parametri forniti."""
    cls = get_strategy(name)
    return cls(*args, **kwargs)


def list_strategies() -> list[str]:
    """Lista dei nomi registrati."""
    return list(strategy_register.keys())


def register_all_strategies():
    register_strategy("EtfEquityLiveAnalysis", EtfEquityLiveAnalysis)
    register_strategy("EtfFiPriceEngine", EtfFiPriceEngine)
    register_strategy("EtfEquityPriceEngine", EtfEquityPriceEngine)
    register_strategy("EtfEquityPriceEngine", EtfEquityPriceEngine)
    register_strategy("FIAnalysis", FIAnalysis)


