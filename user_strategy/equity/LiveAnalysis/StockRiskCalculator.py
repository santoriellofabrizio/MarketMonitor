import logging

import numpy as np
import pandas as pd
from cvxpy import SolverError
from xbbg.blp import bdp
from joblib import Memory

from user_strategy.equity.LiveAnalysis.OptimizationFunctions import linearized_optimization

memory = Memory(".cache/strategy/user_strategy/equity/LiveQuoting/cacheStockRiskCalculator", verbose=False)



logger = logging.getLogger()


class StockRiskCalculator:

    def __init__(self,
                 weight_matrix: pd.DataFrame,
                 spread_estimate: pd.Series | None = None,
                 lambd: float = 1.,
                 max_number_of_trades: int | None = None):
        weight_matrix = weight_matrix.loc[:, [c for c in weight_matrix.columns if not c.startswith('FX')]]
        self.weight_matrix: pd.DataFrame = weight_matrix
        self.instruments = self.weight_matrix.index
        self.spread_estimate: pd.Series = spread_estimate or set_spread_estimate(self.instruments)
        self.lambd: float = lambd
        self.max_number_of_trades: int | None = max_number_of_trades

    def calculate_hedging(self, ctv: pd.Series, spreads: pd.Series | None = None, lambd: None | float = None):
        """
        Calculate hedging trades and evaluate their impact on exposure reduction.

        Args:
            ctv (pd.Series): Current target values for instruments.
            spreads (pd.Series, optional): Spread estimates. Defaults to None.

        Returns:
            tuple: Trades with significant impact, hedging cost, and max exposure reduction.
        """
        lambd = lambd or self.lambd
        try:

            ctv = ctv[self.instruments]
            spreads = spreads[self.instruments]

            spreads = spreads[spreads.apply(lambda x: isinstance(x, (int, float)))]
            spreads = spreads.combine_first(self.spread_estimate) if spreads is not None else self.spread_estimate

            # Controllo strumenti mancanti
            missing_ctv = set(self.instruments) - set(ctv.index)
            if missing_ctv: raise ValueError(f"Missing positions for instruments: {', '.join(missing_ctv)}")

            spreads, ctv = spreads.fillna(1).align(ctv, join="inner")

            if ctv.isna().any():
                logger.warning("Missing values detected in ctv.")
                return None, None, None
            try:
                result = linearized_optimization(spreads, lambd, ctv, self.weight_matrix)
                if result is None:
                    logger.warning("Optimization is not OPTIMAL/QUASI-OPTIMAL.")
                    return None, None, None

                trades, hedging_cost = result
                hedging_cost = trades @ spreads

                current_exposure = self.weight_matrix.T @ ctv
                new_exposure = self.weight_matrix.T @ (ctv + trades)
                max_exposure_reduction = np.max(current_exposure) - np.max(new_exposure)

                significant_trades = trades[trades.abs() > 10000].sort_values(ascending=False, key=lambda x: abs(x))

                return significant_trades, hedging_cost, max_exposure_reduction
            except SolverError:
                logger.warning("Optimization did not converged")
                return None, None, None



        except Exception as e:
            logger.exception("Unexpected error in optimizing hedging. " + str(e))
            return None, None, None


@memory.cache()
def set_spread_estimate(instruments):
    field = "TIME_WAVG_BID_ASK_SPREAD_PCT"
    spread_estimate = bdp([f"{etf} IM EQUITY" for etf in instruments], flds=field)[field.lower()]/100
    spread_estimate.index = [i.split(" ")[0] for i in spread_estimate.index]
    if missing_spreads := (set(instruments) - set(spread_estimate.index)):
        logger.warning(f"Missing spreads for:\n-" + "\n-".join(missing_spreads) +"\n Using 10 BP.")
        for etf in missing_spreads: spread_estimate[etf] = 0.002

    return spread_estimate
