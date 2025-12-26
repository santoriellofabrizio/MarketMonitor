import logging
from datetime import datetime

import cvxpy as cp
import numpy as np
import pandas as pd
from numpy import linalg
from scipy.optimize import minimize, NonlinearConstraint


logger = logging.getLogger()

def linearized_optimization(spreads, lambd, ctv, matrix):

    start = datetime.now()
    n = len(ctv.values)

    x = cp.Variable(n, name="x")
    t = cp.Variable(name="t")
    u = cp.Variable(n, name="u")

    constraints = [
        matrix.values.T @ x <= t,
        matrix.values.T @ x >= -t,
        t >= 0,
        u >= x - ctv.values,
        u >= -(x - ctv.values),
        u >= 0,
    ]

    objective = cp.Minimize(t + lambd * spreads.values @ u)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, verbose=False)
    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        hedging_trades = pd.Series(x.value - ctv.values, index=ctv.index)
        hedging_trades.name = "trades"
        logger.info(f"optimization successful: {(datetime.now() - start).microseconds/1e6:.3f}s")
        return hedging_trades, problem.value
    else:
        return None


def non_convex_optimization(spreads, lambd, ctv, matrix, max_number_of_trades):
    spreads, ctv = spreads.align(ctv, join='inner')

    number_of_trades = lambda x: linalg.norm(x - ctv, 0) - max_number_of_trades
    spread_penalty = lambda x: spreads * lambd @ np.abs(x - ctv)
    objective = lambda x:  np.max(matrix.T @ x) + spread_penalty(x)

    constraints = {"type": "ineq", "fun": number_of_trades}
    x0 = np.zeros_like(ctv.values)
    minimization = minimize(objective, x0, method='SLSQP', constraints= constraints)
    return minimization.x, minimization.value


