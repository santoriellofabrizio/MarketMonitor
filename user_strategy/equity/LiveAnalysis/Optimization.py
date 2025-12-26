import time

import cvxpy as cp
import pandas as pd
import xlwings as xw


def linearized_optimization(spreads, lambd, ctv, matrix):
    n = len(ctv.values)  # Numero di variabili

    # Variabili di ottimizzazione
    x = cp.Variable(n)
    t = cp.Variable()  # Variabile per la norma infinito
    u = cp.Variable(n)  # Variabili per la norma 1

    # Vincoli
    constraints = []

    # Vincoli per la norma infinito
    constraints += [matrix.T.values @ (x) <= t]
    constraints += [matrix.T.values @ (x) >= -t]
    constraints += [t >= 0]

    # Vincoli per la norma 1
    constraints += [u >= x - ctv.values]
    constraints += [u >= -(x - ctv.values)]
    constraints += [u >= 0]

    # Funzione obiettivo
    objective = cp.Minimize(t+ lambd * cp.sum(cp.multiply(spreads.values, u)))

    # Problema di ottimizzazione
    problem = cp.Problem(objective , constraints)
    problem.solve(verbose=True)

    print("Termine norma infinito:", t.value)
    # print("Termine regolarizzazione:", lambd * np.sum(spreads.values * u.value))

    # Risultato finale
    hedging_trades = pd.Series(x.value - ctv, index=ctv.index)
    hedging_trades.name = "trades"
    return hedging_trades, problem.value


wb = xw.Book("min.xlsx")
ws = wb.sheets[0]

while True:
    try:
        data = ws.range("c1").expand().options(pd.DataFrame).value
        matrix = ws.range("am5").expand().options(pd.DataFrame).value
        matrix.fillna(0, inplace=True)
        ctv = data["ctv"]
        spread = data["spread"]
        lambd = ws.range("b1").value
        hedging_trades, hed = linearized_optimization(spread, lambd, ctv, matrix)
        ws.range("I1").value = hedging_trades
        time.sleep(3)
    except Exception as e:
        print(e)
        raise KeyboardInterrupt