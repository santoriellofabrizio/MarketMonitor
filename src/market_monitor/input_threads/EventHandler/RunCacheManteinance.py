import os
import time

import pandas as pd
import xlwings as xw


def load_cache(cache_path):
    """Carica la cache da disco se esiste, altrimenti restituisce un dizionario vuoto."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            import pickle
            return pickle.load(f)
    return {}


def save_cache(cache, cache_path):
    """Salva la cache su disco."""
    with open(cache_path, "wb") as f:
        import pickle
        pickle.dump(cache, f)

# Percorso della cache
cache_path = r"C:\AFMachineLearning\Projects\Trading\MarketMonitorFI\cachedir\instrument_status.pkl"

# Carica la cache
cache = load_cache(cache_path)

# Crea un DataFrame per visualizzare i dati
visualize = pd.DataFrame(cache).T

wb = xw.Book()

# Scrivi il DataFrame nella prima cella
while True:
    try:
        sheet = wb.sheets[0]
        sheet.range("A1").value = visualize
        break
    except:
        time.sleep(1)
        print("... retrying opening file... ")

input("press any key to save cache")


df = sheet.range("A1").expand().options(pd.DataFrame).value
try:
    cache = df.T.to_dict()
    save_cache(cache, cache_path)
    wb.close()
    print("Cache saved successfully")
except Exception as e:
    print("error occured while saving cache")
    raise KeyboardInterrupt