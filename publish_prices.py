import datetime
import json

import pandas as pd
import numpy as np
import pickle
import time
from market_monitor.publishers.redis_publisher import RedisMessaging

# 1. Inizializzazione
gui = RedisMessaging()
path_to_pickle = r"C:\AFMachineLearning\Libraries\MarketMonitor\src\market_monitor\testing\isin_to_ticker.pkl"

with open(path_to_pickle, "rb") as f:
    isin_data = pickle.load(f)
    isins_800 = list(isin_data.keys())

# 2. Generazione e Pubblicazione DATI STATICI (Storico)
# Supponiamo che return_1 sia il live, mentre da 2 a 8 siano storici
num_static_periods = 7
static_columns = [f"return_{i+1}" for i in range(num_static_periods)]
isins_800 = isins_800
df_static = pd.DataFrame(
    np.random.randn(len(isins_800), num_static_periods) * 0.02,
    index=isins_800,
    columns=static_columns
)

print("Pubblicazione dati storici (statici)...")
for col in df_static.columns:
    channel_name = f"market:{col}"
    # export_static_data invia il dato una volta sola

    gui.export_static_data(**{channel_name: (df_static[col]*100).round(2)})
df_static*=100
df_static=df_static.round(2)
time.sleep(10)
gui.export_static_data(big_df=json.dumps(df_static.to_dict(orient="split")))
# 3. Loop Real-Time (Solo per l'ultimo ritorno)
print("Inizio streaming real-time su market:last_return...")

try:
    while True:
        # Generiamo il ritorno "live" (es. variazione giornaliera)
        # Usiamo Series per avere ISIN come chiavi nel JSON
        import pandas as pd
        import datetime

        # Definiamo la data base di Excel
        excel_base_date = datetime.datetime(1899, 12, 30)
        now = datetime.datetime.now()

        # Calcolo del valore seriale
        # (Differenza totale in giorni tra oggi e la base di Excel)
        excel_serial_now = (now - excel_base_date).total_seconds() / 86400

        last_returns = pd.Series(
            excel_serial_now,
            index=isins_800
        )

        # Pubblicazione sul canale live
        gui.export_message("market:return_0", last_returns)

        print(f"Update RT inviato alle {time.strftime('%H:%M:%S')}")
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopping...")