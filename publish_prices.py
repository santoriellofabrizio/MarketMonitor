# scripts/alter_ts_labels.py
import redis
from market_monitor.publishers.timeseries_publisher import TimeSeriesPublisher

def alter_all_ts_labels(host="localhost", port=6380, isin_to_ticker: dict = None):
    """
    Script one-shot: aggiorna labels su tutte le TimeSeries esistenti.
    Legge le label attuali via TS.INFO e aggiunge/sovrascrive quelle mancanti.
    """
    r = redis.Redis(host=host, port=port, decode_responses=True)
    isin_to_ticker = isin_to_ticker or {}

    # Trova tutte le chiavi TS
    keys = list(r.scan_iter(match="ts:*:*", _type="TSDB-TYPE"))
    print(f"Found {len(keys)} TimeSeries keys")

    altered = 0
    errors = 0

    for key in keys:
        try:
            # Leggi info esistenti
            raw = r.execute_command("TS.INFO", key)
            info = dict(zip(raw[::2], raw[1::2]))

            # Estrai labels attuali
            raw_labels = info.get("labels", [])
            current_labels = dict(zip(raw_labels[::2], raw_labels[1::2])) if raw_labels else {}

            # Estrai isin e field dalla chiave (ts:{isin}:{field})
            parts = key.split(":")
            if len(parts) < 3:
                continue
            isin = parts[1]
            field = parts[2]

            # Costruisci nuove label (merge)
            new_labels = {
                "isin": isin,
                "ticker": isin_to_ticker.get(isin, isin),
                **current_labels,  # mantieni esistenti
            }

            # Applica
            r.execute_command(
                "TS.ALTER", key,
                "LABELS", *[item for pair in new_labels.items() for item in pair]
            )
            print(f"  OK {key} -> {new_labels}")
            altered += 1

        except Exception as e:
            print(f"  ERR {key}: {e}")
            errors += 1

    print(f"\nDone: {altered} altered, {errors} errors")


if __name__ == "__main__":
    from market_monitor.config import load_isin_to_ticker  # adatta al tuo config
    isin_to_ticker = load_isin_to_ticker()
    alter_all_ts_labels(host="localhost", port=6380, isin_to_ticker=isin_to_ticker)