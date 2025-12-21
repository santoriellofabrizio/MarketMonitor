import redis
import json
import time
import random

# Connetti a Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

print("Pubblicando messaggi su Redis...")
print("Premi Ctrl+C per fermare\n")

count = 0
start = time.time()

try:
    while True:
        # 10 canali diversi
        for ch in range(1, 2000):
            bid = round(1.2000 + random.uniform(0, 0.1), 4)

            data = {
                "bid": bid,
                "ask": round(bid + 0.0002, 4),
                "last": round(bid + 0.0001, 4),
                "volume": random.randint(1000, 10000),
                "high": round(bid + 0.001, 4),
                "low": round(bid - 0.001, 4)
            }

            r.publish(f'prices{ch}', json.dumps(data))
            count += 1

        # Progress ogni 100 messaggi
        if count % 100 == 0:
            elapsed = time.time() - start
            rate = count / elapsed
            print(f"\r{count} messaggi inviati ({rate:.1f} msg/sec)", end='', flush=True)

        time.sleep(1)  # 50 msg/sec

except KeyboardInterrupt:
    elapsed = time.time() - start
    print(f"\n\nFinito!")
    print(f"Durata: {elapsed:.1f} secondi")
    print(f"Messaggi: {count}")
    print(f"Rate: {count / elapsed:.1f} msg/sec")