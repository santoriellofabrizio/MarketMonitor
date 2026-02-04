# MarketMonitor - Deployment Ibrido

## Cos'è l'approccio ibrido?

L'approccio **ibrido** combina i vantaggi di un eseguibile standalone con la flessibilità delle strategie modificabili:

- **EXE standalone**: Core di MarketMonitor bundled con PyInstaller (non serve Python installato)
- **Strategie esterne**: File `.py` caricati dinamicamente (modificabili senza ricompilare)

```
Deployment finale:
├── run-strategy.exe         # Eseguibile (con Python embedded)
├── _internal/               # Librerie bundled
├── user_strategy/           # Strategie .py (MODIFICABILI!)
│   ├── equity/
│   ├── fixed_income/
│   ├── test_strategy/
│   └── utils/
└── etc/config/
    └── my_config.yaml       # Configurazione
```

---

## Vantaggi

✅ **No Python richiesto** sul computer target
✅ **Strategie modificabili** senza ricompilare l'exe
✅ **File più piccolo** (~100-200MB invece di 500MB)
✅ **Facile aggiornamento** strategie (copia nuovi .py)
✅ **Deployment semplice** (zip + unzip)

---

## Build (Computer di sviluppo)

### Prerequisiti
- Python 3.11+
- PyInstaller: `pip install pyinstaller`
- Dipendenze progetto: `pip install -e .`

### Compilazione

**Linux/Mac:**
```bash
./build_hybrid.sh
```

**Windows:**
```cmd
build_hybrid.bat
```

**Opzioni:**
```bash
./build_hybrid.sh --clean  # Pulisce build precedenti
```

### Output

```
deployment/
├── run-strategy/                          # Directory deployment
│   ├── run-strategy.exe                   # Eseguibile
│   ├── _internal/                         # Librerie
│   ├── user_strategy/                     # Strategie
│   ├── etc/config/                        # Configs
│   └── README_DEPLOYMENT.txt
└── MarketMonitor-hybrid-YYYYMMDD.zip      # Archivio distribuibile
```

---

## Deployment (Computer target)

### 1. Copia i file

**Opzione A - Da archivio:**
```bash
# Estrai archivio
unzip MarketMonitor-hybrid-YYYYMMDD.zip
cd run-strategy/
```

**Opzione B - Manuale:**
```bash
# Copia la directory deployment/run-strategy/
cp -r deployment/run-strategy/ /path/to/deployment/
cd /path/to/deployment/run-strategy/
```

### 2. Configura

Modifica `etc/config/config_hybrid_deployment_example.yaml`:

```yaml
load_strategy_info:
  # Path RELATIVO alla strategia
  package_path: ./user_strategy/test_strategy

  # File Python (senza .py)
  module_name: SimplePriceMonitorStrategy

  # Classe da caricare
  class_name: SimplePriceMonitorStrategy

market_monitor:
  # ... parametri strategia ...

redis_data_distributor:
  activate: true
  redis_params:
    redis_host: 'localhost'
    redis_port: 6379
```

### 3. Esegui

**Linux:**
```bash
./run-strategy etc/config/config_hybrid_deployment_example.yaml
```

**Windows:**
```cmd
run-strategy.exe etc\config\config_hybrid_deployment_example.yaml
```

**Con alias config:**
```bash
# Rinomina config_hybrid_deployment_example.yaml -> my_strategy.yaml
./run-strategy my_strategy
```

---

## Modificare le strategie

Le strategie sono file `.py` modificabili! Basta editare:

```bash
# Modifica strategia esistente
vim user_strategy/test_strategy/SimplePriceMonitorStrategy.py

# Aggiungi nuova strategia
cp user_strategy/test_strategy/SimplePriceMonitorStrategy.py \
   user_strategy/test_strategy/MyNewStrategy.py

# Modifica il config per usare la nuova strategia
# load_strategy_info:
#   module_name: MyNewStrategy
#   class_name: MyNewStrategy
```

**Non serve ricompilare l'exe!** Le modifiche sono caricate al prossimo avvio.

---

## Strategie disponibili

### Test Strategies
- `user_strategy/test_strategy/SimplePriceMonitorStrategy.py`
- `user_strategy/test_strategy/TestTradeManagerStrategy.py`
- `user_strategy/test_strategy/PriceSpreadAnalyzerStrategy.py`
- `user_strategy/test_strategy/TradeAccumulatorStrategy.py`

### Equity Strategies
- `user_strategy/equity/LiveAnalysis/`
- `user_strategy/equity/LiveQuoting/`
- `user_strategy/equity/NavChecking/`

### Fixed Income Strategies
- `user_strategy/fixed_income/EtfFiStrategy.py`

---

## Troubleshooting

### Import errors

**Problema:** `ModuleNotFoundError: No module named 'user_strategy'`

**Soluzione:** Verifica che:
1. La cartella `user_strategy/` sia accanto all'exe
2. Il config usi path relativi: `./user_strategy/...` o `user_strategy/...`

### Path assoluti nei config

**Problema:** Config con path Windows assoluti (`C:\...`)

**Soluzione:** Converti in path relativi:
```yaml
# PRIMA (non funziona):
package_path: C:\AFMachineLearning\Libraries\MarketMonitor\user_strategy\equity

# DOPO (funziona):
package_path: ./user_strategy/equity
```

### Bloomberg non funziona

**Problema:** `blpapi` not found

**Soluzione:** Bloomberg API va installato separatamente:
```bash
# Windows
pip install blpapi

# Linux (richiede compilazione)
# Segui docs Bloomberg
```

### Logs

I log sono in `logs/market_monitor_hybrid.log`:
```bash
tail -f logs/market_monitor_hybrid.log
```

---

## Aggiornamento strategie

Per aggiornare solo le strategie (senza ricompilare l'exe):

```bash
# Computer di sviluppo: crea package strategie
cd MarketMonitor/
tar -czf user_strategy_update.tar.gz user_strategy/

# Computer target: applica update
tar -xzf user_strategy_update.tar.gz
# Restart run-strategy.exe
```

---

## Aggiornamento completo

Per aggiornare anche il core MarketMonitor:

```bash
# Ricompila su dev machine
./build_hybrid.sh --clean

# Deploy su target
# (sostituisci tutto tranne user_strategy/ se hai modifiche custom)
```

---

## Limitazioni

- ❌ **Librerie native**: Bloomberg blpapi richiede installazione separata
- ❌ **OS-specific**: Build Windows funziona solo su Windows (e viceversa)
- ⏱️ **Startup lento**: ~2-5 secondi per avviare (Python embedded)
- 📦 **Dimensioni**: ~100-200MB (vs ~10MB pip package)

---

## Confronto con altre soluzioni

| Approccio | Pro | Contro | Caso d'uso |
|-----------|-----|--------|------------|
| **Ibrido (questo)** | No Python richiesto, strategie modificabili | Build per ogni OS, librerie native separate | **Distribuzione interna/demo** |
| **pip install** | Leggero, standard Python | Richiede Python + accesso NAS | Sviluppo, deployment con infra |
| **Docker** | Massima portabilità | Richiede Docker | Produzione cloud/server |

---

## FAQ

**Q: Posso usare strategie custom?**
A: Sì! Basta aggiungere i file `.py` in `user_strategy/` e configurare il YAML.

**Q: Funziona senza accesso al NAS?**
A: Sì, se le dipendenze `sfm-*` sono state bundled in fase di build.

**Q: Posso distribuire via USB?**
A: Sì, copia `run-strategy/` su USB e funziona out-of-the-box.

**Q: Posso avere più config per strategie diverse?**
A: Sì, crea più file YAML in `etc/config/` e carica quello che vuoi.

---

## Supporto

Per problemi o domande:
1. Controlla `logs/market_monitor_hybrid.log`
2. Verifica path relativi nel config
3. Testa con strategia di esempio (`SimplePriceMonitorStrategy`)

---

**Buon deployment! 🚀**
