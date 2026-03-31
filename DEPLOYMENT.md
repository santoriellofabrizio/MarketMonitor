# MarketMonitor — Guida al Deployment

## Indice

1. [Architettura](#1-architettura)
2. [Struttura della cartella di deployment](#2-struttura-della-cartella-di-deployment)
3. [Come buildare gli eseguibili](#3-come-buildare-gli-eseguibili)
4. [Come avviare le strategie](#4-come-avviare-le-strategie)
5. [Come aggiungere una libreria interna](#5-come-aggiungere-una-libreria-interna)
6. [Come aggiungere una strategia](#6-come-aggiungere-una-strategia)
7. [Come collegare il Control Panel](#7-come-collegare-il-control-panel)

---

## 1. Architettura

Il sistema è composto da tre eseguibili indipendenti che comunicano via **Redis**:

```
  run-control-panel.exe
  ┌──────────────────────────────────────────┐
  │  GUI PyQt5                               │
  │  - Seleziona/avvia strategie             │
  │  - Invia comandi (es. "reload data")     │
  │  - Mostra status e log in real-time      │
  └──────────┬─────────────────┬─────────────┘
             │ QProcess        │ Redis pub/sub
             │ (lancia)        │ (engine:commands / engine:status)
             ▼                 ▼
  run-strategy.exe          Redis
  ┌──────────────────┐       ┌─────────┐
  │  Framework core  │◄─────►│         │
  │  (bundlato)      │       │ pub/sub │
  │  + strategia     │       │         │
  │  esterna (.py)   │       └─────────┘
  └──────────────────┘
```

- **`run-strategy.exe`** — esegue la logica di business. ONEDIR (startup immediato).
- **`run-control-panel.exe`** — GUI per avviare/fermare/comandare strategie. ONEFILE.
- **`run-dashboard.exe`** — viewer dati standalone. ONEFILE.

Le **strategie** e le **configurazioni YAML** sono file esterni (non nel bundle), copiati a mano nella cartella di deployment.

---

## 2. Struttura della cartella di deployment

```
MarketMonitor_Deploy\
├── run-strategy.exe           ← avviato da Control Panel (QProcess) o da CLI
├── run-control-panel.exe      ← avviato dall'utente; lancia run-strategy
├── run-dashboard.exe          ← viewer dati standalone
├── _internal\                 ← librerie Python di run-strategy (non toccare)
├── user_strategy\             ← strategie .py (copiate a mano, vedi §6)
│   ├── equity\
│   │   └── MyStrategy\
│   │       └── MyStrategy.py
│   └── utils\
└── etc\
    └── config\                ← file YAML (copiati a mano)
        ├── aliases.yaml       ← AGGIORNARE i path dopo deployment!
        └── configMyStrategy.yaml
```

> **Attenzione**: dopo aver copiato la cartella in produzione, aggiornare tutti i path assoluti nei file YAML (in particolare `aliases.yaml` e `load_strategy_info.package_path`).

---

## 3. Come buildare gli eseguibili

### Prerequisiti

- `uv` installato ([docs.astral.sh/uv](https://docs.astral.sh/uv/))
- Accesso al NAS (`\\nas1bes\AreaFinanza\...`) per `uv sync`
- Python 3.11+

### Build

```bat
cd C:\path\al\progetto\MarketMonitor
build_exe.bat
```

Lo script esegue in sequenza:

| Step | Azione |
|------|--------|
| `0/4` | `uv sync` — installa tutte le dipendenze nel venv |
| `1/4` | `uv run pyinstaller run_strategy.spec` — build run-strategy |
| `2/4` | `uv run pyinstaller run_control_panel.spec` — build run-control-panel |
| `3/4` | `uv run pyinstaller run_dashboard.spec` — build run-dashboard |
| `4/4` | Assembla `dist\MarketMonitor_Deploy\` |

> **Perché `uv run pyinstaller` e non `pyinstaller` diretto?**
> PyInstaller deve girare nel venv uv per trovare le librerie interne (sfm_*).
> La chiamata raw `pyinstaller` userebbe il Python di sistema, dove queste
> librerie non sono installate, e le ometterebbe silenziosamente dal bundle.

---

## 4. Come avviare le strategie

### Tramite Control Panel (consigliato)

```bat
run-control-panel.exe
run-control-panel.exe --config configNomeStrategia
```

Il pannello mostra la lista delle strategie configurate. Premere **▶ Start** per avviarle — il control panel lancia `run-strategy.exe` come sottoprocesso nella stessa cartella.

### Da CLI

```bat
run-strategy.exe --list                          ← lista configs disponibili
run-strategy.exe configNomeStrategia             ← per nome
run-strategy.exe C:\path\assoluto\config.yaml    ← per path
```

---

## 5. Come aggiungere una libreria interna

Le librerie interne (pacchetti `sfm_*`) sono wheel file sul NAS e vengono gestite da uv.

### Step 1 — `pyproject.toml`

Aggiungere la libreria nella sezione `[project.dependencies]`:

```toml
[project.dependencies]
...
"sfm-nome-libreria>=1.0.12345",
```

Il NAS è già configurato come sorgente in `[tool.uv]`:

```toml
[tool.uv]
no-index = true
find-links = ['\\nas1bes\AreaFinanza\SHARE\PROGETTI ML\Python\python3_11\python packages']
```

### Step 2 — Aggiornare `uv.lock`

Con accesso al NAS attivo:

```bat
uv lock
```

Questo risolve la versione esatta del pacchetto e aggiorna `uv.lock`.
Committare `uv.lock` insieme a `pyproject.toml`.

### Step 3 — `.spec` PyInstaller

Per ogni libreria che contiene moduli caricati dinamicamente (es. sottomoduli lazy), aggiungere in `run_strategy.spec` e `run_control_panel.spec`:

```python
# Sezione 0 (collect_all)
datas_mia_lib, binaries_mia_lib, hiddenimports_mia_lib = collect_all('sfm_nome_modulo')

# Sezione hiddenimports
*hiddenimports_mia_lib,

# Sezione datas
*datas_mia_lib,

# Sezione binaries
*binaries_mia_lib,
```

Le librerie `sfm_datalibrary` e `sfm_data_provider` sono già configurate come riferimento.

### Step 4 — Rebuild

```bat
build_exe.bat
```

---

## 6. Come aggiungere una strategia

Le strategie sono file `.py` esterni al bundle — **non è necessario ricompilare l'exe**.

### Struttura della classe

Una strategia estende `StrategyUI` (sincrona) o `StrategyUIAsync` (asincrona):

```python
# user_strategy/equity/MyStrategy/MyStrategy.py
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI
import pandas as pd

class MyStrategy(StrategyUI):

    def __init__(self, instruments: list, **kwargs):
        super().__init__(**kwargs)
        self.instruments = instruments

    def on_market_data_setting(self):
        """Chiamato una volta all'avvio: imposta i ticker da sottoscrivere."""
        self.market_data.securities = self.instruments

    def update_HF(self):
        """Update ad alta frequenza (es. ogni 2 secondi).
        Ritorna dati da scrivere su Excel/Redis, oppure None."""
        # accesso ai dati: self.market_data.get("ISIN", "LAST_PRICE")
        pass

    def update_LF(self):
        """Update a bassa frequenza (es. ogni 60 secondi)."""
        pass

    def on_trade(self, trades):
        """Callback alla ricezione di un trade."""
        pass
```

### File YAML di configurazione

```yaml
# etc/config/configMyStrategy.yaml

load_strategy_info:
  # Path ASSOLUTO alla cartella che contiene la strategia
  package_path: C:\Deployment\user_strategy\equity\MyStrategy
  module_name: MyStrategy      # nome del file .py (senza estensione)
  class_name: MyStrategy       # nome della classe

market_monitor:
  instruments: [ISIN1, ISIN2]

  tasks:
    update_HF:
      activate: true
      frequency: 2       # secondi
    update_LF:
      activate: true
      frequency: 60
    trade:
      activate: true
      synchronous: true
      frequency: 1
    command_listener:    # necessario per il Control Panel
      activate: true
      frequency: 3

  redis_data_export:
    activate: true
    channel_redis: my_strategy_trades
    redis_params:
      redis_host: localhost
      redis_port: 6379
      redis_db: 0

logging:
  log_level: INFO
  log_level_console: WARNING
  log_name: MyStrategy.log
```

### Deployment della strategia

1. Copiare `MyStrategy.py` (e file helper) in `MarketMonitor_Deploy\user_strategy\equity\MyStrategy\`
2. Copiare `configMyStrategy.yaml` in `MarketMonitor_Deploy\etc\config\`
3. Aggiornare `package_path` nel YAML con il path assoluto sul computer target
4. Avviare con `run-control-panel.exe` oppure `run-strategy.exe configMyStrategy`

---

## 7. Come collegare il Control Panel

Il Control Panel è **completamente disaccoppiato** dalla strategia: comunica solo via Redis.

### Configurazione minima nel YAML

```yaml
market_monitor:
  tasks:
    command_listener:   # OBBLIGATORIO per ricevere comandi dal panel
      activate: true
      frequency: 3
```

### Aggiungere comandi personalizzati

Nel YAML, aggiungere la sezione `control_panel`:

```yaml
control_panel:
  title: "My Strategy Panel"
  commands:
    - label: "Reload Data"
      action: "reload_data"
      payload: {}
      description: "Ricarica i dati dal database"

    - label: "Set Alpha"
      action: "set_alpha"
      type: "float_input"
      min: 0.0
      max: 1.0
      default: 1.0
      step: 0.05
      description: "Peso del modello nel pricing"
```

### Implementare gli handler nella strategia

```python
class MyStrategy(StrategyUI):

    def handle_command(self, command: dict):
        """Riceve i comandi dal Control Panel via Redis (engine:commands)."""
        action = command.get("action")
        payload = command.get("payload", {})

        if action == "reload_data":
            self._reload_data()
            return {"status": "ok", "message": "Dati ricaricati"}

        if action == "set_alpha":
            self.alpha = payload.get("value", 1.0)
            return {"status": "ok", "alpha": self.alpha}
```

### Avvio

Il Control Panel e la strategia sono **processi separati**. Li si può avviare in qualsiasi ordine:

```bat
:: Opzione 1: avviare tutto dal Control Panel
run-control-panel.exe --config configMyStrategy

:: Opzione 2: avviare separatamente (Redis deve essere attivo)
run-strategy.exe configMyStrategy          ← finestra 1
run-control-panel.exe                      ← finestra 2
```

Il panel si connette automaticamente alla strategia in esecuzione tramite i canali Redis configurati (default: `engine:commands`, `engine:status`, `engine:lifecycle`).
