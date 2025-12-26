"""
Logica comune per tutti i CLI entry points.

Questo modulo contiene:
- Setup environment e logging
- Parsing argomenti
- Gestione configurazione (con selezione interattiva)
- Shutdown threads
"""
import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ruamel.yaml import YAML

from market_monitor.entry import get_project_root
from market_monitor.utils.config_helpers import (
    ENV_DEFAULT_VAR,
    available_names,
    list_available,
    load_config,
    print_config_details,
    resolve_config_entry,
    suggest_names,
    load_aliases,
    _unique_configs,
    _read_description,
)

warnings.simplefilter(action='ignore', category=FutureWarning)
_reader = YAML(typ='safe')


# =============================================================================
# INTERACTIVE CONFIG SELECTION
# =============================================================================

def _get_config_choices() -> List[Dict[str, str]]:
    """
    Costruisce la lista di scelte per il menu interattivo.
    
    Returns:
        Lista di dict con 'name' (display) e 'value' (config name)
    """
    choices = []
    aliases = load_aliases()
    configs = _unique_configs()
    
    # Prima gli alias (sono shortcut comuni)
    for alias, config_name in sorted(aliases.items()):
        desc = _read_description(configs.get(config_name))
        label = f"[alias] {alias} → {config_name}"
        if desc:
            label += f"  ({desc})"
        choices.append({"name": label, "value": alias})
    
    # Poi le config dirette
    for name, path in sorted(configs.items()):
        # Salta se già presente come alias target
        if name in aliases.values():
            continue
        desc = _read_description(path)
        label = name
        if desc:
            label += f"  ({desc})"
        choices.append({"name": label, "value": name})
    
    return choices


def select_config_interactive() -> str:
    """
    Mostra un menu interattivo per selezionare la configurazione.
    
    Returns:
        Nome della config selezionata
    
    Raises:
        SystemExit: Se l'utente annulla o non ci sono config
    """
    try:
        import questionary
        from questionary import Style
    except ImportError:
        print("❌ questionary non installato. Installa con: pip install questionary")
        print("\nConfigurazioni disponibili:")
        list_available()
        sys.exit(1)
    
    choices = _get_config_choices()
    
    if not choices:
        print("❌ Nessuna configurazione trovata.")
        sys.exit(1)
    
    # Stile personalizzato
    custom_style = Style([
        ('qmark', 'fg:cyan bold'),
        ('question', 'fg:white bold'),
        ('answer', 'fg:green bold'),
        ('pointer', 'fg:cyan bold'),
        ('highlighted', 'fg:cyan bold'),
        ('selected', 'fg:green'),
    ])
    
    # Menu con autocomplete/fuzzy search
    result = questionary.select(
        "Seleziona configurazione:",
        choices=[c["name"] for c in choices],
        style=custom_style,
        use_shortcuts=True,
        use_arrow_keys=True,
        use_jk_keys=True,
    ).ask()
    
    if result is None:
        print("\n❌ Selezione annullata.")
        sys.exit(1)
    
    # Trova il value corrispondente
    for c in choices:
        if c["name"] == result:
            return c["value"]
    
    return result


def select_config_with_autocomplete() -> str:
    """
    Mostra un prompt con autocomplete per la configurazione.
    
    Returns:
        Nome della config selezionata
    """
    try:
        import questionary
    except ImportError:
        return select_config_interactive()
    
    names = available_names()
    
    if not names:
        print("❌ Nessuna configurazione trovata.")
        sys.exit(1)
    
    result = questionary.autocomplete(
        "Config (Tab per suggerimenti):",
        choices=names,
        match_middle=True,
        validate=lambda x: x in names or "Config non valida",
    ).ask()
    
    if result is None:
        print("\n❌ Selezione annullata.")
        sys.exit(1)
    
    return result


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def build_parser(
    description: str = "MarketMonitor CLI",
    extra_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
) -> argparse.ArgumentParser:
    """
    Costruisce il parser con gli argomenti standard e opzionalmente extra.
    
    Args:
        description: Descrizione del comando
        extra_args: Funzione che aggiunge argomenti extra al parser
    
    Returns:
        ArgumentParser configurato
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "config_name",
        nargs="?",
        help=f"Config name, alias, o path. Se omesso: menu interattivo",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="Lista config e alias disponibili",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Forza selezione interattiva",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Mostra dettagli config ed esci",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida config senza avviare",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Output verboso",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Livello di logging (default: INFO)",
    )
    
    if extra_args:
        extra_args(parser)
    
    return parser


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    name: str = "market_monitor",
) -> logging.Logger:
    """
    Configura il logging per l'applicazione.
    
    Args:
        level: Livello di logging
        log_file: Path opzionale per file di log
        name: Nome del logger
    
    Returns:
        Logger configurato
    """
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='w'))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )
    
    return logging.getLogger(name)


def resolve_config(
    config: Optional[Union[str, dict, Path]],
    args: Optional[argparse.Namespace] = None,
    interactive: bool = True,
) -> dict:
    """
    Risolve la configurazione da varie fonti.
    
    Args:
        config: Può essere:
            - None: usa args.config_name, env var, o selezione interattiva
            - str/Path: path al file YAML
            - dict: config già caricata
        args: Namespace da argparse (usato se config è None)
        interactive: Se True, mostra menu interattivo quando manca config
    
    Returns:
        Dizionario di configurazione
    
    Raises:
        SystemExit: Se config non trovata o comandi informativi
    """
    # Config già come dict
    if isinstance(config, dict):
        return config
    
    # Config come path diretto
    if isinstance(config, (str, Path)):
        path = Path(config)
        if path.suffix in ('.yaml', '.yml') and path.exists():
            return load_config(path)
    
    # Risolvi da nome/alias
    config_name = config if isinstance(config, str) else None
    
    if config_name is None and args:
        config_name = args.config_name
    
    if config_name is None:
        config_name = os.environ.get(ENV_DEFAULT_VAR)
    
    # Forza interattivo se richiesto
    force_interactive = args and hasattr(args, 'interactive') and args.interactive
    
    # Se ancora nessuna config, prova selezione interattiva
    if not config_name:
        if interactive or force_interactive:
            if sys.stdin.isatty():
                print(f"💡 Nessuna config specificata. Avvio selezione interattiva...\n")
                config_name = select_config_interactive()
            else:
                # Non interattivo (es. pipe, script)
                print(f"❌ Nessuna configurazione specificata.")
                print(f"   Usa: comando <config_name> oppure imposta ${ENV_DEFAULT_VAR}")
                print("\nConfigurazioni disponibili:")
                list_available()
                sys.exit(1)
        else:
            print(f"❌ Nessuna configurazione specificata.")
            list_available()
            sys.exit(1)
    
    try:
        entry = resolve_config_entry(config_name)
        print(f"📁 Config: {entry.path}")
        print(f"📂 Working dir: {get_project_root()}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        if suggestions := suggest_names(config_name):
            print("\nForse intendevi:")
            for s in suggestions:
                print(f"  - {s}")
        sys.exit(1)
    
    # Gestione comandi informativi
    if args and args.describe:
        print_config_details(entry)
        sys.exit(0)
    
    loaded_config = load_config(entry.path)
    
    if args and args.dry_run:
        print_config_details(entry)
        print("\n✅ Config validata con successo (dry-run).")
        sys.exit(0)
    
    return loaded_config


def handle_list_command(args: argparse.Namespace) -> None:
    """Gestisce il comando --list."""
    if args.list:
        list_available()
        sys.exit(0)


# =============================================================================
# LIFECYCLE MANAGEMENT
# =============================================================================

def shutdown_threads(
    threads: List[Any],
    monitor: Optional[Any] = None,
    timeout: float = 5.0,
) -> None:
    """
    Shutdown pulito di threads e monitor.
    
    Args:
        threads: Lista di thread da fermare
        monitor: Monitor principale (opzionale)
        timeout: Timeout per join dei thread
    """
    print("\n[SHUTDOWN] Chiusura in corso...")
    
    # Stop monitor
    if monitor and hasattr(monitor, 'stop'):
        try:
            monitor.stop()
        except Exception as e:
            print(f"  ⚠️  Errore stop monitor: {e}")
    
    # Stop threads
    for t in threads:
        if hasattr(t, 'stop'):
            try:
                t.stop()
            except Exception as e:
                print(f"  ⚠️  Errore stop thread {t}: {e}")
    
    # Join threads
    all_threads = threads + ([monitor] if monitor else [])
    for t in all_threads:
        if t and hasattr(t, 'is_alive') and t.is_alive():
            try:
                t.join(timeout=timeout)
            except Exception as e:
                print(f"  ⚠️  Errore join thread {t}: {e}")
    
    print("✅ Shutdown completato.")


# =============================================================================
# RUNNER BASE
# =============================================================================

class BaseRunner:
    """
    Classe base per i runner CLI.
    
    Fornisce logica comune per:
    - Parsing argomenti
    - Setup logging
    - Risoluzione config (con selezione interattiva)
    - Lifecycle management
    """
    
    name: str = "base"
    description: str = "MarketMonitor CLI"
    log_file: Optional[str] = None
    
    def __init__(self):
        self.logger: Optional[logging.Logger] = None
        self.config: Optional[dict] = None
        self.args: Optional[argparse.Namespace] = None
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Override per aggiungere argomenti specifici."""
        pass
    
    def setup(self) -> None:
        """Override per setup specifico prima dell'esecuzione."""
        pass
    
    def run(self) -> None:
        """Override con la logica principale di esecuzione."""
        raise NotImplementedError("Subclass must implement run()")
    
    def cleanup(self) -> None:
        """Override per cleanup specifico."""
        pass
    
    def execute(
        self,
        config: Optional[Union[str, dict, Path]] = None,
        argv: Optional[List[str]] = None,
    ) -> None:
        """
        Esegue il runner.
        
        Args:
            config: Config diretta (path, dict, o nome)
            argv: Argomenti CLI (default: sys.argv[1:])
        """
        # Parse arguments
        parser = build_parser(
            description=self.description,
            extra_args=self.add_arguments,
        )
        self.args = parser.parse_args(argv)
        
        # Handle --list
        handle_list_command(self.args)
        
        # Setup logging
        log_level = self.args.log_level if hasattr(self.args, 'log_level') else "INFO"
        self.logger = setup_logging(
            level=log_level,
            log_file=self.log_file,
            name=f"market_monitor.{self.name}",
        )
        
        # Resolve config (con supporto interattivo)
        self.config = resolve_config(config, self.args, interactive=True)
        
        # Setup specifico
        self.setup()
        
        # Run
        try:
            self.run()
        except KeyboardInterrupt:
            print("\n⚠️  Interruzione manuale.")
        except Exception as e:
            self.logger.exception(f"Errore fatale: {e}")
            raise
        finally:
            self.cleanup()
