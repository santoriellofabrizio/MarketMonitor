import argparse
import os
import sys
import warnings

from ruamel.yaml import YAML

from market_monitor.builder import Builder
from market_monitor.utils.config_helpers import (
    list_available,
    resolve_config_entry,
    suggest_names,
)

ENV_DEFAULT_VAR = "MARKET_MONITOR_CONFIG"

reader = YAML(typ='safe')
warnings.simplefilter(action='ignore', category=FutureWarning)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-monitor",
        description=(
            "Launch the market monitor using a config name or alias. "
            "Set the environment variable MARKET_MONITOR_CONFIG to choose a default."
        ),
    )
    parser.add_argument(
        "config_name",
        nargs="?",
        help="Config name or alias (see --list). "
             f"Defaults to ${ENV_DEFAULT_VAR} if set.",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available aliases and config files.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Show details about the resolved config and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate the config file without starting threads.",
    )
    return parser


def _load_config(path):
    with open(path, 'r') as stream:
        return reader.load(stream)


def _print_config_details(entry):
    alias_hint = f"(alias for '{entry.alias_of}')" if entry.alias_of else ""
    print(f"Config: {entry.name} {alias_hint}".strip())
    print(f"Path:   {entry.path}")
    if entry.description:
        print(f"Info:   {entry.description}")


def run_monitor(config=None, argv=None):
    if config is None:
        parser = _build_parser()
        args = parser.parse_args(argv)

        if args.list:
            list_available()
            sys.exit(0)

        config_name = args.config_name or os.environ.get(ENV_DEFAULT_VAR)
        if not config_name:
            parser.print_help()
            print("\nNo config provided. "
                  f"Set {ENV_DEFAULT_VAR} or pass a config name/alias. "
                  "Available options:")
            list_available()
            sys.exit(1)

        try:
            config_entry = resolve_config_entry(config_name)
            if config_entry.alias_of:
                print(f"Using alias '{config_name}' -> {config_entry.alias_of}")
            print(f"Loading config: {config_entry.path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            suggestions = suggest_names(config_name)
            if suggestions:
                print("\nDid you mean:")
                for suggestion in suggestions:
                    print(f"  - {suggestion}")
            print("\nTip: Use 'market-monitor --list' to see available configs")
            sys.exit(1)

        if args.describe:
            _print_config_details(config_entry)
            sys.exit(0)

        config = _load_config(config_entry.path)

        if args.dry_run:
            _print_config_details(config_entry)
            print("\nConfig loaded successfully. Exiting (dry-run).")
            sys.exit(0)

    builder = Builder(config)
    threads, monitor = builder.build()

    try:
        for thread in threads:
            thread.start()
        monitor.start()
        monitor.join()

    except KeyboardInterrupt:
        print("\nMarket monitor interrupted by user")
    finally:
        monitor.stop()
        for thread in threads:
            thread.stop()
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=10.0)


if __name__ == "__main__":
    run_monitor()
