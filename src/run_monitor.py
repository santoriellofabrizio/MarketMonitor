import os
import sys
import warnings

from ruamel.yaml import YAML

from market_monitor.builder import Builder
from market_monitor.utils.config_helpers import resolve_config, list_available

reader = YAML(typ='safe')
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_monitor(config=None):
    if config is None:
        if len(sys.argv) < 2:
            print("Usage: market-monitor <config_name|alias>")
            print("\nOptions:")
            print("  -l, --list    List available aliases and configs")
            print("\nExamples:")
            print("  market-monitor prod")
            print("  market-monitor p              # shorthand for prod")
            print("  market-monitor production.yaml")
            print("  market-monitor --list")
            sys.exit(1)

        # Check for --list or -l flag
        if sys.argv[1] in ['--list', '-l']:
            list_available()
            sys.exit(0)

        config_name = sys.argv[1]

        try:
            config_path = resolve_config(config_name)
            print(f"Loading config: {config_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nTip: Use 'market-monitor --list' to see available configs")
            sys.exit(1)

        with open(config_path, 'r') as stream:
            config = reader.load(stream)

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
