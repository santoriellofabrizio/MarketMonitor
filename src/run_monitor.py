import os
import sys
import warnings

from ruamel.yaml import YAML

from market_monitor.builder import Builder

os.environ.setdefault('BBG_ROOT', 'xbbg')

reader = YAML(typ='safe')
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_monitor(config=None):
    if config is None:
        with open(sys.argv[1], 'r') as stream:
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