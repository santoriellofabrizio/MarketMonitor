import os
import sys
import warnings

from ruamel.yaml import YAML

from market_monitor.Builder import Builder

os.environ.setdefault('BBG_ROOT', 'xbbg')

# Ignora gli avvisi FutureWarning
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
    except KeyboardInterrupt:
        print("Interrupted by user")
        for thread in threads:
            thread.stop()
        monitor.stop()


if __name__ == "__main__":
    run_monitor()
