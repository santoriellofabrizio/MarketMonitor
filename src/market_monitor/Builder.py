import asyncio
import importlib

import logging
import os
import sys
from importlib import util
from queue import Queue
from threading import Lock

from sfm_dbconnections.DbConnectionParameters import DbConnectionParameters, OracleConnectionParameters, \
    TimescaleConnectionParameters

from market_monitor.gui.implementations.GUI import GUIDummy
from market_monitor.gui.threaded_GUI.GUIQueue import GUIQueue
from market_monitor.gui.threaded_GUI.QueueDataSource import QueueDataSource
from market_monitor.gui.threaded_GUI.ThreadGUIExcel import ThreadGUIExcel
from market_monitor.gui.threaded_GUI.TradeThreadGUITkinter import TradeThreadGUITkinter
from market_monitor.input_threads.event_handler.BBGEventHandler import BBGEventHandler
from market_monitor.input_threads.trade import TradeThread
from market_monitor.input_threads.bloomberg import BloombergStreamingThread
from market_monitor.input_threads.excel import ExcelStreamingThread
from market_monitor.input_threads.redis import RedisStreamingThread
from market_monitor.live_data_hub.real_time_data_hub import RTData


class Builder:

    def __init__(self, config):
        self.config = config

    def build(self):
        logger = self._setup_logging()
        if "oracle_connection" in self.config:
            self._setup_oracle_connection()
        if "timescale_connection" in self.config:
            self._setup_timescale_connection()

        # Lettura dei parametri principali
        lock = Lock()
        strategy = self._load_strategy_from_metadata()
        user_strategy = strategy(**self.config["market_monitor"])
        threads = []

        self._set_real_time_data(user_strategy, lock)

        if self.config.get("trade_distributor", {}).get("activate", False):
            self._setup_trade_distributor(threads, user_strategy, lock, logger)
        if self.config.get("market_data_distributor", {}).get("activate", False):
            self._setup_market_distributor(threads, user_strategy)
        if self.config.get("excel_data_distributor", {}).get("activate", False):
            self._setup_excel_distributor(threads, user_strategy)
        if self.config.get("redis_data_distributor", {}).get("activate", False):
            self._setup_redis_distributor(threads, user_strategy)

        self._setup_gui(threads, user_strategy)

        return threads, user_strategy

    def _setup_logging(self):
        log_level = self.config['logging'].get('log_level', 'DEBUG').upper()
        log_dir = os.path.join("logging", self.config['logging']['log_name'])

        os.makedirs(os.path.dirname(log_dir), exist_ok=True)

        logging.basicConfig(level=log_level,
                            filename=log_dir,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            filemode='w')

        logger = logging.getLogger()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config['logging'].get('log_level_console', 'DEBUG'))
        logger.addHandler(console_handler)

        return logger

    def _load_strategy_from_metadata(self):
        """
        Carica dinamicamente una classe da un file specifico senza
        basarsi sul sistema di import standard di Python.
        """
        load_strategy_info = self.config["load_strategy_info"]
        package_path = load_strategy_info["package_path"]
        module_name = load_strategy_info["module_name"]
        class_name = load_strategy_info["class_name"]
        file_path = os.path.join(package_path, f"{module_name}.py")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File non trovato: {file_path}")

        try:
            spec = util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)

            if package_path not in sys.path:
                sys.path.insert(0, package_path)

            spec.loader.exec_module(module)
            strategy_class = getattr(module, class_name)

            logging.info(f"Classe {class_name} caricata con successo da {file_path}")
            return strategy_class

        except Exception as e:
            logging.error(f"Errore durante il caricamento dinamico della strategia: {e}")
            raise

    def _set_real_time_data(self, monitor, lock):
        market_data = RTData(lock, **self.config.get("market_data_distributor",{}).get("book_params",{}))
        monitor.set_market_data(market_data)

    def _setup_trade_distributor(self, threads, monitor, lock, logger):
        q_trade = Queue() if self.config["market_monitor"]["tasks"]["trade"]["synchronous"] else asyncio.Queue()
        monitor.set_q_trade(q_trade)
        path = self.config["trade_distributor"]["path"]
        db_name = self.config["trade_distributor"]["db_name"]
        trade_thread = TradeThread(lock, q_trade, path=path, db_name=db_name)
        threads.append(trade_thread)

    def _setup_market_distributor(self, threads, monitor):
        market_data = monitor.market_data
        event_handler = BBGEventHandler(market_data)
        price_distributor_thread = BloombergStreamingThread(event_handler, **self.config["market_data_distributor"]
                                                                                        ["bloomberg_params"])
        threads.append(price_distributor_thread)

    def _setup_excel_distributor(self, threads, monitor):
        excel_distributor_thread = ExcelStreamingThread(**self.config["excel_data_distributor"]["excel_params"])
        excel_distributor_thread.set_real_time_data(monitor.market_data)
        threads.append(excel_distributor_thread)

    def _setup_redis_distributor(self, threads, monitor):
        redis_distributor_thread = RedisStreamingThread(**self.config["redis_data_distributor"]["redis_params"])
        redis_distributor_thread.set_real_time_data(monitor.market_data)
        threads.append(redis_distributor_thread)

    def _setup_gui(self, threads, monitor):
        gui = None
        gui_config = self.config.get("gui", {})
        for gui_name, params in gui_config.items():
            match params.pop("gui_type"):
                case "GUIDummy":
                    gui = GUIDummy()
                case 'TradeThreadGUITkinter':
                    q_gui = Queue(gui_config.get("thread_queue_size", 5))
                    gui = GUIQueue(q_gui)
                    threads.append(TradeThreadGUITkinter(data_source=QueueDataSource(q_gui), **params))
                case "ThreadGUIExcel":
                    q_gui = Queue(gui_config.get("thread_queue_size", 5))
                    gui = GUIQueue(q_gui)
                    threads.append(ThreadGUIExcel(queue=q_gui, **params))
                case _:
                    raise ValueError(f"Unknown GUI type: {params.get('gui_type')}")
            monitor.set_gui(gui_name, gui)

    def _setup_oracle_connection(self):
        db_connection_parameters = DbConnectionParameters
        db_connection_parameters.set_oracle_parameter(OracleConnectionParameters.ENVIRONMENT,
                                                      self.config['oracle_connection']['environment'])
        db_connection_parameters.set_oracle_parameter(OracleConnectionParameters.USERNAME,
                                                      self.config['oracle_connection']['user'])
        db_connection_parameters.set_oracle_parameter(OracleConnectionParameters.PASSWORD,
                                                      self.config['oracle_connection']['password'])
        db_connection_parameters.set_oracle_parameter(OracleConnectionParameters.SCHEMA,
                                                      self.config['oracle_connection']['schema'])
        db_connection_parameters.set_oracle_parameter(OracleConnectionParameters.TNS_NAME,
                                                      self.config['oracle_connection']['tns_name'])

    def _setup_timescale_connection(self):
        db_connection_parameters = DbConnectionParameters
        db_connection_parameters.set_timescale_parameter(TimescaleConnectionParameters.HOST,
                                                         self.config['timescale_connection']['host'])
        db_connection_parameters.set_timescale_parameter(TimescaleConnectionParameters.DB_NAME,
                                                         self.config['timescale_connection']['database'])
        db_connection_parameters.set_timescale_parameter(TimescaleConnectionParameters.PORT,
                                                         int(self.config['timescale_connection']['port']))
        db_connection_parameters.set_timescale_parameter(TimescaleConnectionParameters.USERNAME,
                                                         self.config['timescale_connection']['user'])
        db_connection_parameters.set_timescale_parameter(TimescaleConnectionParameters.PASSWORD,
                                                         self.config['timescale_connection']['password'])
