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


class FlushFileHandler(logging.FileHandler):
    """
    Custom FileHandler that flushes immediately after each write.
    This ensures logs are visible in real-time, not just when the script closes.
    """
    def emit(self, record):
        super().emit(record)
        self.flush()  # Flush immediately after each log


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
        if self.config.get("bloomberg_data_distributor", {}).get("activate", False):
            self._setup_bloomberg_distributor(threads, user_strategy)
        if self.config.get("excel_data_distributor", {}).get("activate", False):
            self._setup_excel_distributor(threads, user_strategy)
        if self.config.get("redis_data_distributor", {}).get("activate", False):
            self._setup_redis_distributor(threads, user_strategy)

        self._setup_gui(threads, user_strategy)

        return threads, user_strategy

    def _setup_logging(self):
        log_level = self.config['logging'].get('log_level', 'DEBUG').upper()
        log_name = self.config['logging']['log_name']
        
        # Create logs directory relative to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Build full log file path
        log_file = os.path.join(log_dir, log_name)

        # Remove any existing handlers to reset logging
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Configure basic logging
        logging.basicConfig(
            level=log_level,
            filename=log_file,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filemode='w',
            force=True  # Force reconfiguration even if already configured
        )
        
        # Get root logger
        logger = logging.getLogger()
        
        # Add file handler explicitly (backup method)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config['logging'].get('log_level_console', 'DEBUG'))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Set the root logger level
        logger.setLevel(log_level)
        
        # Log startup messages
        logging.info(f"=" * 80)
        logging.info(f"Logging initialized at: {log_file}")
        logging.info(f"Log level: {log_level}")
        logging.info(f"=" * 80)
        
        print(f"Logging initialized: {log_file}")  # Debug output to console

        return logger

    def _find_user_strategy_root(self, package_path):
        """
        Trova la directory root che contiene user_strategy.

        Args:
            package_path: Path assoluto alla strategia (es. /path/to/MarketMonitor/user_strategy/equity/LiveAnalysis)

        Returns:
            Path alla root del progetto (es. /path/to/MarketMonitor) o None se non trovato
        """
        current_path = os.path.abspath(package_path)

        # Risali il path finché trovi una directory chiamata "user_strategy"
        while current_path and current_path != os.path.dirname(current_path):
            # Se la directory corrente si chiama "user_strategy"
            if os.path.basename(current_path) == "user_strategy":
                # Ritorna il parent (la root del progetto)
                project_root = os.path.dirname(current_path)
                logging.debug(f"user_strategy root trovata: {project_root}")
                return project_root

            # Risali di un livello
            current_path = os.path.dirname(current_path)

        # Se non troviamo "user_strategy" nel path, proviamo a vedere se esiste una sottocartella
        # user_strategy nella stessa directory del package_path
        search_path = os.path.abspath(package_path)
        for _ in range(5):  # Massimo 5 livelli di risalita
            potential_user_strategy = os.path.join(search_path, "user_strategy")
            if os.path.isdir(potential_user_strategy):
                logging.debug(f"user_strategy trovata in: {search_path}")
                return search_path
            parent = os.path.dirname(search_path)
            if parent == search_path:  # Siamo alla root del filesystem
                break
            search_path = parent

        logging.warning(f"user_strategy root non trovata a partire da: {package_path}")
        return None

    def _load_strategy_from_metadata(self):
        """
        Carica dinamicamente una classe da un file specifico senza
        basarsi sul sistema di import standard di Python.
        """
        load_strategy_info = self.config["load_strategy_info"]
        package_path = load_strategy_info["package_path"]
        module_name = load_strategy_info["module_name"]
        class_name = load_strategy_info["class_name"]
        package_path = os.path.abspath(package_path)
        file_path = os.path.join(package_path, f"{module_name}.py")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File non trovato: {file_path}")

        try:
            spec = util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)

            # Aggiungi il package_path specifico della strategia
            if package_path not in sys.path:
                sys.path.insert(0, package_path)

            # IMPORTANTE: Aggiungi anche la root che contiene user_strategy
            # per permettere import tipo: from user_strategy.utils.trade_manager import TradeManager
            user_strategy_root = self._find_user_strategy_root(package_path)
            if user_strategy_root and user_strategy_root not in sys.path:
                sys.path.insert(0, user_strategy_root)
                logging.info(f"Aggiunto user_strategy root al sys.path: {user_strategy_root}")

            spec.loader.exec_module(module)
            strategy_class = getattr(module, class_name)

            logging.info(f"Classe {class_name} caricata con successo da {file_path}")
            return strategy_class

        except Exception as e:
            logging.error(f"Errore durante il caricamento dinamico della strategia: {e}")
            raise

    def _set_real_time_data(self, monitor, lock):
        market_data = RTData(lock, **self.config.get("market_data_distributor", {}).get("book_params",{}))
        monitor.set_market_data(market_data)

    def _setup_trade_distributor(self, threads, monitor, lock, logger):
        q_trade = Queue() if self.config["market_monitor"]["tasks"]["trade"]["synchronous"] else asyncio.Queue()
        monitor.set_q_trade(q_trade)
        path = self.config["trade_distributor"]["path"]
        trade_thread = TradeThread(lock, q_trade, path=path)
        threads.append(trade_thread)

    def _setup_bloomberg_distributor(self, threads, monitor):
        market_data = monitor.market_data
        event_handler = BBGEventHandler(market_data)
        bloomberg_distributor_thread = BloombergStreamingThread(event_handler, **self.config["bloomberg_data_distributor"].get(
                                                                                        "bloomberg_params", {}))
        threads.append(bloomberg_distributor_thread)

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
