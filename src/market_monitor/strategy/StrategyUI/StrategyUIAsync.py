from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from queue import Queue
from time import time
from typing import Optional, Dict, Any, Union, Coroutine

import pandas as pd
from watchdog.observers import Observer


from market_monitor.gui.implementations.GUI import GUI
from market_monitor.input_threads.trade import TradeType

from market_monitor.live_data_hub.real_time_data_hub import RTData
from market_monitor.utils.config_observer import ConfigChangeHandler


class StrategyUIAsync(ABC):
    """
      Abstract class for asynchronous market monitoring, allowing the scheduling and management of tasks in a non-blocking way.

      Available tasks:
          - High-frequency computations (update HF)
          - Storing data in the database (store on DB)
          - Monitoring trade queue and handling market operations (check trade queue)
          - Monitoring dynamic parameters (observe params)
          - Low-frequency computations (update LF)

      Attributes:
          logger (Logger): Logger instance for recording events and errors.
          q_trade (Queue): Queue for handling market operations.
          gui (GUI): Graphical user interface for interaction with Excel.
          storage (DataStorageUI): Interface for data storage.
          market_data (RTData): Real-time market data book.
          kwargs (dict): Additional configuration arguments.
    """

    def __init__(self, q_trade: None | Queue | asyncio.Queue = None, market_data: RTData = None,
                 gui: GUI = None, storage: None = None, **kwargs):
        """
         Initializes an instance of StrategyUIAsync.rst with the necessary components.

         Args:
             q_trade (Queue): Queue for handling market operations.
             market_data (RTData): Instance of the real-time market data book.
             storage (DataStorageUI): Interface for data storage.
             **kwargs: Additional configuration arguments. Read config for other info's.
         """
        self.input_params = None
        logging.getLogger()
        self.q_trade: None | Queue | asyncio.Queue = q_trade
        self.synchronous_trade_handling: bool = True
        self.GUIs: dict[str, GUI] = {}
        self.market_data: RTData = market_data
        self.running = False
        self.kwargs = kwargs

    def start(self):
        """ Starts the strategy"""
        logging.debug("Entering start method.")
        asyncio.run(self._async_run())

    def _schedule_tasks(self) -> list[Coroutine[Any, Any, None]] | None:
        """
        Schedules the tasks to be executed asynchronously based on the configuration.

        Returns:
            list[asyncio.Task]: List of scheduled asyncio tasks.

        Raises:
            KeyboardInterrupt: If a task is not implemented.
        """
        try:
            self.synchronous_trade_handling = self.kwargs["tasks"]["trade"]["synchronous"]
        except KeyError:
            logging.error("please add synchornous (bool) in tasks->trade config")

        task_dict = {
            "update_LF": self._async_update_LF,
            "trade": self._async_check_trade_queue,
            "update_HF": self._async_update_HF,
            "dynamic_params": self._async_dynamic_params_observer
        }

        try:
            return [task_dict[task](**params) for task, params in self.kwargs["tasks"].items() if
                    params.pop("activate")]
        except KeyError as e:
            logging.critical(f"task not implemented. available are {', '.join(task_dict.keys())}.\n {e}")
            raise KeyboardInterrupt
        except Exception as e:
            logging.critical(f"Unhandled error in scheduling tasks.  {e}")

    async def _async_run(self):
        """
        Manages the asynchronous execution of market monitoring activities.

        This method starts the tasks scheduled for monitoring the market and ensures
        that the market data book is initialized before proceeding.
        """

        logging.debug("Entering _async_run method.")
        if await self._wait_for_book_initialization():
            print("started!")
            tasks = self._schedule_tasks()
            self._on_other_thread_start()
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            logging.error("Book initialization failed.")

    async def _async_dynamic_params_observer(self, dynamic_config_path: str, frequency: float):
        """
        Osserva e aggiorna parametri dinamici tramite callback.

        Args:
            dynamic_config_path: Path al file di configurazione
            frequency: Frequenza di controllo (in secondi)
        """
        logging.debug("Starting config observer")

        event_handler = ConfigChangeHandler(
            dynamic_config_path=dynamic_config_path,
            on_config_change=self._handle_config_change
        )

        observer = Observer()
        observer.name = "DynamicParamsObserver"
        observer.schedule(event_handler, path=str(Path(dynamic_config_path).parent), recursive=False)
        observer.start()

        self._config_observer = observer

        try:
            while not self.running:
                await asyncio.sleep(frequency)
        except Exception as e:
            logging.error(f"Error in config observer: {e}")
        finally:
            observer.stop()
            observer.join()

    def _handle_config_change(self, key: str, old_value: Any, new_value: Any):
        """
        Callback invocato quando un parametro di configurazione cambia.

        Args:
            key: Nome del parametro
            old_value: Valore precedente
            new_value: Nuovo valore
        """
        logging.info(f"Handling config change: {key} = {new_value} (was {old_value})")

        try:
            self.on_config_change(key, old_value, new_value)
        except Exception as e:
            logging.error(f"Error handling config change for {key}: {e}", exc_info=True)

    def on_config_change(self, key: str, old_value: Any, new_value: Any):
        """
        Override questo metodo nella tua strategia per gestire i cambi di configurazione.

        Example:
            def on_config_change(self, key, old_value, new_value):
                if key == "max_position_size":
                    self.max_position = new_value
                elif key == "risk_threshold":
                    self.risk_threshold = new_value
                    self._recalculate_limits()

        Args:
            key: Nome del parametro modificato
            old_value: Valore precedente
            new_value: Nuovo valore
        """
        pass

    async def _async_update_HF(self, frequency, *args, **kwargs):
        """
        Updates data at regular intervals and exports it to the gui.

        Args:
            frequency (int): Frequency of updates (in seconds).
            *args: Additional positional arguments for high-frequency updates.
            **kwargs: Additional keyword arguments for high-frequency updates.

        Returns:
            None: This method does not return a value.
        """
        logging.debug("Entering _async_update_HF method.")
        while not self.running:
            start = time()
            try:
                self.update_HF()
                logging.debug(f"Update HF calculation {time() - start:.4f}s")

            except Exception as e:
                logging.info(f"Error in update HF: {e}")
            await asyncio.sleep(frequency)

    async def _async_check_trade_queue(self, *args, **kwargs):
        """
        Checks the trade queue for updates and handles market operations or position changes.
        Can work both synchronously and asynchronously.

        Args:
            *args: Additional positional arguments for trade checking.
            **kwargs: Additional keyword arguments for trade checking.

        Returns:
            None: This method does not return a value.
        """
        if self.synchronous_trade_handling:
            await self._check_trade_queue_on_time(*args, **kwargs)
        else:
            await self._check_trade_queue_on_event()

    async def _check_trade_queue_on_event(self):
        """
           Checks the trade queue for new items based on events.

           This method waits for new trade items and processes them as they arrive.

           Returns:
               None: This method does not return a value.
           """
        while not self.running:
            logging.debug("Entering _check_trade_queue_on_event method.")
            trade_type, item = await self.q_trade.get()
            start = time()
            self._on_trade(trade_type, item)
            logging.debug(f"trade elaborated {time() - start:.4f}s")

    async def _check_trade_queue_on_time(self, frequency, *args, **kwargs):
        """
        Checks the specified queue for new items and invokes the appropriate handler.

        Args:
            frequency (int): Frequency at which to check the trade queue (in seconds).
            *args: Additional positional arguments for trade checking.
            **kwargs: Additional keyword arguments for trade checking.

        Returns:
            None: This method does not return a value.
        """
        logging.debug("Entering _sync_check_queue_update method.")
        while not self.running:
            logging.debug("Entering _check_trade_queue_on_time method.")
            market_trades, own_trades = [], []
            batch_size = 0
            while not self.q_trade.empty():
                batch_size += 1
                trade_type, item = self.q_trade.get_nowait()
                logging.debug(f"Got trade from the queue_trade. batch: {batch_size}")
                if isinstance(item, pd.DataFrame):
                    if trade_type == TradeType.MARKET:
                        market_trades.append(item)
                    elif trade_type == TradeType.OWN:
                        own_trades.append(item)
                    else:
                        logging.error(f"Unknown trade type: {trade_type}")
            try:
                if len(market_trades):
                    start = time()
                    self._on_trade(TradeType.MARKET, pd.concat(market_trades))
                    logging.debug(f"Task market trades completed ({time() - start:.4f}s)")
                if len(own_trades):
                    start = time()
                    self._on_trade(TradeType.OWN, pd.concat(own_trades))
                    logging.debug(f"Task own trades completed ({time() - start:.4f}s)")
            except Exception as e:
                logging.error(f"Task trade: error in processing trades")
            await asyncio.sleep(frequency)

    async def _async_update_LF(self, frequency):
        """
        Performs periodic checks at regular intervals.

        Args:
            frequency (int): Frequency of periodic checks (in seconds).

        Returns:
            None: This method does not return a value.
        """
        logging.debug("Entering _async_update_LF method.")
        while not self.running:
            try:
                start = time()
                logging.debug("entering _async_update_LF")
                self.update_LF()
                logging.debug(f"_async_update_LF check ({time() - start:.4f})s")
            except Exception as e:
                logging.error(f"Error during periodic check: {e}")
            await asyncio.sleep(frequency)

    async def _wait_for_book_initialization(self):
        """
        Attende l'inizializzazione del libro con un numero massimo di tentativi.
        Returns:
            bool: True se l'inizializzazione è riuscita, False altrimenti.
        """
        logging.debug("Entering _wait_for_book_initialization method.")
        while not self.wait_for_book_initialization():
            await asyncio.sleep(3)
            logging.info("Waiting for _real_time_data initialization...")
        self.on_book_initialized()
        return True

    def wait_for_book_initialization(self):
        return True

    async def shutdown(self):
        """
        Arresta la strategia e chiude la gui.
        """
        logging.debug("Entering shutdown method.")
        self.running = True  # Segnala la chiusura agli altri cicli

        try:
            # Chiude la gui in modo sicuro
            for _, gui in self.GUIs.items():
                 gui.close()

            # Cancella tutti i task pendenti
            tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

            for task in tasks:
                task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)

            logging.info("Shutdown completo.")

        except Exception as e:
            logging.error(f"Errore nello shutdown: {e}", exc_info=True)

    def set_gui(self,gui_name: str, gui: GUI):
        """ sets gui for the strategy"""
        self.GUIs[gui_name] = gui

    def set_q_trade(self, queue_: Queue | asyncio.Queue):
        """ Sets trade queue for the strategy."""
        if isinstance(queue_, asyncio.Queue): self.synchronous_trade_handling = False
        self.q_trade = queue_

    def set_market_data(self, market_data: RTData):
        """ Sets book object for the strategy."""
        self.market_data = market_data
        self._on_market_data_setting()

    def on_book_initialized(self):
        pass

    def _on_trade(self, trade_type: TradeType, trade: pd.DataFrame) -> Optional[Dict[str, Any]]:

        try:
            start = time()
            logging.debug("Entering _on_trade method.")
            if trade_type == TradeType.MARKET:
                self.on_trade(trade)
            elif trade_type == TradeType.OWN:
                self._on_my_trade(trade)
            else:
                logging.error(f"Invalid trade_type: {trade_type}")
            logging.debug(f"_on_trade takes ({time() - start:.4f})s")
        except Exception as e:
            logging.error(f"error in _on_trade method: {e}")
        return

    def _on_my_trade(self, trade: pd.DataFrame):
        logging.info("Entering on_my_trade method.")
        self.on_my_trade(trade)

    def _on_market_data_setting(self):
        self.on_market_data_setting()
        pass

    def _on_other_thread_start(self):
        self.on_other_thread_start()
        pass

    def _on_start_strategy(self):

        logging.debug("Entering _on_start_monitor method.")
        self.on_start_strategy()

    @abstractmethod
    def update_HF(self, *args, **kwargs):
        """
        Abstract method to perform low-frequency updates.

        Args:
            *args: Positional arguments for low-frequency update.
            **kwargs: Keyword arguments for low-frequency update.

        Returns:
            None: This method does not return a value.
        """
        logging.debug("Entering update_on_time method.")
        return None, None, None

    def update_LF(self, *args, **kwargs):
        """
        Abstract method to perform low-frequency updates.

        Args:
            *args: Positional arguments for low-frequency update.
            **kwargs: Keyword arguments for low-frequency update.

        Returns:
            None: This method does not return a value.
        """
        logging.debug("Entering periodic_check method.")
        pass

    def export_data(self, *args, **kwargs):
        pass

    def stop(self):
        """Stop non bloccante."""
        self.on_stop()
        logging.debug("Entering stop method.")

        # Se hai un event loop attivo
        if hasattr(self, '_loop') and self._loop and self._loop.is_running():
            # Schedule shutdown nell'event loop esistente
            asyncio.run_coroutine_threadsafe(self.shutdown(), self._loop)
        else:
            # Altrimenti crea nuovo loop
            try:
                asyncio.run(self.shutdown())
            except RuntimeError as e:
                logging.warning(f"Impossibile eseguire shutdown async: {e}")

    def on_start_strategy(self):
        """Callback invoked when a strategy is started"""
        pass

    def on_trade(self, trades: pd.DataFrame):
        """
        method to handle trade actions.

        Args:
            trades: Dataframe containing trade data: ["last_update", "price", "quantity", "ctv", "side", "own_trade"]

        Returns:
            None: This method does not return a value.
        """
        pass

    def on_my_trade(self, trades):
        """
             method to handle own trade actions.

             Args:
                 trades: Dataframe containing trade data: ["last_update", "price", "quantity", "ctv", "side", "own_trade"]
             Returns:
                 None: This method does not return a value.
        """
        pass

    def on_other_thread_start(self):
        pass

    def on_market_data_setting(self):
        pass

    def on_stop(self):
        pass
