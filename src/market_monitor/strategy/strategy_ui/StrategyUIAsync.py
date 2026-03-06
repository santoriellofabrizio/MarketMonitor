from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from queue import Queue
from time import time
from typing import Optional, Dict, Any, Union, Coroutine

import pandas as pd
from market_monitor.gui.implementations.GUI import GUI
from market_monitor.input_threads.trade import TradeType

from market_monitor.live_data_hub.real_time_data_hub import RTData
from market_monitor.utils.command_listener import CommandListener
from market_monitor.live_data_hub.subscription_service import SubscriptionService
from market_monitor.utils.config_observer import ConfigChangeHandler

logger = logging.getLogger(__name__)


class StrategyUIAsync(ABC):
    """
      Abstract class for asynchronous market monitoring, allowing the scheduling and management of tasks in a non-blocking way.

      Available tasks:
          - High-frequency computations (update HF)
          - Low-frequency computations (update LF)
          - Monitoring trade queue and handling market operations (check trade queue)
          - Command listener via Redis pub/sub (command_listener)

      Attributes:
          logger (Logger): Logger instance for recording events and errors.
          q_trade (Queue): Queue for handling market operations.
          gui (GUI): Graphical user interface for interaction with Excel.
          storage (DataStorageUI): Interface for data storage.
          market_data (RTData): Real-time market data book.
          kwargs (dict): Additional configuration arguments.
    """

    def __init__(self, q_trade: None | Queue | asyncio.Queue = None, market_data: RTData = None, **kwargs):
        """
         Initializes an instance of StrategyUIAsync.rst with the necessary components.

         Args:
             q_trade (Queue): Queue for handling market operations.
             market_data (RTData): Instance of the real-time market data book.
             storage (DataStorageUI): Interface for data storage.
             **kwargs: Additional configuration arguments. Read config for other info's.
         """
        self.input_params = None
        self.q_trade: None | Queue | asyncio.Queue = q_trade
        self.synchronous_trade_handling: bool = True
        self.GUIs: dict[str, GUI] = {}
        self.market_data: RTData = market_data
        self.global_subscription_service: Optional[SubscriptionService] = None
        self.running = False
        self.kwargs = kwargs

    def set_subscription_service(self, subscription_service: SubscriptionService):
        self.global_subscription_service = subscription_service

    def start(self):
        """ Starts the strategy"""
        logger.debug("Entering start method.")
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
            logger.error("please add synchornous (bool) in tasks->trade config")

        task_dict = {
            "update_LF": self._async_update_LF,
            "trade": self._async_check_trade_queue,
            "update_HF": self._async_update_HF,
            "command_listener": self._async_command_listener,
        }

        try:
            return [task_dict[task](**params) for task, params in self.kwargs["tasks"].items() if
                    params.pop("activate")]
        except KeyError as e:
            logger.critical(f"task not implemented. available are {', '.join(task_dict.keys())}.\n {e}")
            raise KeyboardInterrupt
        except Exception as e:
            logger.critical(f"Unhandled error in scheduling tasks.  {e}")

    async def _async_run(self):
        """
        Manages the asynchronous execution of market monitoring activities.

        This method starts the tasks scheduled for monitoring the market and ensures
        that the market data book is initialized before proceeding.
        """

        logger.debug("Entering _async_run method.")
        if await self._wait_for_book_initialization():
            print("started!")
            tasks = self._schedule_tasks()
            self._on_other_thread_start()
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            logger.error("Book initialization failed.")

    # =========================================================================
    # Command listener (Redis pub/sub)
    # =========================================================================

    async def _async_command_listener(self, redis_client_attr: str = "publisher.gui.redis_client",
                                      channel: str = "engine:commands",
                                      status_channel: str = "engine:status", **kwargs):
        """
        Avvia un CommandListener su un canale Redis pub/sub.
        Il listener gira in un daemon thread e invoca on_command per ogni comando ricevuto.

        Config example:
            tasks:
              command_listener:
                activate: true
                redis_client_attr: publisher.gui.redis_client
                channel: engine:commands
                status_channel: engine:status

        Args:
            redis_client_attr: Attributo dot-separated per raggiungere il redis_client (es. "publisher.gui.redis_client")
            channel: Canale Redis su cui ascoltare i comandi
            status_channel: Canale Redis su cui pubblicare lo status
        """
        # Risolvi il redis_client dall'attributo dot-separated

        self._command_listener = CommandListener(
            on_command=self._handle_command,
            channel=channel,
            status_channel=status_channel)

        self._command_listener.start()

        try:
            while not self.running:
                await asyncio.sleep(1)
        finally:
            self._command_listener.stop()

    def _handle_command(self, action: str, payload: dict):
        """
        Callback invocato quando un comando viene ricevuto via Redis.

        Args:
            action: Nome dell'azione (es. "reload_beta")
            payload: Payload completo del messaggio JSON
        """
        logger.info(f"Handling command: {action}")

        try:
            self.on_command(action, payload)
        except Exception as e:
            logger.error(f"Error handling command '{action}': {e}", exc_info=True)
            raise

    def on_command(self, action: str, payload: dict):
        """
        Override questo metodo nella tua strategia per gestire i comandi via Redis.

        Il risultato (success/error) viene automaticamente pubblicato su engine:status
        dal CommandListener. Se il metodo lancia un'eccezione, lo status sarà "error".

        Trigger:
            redis-cli PUBLISH engine:commands '{"action": "reload_beta"}'

        Example:
            def on_command(self, action, payload):
                if action == "reload_beta":
                    self._reload_beta_matrices()
                    self.models.predict_all(self.mid_eur)
                elif action == "reset_cache":
                    self.publisher.gui.clear_change_cache()

        Args:
            action: Nome dell'azione richiesta
            payload: Payload completo del messaggio (contiene almeno {"action": "..."})
        """
        logger.warning(f"Unhandled command: '{action}'. Override on_command() in your strategy.")

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
        logger.debug("Entering _async_update_HF method.")
        while not self.running:
            start = time()
            try:
                self.update_HF()
                logger.debug(f"Update HF calculation {time() - start:.4f}s")

            except Exception as e:
                logger.info(f"Error in update HF: {e}")
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
            logger.debug("Entering _check_trade_queue_on_event method.")
            trade_type, item = await self.q_trade.get()
            start = time()
            self._on_trade(trade_type, item)
            logger.debug(f"trade elaborated {time() - start:.4f}s")

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
        logger.debug("Entering _sync_check_queue_update method.")
        while not self.running:
            logger.debug("Entering _check_trade_queue_on_time method.")
            market_trades, own_trades = [], []
            batch_size = 0
            while not self.q_trade.empty():
                batch_size += 1
                trade_type, item = self.q_trade.get_nowait()
                logger.debug(f"Got trade from the queue_trade. batch: {batch_size}")
                if isinstance(item, pd.DataFrame):
                    if trade_type == TradeType.MARKET:
                        market_trades.append(item)
                    elif trade_type == TradeType.OWN:
                        own_trades.append(item)
                    else:
                        logger.error(f"Unknown trade type: {trade_type}")
            try:
                if len(market_trades):
                    start = time()
                    self._on_trade(TradeType.MARKET, pd.concat(market_trades))
                    logger.debug(f"Task market trades completed ({time() - start:.4f}s)")
                if len(own_trades):
                    start = time()
                    self._on_trade(TradeType.OWN, pd.concat(own_trades))
                    logger.debug(f"Task own trades completed ({time() - start:.4f}s)")
            except Exception as e:
                logger.error(f"Task trade: error in processing trades")
            await asyncio.sleep(frequency)

    async def _async_update_LF(self, frequency):
        """
        Performs periodic checks at regular intervals.

        Args:
            frequency (int): Frequency of periodic checks (in seconds).

        Returns:
            None: This method does not return a value.
        """
        logger.debug("Entering _async_update_LF method.")
        while not self.running:
            try:
                start = time()
                logger.debug("entering _async_update_LF")
                self.update_LF()
                logger.debug(f"_async_update_LF check ({time() - start:.4f})s")
            except Exception as e:
                logger.error(f"Error during periodic check: {e}")
            await asyncio.sleep(frequency)

    async def _wait_for_book_initialization(self):
        """
        Attende l'inizializzazione del libro con un numero massimo di tentativi.
        Returns:
            bool: True se l'inizializzazione è riuscita, False altrimenti.
        """
        logger.debug("Entering _wait_for_book_initialization method.")
        while not self.wait_for_book_initialization():
            await asyncio.sleep(3)
            logger.info("Waiting for _real_time_data initialization...")
        self.on_book_initialized()
        self._publish_lifecycle_event("on_book_initialized")
        return True

    def wait_for_book_initialization(self):
        return True

    async def shutdown(self):
        """
        Arresta la strategia e chiude la gui.
        """
        logger.debug("Entering shutdown method.")
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

            logger.info("Shutdown completo.")

        except Exception as e:
            logger.error(f"Errore nello shutdown: {e}", exc_info=True)

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
            logger.debug("Entering _on_trade method.")
            if trade_type == TradeType.MARKET:
                self.on_trade(trade)
                self._publish_lifecycle_event("on_trade", len(trade))
            elif trade_type == TradeType.OWN:
                self._on_my_trade(trade)
            else:
                logger.error(f"Invalid trade_type: {trade_type}")
            logger.debug(f"_on_trade takes ({time() - start:.4f})s")
        except Exception as e:
            logger.error(f"error in _on_trade method: {e}")
        return

    def _on_my_trade(self, trade: pd.DataFrame):
        logger.info("Entering on_my_trade method.")
        self.on_my_trade(trade)
        self._publish_lifecycle_event("on_my_trade", len(trade))

    def _on_market_data_setting(self):
        self.on_market_data_setting()
        pass

    def _on_other_thread_start(self):
        self.on_other_thread_start()
        pass

    def _on_start_strategy(self):

        logger.debug("Entering _on_start_monitor method.")
        self.on_start_strategy()
        self._publish_lifecycle_event("on_start_strategy")

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
        logger.debug("Entering update_on_time method.")
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
        logger.debug("Entering periodic_check method.")
        pass

    def export_data(self, *args, **kwargs):
        pass

    def _publish_lifecycle_event(self, event_name: str, data=None) -> None:
        pass

    def stop(self):
        """Stop non bloccante."""
        self.on_stop()
        self._publish_lifecycle_event("on_stop")
        logger.debug("Entering stop method.")

        # Se hai un event loop attivo
        if hasattr(self, '_loop') and self._loop and self._loop.is_running():
            # Schedule shutdown nell'event loop esistente
            asyncio.run_coroutine_threadsafe(self.shutdown(), self._loop)
        else:
            # Altrimenti crea nuovo loop
            try:
                asyncio.run(self.shutdown())
            except RuntimeError as e:
                logger.warning(f"Impossibile eseguire shutdown async: {e}")

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
