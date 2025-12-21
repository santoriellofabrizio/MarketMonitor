from typing import Optional, Union

import pandas as pd

from market_monitor.gui.implementations.GUI import GUI
from market_monitor.strategy.StrategyUI.StrategyUIAsync import StrategyUIAsync



class StrategyUI(StrategyUIAsync):
    """
      Abstract Facade for asynchronous market monitoring,
       allowing the scheduling and management of tasks in a non-blocking way.

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
          kwargs (dict): Additional configuration arguments. Read config for other info's
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def on_trade(self, trades):
        """
        method to handle trade actions.

        Args:
            trades: Dataframe containing trade data: ["last_update", "price", "quantity", "ctv", "side", "own_trade"]

        Returns:
            None: This method does not return a value.
        """
        pass

    def update_HF(self, *args, **kwargs) -> Optional[tuple[pd.DataFrame, str, str]]:
        """
        Updates data on a timed NAVs.

         Returns:
            tuple of:
                - dataframe to be displayed.
                in case of an Excel gui:
                    - cell where data is shown
                    - sheet where data is shown
        """
        pass

    def export_data(self, gui_name, *args, **kwargs):
        """ method to export output_NAV to the chosen gui"""
        self.GUIs[gui_name].export_data(*args, **kwargs)

    def update_LF(self, *args, **kwargs):
        """
        Abstract method to perform low-frequency updates.

        Args:
            *args: Positional arguments for low-frequency update.
            **kwargs: Keyword arguments for low-frequency update.

        Returns:
            None: This method does not return a value.
        """
        pass

    def on_my_trade(self, trades: pd.DataFrame) -> None:
        """
        method to handle own trade actions.

        Args:
            trades: Dataframe containing trade data: ["last_update", "price", "quantity", "ctv", "side", "own_trade"]

        Returns:
            None: This method does not return a value.
        """
        pass

    def on_market_data_setting(self):
        """
            callback invoked when market data are set
        Returns:
            None: This method does not return a value

        """
        pass

    def on_other_thread_start(self):
        """
           callback invoked when other thread are started
       Returns:
           None: This method does not return a value

       """
        pass


