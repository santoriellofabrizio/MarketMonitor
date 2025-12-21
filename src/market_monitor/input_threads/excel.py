from __future__ import annotations

import logging
import time
from threading import Thread
from typing import Optional, Union, List, Dict

import pandas as pd
import xlwings as xw

from market_monitor.live_data_hub.real_time_data_hub import RTData


class ExcelStreamingThread(Thread):
    """
    A thread class to handle streaming data from an Excel workbook using xlwings.

    """

    def __init__(self,  **kwargs) -> None:
        """
        Initialize the excel instance.

        Args:
            real_time_data (RTData | pd.DataFrame): The RTBook instance or DataFrame to handle.
            path (str, optional): Path to the Excel file. Defaults to "".
            logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.
            **kwargs: Additional keyword arguments for customization.
                - start_cell (str): Starting cell for data read. Defaults to "A1".
                - sheet_name (str | int): Name or index of the sheet. Defaults to 0.
                - book_reading_frequency (int): Frequency in seconds for reading the workbook. Defaults to 1.
        """

        super().__init__()
        self.sheets: Optional[xw.Book.sheets] = None
        self.wb: None | xw.Book = None
        self.running: bool = False
        self.pass_on_crash: bool = False
        self._real_time_data: RTData | None = None
        self.path: str = kwargs["path"]
        self.logger = logging.getLogger()
        self.start_cell: str = kwargs.get("start_cell", "A1")
        self.excel_range: str = kwargs.get("excel range read", None)
        self.sheet_names: List[str | None] = kwargs.get("sheet_names", [None])
        self.is_single_sheet = False
        self.book_reading_frequency = float(kwargs.get("book_reading_frequency", 1))
        self._saved_ranges = {}
        self.instance_visible: bool = kwargs.get("visible", True)

    def set_real_time_data(self, real_time_data: RTData) -> None:
        self._real_time_data = real_time_data

    def close_excel(self):
        """
        Close the Excel workbook safely.

        Saves changes and logs closure information.
        """
        self.wb.save()
        if self.logger:
            self.logger.info("Closing Excel file")

    def _read_single_sheet(self, sheet_name: str):
            try:
                sheet = self.wb.sheets[sheet_name]
                if sheet_name in self._saved_ranges:
                    table_range = sheet.range(self._saved_ranges[sheet_name])
                else:
                    if self.excel_range is not None:
                        table_range = sheet.range(self.excel_range)
                    else:
                        table_range = sheet.range(self.start_cell).expand()
                    # self._saved_ranges[sheet_name] = table_range.address
                table = table_range.options(pd.DataFrame, header=True, index=True).value
                time.sleep(1)
                return table
            except Exception as e:
                if self.logger:
                    self.logger.info(f"Error while reading '{sheet_name}', retriyng")
                    time.sleep(1)

    def read_sheets(self, sheet_names: Union[str, List[str]] = None) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Read tables from the specified sheet(s) in the Excel workbook and return them as a dictionary of pandas DataFrames.

        Returns:
            Dict[str, Optional[pd.DataFrame]]: Dictionary containing the read data for each sheet, or None if an error occurred.
        """
        sheet_names = self.sheet_names if sheet_names is None else sheet_names
        if isinstance(sheet_names, str): sheet_names = [sheet_names]
        if self.is_single_sheet: return self._read_single_sheet(sheet_names[0])
        else: return {sheet_name: self._read_single_sheet(sheet_name) for sheet_name in sheet_names}

    def on_crash(self):
        """
        Handle workbook crash event.

        If pass_on_crash is False, reopen the workbook.
        """
        if self.pass_on_crash:
            return
        else:
            self.wb = xw.Book(self.path)
            self.sheets = self.wb.sheets

    def close(self):
        """
        Close the Excel application and workbook.

        Saves changes and quits Excel application.
        """
        if not self.instance_visible:
            self.wb.app.visible = True
        if self.logger:
            self.logger.info("Excel application closed.")
        else:
            print("Excel application closed.")

    def open_excel_book(self, path):
        """
        Attempt to open an Excel workbook at the specified path.

        Retry up to 4 times with a 1-second delay between attempts.

        Args:
            path (str): Path to the Excel workbook.

        Returns:
            xw.Book: Opened workbook object.

        Raises:
            Exception: If the workbook cannot be opened after multiple attempts.
        """
        # Check if the workbook is already open

        try:
            for book in xw.books:
                if book.fullname == path or book.name == path:
                    if self.logger:
                        self.logger.info(f"Excel file {path} is already open.")
                    return book  # Return the already open workbook
        except Exception as e:
            self.logger.error(f"cannot scan open workbooks: {e}")
        # Attempt to open the workbook if it's not already open
        max_count, count = 4, 0
        while count < max_count:
            try:
                wb = xw.Book(path)
                if self.logger:
                    self.logger.info(f"Excel file {path} opened successfully.")
                return wb  # Return the workbook if successful
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error opening the file: {e}")
                time.sleep(1)  # Wait for a second before retrying
                count += 1
        raise Exception(f"Can't open excel file: {path}. Try closing and relaunching it.")

    def run(self):
        """
        Main execution method for the thread.

        Opens the Excel workbook and continuously reads and updates the _real_time_data data.
        """
        self.wb = self.open_excel_book(self.path)
        if not self.instance_visible: self.wb.app.visible = False
        while not self.running:
            books = self.read_sheets()
            for name, book in books.items():
                if book is None or book.empty: continue
                self._real_time_data.update_all_data(book, None if self.is_single_sheet else name)
            time.sleep(self.book_reading_frequency)
        self.close()

    def stop(self):
        self.running = True