import pandas as pd

from market_monitor.gui.threaded_GUI.QueueDataSource import QueueDataSource
from market_monitor.gui.threaded_GUI.ThreadGUITkinter import ThreadGUITkinter


class TimescaleThreadGUITkinter(ThreadGUITkinter):
    """
    Interactive gui to visualize trades and other statistics from Timescale.
    """

    def __init__(self, data_source: QueueDataSource, update_interval: int = 3_000):
        super().__init__(data_source=data_source, update_interval=update_interval, name="TradeThreadGUITkinter")

    def _add_legend(self):
        pass

    def _add_tags(self):
        pass

    def _tag_table(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            vals = []
            for c in self.selected_columns:
                val = row[c]
                if isinstance(val, (int, float)):
                    s = f"{val:,.2f}"
                    tmp = s.replace(",", "X").replace(".", ",").replace("X", ".")
                    if tmp.endswith(',00'):
                        tmp = tmp[:-3]
                    vals.append(tmp)
                else:
                    vals.append(str(val))

            self.tree.insert('', 'end', values=vals)
