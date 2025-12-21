import tkinter as tk
from tkinter import ttk
import pandas as pd
from tkinter import font

from market_monitor.gui.threaded_GUI.QueueDataSource import QueueDataSource
from market_monitor.gui.threaded_GUI.ThreadGUITkinter import ThreadGUITkinter


class TradeThreadGUITkinter(ThreadGUITkinter):
    """
    Interactive gui to visualize real-time trades.
    """

    def __init__(self, data_source: QueueDataSource, update_interval: int = 1_000):
        super().__init__(data_source=data_source, update_interval=update_interval)

    def _add_legend(self):
        legend = ttk.Frame(self.root)
        legend.pack(fill=tk.X, padx=5)
        default_font = font.nametofont("TkDefaultFont")
        bold_font = default_font.copy()
        bold_font.configure(weight="bold")
        ttk.Label(legend, text='Legenda:').pack(side=tk.LEFT)
        ttk.Label(legend, text='BUY', background='#b0e0e6').pack(side=tk.LEFT, padx=2)
        ttk.Label(legend, text='SELL', background='#ffcccc').pack(side=tk.LEFT, padx=2)
        # ttk.Label(legend, text='NEW TRADE', background='#fff6b2').pack(side=tk.LEFT, padx=2)
        ttk.Label(legend, text='OWN_TRADE', font=bold_font).pack(side=tk.LEFT, padx=2)

    def _add_tags(self):
        for tag, color in [('own_trade_buy', '#b0e0e6'), ('own_trade_sell', '#ffcccc'), ('new_trade', '#fff6b2')]:
            self.tree.tag_configure(tag, background=color)
            if tag != 'new_trade':
                self.tree.tag_configure(tag, font=('Arial', 10, 'bold'))

    def _tag_table(self, df: pd.DataFrame):
        for index, row in df.iterrows():
            tags = []
            own_trade = int(row.get('own_trade', 0))
            side = row.get('side')
            if own_trade != 0:
                if side == 'bid':
                    tags.append('own_trade_buy')
                elif side == 'ask':
                    tags.append('own_trade_sell')
            # elif index in self.new_trade_indices:
            #     tags.append('new_trade')

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

            self.tree.insert('', 'end', values=vals, tags=tags)
