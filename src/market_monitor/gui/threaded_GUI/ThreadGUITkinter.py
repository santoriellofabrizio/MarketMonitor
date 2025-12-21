import threading
from typing import Optional, List, Dict, Tuple
from abc import ABC, abstractmethod
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import messagebox
import tkinter.font as tkfont

from market_monitor.gui.threaded_GUI.QueueDataSource import QueueDataSource

AGGREGATOR_FUNCTIONS = {
    'sum': lambda s: s.sum(),
    'mean': lambda s: s.mean(),
    'count': lambda s: s.count(),
    'max': lambda s: s.max(),
    'min': lambda s: s.min()
}

COLUMN_TYPES = {
    'ctv': int,
    'quantity': int,
    'price': float,
    'spread_pl': float,
    'spread_pl_model': float,
    'lagged_spread_pl': float,
    'lagged_spread_pl_model': float
}

OPERATORS = ['(None)',
             'equal',
             'not equal',
             'contains',
             'not contains',
             'greater',
             'greater/equal',
             'lower',
             'lower/equal',
             'range',
             'not in range']


class ThreadGUITkinter(threading.Thread, ABC):
    """
    Interactive gui to visualize a DataFrame in real-time.
    """

    def __init__(self, data_source: QueueDataSource,
                 update_interval: int = 3_000,
                 name="ThreadGUITkinter",
                 *args, **kwargs):
        super().__init__(name=name)
        self.data_source = data_source
        self.update_interval = update_interval
        self.running: bool = True
        self.root: Optional[tk.Tk] = None
        self.tree: Optional[ttk.Treeview] = None
        self.sum_frame: Optional[ttk.Frame] = None
        self.dataframe: pd.DataFrame = pd.DataFrame()
        self.selected_columns: List = []
        self.column_vars: Dict = {}
        self.aggregators: Dict = {}
        self.column_types = COLUMN_TYPES or {}
        self.aggregator_vars = {}
        self.filter_vars = {}
        self.quick_search_var = None
        self.quick_search_col_var = None
        self.search_combo = None
        self._pivot_window = None
        self.pivot_sort_state = {}
        self.pivot_tree = None
        self.pivot_vars = {}
        self.pivot_agg_var = None
        self.pivot_filter_var = None
        self.pivot_filter_col_var = None
        self.pivot_sorted_column = None
        self.pivot_sort_reverse: bool = False
        self.pivot_normalize = None
        self.sort_order: List[Tuple[str, str]] = []
        self.filtered_df: Optional[pd.DataFrame] = None
        self.active_filters = {}
        self.column_filter_windows = {}
        self.column_filters_box: Dict[str, List[str]] = {}
        self.column_filters: Dict[str, pd.Series] = {}
        self._columns_window = None  # don't remove even if only usage
        self._aggregators_window = None  # don't remove even if only usage
        self._filter_window = None  # don't remove even if only usage

    def _initialize_gui(self):
        """
        Initialize the main gui layout, including toolbar, search bar, table view, and footer sections.
        """
        self.root = tk.Tk()
        self.root.title('Live DataFrame Viewer')
        self.root.geometry('900x600')

        control = ttk.Frame(self.root)
        control.pack(fill=tk.X, padx=5, pady=5)

        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        left_frame = ttk.Frame(toolbar)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        right_frame = ttk.Frame(toolbar)
        right_frame.pack(side=tk.RIGHT)

        # Update circle
        self.refresh_indicator = tk.Label(left_frame, text='●', fg='gray', font=('Arial', 24))
        self.refresh_indicator.pack(side=tk.LEFT, padx=5)

        # Grey buttons
        btn_filters = ttk.Button(toolbar, text="Set Filters")
        btn_filters.config(command=lambda w=btn_filters: self._show_filters(w))
        btn_filters.pack(side=tk.LEFT, padx=5)

        btn_pivot = ttk.Button(toolbar, text="Pivot Table")
        btn_pivot.config(command=lambda w=btn_pivot: self._show_pivot(w))
        btn_pivot.pack(side=tk.LEFT, padx=5)

        btn_columns = ttk.Button(toolbar, text="Show Columns")
        btn_columns.config(command=lambda w=btn_columns: self._show_columns(w))
        btn_columns.pack(side=tk.LEFT, padx=5)

        btn_aggr = ttk.Button(toolbar, text="Choose Aggregators")
        btn_aggr.config(command=lambda w=btn_aggr: self._show_aggregators(w))
        btn_aggr.pack(side=tk.LEFT, padx=5)

        # Colorful buttons
        btn_export = tk.Button(right_frame, text="Export Excel", command=self._export_excel,
                               bg='#90ee90', fg='black', activebackground='#66cc66', activeforeground='black')
        btn_export.pack(side=tk.LEFT, padx=5)

        btn_reset = tk.Button(right_frame, text="Reset Filters", command=self._reset_filters,
                              bg='red', fg='black', activebackground='#cc0000', activeforeground='black')
        btn_reset.pack(side=tk.LEFT, padx=5)

        btn_export = tk.Button(right_frame, text="Manual Refresh", command=lambda: self._schedule_refresh(False),
                               bg='yellow', fg='black', activebackground='#e6c300', activeforeground='black')
        btn_export.pack(side=tk.LEFT, padx=5)

        # Rapid search 🔍
        search = ttk.Frame(self.root)
        search.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(search, text='🔍').pack(side=tk.LEFT)
        self.quick_search_var = tk.StringVar()
        ttk.Entry(search, textvariable=self.quick_search_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.quick_search_col_var = tk.StringVar(value='All')
        self.search_combo = ttk.Combobox(
            search,
            textvariable=self.quick_search_col_var,
            values=['All'],
            state='readonly',
            width=12
        )
        self.search_combo.pack(side=tk.LEFT, padx=5)
        self.quick_search_var.trace_add('write', lambda *args: self._update_table())
        self.search_combo.bind('<<ComboboxSelected>>', lambda e: self._update_table())

        # Treeview for data visualization
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree = ttk.Treeview(frame, show='headings', yscrollcommand=sb.set)
        self.tree.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.tree.yview)

        self.tree.bind("<Button-3>", self._on_right_click_header)

        self._add_tags()
        self._add_legend()

        # Frame for aggregations
        self.sum_frame = ttk.Frame(self.root)
        self.sum_frame.pack(fill=tk.X, padx=5, pady=5)

        self.filter_vars = {}
        filter_frame = ttk.Frame(self.root)
        filter_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        for col in self.selected_columns:
            var = tk.StringVar(value='All')
            self.filter_vars[col] = var
            combo = ttk.Combobox(filter_frame, textvariable=var, state='readonly', width=12)
            combo.pack(side=tk.LEFT, padx=2)
            combo.bind('<<ComboboxSelected>>', lambda e, c=col: self._update_table())

    def _on_right_click_header(self, event):
        """
        Handle right-clicks on column headers to open a filter popup for that specific column.
        """
        region = self.tree.identify_region(event.x, event.y)
        if region == "heading":
            column_id = self.tree.identify_column(event.x)
            col_index = int(column_id.replace('#', '')) - 1
            if 0 <= col_index < len(self.selected_columns):
                col = self.selected_columns[col_index]
                self._open_filter_menu(col)

    def _open_single_popup(self, window_ref_attr: str, title: str, widget: tk.Widget,
                           content_callback, width: Optional[int] = None, height: Optional[int] = None):
        """
        Method to prevent a double opening of the windows opened by the user buttons.
        """
        existing_win = getattr(self, window_ref_attr)
        if existing_win and tk.Toplevel.winfo_exists(existing_win):
            existing_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title(title)

        # Positions near the button
        x = widget.winfo_rootx()
        y = widget.winfo_rooty()
        if width and height:
            win.geometry(f"{width}x{height}+{x}+{y}")
        else:
            win.geometry(f"+{x}+{y}")

        # Reset the position reference once the window is closed
        def on_close():
            setattr(self, window_ref_attr, None)
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)
        setattr(self, window_ref_attr, win)
        content_callback(win)

    def _open_filter_menu(self, col):
        """
        Create and display a custom filter menu next to the mouse for selecting allowed values in a column.
        """
        unique_count = self.dataframe[col].nunique(dropna=True)
        if unique_count > 1_000:
            messagebox.showinfo("Filter unavailable",
                                f"Too many unique values in column '{col}' ({unique_count:_}).")
            return

        if col in self.column_filter_windows:
            win = self.column_filter_windows[col]
            if win.winfo_exists():
                win.lift()
                win.focus_force()
                return
            else:
                del self.column_filter_windows[col]

        popup = tk.Toplevel(self.root)
        popup.wm_overrideredirect(True)
        x, y = self.tree.winfo_pointerxy()
        popup.wm_geometry(f"+{x}+{y}")
        self.column_filter_windows[col] = popup

        all_values = sorted(self.dataframe.copy()[col].dropna().unique())
        previous_selection = set(self.active_filters.get(col, all_values))
        selected_values = [v for v in all_values if v in previous_selection]

        vars_dict = {}
        select_all_var = tk.BooleanVar(value=(len(selected_values) == len(all_values)))

        frame = tk.Frame(popup, bd=1, relief="solid", background="white")
        frame.pack(fill="both", expand=True, padx=2, pady=2)

        def apply_filter():
            new_values = [v for v, var in vars_dict.items() if var.get()]
            if new_values and len(new_values) < len(all_values):
                self.active_filters[col] = new_values
                self.column_filters_box[col] = new_values
            else:
                self.active_filters.pop(col, None)
                self.column_filters_box.pop(col, None)
            self._update_table()
            popup.destroy()
            self.column_filter_windows.pop(col, None)
            scrollable_frame.unbind_all("<MouseWheel>")
            scrollable_frame.unbind_all("<Button-4>")
            scrollable_frame.unbind_all("<Button-5>")

        def cancel_filter():
            popup.destroy()
            self.column_filter_windows.pop(col, None)
            scrollable_frame.unbind_all("<MouseWheel>")
            scrollable_frame.unbind_all("<Button-4>")
            scrollable_frame.unbind_all("<Button-5>")

        button_frame = tk.Frame(frame, background="white")
        button_frame.pack(fill='x', pady=(5, 5), padx=10)
        ttk.Button(button_frame, text="Apply", command=apply_filter).pack(side="left", padx=(0, 5))
        ttk.Button(button_frame, text="Cancel", command=cancel_filter).pack(side="right", padx=(5, 0))

        def toggle_all():
            check = select_all_var.get()
            for v in all_values:
                vars_dict[v].set(check)

        tk.Checkbutton(frame, text="Select All", variable=select_all_var, command=toggle_all, background="white").pack(
            anchor='w', padx=10, pady=(0, 5)
        )

        canvas = tk.Canvas(frame, height=220, width=200, background="white", highlightthickness=0)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, background="white")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=(0, 5))
        scrollbar.pack(side="right", fill="y", pady=(0, 5))

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_linux(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        scrollable_frame.bind_all("<MouseWheel>", _on_mousewheel)
        scrollable_frame.bind_all("<Button-4>", _on_mousewheel_linux)
        scrollable_frame.bind_all("<Button-5>", _on_mousewheel_linux)

        for v in all_values:
            var = tk.BooleanVar(value=(v in selected_values))
            cb = tk.Checkbutton(
                scrollable_frame,
                text=str(v),
                variable=var,
                background="white",
                anchor='w',
                wraplength=180,
                justify='left'
            )
            cb.pack(anchor='w', padx=10, fill='x')
            vars_dict[v] = var

    @abstractmethod
    def _add_legend(self):
        """
        Abstract method to define a visual legend describing the meaning of tags.
        """
        pass

    @abstractmethod
    def _add_tags(self):
        """
        Abstract method to define and apply row-level tags (e.g., colors or styles)
        """
        pass

    def _export_excel(self):
        """
        Export the currently displayed data (with filters applied) to an Excel file via file dialog.
        """
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if not file_path:
            return

        df = self.dataframe.copy() if self.filtered_df is None else self.filtered_df
        df.to_excel(file_path, index=False)

    def _show_filters(self, widget: tk.Widget):
        """
        Show a popup window with filters for each column.
        """
        def populate(win):
            for col in self.dataframe.columns:
                row = ttk.Frame(win)
                row.pack(fill=tk.X, pady=2)

                ttk.Label(row, text=col, width=15).pack(side=tk.LEFT)

                # Reuse or create filter vars
                if col in self.filter_vars:
                    op_var, val_var = self.filter_vars[col]
                else:
                    op_var = tk.StringVar(value='(None)')
                    val_var = tk.StringVar(value='')
                    self.filter_vars[col] = (op_var, val_var)

                op_menu = ttk.Combobox(row, textvariable=op_var, values=OPERATORS, state='readonly', width=15)
                op_menu.pack(side=tk.LEFT, padx=5)

                ttk.Entry(row, textvariable=val_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

            ttk.Button(win, text='Apply',
                       command=lambda: [self._apply_and_refresh(), win.destroy()]).pack(pady=5)

        self._open_single_popup("_filter_window", "Set Filters", widget, populate)

    def _apply_and_refresh(self):
        """
        Apply filter criteria from the filter window and refreshes the table display.
        """
        self._apply_filters(self.dataframe.copy())
        self._update_table()

    def _apply_filters(self, df: pd.DataFrame):
        """
        Apply value-based filters (equality, range, contains, etc.) to a copy of the DataFrame.
        """
        for col, (op_var, val_var) in self.filter_vars.items():
            op = op_var.get()
            val = val_var.get().strip()
            if not val or op == '(None)':
                continue

            series = df[col]

            try:
                if pd.api.types.is_numeric_dtype(series):
                    if ':' not in val and '-' not in val:
                        val = float(val)
                elif col == 'timestamp':
                    try:
                        series = pd.to_datetime(series, dayfirst=True)
                        val = pd.to_datetime(val, dayfirst=True)
                    except ValueError:
                        print('Error in formatting timestamp')
                        continue
            except Exception as e:
                print(f"Error converting value for column {col}: {e}")
                continue

            try:
                if op == '(None)':
                    pass
                elif op == 'equal':
                    self.column_filters[col] = (series == val)
                elif op == 'not equal':
                    self.column_filters[col] = (series != val)
                elif op == 'contains':
                    self.column_filters[col] = (series.astype(str).str.contains(str(val), case=False, na=False))
                elif op == 'not contains':
                    self.column_filters[col] = (~series.astype(str).str.contains(str(val), case=False, na=False))
                elif op == 'greater':
                    self.column_filters[col] = (series > val)
                elif op == 'greater/equal':
                    self.column_filters[col] = (series >= val)
                elif op == 'lower':
                    self.column_filters[col] = (series < val)
                elif op == 'lower/equal':
                    self.column_filters[col] = (series <= val)
                elif op == 'range':
                    operator = ':' if ':' in val else '-'
                    lower_bound = float(val.split(operator)[0])
                    upper_bound = float(val.split(operator)[1])
                    self.column_filters[col] = ((series >= lower_bound) & (series <= upper_bound))
                elif op == 'not in range':
                    operator = ':' if ':' in val else '-'
                    lower_bound = float(val.split(operator)[0])
                    upper_bound = float(val.split(operator)[1])
                    self.column_filters[col] = ((series < lower_bound) | (series > upper_bound))
            except Exception as e:
                print(f"Error applying filter on {col} with operator {op}: {e}")

    def _flash_refresh_indicator(self):
        """
        Blimp the gray refresh button in green for 0.25 seconds.
        """
        self.refresh_indicator.configure(foreground='green')
        self.root.after(250, lambda: self.refresh_indicator.configure(foreground='gray'))

    def _schedule_refresh(self, periodic: bool = True):
        """
        Refresh called every self.update_interval milliseconds or by the 'Manual Refresh' button.
        """
        self._process_queue()
        self._apply_filters(self.dataframe.copy())
        self._update_table()
        self._flash_refresh_indicator()
        if self.running and periodic:
            self.root.after(self.update_interval, self._schedule_refresh)

    def _process_queue(self):
        """
        Fetch and process new data from the data source, merging it into the main DataFrame.
        """
        df = self.data_source.get_data()
        if df is None or df.empty:
            return

        self.new_trade_indices = list(set(df.index).difference(set(self.dataframe.index)))
        df2 = df.copy()
        if 'price' in df2.columns and 'quantity' in df2.columns and 'ctv' not in df2.columns:
            quantity_col_index = df2.columns.get_loc('quantity')
            price_col_index = df2.columns.get_loc('price')
            insert_position = max(quantity_col_index, price_col_index) + 1
            df2.insert(insert_position, 'ctv', df2['price'] * df2['quantity'])
        if self.column_types:
            try:
                df2 = df2.astype(self.column_types)
            except Exception:
                pass
        if 'trade_index' in df2.columns:
            df2 = df2.set_index('trade_index')
        if self.dataframe.empty:
            self.dataframe = df2
        else:
            combined = pd.concat([self.dataframe, df2])
            self.dataframe = combined.groupby(level=0).last()
        if 'timestamp' in self.dataframe.columns:
            self.dataframe.sort_values('timestamp', ascending=False, inplace=True)

    def _adjust_column_widths(self, min_width=50, max_width=300):
        """
        Adjust the width of each column of the DataFrame based on the length of the values.
        """
        font = tkfont.nametofont("TkDefaultFont")
        for col in self.selected_columns:
            if col not in self.dataframe.columns:
                continue

            values = self.dataframe[col].dropna().astype(str).tolist()
            if not values:
                max_text = col
            else:
                max_text = max([col] + values, key=len)

            is_numeric = False
            try:
                float(str(max_text).replace(',', '').replace('%', ''))
                is_numeric = True
            except ValueError:
                pass

            text_width = font.measure(str(max_text))
            final_width = max(min_width, text_width + 20)
            if is_numeric:
                final_width = max(min_width, min(final_width, 80))
            else:
                final_width = min(final_width, max_width)
            self.tree.column(col, width=final_width, anchor='center')

    def _update_table(self):
        """
        Refresh the data shown in the treeview based on filters, search, and sorting settings.
        """
        self.tree.delete(*self.tree.get_children())

        if self.dataframe.empty:
            return

        if not self.selected_columns:
            self._init_columns()

        self.tree['columns'] = self.selected_columns

        for col in self.selected_columns:
            self.tree.heading(col, text=col, command=lambda c=col: self._toggle_sort(c))
        self._adjust_column_widths()

        df = self.dataframe.copy()

        # Filter box
        for col, allowed_values in self.column_filters_box.items():
            try:
                df = df[df[col].isin(allowed_values)]
            except Exception as e:
                print(e)

        # Set filters
        for col, mask in self.column_filters.items():
            try:
                index_df = list(df.index)
                mask = mask[mask.index.isin(index_df)]
                df = df[mask]
            except Exception as e:
                print(e)

        # Quick search
        search = self.quick_search_var.get().strip().lower()
        sel = self.quick_search_col_var.get()
        if search:
            if sel != 'All':
                df = df[df[sel].astype(str).str.lower().str.contains(search)]
            else:
                mask = df[self.selected_columns].astype(str).apply(
                    lambda r: r.str.lower().str.contains(search).any(), axis=1
                )
                df = df[mask]

        # Sort DataFrame
        df = df.sort_values(by=[column_sort[0] for column_sort in self.sort_order],
                            ascending=[column_sort[1] == 'asc' for column_sort in self.sort_order])

        for c in self.selected_columns:
            heading = c
            for col_, dir_ in self.sort_order:
                if col_ == c:
                    heading += ' ↑' if dir_ == 'asc' else ' ↓'
                    break

            is_filtered = ((c in self.column_filters_box and self.column_filters_box[c] != set()) or
                           (c in self.column_filters and not self.column_filters[c].all()))
            if is_filtered:
                heading += " 🧪"

            self.tree.heading(c, text=heading)

        self._update_sums(df)
        self.filtered_df = df

        # Limit the DataFrame to the first 1_000 rows (only in view-mode, for sorting, filters etc. it always considers
        # the whole DataFrame).
        df = df.head(1_000)

        # Add tags only to visible df
        self._tag_table(df)

        if self._pivot_window and self._pivot_window.winfo_exists():
            self._refresh_pivot()

    def _init_columns(self):
        self.selected_columns = list(self.dataframe.columns)
        self.search_combo['values'] = ['All'] + self.selected_columns
        self.column_vars = {c: tk.BooleanVar(master=self.root, value=True) for c in self.selected_columns}

    @abstractmethod
    def _tag_table(self, df: pd.DataFrame):
        """
        Abstract method to populate the table and apply row tags (styles/colors).
        """
        pass

    def _show_columns(self, widget: tk.Widget):
        """
        Open the 'Show Columns' window, in which the user can select which column are visible and which aren't.
        """
        def populate(win):
            control_frame = ttk.Frame(win)
            control_frame.pack(fill=tk.X, pady=5)
            ttk.Button(control_frame, text='Select All',
                       command=lambda: [v.set(True) for v in self.column_vars.values()]).pack(side=tk.LEFT, padx=5)
            ttk.Button(control_frame, text='Select None',
                       command=lambda: [v.set(False) for v in self.column_vars.values()]).pack(side=tk.LEFT, padx=5)

            # Checkbox list for every column
            check_frame = ttk.Frame(win)
            check_frame.pack(fill=tk.BOTH, expand=True)
            for col, var in self.column_vars.items():
                ttk.Checkbutton(check_frame, text=col, variable=var).pack(anchor='w')

            ttk.Button(win, text='Apply',
                       command=lambda: [self._apply_columns(), win.destroy()]).pack(pady=5)

        self._open_single_popup("_columns_window", "Show Columns",
                                widget, populate, 250, 300)

    def _apply_columns(self):
        """
        Apply the column visibility settings chosen in the 'Show Columns' popup.
        """
        self.selected_columns = [c for c, v in self.column_vars.items() if v.get()]
        self._update_table()

    def _show_aggregators(self, widget: tk.Widget):
        """
        Show the window aggregators.
        """
        def populate(win):
            for col in self.selected_columns:
                # Default: 'sum' for numeric columns, 'None' for other
                default = 'None'
                var = tk.StringVar(master=self.root, value=self.aggregators.get(col, default))

                ttk.Label(win, text=col).pack(anchor='w', padx=5, pady=2)
                cb = ttk.Combobox(win, textvariable=var, values=list(AGGREGATOR_FUNCTIONS.keys()) + ['None'])
                cb.pack(fill=tk.X, padx=5)
                self.aggregator_vars[col] = var

            ttk.Button(win, text='Apply',
                       command=lambda: [self._apply_aggregators(), win.destroy()]).pack(pady=8)

        self._open_single_popup("_aggregators_window", "Choose Aggregators",
                                widget, populate, 300, 500)

    def _apply_aggregators(self):
        """
        Save selected aggregator functions from the aggregator window into the internal aggregator state.
        """
        for col, var in self.aggregator_vars.items():
            self.aggregators[col] = var.get()
        self._update_table()

    def _show_pivot(self, widget: tk.Widget):
        """
        Show the pivot table interface.
        """
        def populate(win):
            win.title('Pivot Table')
            frame = ttk.Frame(win)
            frame.pack(padx=10, pady=10)

            # Combobox for rows, columns, values
            for key in ['rows', 'columns', 'values']:
                ttk.Label(frame, text=key.capitalize()).pack(side=tk.LEFT)
                var = tk.StringVar(master=self.root)
                cb = ttk.Combobox(
                    frame,
                    textvariable=var,
                    values=['None'] + self.dataframe.columns.tolist(),
                    state='readonly',
                    width=12
                )
                cb.pack(side=tk.LEFT, padx=5)
                cb.set('None')
                self.pivot_vars[key] = var

            self.pivot_agg_var = tk.StringVar(master=self.root, value='sum')
            cb2 = ttk.Combobox(
                frame,
                textvariable=self.pivot_agg_var,
                values=list(AGGREGATOR_FUNCTIONS.keys()),
                state='readonly',
                width=8
            )
            cb2.pack(side=tk.LEFT, padx=5)
            ttk.Button(frame, text='Apply', command=self._refresh_pivot).pack(side=tk.LEFT)

            # Normalize buttons
            norm_frame = ttk.Frame(win)
            norm_frame.pack(fill=tk.X, padx=10, pady=5)
            ttk.Button(norm_frame, text='Normalize Rows', command=lambda: self._set_normalization('rows')).pack(
                side=tk.LEFT, padx=5)
            ttk.Button(norm_frame, text='Normalize Columns', command=lambda: self._set_normalization('columns')).pack(
                side=tk.LEFT, padx=5)
            ttk.Button(norm_frame, text='Clear Normalize', command=lambda: self._set_normalization(None)).pack(
                side=tk.LEFT, padx=5)

            # Filter section
            filter_frame = ttk.Frame(win)
            filter_frame.pack(fill=tk.X, padx=10, pady=5)
            ttk.Label(filter_frame, text='Filter Column:').pack(side=tk.LEFT)
            self.pivot_filter_col_var = tk.StringVar(master=self.root, value='None')
            self.pivot_filter_col_cb = ttk.Combobox(
                filter_frame,
                textvariable=self.pivot_filter_col_var,
                values=['None'] + self.dataframe.columns.tolist(),
                state='readonly',
                width=12
            )
            self.pivot_filter_col_cb.pack(side=tk.LEFT, padx=5)

            ttk.Label(filter_frame, text='Filter Value:').pack(side=tk.LEFT)
            self.pivot_filter_var = tk.StringVar(master=self.root, value='')
            ttk.Entry(filter_frame, textvariable=self.pivot_filter_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Button(filter_frame, text='Filter', command=self._refresh_pivot).pack(side=tk.LEFT, padx=5)

            # Output Treeview
            self.pivot_tree = ttk.Treeview(win, show='headings')
            self.pivot_tree.pack(fill=tk.BOTH, expand=True)

        self._open_single_popup("_pivot_window", "Pivot Table", widget, populate)

    def _set_normalization(self, mode):
        """
        Set the normalization mode for pivot data (by rows, columns, or none).
        """
        self.pivot_normalize = mode
        if self._pivot_window and self._pivot_window.winfo_exists():
            self._refresh_pivot()

    def _refresh_pivot(self):
        """
        Update and render the pivot table view based on user-selected settings and filters.
        """
        rows = self.pivot_vars['rows'].get()
        cols = self.pivot_vars['columns'].get()
        vals = self.pivot_vars['values'].get()
        if rows == 'None' or vals == 'None':
            return
        df = self.filtered_df

        # Apply pivot filters
        filter_col = self.pivot_filter_col_var.get()
        filter_val = self.pivot_filter_var.get().strip().lower()
        if filter_col != 'None' and filter_val:
            try:
                df = df[df[filter_col].astype(str).str.lower().str.contains(filter_val)]
            except Exception as e:
                print(f"Filter error: {e}")
                return

        try:
            if cols == 'None':
                pt = df.groupby(rows)[vals].agg(self.pivot_agg_var.get()).fillna(0)
                pt = pt.to_frame(name=vals)
                cols_list = [rows, vals]
            else:
                pt = df.pivot_table(index=rows, columns=cols, values=vals, aggfunc=self.pivot_agg_var.get()).fillna(0)
                cols_list = [rows] + list(map(str, pt.columns))
                pt.columns = list(map(str, pt.columns))
        except:
            return
        if self.pivot_normalize == 'rows':
            pt = pt.div(pt.sum(axis=1), axis=0) * 100
        elif self.pivot_normalize == 'columns':
            pt = pt.div(pt.sum(axis=0), axis=1) * 100

        self.pivot_tree.delete(*self.pivot_tree.get_children())
        self.pivot_tree['columns'] = list(map(str, cols_list))
        for c in cols_list:
            c_str = str(c)
            reverse_flag = (self.pivot_sorted_column == c_str and not self.pivot_sort_reverse)
            self.pivot_tree.heading(c_str, text=c_str,
                                    command=lambda c=c_str, rev=reverse_flag: self._sort_pivot(c, rev))
            self.pivot_tree.column(c_str, anchor='center')
        if cols == 'None':
            for idx, row in pt.iterrows():
                val = row[vals]
                if isinstance(val, (int, float)):
                    s = f"{val:,.2f}"
                    tmp = s.replace(",", "X").replace(".", ",").replace("X", ".")
                    if tmp.endswith(',00'):
                        tmp = tmp[:-3]
                    self.pivot_tree.insert('', 'end', values=[idx, tmp])
                else:
                    self.pivot_tree.insert('', 'end', values=[idx, str(val)])
        else:
            for idx, row in pt.iterrows():
                formatted_row = []
                for c in pt.columns:
                    val = row[c]
                    if isinstance(val, (int, float)):
                        s = f"{val:,.2f}"
                        tmp = s.replace(",", "X").replace(".", ",").replace("X", ".")
                        if tmp.endswith(',00'):
                            tmp = tmp[:-3]
                        formatted_row.append(tmp)
                    else:
                        formatted_row.append(str(val))
                self.pivot_tree.insert('', 'end', values=[idx] + formatted_row)
        if self.pivot_sorted_column in self.pivot_tree['columns']:
            self._sort_pivot(self.pivot_sorted_column, clicked=False)
        else:
            self._update_pivot_headings()

    def _sort_pivot(self, col, clicked=False):
        """
        Sort the pivot table as soon as a click happened in the headings.
        """
        if clicked:
            prev_state = self.pivot_sort_state.get(col)
            if prev_state is None:
                self.pivot_sort_state = {col: 'asc'}
            elif prev_state == 'asc':
                self.pivot_sort_state = {col: 'desc'}
            elif prev_state == 'desc':
                self.pivot_sort_state = {}

        # Get actual state
        state = self.pivot_sort_state.get(col)
        reverse = (state == 'desc')

        items = [(self.pivot_tree.set(i, col), i) for i in self.pivot_tree.get_children('')]
        try:
            items.sort(key=lambda t: float(t[0].replace(',', '.')), reverse=reverse)
        except:
            items.sort(key=lambda t: t[0], reverse=reverse)

        for idx, (_, i) in enumerate(items):
            self.pivot_tree.move(i, '', idx)

        # Update headings
        self.pivot_sorted_column = col
        self.pivot_sort_reverse = reverse
        self._update_pivot_headings()

    def _update_pivot_headings(self):
        """
        Update pivot heading inserting the arrows if a column is sorted.
        """
        for c in self.pivot_tree['columns']:
            state = self.pivot_sort_state.get(c)
            arrow = ''
            if state == 'asc':
                arrow = ' ↑'
            elif state == 'desc':
                arrow = ' ↓'
            self.pivot_tree.heading(c, text=c + arrow,
                                    command=lambda c=c: self._sort_pivot(c, clicked=True))

    def _toggle_sort(self, col):
        """
        Toggle the sorting state for a given column.

        This method manages the sorting directions when a column header is clicked:
        - If the column is not currently sorted, it adds it with ascending ('asc') order.
        - If the column is already sorted ascending, it changes the order to descending ('desc').
        - If the column is sorted descending, it removes the sorting for that column.

        Multiple columns can be sorted simultaneously, with the priority given to the order
        in which columns were clicked by. When a column’s sorting is removed, the priority
        shifts to the next active column in the list.

        This allows toggling through sorting states (ascending → descending → no sort)
        on each column click, supporting multi-column sorting with prioritization based
        on click order.
        """
        found = next(((i, d) for i, (c, d) in enumerate(self.sort_order) if c == col), None)

        if found is None:
            self.sort_order.append((col, 'asc'))
        else:
            idx, direction = found
            if direction == 'asc':
                self.sort_order[idx] = (col, 'desc')
            elif direction == 'desc':
                self.sort_order.pop(idx)

        self._update_table()

    def _update_sums(self, df):
        """
        Display aggregate summary values (e.g. sum, mean) below the table for selected numeric columns.
        """
        for w in self.sum_frame.winfo_children():
            w.destroy()
        for col, agg in self.aggregators.items():
            if agg != 'None' and col in df and pd.api.types.is_numeric_dtype(df[col]):
                val = AGGREGATOR_FUNCTIONS[agg](df[col])
                s = f"{val:,.2f}"
                tmp = s.replace(",", "X").replace(".", ",").replace("X", ".")
                if tmp.endswith(',00'):
                    tmp = tmp[:-3]
                ttk.Label(self.sum_frame, text=f"{col} ({agg}) : {tmp}   ").pack(side=tk.LEFT)

    def _reset_filters(self):
        """
        Clear all filters, sorting, and search settings, restoring the default unfiltered view.
        """
        for op_var, val_var in self.filter_vars.values():
            op_var.set('(None)')
            val_var.set('')

        self.quick_search_var.set('')
        self.quick_search_col_var.set('All')

        # Reset all columns
        for var in self.column_vars.values():
            var.set(True)
        self.selected_columns = list(self.dataframe.columns)
        self.column_filters = {}

        self.filtered_df = None

        # Reset sorting
        self.sort_order = []

        # Reset visible columns
        self.selected_columns = list(self.dataframe.columns)

        self.active_filters = {}
        self.column_filters_box = {}
        self._update_table()

    def run(self):
        """
        Start the Tkinter main loop and schedules periodic data updates.
        """
        self._initialize_gui()
        self._schedule_refresh()
        self.root.mainloop()

    def stop(self):
        """
        Stop the gui loop and closes the window.
        """
        self.running = False
        if self.root:
            self.root.destroy()
