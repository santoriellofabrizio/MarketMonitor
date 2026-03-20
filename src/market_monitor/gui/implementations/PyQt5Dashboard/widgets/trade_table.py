"""
Widget TradeTable con filtri avanzati AND/OR e infinite scrolling
"""

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import pandas as pd
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QObject, QEvent
from PyQt5.QtGui import QColor, QCursor, QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QPushButton, QGroupBox, QHeaderView, QSpinBox, QCheckBox,
    QMenu, QAction, QWidgetAction, QScrollArea, QFrame,
    QDialog, QDialogButtonBox, QListWidget, QListWidgetItem,
    QComboBox, QLineEdit, QColorDialog, QFormLayout, QStackedWidget,
    QSlider,
)

from market_monitor.gui.implementations.PyQt5Dashboard.common import safe_concat
from market_monitor.gui.implementations.PyQt5Dashboard.widgets.filter import AdvancedFilterDialog, FilterGroup

_CF_OPERATORS = [
    ">", ">=", "<", "<=", "==", "!=",
    "between",
    "contains", "not contains", "starts with", "ends with",
    "is empty", "not empty",
    "color scale",
    "formula",
]


# ==============================================================================
# CFRule — singola regola di conditional formatting
# ==============================================================================
@dataclass
class CFRule:
    """Regola di conditional formatting per una colonna."""

    operator: str
    value: str = ""
    value2: str = ""           # "between": secondo valore; "color scale": max
    value_is_col: bool = False  # True → value è il nome di un'altra colonna
    value2_is_col: bool = False # True → value2 è il nome di un'altra colonna
    bg_color: Optional[tuple] = None    # (r,g,b)  — per "color scale": colore min
    fg_color: Optional[tuple] = None    # (r,g,b)
    bold: bool = False
    apply_to_row: bool = False  # True → applica il formato all'intera riga
    condition_col: str = ""     # Se non vuoto, la condizione usa il valore di questa colonna
    # Campi specifici per "color scale"
    scale_max_color: Optional[tuple] = None   # colore al valore massimo
    scale_mid_value: str = ""                 # valore del punto medio (opzionale)
    scale_mid_color: Optional[tuple] = None   # colore al punto medio

    # ------------------------------------------------------------------
    # matching
    # ------------------------------------------------------------------
    def matches(self, cell_val, row_data: Optional[dict] = None) -> bool:
        """
        Verifica se la cella soddisfa la condizione.
        row_data: dizionario {col_name: valore} della riga corrente,
                  usato per risolvere i riferimenti a colonne.
        """
        if self.operator == "color scale":
            return True  # si applica sempre; il colore viene interpolato

        if self.operator == "formula":
            if not self.value or row_data is None:
                return False
            try:
                return bool(eval(self.value, {"__builtins__": {}}, dict(row_data)))  # noqa: S307
            except Exception:
                return False

        # Se condition_col è impostato, testa il valore di quella colonna invece della cella corrente
        if self.condition_col and row_data is not None and self.condition_col in row_data:
            cell_val = row_data[self.condition_col]

        op = self.operator
        is_empty = (
            cell_val is None
            or (isinstance(cell_val, float) and pd.isna(cell_val))
            or str(cell_val).strip() == ""
        )
        if op == "is empty":  return is_empty
        if op == "not empty": return not is_empty
        if is_empty:          return False

        # Risolvi riferimenti a colonna
        def _resolve(ref: str, is_col: bool) -> str:
            if is_col and row_data and ref in row_data:
                v = row_data[ref]
                return "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
            return ref

        cmp1 = _resolve(self.value,  self.value_is_col)
        cmp2 = _resolve(self.value2, self.value2_is_col)

        # Confronto numerico
        try:
            num = float(cell_val)
            if op in (">", ">=", "<", "<=", "==", "!=", "between"):
                v1 = float(cmp1)
                if op == ">":     return num > v1
                if op == ">=":    return num >= v1
                if op == "<":     return num < v1
                if op == "<=":    return num <= v1
                if op == "==":    return num == v1
                if op == "!=":    return num != v1
                if op == "between":
                    v2 = float(cmp2)
                    lo, hi = (v1, v2) if v1 <= v2 else (v2, v1)
                    return lo <= num <= hi
        except (ValueError, TypeError):
            pass

        # Confronto testuale
        s = str(cell_val)
        v = str(cmp1)
        if op == "==":           return s == v
        if op == "!=":           return s != v
        if op == "contains":     return v.lower() in s.lower()
        if op == "not contains": return v.lower() not in s.lower()
        if op == "starts with":  return s.lower().startswith(v.lower())
        if op == "ends with":    return s.lower().endswith(v.lower())
        return False

    # ------------------------------------------------------------------
    # formatting
    # ------------------------------------------------------------------
    def apply_to_item(
        self,
        item: QTableWidgetItem,
        val=None,
        col_min: Optional[float] = None,
        col_max: Optional[float] = None,
    ):
        if self.operator == "color scale":
            color = self._scale_color(val, col_min, col_max)
            if color:
                item.setBackground(color)
            return
        if self.bg_color:
            item.setBackground(QColor(*self.bg_color))
        if self.fg_color:
            item.setForeground(QColor(*self.fg_color))
        if self.bold:
            f = item.font()
            f.setBold(True)
            item.setFont(f)

    def _scale_color(
        self, val, col_min: Optional[float], col_max: Optional[float]
    ) -> Optional[QColor]:
        """Interpola il colore per 'color scale'."""
        if val is None or not isinstance(val, (int, float)):
            return None
        try:
            if pd.isna(val):
                return None
        except Exception:
            return None

        try:
            lo = float(self.value)  if self.value  else col_min
            hi = float(self.value2) if self.value2 else col_max
        except (ValueError, TypeError):
            lo, hi = col_min, col_max

        if lo is None or hi is None or hi <= lo:
            return None

        t = max(0.0, min(1.0, (float(val) - lo) / (hi - lo)))
        c1 = self.bg_color        or (255, 255, 255)
        c2 = self.scale_max_color or (255, 0,   0  )

        if self.scale_mid_value and self.scale_mid_color:
            try:
                mid   = float(self.scale_mid_value)
                t_mid = max(0.0, min(1.0, (mid - lo) / (hi - lo)))
                if t <= t_mid:
                    t_loc = (t / t_mid) if t_mid > 0 else 0.0
                    ca, cb = c1, self.scale_mid_color
                else:
                    t_loc = ((t - t_mid) / (1.0 - t_mid)) if (1.0 - t_mid) > 0 else 1.0
                    ca, cb = self.scale_mid_color, c2
            except (ValueError, TypeError):
                ca, cb, t_loc = c1, c2, t
        else:
            ca, cb, t_loc = c1, c2, t

        def lerp(a, b, x): return max(0, min(255, int(a + x * (b - a))))
        return QColor(lerp(ca[0], cb[0], t_loc),
                      lerp(ca[1], cb[1], t_loc),
                      lerp(ca[2], cb[2], t_loc))

    # ------------------------------------------------------------------
    # descrizione (per la lista nel dialog)
    # ------------------------------------------------------------------
    def describe(self) -> str:
        if self.operator == "color scale":
            mn = self.value  or "auto"
            mx = self.value2 or "auto"
            return f"color scale [{mn} … {mx}]"
        if self.operator == "formula":
            base = f"formula: {self.value}"
            return base + ("  [→ row]" if self.apply_to_row else "")
        if self.operator in ("is empty", "not empty"):
            base = self.operator
        else:
            v = f"col:{self.value}" if self.value_is_col else self.value
            if self.operator == "between":
                v2 = f"col:{self.value2}" if self.value2_is_col else self.value2
                base = f"between {v} and {v2}"
            else:
                base = f"{self.operator} {v}"
        if self.condition_col:
            base = f"[se {self.condition_col}] " + base
        return base + ("  [→ row]" if self.apply_to_row else "")

    # ------------------------------------------------------------------
    # serializzazione JSON
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "operator":        self.operator,
            "value":           self.value,
            "value2":          self.value2,
            "value_is_col":    self.value_is_col,
            "value2_is_col":   self.value2_is_col,
            "bg_color":        list(self.bg_color)        if self.bg_color        else None,
            "fg_color":        list(self.fg_color)        if self.fg_color        else None,
            "bold":            self.bold,
            "apply_to_row":    self.apply_to_row,
            "condition_col":   self.condition_col,
            "scale_max_color": list(self.scale_max_color) if self.scale_max_color else None,
            "scale_mid_value": self.scale_mid_value,
            "scale_mid_color": list(self.scale_mid_color) if self.scale_mid_color else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CFRule":
        def _t(v): return tuple(v) if v else None
        return cls(
            operator=        d.get("operator",        "=="),
            value=           d.get("value",           ""),
            value2=          d.get("value2",          ""),
            value_is_col=    d.get("value_is_col",    False),
            value2_is_col=   d.get("value2_is_col",   False),
            bg_color=        _t(d.get("bg_color")),
            fg_color=        _t(d.get("fg_color")),
            bold=            d.get("bold",            False),
            apply_to_row=    d.get("apply_to_row",    False),
            condition_col=   d.get("condition_col",   ""),
            scale_max_color= _t(d.get("scale_max_color")),
            scale_mid_value= d.get("scale_mid_value", ""),
            scale_mid_color= _t(d.get("scale_mid_color")),
        )


# ==============================================================================
# CFRuleEditDialog — aggiunta / modifica di una singola regola
# ==============================================================================
class CFRuleEditDialog(QDialog):
    """
    Dialog per aggiungere o modificare una regola CF.

    Funzionalità:
    • Operatori normali: confronto valore letterale o valore di un'altra colonna
    • "color scale": gradiente fra min e max con punto medio opzionale
    """

    def __init__(
        self,
        col_name: str,
        col_dtype=None,
        rule: Optional[CFRule] = None,
        available_columns: Optional[List[str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(("Modifica" if rule else "Aggiungi") + f" Regola — {col_name}")
        self.setMinimumWidth(480)

        self._available_columns: List[str] = available_columns or []
        is_scale = rule and rule.operator == "color scale"

        # colori interni
        self._bg_color:         Optional[tuple] = rule.bg_color         if rule else None
        self._fg_color:         Optional[tuple] = rule.fg_color         if rule else None
        self._scale_max_color:  Optional[tuple] = rule.scale_max_color  if rule else None
        self._scale_mid_color:  Optional[tuple] = rule.scale_mid_color  if rule else None

        root = QVBoxLayout(self)

        # ── Selezione operatore ──────────────────────────────────────────────
        op_row = QWidget()
        op_h = QHBoxLayout(op_row)
        op_h.setContentsMargins(0, 0, 0, 0)
        op_h.addWidget(QLabel("Operatore:"))
        self.op_combo = QComboBox()
        self.op_combo.addItems(_CF_OPERATORS)
        if rule:
            idx = self.op_combo.findText(rule.operator)
            if idx >= 0:
                self.op_combo.setCurrentIndex(idx)
        self.op_combo.currentTextChanged.connect(self._on_op_changed)
        op_h.addWidget(self.op_combo)
        op_h.addStretch()
        root.addWidget(op_row)

        # ── Condizione (nascosta per "color scale") ──────────────────────────
        self.cond_box = QGroupBox("Condizione")
        cond_v = QVBoxLayout(self.cond_box)

        # ── Formula row (visibile solo quando op == "formula") ────────────────
        self.formula_row = QWidget()
        formula_h = QHBoxLayout(self.formula_row)
        formula_h.setContentsMargins(0, 0, 0, 0)
        formula_h.addWidget(QLabel("Expr:"))
        is_formula = rule and rule.operator == "formula"
        self.formula_edit = QLineEdit(rule.value if is_formula else "")
        self.formula_edit.setPlaceholderText("es. spread_pl / ctv < 0.01")
        cols_hint = ", ".join(self._available_columns[:8])
        self.formula_edit.setToolTip(
            f"Espressione Python con i nomi delle colonne come variabili.\n"
            f"Colonne disponibili: {cols_hint}…\n"
            f"Esempio: spread_pl / ctv < 0.01"
        )
        formula_h.addWidget(self.formula_edit)
        cond_v.addWidget(self.formula_row)

        # riga valore
        self.cond_val_row = QWidget()
        cond_h = QHBoxLayout(self.cond_val_row)
        cond_h.setContentsMargins(0, 0, 0, 0)
        cond_v.addWidget(self.cond_val_row)

        cond_h.addWidget(QLabel("Value"))

        # val1: QStackedWidget (pagina 0 = testo, pagina 1 = colonna)
        self.val1_stack = QStackedWidget()
        val1_literal = rule.value if (rule and not rule.value_is_col and not is_scale and not is_formula) else ""
        self.val1_edit = QLineEdit(val1_literal)
        self.val1_edit.setPlaceholderText("valore")
        self.val1_col_combo = QComboBox()
        self.val1_col_combo.addItems(self._available_columns)
        if rule and rule.value_is_col:
            idx = self.val1_col_combo.findText(rule.value)
            if idx >= 0:
                self.val1_col_combo.setCurrentIndex(idx)
        self.val1_stack.addWidget(self.val1_edit)
        self.val1_stack.addWidget(self.val1_col_combo)
        if rule and rule.value_is_col:
            self.val1_stack.setCurrentIndex(1)

        self.val1_col_btn = QPushButton("⧫")
        self.val1_col_btn.setCheckable(True)
        self.val1_col_btn.setFixedWidth(28)
        self.val1_col_btn.setToolTip("Usa il valore di un'altra colonna come confronto")
        self.val1_col_btn.setChecked(bool(rule and rule.value_is_col))
        self.val1_col_btn.clicked.connect(
            lambda c: self.val1_stack.setCurrentIndex(1 if c else 0)
        )
        cond_h.addWidget(self.val1_stack)
        cond_h.addWidget(self.val1_col_btn)

        self.val2_label = QLabel("and")
        cond_h.addWidget(self.val2_label)

        # val2: stesso schema
        self.val2_stack = QStackedWidget()
        val2_literal = rule.value2 if (rule and not rule.value2_is_col) else ""
        self.val2_edit = QLineEdit(val2_literal)
        self.val2_edit.setPlaceholderText("valore")
        self.val2_col_combo = QComboBox()
        self.val2_col_combo.addItems(self._available_columns)
        if rule and rule.value2_is_col:
            idx = self.val2_col_combo.findText(rule.value2)
            if idx >= 0:
                self.val2_col_combo.setCurrentIndex(idx)
        self.val2_stack.addWidget(self.val2_edit)
        self.val2_stack.addWidget(self.val2_col_combo)
        if rule and rule.value2_is_col:
            self.val2_stack.setCurrentIndex(1)

        self.val2_col_btn = QPushButton("⧫")
        self.val2_col_btn.setCheckable(True)
        self.val2_col_btn.setFixedWidth(28)
        self.val2_col_btn.setToolTip("Usa il valore di un'altra colonna come confronto")
        self.val2_col_btn.setChecked(bool(rule and rule.value2_is_col))
        self.val2_col_btn.clicked.connect(
            lambda c: self.val2_stack.setCurrentIndex(1 if c else 0)
        )
        cond_h.addWidget(self.val2_stack)
        cond_h.addWidget(self.val2_col_btn)

        # riga "Se colonna:" — condition_col
        cond_col_row = QWidget()
        cond_col_h = QHBoxLayout(cond_col_row)
        cond_col_h.setContentsMargins(0, 0, 0, 0)
        cond_col_h.addWidget(QLabel("Se colonna:"))
        self.condition_col_combo = QComboBox()
        self.condition_col_combo.addItem("(questa colonna)")
        self.condition_col_combo.addItems(self._available_columns)
        existing_cond_col = rule.condition_col if rule else ""
        if existing_cond_col:
            idx = self.condition_col_combo.findText(existing_cond_col)
            if idx >= 0:
                self.condition_col_combo.setCurrentIndex(idx)
        cond_col_h.addWidget(self.condition_col_combo)
        cond_col_h.addStretch()
        cond_v.addWidget(cond_col_row)

        root.addWidget(self.cond_box)

        # ── Intervallo color scale (nascosto per operatori normali) ──────────
        self.scale_range_box = QGroupBox("Intervallo  (vuoto = auto da min/max colonna)")
        sr_form = QFormLayout(self.scale_range_box)
        self.scale_min_edit = QLineEdit(rule.value  if (rule and is_scale) else "")
        self.scale_min_edit.setPlaceholderText("min  (vuoto = auto)")
        self.scale_max_edit = QLineEdit(rule.value2 if (rule and is_scale) else "")
        self.scale_max_edit.setPlaceholderText("max  (vuoto = auto)")
        sr_form.addRow("Min:", self.scale_min_edit)
        sr_form.addRow("Max:", self.scale_max_edit)
        root.addWidget(self.scale_range_box)

        # ── Formato normale (nascosto per "color scale") ─────────────────────
        self.fmt_box = QGroupBox("Formato")
        fmt_form = QFormLayout(self.fmt_box)

        bg_row = self._make_color_row(
            self._bg_color,
            lambda c: self._set_color("bg", c),
            lambda: self._set_color("bg", None),
            "Colore di sfondo",
        )
        self.bg_swatch = bg_row[0]
        fmt_form.addRow("Background:", bg_row[1])

        fg_row = self._make_color_row(
            self._fg_color,
            lambda c: self._set_color("fg", c),
            lambda: self._set_color("fg", None),
            "Colore del testo",
        )
        self.fg_swatch = fg_row[0]
        fmt_form.addRow("Text color:", fg_row[1])

        self.bold_cb = QCheckBox("Bold")
        self.bold_cb.setChecked(rule.bold if rule else False)
        fmt_form.addRow("Font:", self.bold_cb)

        self.apply_to_row_cb = QCheckBox("Apply to entire row")
        self.apply_to_row_cb.setToolTip(
            "Quando la condizione è soddisfatta, applica il formato a tutta la riga"
        )
        self.apply_to_row_cb.setChecked(rule.apply_to_row if rule else False)
        fmt_form.addRow("Scope:", self.apply_to_row_cb)

        root.addWidget(self.fmt_box)

        # ── Colori scala (nascosto per operatori normali) ────────────────────
        self.scale_fmt_box = QGroupBox("Colori della scala")
        sf_form = QFormLayout(self.scale_fmt_box)

        sc_min_row = self._make_color_row(
            self._bg_color,
            lambda c: self._set_color("scale_min", c),
            None,
            "Colore al valore minimo",
        )
        self.scale_min_swatch = sc_min_row[0]
        sf_form.addRow("Colore min:", sc_min_row[1])

        sc_max_row = self._make_color_row(
            self._scale_max_color,
            lambda c: self._set_color("scale_max", c),
            None,
            "Colore al valore massimo",
        )
        self.scale_max_swatch = sc_max_row[0]
        sf_form.addRow("Colore max:", sc_max_row[1])

        # Punto medio
        has_mid = bool(rule and rule.scale_mid_value)
        self.mid_enable_cb = QCheckBox("Punto medio")
        self.mid_enable_cb.setChecked(has_mid)
        self.mid_enable_cb.stateChanged.connect(self._on_mid_toggled)
        sf_form.addRow(self.mid_enable_cb)

        self.mid_container = QWidget()
        mid_form = QFormLayout(self.mid_container)
        mid_form.setContentsMargins(16, 0, 0, 0)
        self.mid_value_edit = QLineEdit(rule.scale_mid_value if rule else "")
        self.mid_value_edit.setPlaceholderText("valore")
        mid_form.addRow("Valore:", self.mid_value_edit)
        sc_mid_row = self._make_color_row(
            self._scale_mid_color,
            lambda c: self._set_color("scale_mid", c),
            None,
            "Colore al punto medio",
        )
        self.scale_mid_swatch = sc_mid_row[0]
        mid_form.addRow("Colore:", sc_mid_row[1])
        self.mid_container.setVisible(has_mid)
        sf_form.addRow(self.mid_container)

        root.addWidget(self.scale_fmt_box)

        # ── Bottoni dialog ───────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self._on_op_changed(self.op_combo.currentText())

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _make_color_row(self, color, on_pick, on_none, title: str):
        """Crea una riga [swatch][Pick][None] e restituisce (swatch, container)."""
        swatch = QLabel()
        swatch.setFixedSize(54, 22)
        self._refresh_swatch(swatch, color)
        pick_btn = QPushButton("Pick…")
        pick_btn.clicked.connect(lambda: self._pick_color(swatch, on_pick, title))
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(swatch)
        h.addWidget(pick_btn)
        if on_none is not None:
            none_btn = QPushButton("None")
            none_btn.clicked.connect(lambda: (on_none(), self._refresh_swatch(swatch, None)))
            h.addWidget(none_btn)
        h.addStretch()
        return swatch, container

    def _pick_color(self, swatch: QLabel, callback, title: str):
        c = QColorDialog.getColor(QColor(200, 200, 200), self, title)
        if c.isValid():
            color = (c.red(), c.green(), c.blue())
            callback(color)
            self._refresh_swatch(swatch, color)

    def _set_color(self, which: str, color: Optional[tuple]):
        if which == "bg":
            self._bg_color = color
            self._refresh_swatch(self.bg_swatch, color)
        elif which == "fg":
            self._fg_color = color
            self._refresh_swatch(self.fg_swatch, color)
        elif which == "scale_min":
            self._bg_color = color
            self._refresh_swatch(self.scale_min_swatch, color)
        elif which == "scale_max":
            self._scale_max_color = color
            self._refresh_swatch(self.scale_max_swatch, color)
        elif which == "scale_mid":
            self._scale_mid_color = color
            self._refresh_swatch(self.scale_mid_swatch, color)

    @staticmethod
    def _refresh_swatch(label: QLabel, color: Optional[tuple]):
        if color:
            label.setStyleSheet(
                f"background-color: rgb({color[0]},{color[1]},{color[2]});"
                "border: 1px solid #888;"
            )
            label.setText("")
        else:
            label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #aaa;")
            label.setText("  —  ")

    def _on_op_changed(self, op: str):
        is_scale   = op == "color scale"
        is_formula = op == "formula"
        no_val     = op in ("is empty", "not empty", "color scale", "formula")
        is_between = op == "between"

        self.cond_box.setVisible(not is_scale)
        self.scale_range_box.setVisible(is_scale)
        self.fmt_box.setVisible(not is_scale)
        self.scale_fmt_box.setVisible(is_scale)

        if not is_scale:
            self.formula_row.setVisible(is_formula)
            self.cond_val_row.setVisible(not is_formula)
            if not is_formula:
                self.val1_stack.setVisible(not no_val)
                self.val1_col_btn.setVisible(not no_val)
                self.val2_label.setVisible(is_between)
                self.val2_stack.setVisible(is_between)
                self.val2_col_btn.setVisible(is_between)

    def _on_mid_toggled(self, state: int):
        self.mid_container.setVisible(state == Qt.Checked)

    # ── Lettura risultato ────────────────────────────────────────────────────

    def get_rule(self) -> CFRule:
        op = self.op_combo.currentText()
        if op == "color scale":
            return CFRule(
                operator="color scale",
                value=self.scale_min_edit.text().strip(),
                value2=self.scale_max_edit.text().strip(),
                bg_color=self._bg_color,
                scale_max_color=self._scale_max_color,
                scale_mid_value=(
                    self.mid_value_edit.text().strip()
                    if self.mid_enable_cb.isChecked() else ""
                ),
                scale_mid_color=(
                    self._scale_mid_color if self.mid_enable_cb.isChecked() else None
                ),
            )

        if op == "formula":
            return CFRule(
                operator="formula",
                value=self.formula_edit.text().strip(),
                bg_color=self._bg_color,
                fg_color=self._fg_color,
                bold=self.bold_cb.isChecked(),
                apply_to_row=self.apply_to_row_cb.isChecked(),
            )

        v1_col = self.val1_col_btn.isChecked()
        v1 = self.val1_col_combo.currentText() if v1_col else self.val1_edit.text().strip()
        v2_col = self.val2_col_btn.isChecked()
        v2 = self.val2_col_combo.currentText() if v2_col else self.val2_edit.text().strip()

        cond_col_text = self.condition_col_combo.currentText()
        cond_col = "" if cond_col_text == "(questa colonna)" else cond_col_text

        return CFRule(
            operator=op,
            value=v1,
            value2=v2,
            value_is_col=v1_col,
            value2_is_col=v2_col,
            bg_color=self._bg_color,
            fg_color=self._fg_color,
            bold=self.bold_cb.isChecked(),
            apply_to_row=self.apply_to_row_cb.isChecked(),
            condition_col=cond_col,
        )


# ==============================================================================
# CFRulesDialog — gestione dell'elenco di regole di una colonna
# ==============================================================================
class CFRulesDialog(QDialog):
    """Dialog per gestire le regole CF di una colonna (stile Excel)."""

    def __init__(
        self,
        col_name: str,
        col_dtype,
        rules: List[CFRule],
        available_columns: Optional[List[str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Conditional Formatting — {col_name}")
        self.setMinimumSize(540, 380)
        self._col_name = col_name
        self._col_dtype = col_dtype
        self._available_columns: List[str] = available_columns or []
        self._rules: List[CFRule] = list(rules)

        root = QVBoxLayout(self)

        hint = QLabel(
            "Le regole vengono applicate dall'alto verso il basso. "
            "Vince la prima che corrisponde (come in Excel)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555; font-style: italic;")
        root.addWidget(hint)

        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.itemDoubleClicked.connect(self._edit_rule)
        root.addWidget(self.list_widget)

        # Toolbar
        toolbar = QWidget()
        tb_h = QHBoxLayout(toolbar)
        tb_h.setContentsMargins(0, 0, 0, 0)

        add_btn = QPushButton("➕ Aggiungi")
        add_btn.clicked.connect(self._add_rule)
        tb_h.addWidget(add_btn)

        edit_btn = QPushButton("✏️ Modifica")
        edit_btn.clicked.connect(self._edit_rule)
        tb_h.addWidget(edit_btn)

        del_btn = QPushButton("🗑 Elimina")
        del_btn.clicked.connect(self._delete_rule)
        tb_h.addWidget(del_btn)

        tb_h.addStretch()

        up_btn = QPushButton("▲")
        up_btn.setFixedWidth(34)
        up_btn.clicked.connect(self._move_up)
        tb_h.addWidget(up_btn)

        dn_btn = QPushButton("▼")
        dn_btn.setFixedWidth(34)
        dn_btn.clicked.connect(self._move_down)
        tb_h.addWidget(dn_btn)

        root.addWidget(toolbar)

        clr_btn = QPushButton("Cancella tutte le regole")
        clr_btn.clicked.connect(self._clear_all)
        root.addWidget(clr_btn)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self._rebuild_list()

    def _rebuild_list(self):
        cur = self.list_widget.currentRow()
        self.list_widget.clear()
        for rule in self._rules:
            text = f"Value {rule.describe()}"
            if rule.operator == "color scale":
                parts = []
                if rule.bg_color:        parts.append(f"min rgb{rule.bg_color}")
                if rule.scale_max_color: parts.append(f"max rgb{rule.scale_max_color}")
                if rule.scale_mid_value: parts.append(f"mid={rule.scale_mid_value}")
                if parts: text += f"   →   {', '.join(parts)}"
            else:
                parts = []
                if rule.bg_color: parts.append(f"bg: rgb{rule.bg_color}")
                if rule.fg_color: parts.append(f"testo: rgb{rule.fg_color}")
                if rule.bold:     parts.append("bold")
                if parts: text += f"   →   {', '.join(parts)}"

            item = QListWidgetItem(text)
            preview_color = rule.bg_color or rule.scale_max_color
            if preview_color:
                item.setBackground(QColor(*preview_color))
            if rule.fg_color:
                item.setForeground(QColor(*rule.fg_color))
            if rule.bold:
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            self.list_widget.addItem(item)
        if 0 <= cur < self.list_widget.count():
            self.list_widget.setCurrentRow(cur)

    def _idx(self) -> int:
        return self.list_widget.currentRow()

    def _add_rule(self):
        dlg = CFRuleEditDialog(
            self._col_name, self._col_dtype,
            available_columns=self._available_columns, parent=self,
        )
        if dlg.exec_():
            self._rules.append(dlg.get_rule())
            self._rebuild_list()
            self.list_widget.setCurrentRow(len(self._rules) - 1)

    def _edit_rule(self):
        idx = self._idx()
        if idx < 0:
            return
        dlg = CFRuleEditDialog(
            self._col_name, self._col_dtype,
            rule=self._rules[idx],
            available_columns=self._available_columns, parent=self,
        )
        if dlg.exec_():
            self._rules[idx] = dlg.get_rule()
            self._rebuild_list()

    def _delete_rule(self):
        idx = self._idx()
        if idx < 0:
            return
        self._rules.pop(idx)
        self._rebuild_list()

    def _move_up(self):
        idx = self._idx()
        if idx <= 0:
            return
        self._rules[idx - 1], self._rules[idx] = self._rules[idx], self._rules[idx - 1]
        self._rebuild_list()
        self.list_widget.setCurrentRow(idx - 1)

    def _move_down(self):
        idx = self._idx()
        if idx < 0 or idx >= len(self._rules) - 1:
            return
        self._rules[idx], self._rules[idx + 1] = self._rules[idx + 1], self._rules[idx]
        self._rebuild_list()
        self.list_widget.setCurrentRow(idx + 1)

    def _clear_all(self):
        self._rules.clear()
        self._rebuild_list()

    def get_rules(self) -> List[CFRule]:
        return list(self._rules)


class NumericTableWidgetItem(QTableWidgetItem):
    """
    QTableWidgetItem che supporta sorting numerico.
    Memorizza il valore originale e sovrascrive __lt__ per confronto corretto.
    """

    def __init__(self, display_text: str, sort_value=None):
        super().__init__(display_text)
        self._sort_value = sort_value

    def __lt__(self, other):
        if not isinstance(other, NumericTableWidgetItem):
            return super().__lt__(other)

        # Se entrambi hanno valori numerici, confronta numericamente
        self_val = self._sort_value
        other_val = other._sort_value

        # Gestisci None/NaN come minori di tutto
        self_is_none = self_val is None or (isinstance(self_val, float) and pd.isna(self_val))
        other_is_none = other_val is None or (isinstance(other_val, float) and pd.isna(other_val))

        if self_is_none and other_is_none:
            return False
        if self_is_none:
            return True  # None va in fondo
        if other_is_none:
            return False

        # Confronto numerico se entrambi sono numeri
        if isinstance(self_val, (int, float)) and isinstance(other_val, (int, float)):
            return self_val < other_val

        # Confronto datetime
        if isinstance(self_val, (pd.Timestamp, datetime.datetime)) and isinstance(other_val, (pd.Timestamp, datetime.datetime)):
            return self_val < other_val

        # Fallback a confronto stringa
        return self.text() < other.text()


class TradeTableWidget(QWidget):
    """
    Visualizza trades con:
    - filtri avanzati AND/OR
    - infinite scrolling
    - formatting per-colonna
    """

    filtered_data_changed = pyqtSignal(pd.DataFrame)

    # ==========================================================
    # INIT
    # ==========================================================
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        max_rows: int = 1000,
        dedup_column: str = "trade_index",
        datetime_columns="timestamp",
        datetime_format: str = "%H:%M:%S.%f",
        cf_rules_path: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)

        # ---- Data ----
        self.all_data = pd.DataFrame()
        self.filtered_data = pd.DataFrame()

        self.visible_columns = columns or []
        self.max_rows = max_rows
        self.dedup_column = dedup_column

        self.datetime_columns = (
            datetime_columns if isinstance(datetime_columns, list)
            else [datetime_columns]
        )
        self.datetime_format = datetime_format
        self.column_decimals: dict[str, int] = {}

        # ---- Conditional Formatting ----
        # Regole per-colonna stile Excel: {col_name: [CFRule, ...]}
        self._cf_rules: dict[str, List[CFRule]] = {}
        # Percorso file di persistenza
        self._cf_rules_path: str = cf_rules_path or str(
            Path.home() / ".config" / "marketmonitor" / "cf_rules.json"
        )
        self._load_cf_rules()

        # ---- Filters ----
        self.active_filter: Optional[FilterGroup] = None
        # Filtri per valori colonna: {col_name: set di valori esclusi}
        self._column_value_filters: dict[str, set] = {}

        # ---- Infinite scroll ----
        self.displayed_rows = 0
        self.rows_per_batch = 100
        self.is_loading = False

        # ---- Row height / zoom ----
        self._row_height: int = 22        # default px
        self._ROW_HEIGHT_MIN: int = 6
        self._ROW_HEIGHT_MAX: int = 60
        self._ROW_HEIGHT_STEP: int = 2
        self._font_size_pt: int = 0       # 0 = auto (scales with row height)

        self._setup_ui()

    # ==========================================================
    # UI
    # ==========================================================
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # ---------- CONTROLS ----------
        controls = QGroupBox("Table Controls")
        controls_layout = QHBoxLayout()

        self.autoscroll_checkbox = QCheckBox("Auto-scroll")
        self.autoscroll_checkbox.setChecked(True)
        controls_layout.addWidget(self.autoscroll_checkbox)

        self.advanced_filter_btn = QPushButton("🔍 Advanced Filter")
        self.advanced_filter_btn.clicked.connect(
            self._show_advanced_filter_dialog
        )
        controls_layout.addWidget(self.advanced_filter_btn)

        clear_btn = QPushButton("Clear Filters")
        clear_btn.clicked.connect(self._clear_all_filters)
        controls_layout.addWidget(clear_btn)

        clear_cf_btn = QPushButton("🚫 Remove All CF")
        clear_cf_btn.setToolTip("Rimuove tutte le regole di conditional formatting")
        clear_cf_btn.clicked.connect(self._remove_all_cf)
        controls_layout.addWidget(clear_cf_btn)

        controls_layout.addStretch()

        # ---- Zoom button ----
        self._zoom_btn = QPushButton("🔍 Zoom")
        self._zoom_btn.setToolTip("Adjust row height and font size  (Ctrl+wheel to change row height)")
        self._zoom_btn.clicked.connect(self._show_zoom_menu)
        controls_layout.addWidget(self._zoom_btn)

        self.filter_info_label = QLabel("No filters active")
        controls_layout.addWidget(self.filter_info_label)

        self.info_label = QLabel("No data")
        controls_layout.addWidget(self.info_label)

        controls.setLayout(controls_layout)
        layout.addWidget(controls)

        # ---------- TABLE ----------
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionsMovable(True)
        header.sectionMoved.connect(self._on_section_moved)
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(
            self._show_header_context_menu
        )
        # Doppio click per filtro valori
        header.sectionDoubleClicked.connect(self._show_column_filter_menu)

        self.table.verticalScrollBar().valueChanged.connect(
            self._on_scroll
        )

        # Ctrl+wheel → zoom
        self.table.installEventFilter(self)
        self.table.viewport().installEventFilter(self)

        # Store default font size (read after table is created so Qt has set it)
        self._default_font_pt: int = max(self.table.font().pointSize(), 9)

        # Apply initial zoom
        self._apply_zoom()

        layout.addWidget(self.table)

    # ==========================================================
    # COLUMN REORDER (header drag)
    # ==========================================================
    def _on_section_moved(self, logical: int, old_visual: int, new_visual: int):
        """Aggiorna visible_columns quando l'utente trascina un'intestazione di colonna."""
        header = self.table.horizontalHeader()
        n = self.table.columnCount()
        new_order = [
            self.table.horizontalHeaderItem(header.logicalIndex(vi)).text()
            for vi in range(n)
        ]
        self.visible_columns = new_order
        self._refresh_view()

    # ==========================================================
    # ZOOM
    # ==========================================================

    def _apply_zoom(self):
        """Apply _row_height and _font_size_pt to the table."""
        # Row height
        vh = self.table.verticalHeader()
        vh.setDefaultSectionSize(self._row_height)
        vh.setSectionResizeMode(QHeaderView.Fixed)

        # Font: manual override or auto-scale with row height
        if self._font_size_pt > 0:
            pt = self._font_size_pt
        else:
            # Auto: scale proportionally, threshold = default row height (22)
            pt = max(6, round(self._default_font_pt * self._row_height / 22))
            pt = min(pt, self._default_font_pt * 3)

        f = self.table.font()
        f.setPointSize(pt)
        self.table.setFont(f)

        self._update_zoom_btn_label()

    # keep old name as alias for dashboard_state compatibility
    def _apply_row_height(self):
        self._apply_zoom()

    def _update_zoom_btn_label(self):
        pt = self._font_size_pt if self._font_size_pt > 0 else self.table.font().pointSize()
        self._zoom_btn.setText(f"🔍 {self._row_height}px / {pt}pt")

    def _zoom_in(self):
        new_h = min(self._row_height + self._ROW_HEIGHT_STEP, self._ROW_HEIGHT_MAX)
        if new_h != self._row_height:
            self._row_height = new_h
            self._apply_zoom()

    def _zoom_out(self):
        new_h = max(self._row_height - self._ROW_HEIGHT_STEP, self._ROW_HEIGHT_MIN)
        if new_h != self._row_height:
            self._row_height = new_h
            self._apply_zoom()

    def _show_zoom_menu(self):
        """Show a dropdown menu with row height and font size sliders."""
        menu = QMenu(self)
        menu.setMinimumWidth(260)

        def _make_slider_widget(label_text, value, lo, hi, step, on_change, reset_val):
            w = QWidget()
            w.setContentsMargins(8, 4, 8, 4)
            row = QHBoxLayout(w)
            row.setSpacing(6)

            lbl = QLabel(label_text)
            lbl.setFixedWidth(68)
            row.addWidget(lbl)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(lo, hi)
            slider.setSingleStep(step)
            slider.setPageStep(step * 2)
            slider.setValue(value)
            slider.setFixedWidth(110)
            row.addWidget(slider)

            val_lbl = QLabel(str(value))
            val_lbl.setFixedWidth(28)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(val_lbl)

            reset_btn = QPushButton("↺")
            reset_btn.setFixedWidth(22)
            reset_btn.setToolTip(f"Reset to {reset_val}")
            row.addWidget(reset_btn)

            def _slider_moved(v):
                val_lbl.setText(str(v))
                on_change(v)

            slider.valueChanged.connect(_slider_moved)
            reset_btn.clicked.connect(lambda: slider.setValue(reset_val))

            return w

        # ── Row Height ────────────────────────────────────────────
        rh_title = QAction("  Row Height", menu)
        rh_title.setEnabled(False)
        menu.addAction(rh_title)

        rh_widget = _make_slider_widget(
            "Height (px)",
            self._row_height,
            self._ROW_HEIGHT_MIN, self._ROW_HEIGHT_MAX, self._ROW_HEIGHT_STEP,
            lambda v: (setattr(self, '_row_height', v), self._apply_zoom()),
            22,
        )
        rh_action = QWidgetAction(menu)
        rh_action.setDefaultWidget(rh_widget)
        menu.addAction(rh_action)

        menu.addSeparator()

        # ── Font Size ─────────────────────────────────────────────
        fs_title = QAction("  Font Size", menu)
        fs_title.setEnabled(False)
        menu.addAction(fs_title)

        cur_font_pt = (self._font_size_pt if self._font_size_pt > 0
                       else self.table.font().pointSize())
        fs_widget = _make_slider_widget(
            "Font (pt)",
            cur_font_pt,
            6, 24, 1,
            lambda v: (setattr(self, '_font_size_pt', v), self._apply_zoom()),
            0,  # reset → auto
        )
        fs_action = QWidgetAction(menu)
        fs_action.setDefaultWidget(fs_widget)
        menu.addAction(fs_action)

        # "Auto font" toggle
        auto_action = QAction("  Auto font (scales with row height)", menu)
        auto_action.setCheckable(True)
        auto_action.setChecked(self._font_size_pt == 0)
        def _toggle_auto(checked):
            self._font_size_pt = 0 if checked else self.table.font().pointSize()
            self._apply_zoom()
        auto_action.triggered.connect(_toggle_auto)
        menu.addAction(auto_action)

        menu.addSeparator()

        reset_all = QAction("  Reset all", menu)
        def _reset():
            self._row_height = 22
            self._font_size_pt = 0
            self._apply_zoom()
        reset_all.triggered.connect(_reset)
        menu.addAction(reset_all)

        menu.exec_(self._zoom_btn.mapToGlobal(
            self._zoom_btn.rect().bottomLeft()
        ))

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Wheel
                and event.modifiers() & Qt.ControlModifier):
            delta = event.angleDelta().y()
            if delta > 0:
                self._zoom_in()
            elif delta < 0:
                self._zoom_out()
            return True          # consume event
        return super().eventFilter(obj, event)

    # ==========================================================
    # HEADER CONTEXT MENU
    # ==========================================================
    def _show_header_context_menu(self, pos: QPoint):
        col = self.table.horizontalHeader().logicalIndexAt(pos)
        if col < 0:
            return

        name = self.table.horizontalHeaderItem(col).text()
        if name not in self.filtered_data.columns:
            return

        series = self.filtered_data[name]
        menu = QMenu(self)

        title = QAction(f"📊 {name}", self)
        title.setEnabled(False)
        menu.addAction(title)
        menu.addSeparator()

        if pd.api.types.is_numeric_dtype(series):
            spin = QSpinBox()
            spin.setRange(0, 6)
            spin.setValue(self.column_decimals.get(name, 2))

            spin.valueChanged.connect(
                lambda v, c=name: self._set_column_decimals(c, v)
            )

            w = QWidget()
            l = QHBoxLayout(w)
            l.addWidget(QLabel("Decimals"))
            l.addWidget(spin)

            act = QWidgetAction(self)
            act.setDefaultWidget(w)
            menu.addAction(act)

        menu.addSeparator()
        n_rules = len(self._cf_rules.get(name, []))
        cf_label = f"🎨 Conditional Formatting…" + (f" ({n_rules})" if n_rules else "")
        cf_action = QAction(cf_label, self)
        cf_action.triggered.connect(lambda checked=False, c=name: self._show_cf_rules_dialog(c))
        menu.addAction(cf_action)

        menu.exec_(QCursor.pos())

    def _show_column_filter_menu(self, col_index: int):
        """Mostra menu con checkbox per filtrare i valori della colonna."""
        if col_index < 0 or self.all_data.empty:
            return

        header_item = self.table.horizontalHeaderItem(col_index)
        if not header_item:
            return

        col_name = header_item.text()
        if col_name not in self.all_data.columns:
            return

        # Ottieni valori unici (usa all_data per avere tutti i valori possibili)
        unique_values = self.all_data[col_name].dropna().unique()

        # Limita a 100 valori per performance
        if len(unique_values) > 100:
            unique_values = unique_values[:100]

        # Ordina i valori
        try:
            unique_values = sorted(unique_values)
        except TypeError:
            unique_values = sorted(unique_values, key=str)

        # Valori attualmente esclusi (copia per stato temporaneo)
        excluded = self._column_value_filters.get(col_name, set()).copy()
        # Stato temporaneo per questo menu
        temp_excluded = [excluded]  # Lista per permettere modifica nella closure

        # Crea menu
        menu = QMenu(self)
        menu.setMinimumWidth(200)

        # Titolo
        title = QAction(f"🔍 Filter: {col_name}", self)
        title.setEnabled(False)
        menu.addAction(title)
        menu.addSeparator()

        # Select All / Unselect All (usando QPushButton per non chiudere il menu)
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.setContentsMargins(5, 2, 5, 2)
        btn_layout.setSpacing(5)

        select_all_btn = QPushButton("✓ Select All")
        select_all_btn.clicked.connect(
            lambda: self._filter_select_all_temp(checkboxes, temp_excluded)
        )
        btn_layout.addWidget(select_all_btn)

        unselect_all_btn = QPushButton("✗ Unselect All")
        unselect_all_btn.clicked.connect(
            lambda: self._filter_unselect_all_temp(unique_values, checkboxes, temp_excluded)
        )
        btn_layout.addWidget(unselect_all_btn)

        btn_action = QWidgetAction(self)
        btn_action.setDefaultWidget(btn_container)
        menu.addAction(btn_action)

        menu.addSeparator()

        # Container scrollabile per checkbox
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        scroll_layout.setSpacing(2)

        checkboxes = []
        for val in unique_values:
            display_text = str(val) if not pd.isna(val) else "(empty)"
            cb = QCheckBox(display_text)
            cb.setChecked(val not in temp_excluded[0])
            cb.stateChanged.connect(
                lambda state, v=val: self._on_filter_checkbox_temp(v, state, temp_excluded)
            )
            scroll_layout.addWidget(cb)
            checkboxes.append((val, cb))

        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(300)
        scroll_area.setFrameShape(QFrame.NoFrame)

        scroll_action = QWidgetAction(self)
        scroll_action.setDefaultWidget(scroll_area)
        menu.addAction(scroll_action)

        menu.addSeparator()

        # Bottone Filter per applicare
        filter_btn = QPushButton("Filter")
        filter_btn.clicked.connect(
            lambda: self._apply_column_filter(col_name, temp_excluded[0], menu)
        )
        filter_action = QWidgetAction(self)
        filter_action.setDefaultWidget(filter_btn)
        menu.addAction(filter_action)

        # Mostra menu alla posizione dell'header
        header = self.table.horizontalHeader()
        pos = header.mapToGlobal(QPoint(header.sectionPosition(col_index), header.height()))
        menu.exec_(pos)

    def _on_filter_checkbox_temp(self, value, state: int, temp_excluded: list):
        """Aggiorna stato temporaneo (non applica ancora il filtro)."""
        if state == Qt.Checked:
            temp_excluded[0].discard(value)
        else:
            temp_excluded[0].add(value)

    def _filter_select_all_temp(self, checkboxes, temp_excluded: list):
        """Seleziona tutti (stato temporaneo)."""
        temp_excluded[0].clear()
        for val, cb in checkboxes:
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)

    def _filter_unselect_all_temp(self, values, checkboxes, temp_excluded: list):
        """Deseleziona tutti (stato temporaneo)."""
        temp_excluded[0] = set(values)
        for val, cb in checkboxes:
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)

    def _apply_column_filter(self, col_name: str, excluded: set, menu: QMenu):
        """Applica il filtro colonna e chiude il menu."""
        if excluded:
            self._column_value_filters[col_name] = excluded.copy()
        elif col_name in self._column_value_filters:
            del self._column_value_filters[col_name]

        self._apply_filters()
        self._update_filter_info()
        menu.close()

    def _update_filter_info(self):
        """Aggiorna label info filtri."""
        filters_active = []

        if self.active_filter and self.active_filter.conditions:
            filters_active.append("Advanced")

        for col_name, excluded in self._column_value_filters.items():
            if excluded:
                filters_active.append(f"{col_name}({len(excluded)})")

        if filters_active:
            self.filter_info_label.setText(f"Filters: {', '.join(filters_active)}")
        else:
            self.filter_info_label.setText("No filters active")

    # ==========================================================
    # FILTERS
    # ==========================================================
    def _show_advanced_filter_dialog(self):
        if self.all_data.empty:
            return

        dlg = AdvancedFilterDialog(
            columns=list(self.all_data.columns),
            data=self.all_data,
            current_filter=self.active_filter,
            parent=self,
        )

        if dlg.exec_():
            self.active_filter = dlg.get_filter()
            self._apply_filters()

    def _clear_all_filters(self):
        self.active_filter = None
        self._column_value_filters.clear()
        self._apply_filters()
        self._update_filter_info()

    def _apply_filters(self):
        if self.all_data.empty:
            self.filtered_data = pd.DataFrame()
            self._refresh_view()
            return

        # 1. Applica filtro avanzato (se presente)
        if not self.active_filter or not self.active_filter.conditions:
            self.filtered_data = self.all_data.copy()
        else:
            try:
                mask = self.active_filter.apply(self.all_data)

                # Assicuriamoci che mask sia una Series con gli stessi indici
                if isinstance(mask, pd.Series):
                    # Reindex per garantire corrispondenza con all_data
                    mask = mask.reindex(self.all_data.index, fill_value=False)

                self.filtered_data = self.all_data[mask]

            except Exception as e:
                # Mostra errore all'utente invece di crashare
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Filter Error",
                    f"Failed to apply filter: {str(e)}\n\n"
                    f"Please check your filter conditions.\n"
                    f"Common issues:\n"
                    f"- Invalid data type (e.g., text in numeric/date fields)\n"
                    f"- Invalid date format\n"
                    f"- Malformed comparison values"
                )
                # Reset al filtro precedente (nessun filtro)
                self.active_filter = None
                self.filtered_data = self.all_data.copy()

        # 2. Applica filtri per valori colonna (checkbox)
        if self._column_value_filters:
            for col_name, excluded_values in self._column_value_filters.items():
                if col_name in self.filtered_data.columns and excluded_values:
                    self.filtered_data = self.filtered_data[
                        ~self.filtered_data[col_name].isin(excluded_values)
                    ]

        self.displayed_rows = 0
        self._refresh_view()
        self.filtered_data_changed.emit(self.filtered_data)

    # ==========================================================
    # DATA UPDATE
    # ==========================================================
    def update_data(self, df: pd.DataFrame):
        if df is None or df.empty:
            return

        if self.all_data.empty:
            self.all_data = df.copy()
        else:
            self.all_data = safe_concat(
                [self.all_data, df], ignore_index=True
            )

            if self.dedup_column in self.all_data.columns:
                sort_cols = (
                    ['timestamp', self.dedup_column]
                    if 'timestamp' in self.all_data.columns
                    else [self.dedup_column]
                )
                try:
                    self.all_data = self.all_data.sort_values(
                        by=sort_cols, ascending=True, na_position='first'
                    )
                except Exception:
                    pass
                self.all_data = (
                    self.all_data
                    .groupby(self.dedup_column, sort=False)
                    .last()
                    .reset_index()
                )

                # Re-sort descending so newest trades appear at top
                try:
                    self.all_data = self.all_data.sort_values(
                        by=sort_cols, ascending=False, na_position='last'
                    )
                except Exception:
                    pass

        if not self.visible_columns:
            self.visible_columns = list(self.all_data.columns)

        self._apply_filters()

    # ==========================================================
    # TABLE RENDERING
    # ==========================================================
    def _refresh_view(self):
        """Reset completo - solo quando filtri cambiano"""
        # Salva larghezze colonne correnti prima di azzerarle
        if self.table.columnCount() > 0:
            self._saved_col_widths = {
                self.table.horizontalHeaderItem(i).text(): self.table.columnWidth(i)
                for i in range(self.table.columnCount())
                if self.table.horizontalHeaderItem(i)
            }
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.displayed_rows = 0
        self._load_more_rows()

    def _load_more_rows(self):
        if self.is_loading or self.filtered_data.empty:
            return

        df = self.filtered_data[self.visible_columns]
        if self.displayed_rows >= len(df):
            return

        self.is_loading = True
        start = self.displayed_rows
        end = min(start + self.rows_per_batch, len(df))
        self.displayed_rows = end

        self._render_rows(df, start, end)
        self._update_info()
        self.is_loading = False

    def _render_rows(self, df: pd.DataFrame, start: int, end: int):
        self.table.setSortingEnabled(False)

        self.table.setRowCount(end)
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns.tolist())

        # Ripristina larghezze colonne salvate (se presenti)
        saved = getattr(self, "_saved_col_widths", {})
        if saved:
            for i, col in enumerate(df.columns):
                if col in saved:
                    self.table.setColumnWidth(i, saved[col])

        own_idx = df.columns.get_loc("own_trade") if "own_trade" in df else None

        # Pre-calcola min/max per colonne numeriche (serve per regole "color scale").
        cf_ranges: dict[str, tuple[float, float]] = {}
        needs_ranges = any(
            r.operator == "color scale"
            for rules in self._cf_rules.values()
            for r in rules
        )
        if needs_ranges:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    vals = df[col].dropna()
                    if len(vals) > 0:
                        cf_ranges[col] = (float(vals.min()), float(vals.max()))

        col_list = df.columns.tolist()
        # Lista completa colonne (incluse quelle non visibili) per risolvere
        # i riferimenti a colonne nelle regole CF
        full_col_list = self.filtered_data.columns.tolist()

        for i, row in enumerate(df.iloc[start:end].itertuples(index=False), start):
            row_color = None
            is_own = False

            if own_idx is not None:
                is_own = bool(row[own_idx])

            # ---- Bug fix: row_dict usa TUTTE le colonne, non solo le visibili ----
            # Questo permette alle regole CF di fare riferimento a colonne nascoste
            row_dict = self.filtered_data.iloc[i].to_dict()

            # ---- Pre-calcola la regola "apply_to_row" per questa riga ----
            # Prima regola con apply_to_row=True che fa match → formato per tutta la riga
            row_rule = None
            for _col, _rules in self._cf_rules.items():
                _val = row_dict.get(_col)
                for _r in _rules:
                    if _r.apply_to_row and _r.operator != "color scale":
                        if _r.matches(_val, row_data=row_dict):
                            row_rule = _r
                            break
                if row_rule:
                    break

            for j, val in enumerate(row):
                col_name = col_list[j]
                text = self._format_value(col_name, val)
                item = NumericTableWidgetItem(text, sort_value=val)
                item.setTextAlignment(Qt.AlignCenter)

                # ---- Colore sfondo ----
                # Priorità: regole per-cella utente > row_rule > gradiente CF > BID/ASK
                user_rules = self._cf_rules.get(col_name, [])
                rule_applied = False
                for rule in user_rules:
                    if rule.operator == "color scale":
                        c_min, c_max = cf_ranges.get(col_name, (None, None))
                        rule.apply_to_item(item, val=val, col_min=c_min, col_max=c_max)
                        rule_applied = True
                        break
                    elif rule.matches(val, row_data=row_dict):
                        rule.apply_to_item(item)
                        rule_applied = True
                        break

                if not rule_applied:
                    if row_rule:
                        row_rule.apply_to_item(item)
                    elif row_color:
                        item.setBackground(row_color)

                # ---- Grassetto per own_trade ----
                if is_own:
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)

                self.table.setItem(i, j, item)

        self.table.setSortingEnabled(True)

    # ==========================================================
    # CONDITIONAL FORMATTING
    # ==========================================================
    # FORMATTING
    # ==========================================================
    def _format_value(self, col: str, value) -> str:
        if pd.isna(value):  
            return ""

        # Priorità: Se è un datetime/Timestamp, formattalo sempre come tale
        if isinstance(value, (pd.Timestamp, datetime.datetime)):
            return self._format_datetime(value)
        
        # Se la colonna è marcata come datetime, formattala
        if col in self.datetime_columns:
            return self._format_datetime(value)

        # Numeri float
        if isinstance(value, float):
            d = self.column_decimals.get(col, 2)
            return f"{value:.{d}f}".rstrip("0").rstrip(".")

        return str(value)

    def _format_datetime(self, value) -> str:
        """Formatta datetime come HH:MM:SS.fff"""
        try:
            # Se è già un pandas Timestamp o datetime, usalo direttamente
            if isinstance(value, (pd.Timestamp, datetime.datetime)):
                dt = value
            # Se è stringa, converti
            elif isinstance(value, str):
                dt = pd.to_datetime(value)
            # Se è numero (Unix timestamp), converti
            elif isinstance(value, (int, float)):
                # Gestisce sia secondi che millisecondi
                dt = pd.to_datetime(value, unit='s' if value < 1e12 else 'ms')
            else:
                return str(value)
            
            # Formatta con il formato richiesto
            txt = dt.strftime(self.datetime_format)
            # Taglia microsecondi se richiesto
            return txt[:-3] if "%f" in self.datetime_format else txt
            
        except Exception as e:
            return str(value)

    # ==========================================================
    # SCROLL
    # ==========================================================
    def _on_scroll(self, v):
        sb = self.table.verticalScrollBar()
        if v >= sb.maximum() * 0.9:
            self._load_more_rows()

    # ==========================================================
    # API
    # ==========================================================
    def set_visible_columns(self, cols: List[str]):
        """Imposta le colonne visibili e aggiorna la vista"""
        if not cols:
            return

        # Aggiorna le colonne visibili
        self.visible_columns = cols

        # Forza il refresh completo della vista
        self.displayed_rows = 0
        self._refresh_view()

        # Opzionale: emetti il segnale per notificare il cambio
        # (solo se filtri erano già applicati)
        if not self.filtered_data.empty:
            self.filtered_data_changed.emit(self.filtered_data)

    def get_available_columns(self) -> List[str]:
        return self.all_data.columns.tolist()

    def get_visible_columns(self) -> List[str]:
        return self.visible_columns

    def clear(self):
        self.all_data = pd.DataFrame()
        self.filtered_data = pd.DataFrame()
        self.active_filter = None
        self.displayed_rows = 0
        self.table.clear()
        self.info_label.setText("No data")

    # ==========================================================
    # UI HELPERS
    # ==========================================================
    def _update_info(self):
        tot = len(self.all_data)
        flt = len(self.filtered_data)
        self.info_label.setText(
            f"Total: {tot} | Filtered: {flt} | Showing: {self.displayed_rows}"
        )

    def _set_column_decimals(self, col: str, d: int):
        self.column_decimals[col] = d
        self._refresh_view()

    # ==========================================================
    # CONDITIONAL FORMATTING RULES (per-column, stile Excel)
    # ==========================================================
    def _remove_all_cf(self):
        """Rimuove tutte le regole CF e aggiorna la vista."""
        self._cf_rules.clear()
        self._save_cf_rules()
        self._refresh_view()

    def _show_cf_rules_dialog(self, col_name: str):
        """Apre il dialog di gestione delle regole CF per una colonna."""
        col_dtype = (
            self.filtered_data[col_name].dtype
            if not self.filtered_data.empty and col_name in self.filtered_data.columns
            else None
        )
        available_cols = [
            c for c in self.all_data.columns if c != col_name
        ] if not self.all_data.empty else []

        existing = self._cf_rules.get(col_name, [])
        dlg = CFRulesDialog(
            col_name, col_dtype, existing,
            available_columns=available_cols, parent=self,
        )
        if dlg.exec_():
            new_rules = dlg.get_rules()
            if new_rules:
                self._cf_rules[col_name] = new_rules
            elif col_name in self._cf_rules:
                del self._cf_rules[col_name]
            self._save_cf_rules()
            self._refresh_view()

    def _save_cf_rules(self):
        """Salva tutte le regole CF in JSON."""
        try:
            path = Path(self._cf_rules_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                col: [r.to_dict() for r in rules]
                for col, rules in self._cf_rules.items()
                if rules
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # salvataggio silenzioso

    def _load_cf_rules(self):
        """Carica le regole CF da JSON (se il file esiste)."""
        try:
            path = Path(self._cf_rules_path)
            if not path.exists():
                return
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._cf_rules = {
                col: [CFRule.from_dict(d) for d in rules_list]
                for col, rules_list in data.items()
            }
        except Exception:
            self._cf_rules = {}
