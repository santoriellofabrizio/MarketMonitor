"""
Advanced Filter Dialog con supporto AND/OR e liste multiple
"""
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QComboBox, QLineEdit, QListWidget, QListWidgetItem,
                             QGroupBox, QRadioButton, QButtonGroup, QTextEdit,
                             QDialogButtonBox, QFrame, QScrollArea, QWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import pandas as pd
import re
from typing import List, Dict, Any, Optional
from enum import Enum


class FilterOperator(Enum):
    """Operatori di filtro"""
    CONTAINS = "contains"
    NOT_CONTAINS = "not contains"
    EQUALS = "="
    NOT_EQUALS = "!="
    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    BETWEEN = "between"
    IN_LIST = "in list"
    NOT_IN_LIST = "not in list"
    STARTS_WITH = "starts with"
    ENDS_WITH = "ends with"


class FilterCondition:
    """Singola condizione di filtro"""

    def __init__(self, column: str, operator: FilterOperator, value: Any, value2: Any = None):
        self.column = column
        self.operator = operator
        self.value = value
        self.value2 = value2

    def _clean_value(self, value: Any) -> str:
        """Rimuove virgolette extra e pulisce il valore"""
        value_str = str(value).strip()
        # Rimuovi virgolette doppie o singole all'inizio e fine
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            value_str = value_str[1:-1]
        return value_str

    def _convert_to_comparable(self, series: pd.Series, value: Any):
        """Converte il valore nel tipo corretto per il confronto"""
        value = self._clean_value(value)
        
        # PRIMA: Gestione timestamp/datetime
        # Controlla PRIMA del check numerico perché timestamp numerici sono float!
        if series.name in ['timestamp', 'time', 'datetime', 'last_update'] or 'time' in str(series.name).lower():
            try:
                # Se il valore input contiene ':' è un formato time (HH:MM:SS)
                if ':' in value:
                    from datetime import datetime, time as dt_time
                    
                    # Se la serie è datetime, converti l'input in datetime di oggi
                    if pd.api.types.is_datetime64_any_dtype(series):
                        # Parse come time e crea datetime per oggi
                        time_obj = pd.to_datetime(value, format='%H:%M:%S').time()
                        today = pd.Timestamp.now().date()
                        return pd.Timestamp.combine(today, time_obj)
                    else:
                        # Serie numerica: controlla il range per capire il formato
                        sample_val = series.dropna().iloc[0] if len(series.dropna()) > 0 else 0
                        
                        if sample_val > 1e9:  # Timestamp Unix (secondi dal 1970)
                            time_obj = pd.to_datetime(value, format='%H:%M:%S').time()
                            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                            target_datetime = datetime.combine(today.date(), time_obj)
                            return target_datetime.timestamp()  # Timestamp Unix
                        else:  # Secondi dall'inizio della giornata (0-86400)
                            time_obj = pd.to_datetime(value, format='%H:%M:%S').time()
                            return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
                else:
                    # Già un numero o stringa data completa
                    if pd.api.types.is_datetime64_any_dtype(series):
                        return pd.to_datetime(value)
                    else:
                        return float(value)
            except Exception as e:
                # Fallback: prova conversione standard
                try:
                    if pd.api.types.is_datetime64_any_dtype(series):
                        return pd.to_datetime(value)
                    else:
                        return float(value)
                except:
                    raise ValueError(f"Cannot convert '{value}' to timestamp: {e}")
        
        # Gestione datetime veri (dtype datetime64)
        elif pd.api.types.is_datetime64_any_dtype(series):
            try:
                return pd.to_datetime(value)
            except:
                raise ValueError(f"Cannot convert '{value}' to datetime")
        
        # Gestione numerica generica
        elif pd.api.types.is_numeric_dtype(series):
            return float(value)
        
        # Default: string
        return value

    def apply(self, series: pd.Series) -> pd.Series:
        """Applica il filtro e ritorna maschera booleana"""
        try:
            if self.operator == FilterOperator.CONTAINS:
                return series.astype(str).str.lower().str.contains(str(self.value).lower(), na=False, regex=False)

            elif self.operator == FilterOperator.NOT_CONTAINS:
                return ~series.astype(str).str.lower().str.contains(str(self.value).lower(), na=False, regex=False)

            elif self.operator == FilterOperator.STARTS_WITH:
                return series.astype(str).str.lower().str.startswith(str(self.value).lower(), na=False)

            elif self.operator == FilterOperator.ENDS_WITH:
                return series.astype(str).str.lower().str.endswith(str(self.value).lower(), na=False)

            elif self.operator == FilterOperator.EQUALS:
                if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
                    comp_value = self._convert_to_comparable(series, self.value)
                    return series == comp_value
                cleaned_value = self._clean_value(self.value)
                return series.astype(str).str.lower() == cleaned_value.lower()

            elif self.operator == FilterOperator.NOT_EQUALS:
                if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
                    comp_value = self._convert_to_comparable(series, self.value)
                    return series != comp_value
                cleaned_value = self._clean_value(self.value)
                return series.astype(str).str.lower() != cleaned_value.lower()

            elif self.operator == FilterOperator.GREATER:
                comp_value = self._convert_to_comparable(series, self.value)
                return series > comp_value

            elif self.operator == FilterOperator.GREATER_EQUAL:
                comp_value = self._convert_to_comparable(series, self.value)
                return series >= comp_value

            elif self.operator == FilterOperator.LESS:
                comp_value = self._convert_to_comparable(series, self.value)
                return series < comp_value

            elif self.operator == FilterOperator.LESS_EQUAL:
                comp_value = self._convert_to_comparable(series, self.value)
                return series <= comp_value

            elif self.operator == FilterOperator.BETWEEN:
                comp_value1 = self._convert_to_comparable(series, self.value)
                comp_value2 = self._convert_to_comparable(series, self.value2)
                return (series >= comp_value1) & (series <= comp_value2)

            elif self.operator in (FilterOperator.IN_LIST, FilterOperator.NOT_IN_LIST):
                # Supporto separatori multipli: newline, comma, semicolon
                value_str = str(self.value)

                # Splitta usando regex per supportare tutti i separatori
                values = re.split(r'[,;\n]+', value_str)
                # Pulisci spazi bianchi e virgolette
                values = [self._clean_value(v).lower() for v in values if v.strip()]

                # Confronta case-insensitive
                mask = series.astype(str).str.lower().isin(values)

                if self.operator == FilterOperator.NOT_IN_LIST:
                    return ~mask
                return mask

        except Exception as e:
            print(f"⚠️  Error applying filter {self.column} {self.operator.value}: {e}")
            return pd.Series([True] * len(series))

        return pd.Series([True] * len(series))

    def __repr__(self):
        if self.operator == FilterOperator.BETWEEN:
            return f"{self.column} {self.operator.value} [{self.value}, {self.value2}]"
        elif self.operator in (FilterOperator.IN_LIST, FilterOperator.NOT_IN_LIST):
            # Mostra preview valori
            values = re.split(r'[,;\n]+', str(self.value))
            values = [v.strip() for v in values if v.strip()]
            if len(values) <= 3:
                return f"{self.column} {self.operator.value} [{', '.join(values)}]"
            else:
                preview = ', '.join(values[:3])
                return f"{self.column} {self.operator.value} [{preview}, +{len(values)-3} more]"
        else:
            # Tronca valori lunghi
            val_str = str(self.value)
            if len(val_str) > 30:
                val_str = val_str[:27] + "..."
            return f"{self.column} {self.operator.value} '{val_str}'"


class FilterGroup:
    """Gruppo di condizioni con logica AND/OR"""

    def __init__(self, logic: str = "AND"):
        self.logic = logic  # "AND" o "OR"
        self.conditions: List[FilterCondition] = []

    def add_condition(self, condition: FilterCondition):
        self.conditions.append(condition)

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Applica il gruppo di filtri"""
        if not self.conditions:
            return pd.Series([True] * len(df))

        # Applica prima condizione
        result = self.conditions[0].apply(df[self.conditions[0].column])

        # Combina con le altre
        for condition in self.conditions[1:]:
            mask = condition.apply(df[condition.column])

            if self.logic == "AND":
                result = result & mask
            else:  # OR
                result = result | mask

        return result

    def __repr__(self):
        if not self.conditions:
            return "Empty filter group"

        cond_strs = [str(c) for c in self.conditions]
        return f" {self.logic} ".join(cond_strs)


class AdvancedFilterDialog(QDialog):
    """
    Dialog avanzato per filtri con supporto:
    - Logica AND/OR tra condizioni
    - Liste multiple (separatori: newline, comma, semicolon)
    - Operatori multipli
    - Preview valori disponibili
    """

    def __init__(self, columns: List[str], data: pd.DataFrame,
                 current_filter: Optional[FilterGroup] = None,
                 parent=None):
        super().__init__(parent)

        self.columns = columns
        self.data = data
        self.filter_result: Optional[FilterGroup] = current_filter if current_filter else FilterGroup()

        self.setWindowTitle("🔍 Advanced Filter")
        self.setMinimumSize(700, 600)

        self._setup_ui()

        # Carica filtro esistente se presente
        if current_filter and current_filter.conditions:
            self._load_existing_filter(current_filter)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ========== Header ==========
        header = QLabel("<b>Configure Advanced Filter</b>")
        header.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(header)

        # ========== Logic Selector ==========
        logic_group = QGroupBox("Combine Conditions With")
        logic_layout = QHBoxLayout()

        self.logic_btn_group = QButtonGroup()

        self.and_radio = QRadioButton("AND (all must match)")
        self.and_radio.setChecked(True)
        logic_layout.addWidget(self.and_radio)
        self.logic_btn_group.addButton(self.and_radio)

        self.or_radio = QRadioButton("OR (any can match)")
        logic_layout.addWidget(self.or_radio)
        self.logic_btn_group.addButton(self.or_radio)

        logic_layout.addStretch()
        logic_group.setLayout(logic_layout)
        layout.addWidget(logic_group)

        # ========== Conditions List ==========
        conditions_group = QGroupBox("Filter Conditions")
        conditions_layout = QVBoxLayout()

        # Scroll area per condizioni
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(200)

        self.conditions_container = QWidget()
        self.conditions_layout = QVBoxLayout(self.conditions_container)
        self.conditions_layout.setAlignment(Qt.AlignTop)

        scroll.setWidget(self.conditions_container)
        conditions_layout.addWidget(scroll)

        # Bottone aggiungi condizione
        add_btn = QPushButton("➕ Add Condition")
        add_btn.clicked.connect(self._add_condition_row)
        conditions_layout.addWidget(add_btn)

        conditions_group.setLayout(conditions_layout)
        layout.addWidget(conditions_group)

        # ========== Preview ==========
        self.preview_label = QLabel("Preview: No conditions")
        self.preview_label.setStyleSheet(
            "background-color: #f0f0f0; padding: 10px; "
            "border: 1px solid #ccc; border-radius: 3px; "
            "font-family: monospace; font-size: 11px;"
        )
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

        # ========== Buttons ==========
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )

        # Bottone "Clear All"
        clear_all_btn = QPushButton("🗑️ Clear All")
        clear_all_btn.clicked.connect(self._clear_all_conditions)
        button_box.addButton(clear_all_btn, QDialogButtonBox.ActionRole)

        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Aggiungi prima condizione vuota
        if not self.filter_result.conditions:
            self._add_condition_row()

    def _add_condition_row(self):
        """Aggiunge una riga per configurare una condizione"""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(5, 5, 5, 5)

        # Colonna
        column_combo = QComboBox()
        column_combo.addItems(self.columns)
        column_combo.setMinimumWidth(120)
        column_combo.currentTextChanged.connect(lambda: self._update_preview())
        row_layout.addWidget(column_combo)

        # Operatore
        operator_combo = QComboBox()
        self._populate_operators(operator_combo)
        operator_combo.setMinimumWidth(120)
        operator_combo.currentTextChanged.connect(
            lambda: self._on_operator_changed(row_widget, operator_combo)
        )
        row_layout.addWidget(operator_combo)

        # Valore 1
        value1_input = QLineEdit()
        value1_input.setPlaceholderText("Value...")
        value1_input.textChanged.connect(lambda: self._update_preview())
        row_layout.addWidget(value1_input)

        # Valore 2 (per BETWEEN)
        value2_input = QLineEdit()
        value2_input.setPlaceholderText("End value...")
        value2_input.setVisible(False)
        value2_input.textChanged.connect(lambda: self._update_preview())
        row_layout.addWidget(value2_input)

        # Bottone lista
        list_btn = QPushButton("📋 List")
        list_btn.setToolTip("Enter multiple values (one per line)")
        list_btn.setVisible(False)
        list_btn.clicked.connect(
            lambda: self._show_list_dialog(value1_input, column_combo.currentText())
        )
        row_layout.addWidget(list_btn)

        # Bottone rimuovi
        remove_btn = QPushButton("❌")
        remove_btn.setMaximumWidth(40)
        remove_btn.clicked.connect(lambda: self._remove_condition_row(row_widget))
        row_layout.addWidget(remove_btn)

        # Salva riferimenti
        row_widget.column_combo = column_combo
        row_widget.operator_combo = operator_combo
        row_widget.value1_input = value1_input
        row_widget.value2_input = value2_input
        row_widget.list_btn = list_btn

        self.conditions_layout.addWidget(row_widget)
        self._update_preview()

    def _populate_operators(self, combo: QComboBox):
        """Popola combo con operatori"""
        combo.addItem("Contains", FilterOperator.CONTAINS)
        combo.addItem("Not contains", FilterOperator.NOT_CONTAINS)
        combo.addItem("Equals (=)", FilterOperator.EQUALS)
        combo.addItem("Not equals (≠)", FilterOperator.NOT_EQUALS)
        combo.addItem("Greater than (>)", FilterOperator.GREATER)
        combo.addItem("Greater or equal (≥)", FilterOperator.GREATER_EQUAL)
        combo.addItem("Less than (<)", FilterOperator.LESS)
        combo.addItem("Less or equal (≤)", FilterOperator.LESS_EQUAL)
        combo.addItem("Between", FilterOperator.BETWEEN)
        combo.addItem("In list", FilterOperator.IN_LIST)
        combo.addItem("Not in list", FilterOperator.NOT_IN_LIST)
        combo.addItem("Starts with", FilterOperator.STARTS_WITH)
        combo.addItem("Ends with", FilterOperator.ENDS_WITH)

    def _on_operator_changed(self, row_widget, operator_combo):
        """Mostra/nasconde campi in base all'operatore"""
        operator = operator_combo.currentData()

        # BETWEEN mostra value2
        row_widget.value2_input.setVisible(operator == FilterOperator.BETWEEN)

        # IN_LIST/NOT_IN_LIST mostra bottone lista
        is_list_op = operator in (FilterOperator.IN_LIST, FilterOperator.NOT_IN_LIST)
        row_widget.list_btn.setVisible(is_list_op)

        if is_list_op:
            row_widget.value1_input.setPlaceholderText("Values (comma/newline/semicolon separated)")
        else:
            row_widget.value1_input.setPlaceholderText("Value...")

        self._update_preview()

    def _show_list_dialog(self, value_input: QLineEdit, column_name: str):
        """Mostra dialog per inserire lista valori"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"List Values: {column_name}")
        dialog.setMinimumSize(500, 400)

        layout = QVBoxLayout(dialog)

        # Info
        info = QLabel(
            "Enter values, one per line.\n"
            "You can also paste comma-separated or semicolon-separated values."
        )
        info.setStyleSheet("padding: 10px; background-color: #e8f4f8;")
        layout.addWidget(info)

        # Text area
        text_edit = QTextEdit()
        text_edit.setPlaceholderText("ticker1\nticker2\nticker3\n...\n\nOR: ticker1, ticker2, ticker3")

        # Precompila con valori esistenti
        if value_input.text():
            text_edit.setPlainText(value_input.text().replace(',', '\n').replace(';', '\n'))

        layout.addWidget(text_edit)

        # Preview valori unici
        if not self.data.empty and column_name in self.data.columns:
            unique_values = self.data[column_name].dropna().unique()
            preview_label = QLabel(
                f"Available values in {column_name}: " +
                ', '.join([str(v) for v in unique_values[:10]]) +
                (f" (+{len(unique_values)-10} more)" if len(unique_values) > 10 else "")
            )
            preview_label.setStyleSheet("font-size: 10px; color: #666; padding: 5px;")
            preview_label.setWordWrap(True)
            layout.addWidget(preview_label)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_() == QDialog.Accepted:
            # Normalizza input (converti tutto in comma-separated)
            values_text = text_edit.toPlainText()
            # Mantieni formato originale (può essere newline, comma o semicolon)
            value_input.setText(values_text)
            self._update_preview()

    def _remove_condition_row(self, row_widget):
        """Rimuove una riga di condizione"""
        self.conditions_layout.removeWidget(row_widget)
        row_widget.deleteLater()
        self._update_preview()

    def _clear_all_conditions(self):
        """Rimuove tutte le condizioni"""
        while self.conditions_layout.count() > 0:
            item = self.conditions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Aggiungi una riga vuota
        self._add_condition_row()

    def _update_preview(self):
        """Aggiorna preview del filtro"""
        logic = "AND" if self.and_radio.isChecked() else "OR"

        # Costruisci filter group temporaneo
        temp_group = FilterGroup(logic)

        # Raccogli condizioni
        for i in range(self.conditions_layout.count()):
            row_widget = self.conditions_layout.itemAt(i).widget()
            if not row_widget:
                continue

            column = row_widget.column_combo.currentText()
            operator = row_widget.operator_combo.currentData()
            value1 = row_widget.value1_input.text().strip()
            value2 = row_widget.value2_input.text().strip()

            if not value1:
                continue

            condition = FilterCondition(column, operator, value1, value2 if value2 else None)
            temp_group.add_condition(condition)

        # Mostra preview
        if not temp_group.conditions:
            self.preview_label.setText("Preview: No conditions")
            self.preview_label.setStyleSheet(
                "background-color: #f0f0f0; padding: 10px; "
                "border: 1px solid #ccc; border-radius: 3px; "
                "font-family: monospace; font-size: 11px;"
            )
        else:
            preview_text = f"Filter: {temp_group}"
            self.preview_label.setText(preview_text)
            self.preview_label.setStyleSheet(
                "background-color: #e8f4f8; padding: 10px; "
                "border: 1px solid #0066cc; border-radius: 3px; "
                "font-family: monospace; font-size: 11px; color: #003366;"
            )

    def _load_existing_filter(self, filter_group: FilterGroup):
        """Carica un filtro esistente nel dialog"""
        # Imposta logica
        if filter_group.logic == "OR":
            self.or_radio.setChecked(True)
        else:
            self.and_radio.setChecked(True)

        # Rimuovi riga vuota iniziale
        if self.conditions_layout.count() > 0:
            item = self.conditions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Aggiungi condizioni
        for condition in filter_group.conditions:
            self._add_condition_row()

            # Configura ultima riga aggiunta
            row_widget = self.conditions_layout.itemAt(
                self.conditions_layout.count() - 1
            ).widget()

            # Imposta colonna
            index = row_widget.column_combo.findText(condition.column)
            if index >= 0:
                row_widget.column_combo.setCurrentIndex(index)

            # Imposta operatore
            for i in range(row_widget.operator_combo.count()):
                if row_widget.operator_combo.itemData(i) == condition.operator:
                    row_widget.operator_combo.setCurrentIndex(i)
                    break

            # Imposta valori
            row_widget.value1_input.setText(str(condition.value))
            if condition.value2:
                row_widget.value2_input.setText(str(condition.value2))

        self._update_preview()

    def _on_accept(self):
        """Costruisci e salva il filtro"""
        logic = "AND" if self.and_radio.isChecked() else "OR"
        filter_group = FilterGroup(logic)

        # Raccogli condizioni
        for i in range(self.conditions_layout.count()):
            row_widget = self.conditions_layout.itemAt(i).widget()
            if not row_widget:
                continue

            column = row_widget.column_combo.currentText()
            operator = row_widget.operator_combo.currentData()
            value1 = row_widget.value1_input.text().strip()
            value2 = row_widget.value2_input.text().strip()

            if not value1:
                continue

            condition = FilterCondition(column, operator, value1, value2 if value2 else None)
            filter_group.add_condition(condition)

        self.filter_result = filter_group if filter_group.conditions else None
        self.accept()

    def get_filter(self) -> Optional[FilterGroup]:
        """Ritorna il filtro configurato"""
        return self.filter_result
