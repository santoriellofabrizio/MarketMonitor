"""  
DashboardState - Sistema COMPLETO di salvataggio/caricamento

AGGIORNAMENTI:
✅ Salva/ripristina filtri avanzati (FilterGroup)
✅ Salva/ripristina configurazioni complete Pivot
✅ Salva/ripristina configurazioni complete Chart
✅ Gestione errori migliorata
✅ Backward compatibility

SOSTITUISCI il file dashboard_state.py con questo codice
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
import pandas as pd
from PyQt5.QtCore import QByteArray
from PyQt5.QtGui import QColor


class DashboardStateError(Exception):
    """Eccezione per errori di dashboard state"""
    pass


class DashboardState:
    """
    Gestisce il salvataggio e caricamento dello stato completo della TradeDashboard.

    NOVITÀ v1.1:
    - Salva filtri avanzati (FilterGroup) per Pivot e Chart
    - Salva configurazioni complete dei widget
    - Migliore gestione errori
    """

    VERSION = "1.1"

    def __init__(self, config_dir: Optional[Path] = None, app_name: str = "TradeDashboard"):
        self.app_name = app_name

        if config_dir is None:
            config_dir = Path.home() / f".{app_name.lower()}" / "dashboards"

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DashboardState initialized. Config dir: {self.config_dir}")

    # ==================== SAVE ====================

    def save_dashboard(self, name: str, dashboard,
                      description: str = "", tags: List[str] = None) -> Path:
        """
        Salva lo stato completo della dashboard.
        """
        self.logger.info(f"Saving dashboard: {name}")

        try:
            state = {
                'metadata': self._create_metadata(name, description, tags),
                'geometry': self._save_geometry(dashboard),
                'trade_table': self._save_trade_table(dashboard.trade_table),
                'detached_windows': self._save_detached_windows(dashboard),
                'preferences': self._save_preferences(dashboard),
                'metrics': self._save_metrics_state(dashboard),
            }

            # Sanitize filename
            safe_name = self._sanitize_filename(name)
            output_file = self.config_dir / f"{safe_name}.json"

            # Backup se esiste già
            if output_file.exists():
                backup_file = self.config_dir / f"{safe_name}.backup.json"
                output_file.rename(backup_file)
                self.logger.info(f"Created backup: {backup_file}")

            # Write JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=self._json_serializer, ensure_ascii=False)

            self.logger.info(f"Dashboard saved successfully: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Failed to save dashboard: {e}", exc_info=True)
            raise DashboardStateError(f"Failed to save dashboard: {e}") from e

    def _create_metadata(self, name: str, description: str, tags: List[str]) -> Dict[str, Any]:
        """Crea metadata per la dashboard"""
        return {
            'name': name,
            'description': description or "",
            'tags': tags or [],
            'version': self.VERSION,
            'timestamp': datetime.now().isoformat(),
            'app_name': self.app_name,
        }

    def _save_geometry(self, dashboard) -> Dict[str, Any]:
        """Salva geometria finestra principale"""
        try:
            return {
                'window_geometry': self._qbytearray_to_hex(dashboard.saveGeometry()),
                'window_state': self._qbytearray_to_hex(dashboard.saveState()),
                'is_maximized': dashboard.isMaximized(),
                'is_fullscreen': dashboard.isFullScreen(),
            }
        except Exception as e:
            self.logger.warning(f"Failed to save geometry: {e}")
            return {}

    def _save_trade_table(self, trade_table) -> Dict[str, Any]:
        """Salva configurazione TradeTableWidget"""
        try:
            config = {
                'visible_columns': trade_table.visible_columns,
                'column_widths': self._get_column_widths(trade_table.table),
                'column_decimals': trade_table.column_decimals.copy(),
                'autoscroll': trade_table.autoscroll_checkbox.isChecked(),
                'datetime_format': trade_table.datetime_format,
                'max_rows': trade_table.max_rows,
                'dedup_column': trade_table.dedup_column,
                'sort_column': trade_table.table.horizontalHeader().sortIndicatorSection(),
                'sort_order': int(trade_table.table.horizontalHeader().sortIndicatorOrder()),
            }

            # 🆕 SALVA FILTRI AVANZATI
            if hasattr(trade_table, 'active_filter') and trade_table.active_filter:
                config['advanced_filter'] = self._serialize_filter_group(trade_table.active_filter)

            return config

        except Exception as e:
            self.logger.warning(f"Failed to save trade table: {e}")
            return {}

    def _save_detached_windows(self, dashboard) -> Dict[str, List[Dict]]:
        """Salva configurazione finestre detached CON FILTRI"""
        detached = {
            'pivot_windows': [],
            'chart_windows': [],
            'flow_windows': []
        }

        try:
            # 🆕 PIVOT WINDOWS - Con filtri avanzati
            for window in dashboard.detached_pivots:
                if window.isVisible():
                    pivot_config = {
                        'window_number': window.window_number,
                        'geometry': self._qbytearray_to_hex(window.saveGeometry()),
                    }

                    # Salva configurazione pivot completa
                    if hasattr(window.pivot_widget, 'get_config'):
                        widget_config = window.pivot_widget.get_config()
                        pivot_config['pivot_config'] = widget_config

                        # 🆕 Salva filtri avanzati del pivot
                        if hasattr(window.pivot_widget, 'active_filter') and window.pivot_widget.active_filter:
                            pivot_config['pivot_config']['advanced_filter'] = \
                                self._serialize_filter_group(window.pivot_widget.active_filter)

                    detached['pivot_windows'].append(pivot_config)

            # 🆕 CHART WINDOWS - Con filtri avanzati
            for window in dashboard.detached_charts:
                if window.isVisible():
                    chart_config = {
                        'window_number': window.window_number,
                        'geometry': self._qbytearray_to_hex(window.saveGeometry()),
                    }

                    # Salva configurazione chart completa
                    if hasattr(window.chart_widget, 'get_config'):
                        widget_config = window.chart_widget.get_config()
                        chart_config['chart_config'] = widget_config

                        # 🆕 Salva filtri avanzati del chart
                        if hasattr(window.chart_widget, 'active_filter') and window.chart_widget.active_filter:
                            chart_config['chart_config']['advanced_filter'] = \
                                self._serialize_filter_group(window.chart_widget.active_filter)

                    detached['chart_windows'].append(chart_config)

            # FLOW WINDOWS (basic)
            for window in dashboard.detached_flows:
                if window.isVisible():
                    detached['flow_windows'].append({
                        'window_number': window.window_number,
                        'geometry': self._qbytearray_to_hex(window.saveGeometry()),
                    })

        except Exception as e:
            self.logger.warning(f"Failed to save detached windows: {e}")

        return detached

    def _serialize_filter_group(self, filter_group) -> Dict[str, Any]:
        """
        🆕 Serializza un FilterGroup in JSON-compatible dict

        FilterGroup ha:
        - logic: 'AND' | 'OR'
        - conditions: List[ColumnFilter]

        ColumnFilter ha:
        - column: str
        - operator: FilterOperator (enum)
        - value: Any
        - value2: Optional[Any]
        """
        try:
            serialized = {
                'logic': filter_group.logic,
                'conditions': []
            }

            for condition in filter_group.conditions:
                cond_dict = {
                    'column': condition.column,
                    'operator': condition.operator.value,  # Enum to string
                    'value': self._serialize_filter_value(condition.value),
                    'value2': self._serialize_filter_value(condition.value2) if hasattr(condition, 'value2') else None,
                }
                serialized['conditions'].append(cond_dict)

            return serialized

        except Exception as e:
            self.logger.warning(f"Failed to serialize filter group: {e}")
            return None

    def _serialize_filter_value(self, value) -> Any:
        """Serializza un valore di filtro (gestisce datetime, etc)"""
        if pd.isna(value):
            return None
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        if isinstance(value, (list, tuple)):
            return [self._serialize_filter_value(v) for v in value]
        return value

    def _save_preferences(self, dashboard) -> Dict[str, Any]:
        """Salva preferenze globali"""
        return {
            'paused': dashboard.paused,
            'mode': dashboard.mode,
            'columns': dashboard.columns,
        }

    def _save_metrics_state(self, dashboard) -> Dict[str, Any]:
        """Salva stato del pannello metriche"""
        try:
            return {
                'last_update_count': getattr(dashboard, '_update_count', 0),
            }
        except:
            return {}

    # ==================== LOAD ====================

    def load_dashboard(self, name: str, dashboard) -> Dict[str, Any]:
        """
        Carica lo stato completo della dashboard CON FILTRI.
        """
        self.logger.info(f"Loading dashboard: {name}")

        try:
            safe_name = self._sanitize_filename(name)
            config_file = self.config_dir / f"{safe_name}.json"

            if not config_file.exists():
                raise DashboardStateError(f"Dashboard not found: {name}")

            with open(config_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            # Version check & migration
            if state['metadata']['version'] != self.VERSION:
                self.logger.warning(f"Version mismatch: {state['metadata']['version']} vs {self.VERSION}")
                state = self._migrate_config(state)

            # Restore in ordine
            self._restore_geometry(dashboard, state.get('geometry', {}))
            self._restore_trade_table(dashboard.trade_table, state.get('trade_table', {}))
            self._restore_preferences(dashboard, state.get('preferences', {}))

            # Restore detached windows
            detached = state.get('detached_windows', {})
            if detached:
                self._restore_detached_windows(dashboard, detached)

            self.logger.info(f"Dashboard loaded successfully: {name}")
            return state['metadata']

        except DashboardStateError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to load dashboard: {e}", exc_info=True)
            raise DashboardStateError(f"Failed to load dashboard: {e}") from e

    def _restore_geometry(self, dashboard, geometry: Dict[str, Any]):
        """Ripristina geometria finestra"""
        try:
            if 'window_geometry' in geometry:
                dashboard.restoreGeometry(self._hex_to_qbytearray(geometry['window_geometry']))

            if 'window_state' in geometry:
                dashboard.restoreState(self._hex_to_qbytearray(geometry['window_state']))

            if geometry.get('is_maximized'):
                dashboard.showMaximized()
            elif geometry.get('is_fullscreen'):
                dashboard.showFullScreen()

        except Exception as e:
            self.logger.warning(f"Failed to restore geometry: {e}")

    def _restore_trade_table(self, trade_table, config: Dict[str, Any]):
        """Ripristina configurazione TradeTableWidget CON FILTRI"""
        try:
            # Pulisci dati vecchi
            trade_table.all_data = pd.DataFrame()
            trade_table.filtered_data = pd.DataFrame()
            trade_table.displayed_rows = 0
            trade_table.table.setRowCount(0)
            self.logger.debug("Cleared old trade_table data before restoring configuration")

            # Colonne visibili
            if 'visible_columns' in config:
                trade_table.set_visible_columns(config['visible_columns'])

            # Decimali
            if 'column_decimals' in config:
                trade_table.column_decimals = config['column_decimals'].copy()

            # 🆕 RIPRISTINA FILTRI AVANZATI
            if 'advanced_filter' in config and config['advanced_filter']:
                filter_group = self._deserialize_filter_group(config['advanced_filter'])
                if filter_group:
                    trade_table.active_filter = filter_group

            # Preferenze
            if 'autoscroll' in config:
                trade_table.autoscroll_checkbox.setChecked(config['autoscroll'])

            if 'datetime_format' in config:
                trade_table.datetime_format = config['datetime_format']

            if 'max_rows' in config:
                trade_table.max_rows = config['max_rows']

            if 'dedup_column' in config:
                trade_table.dedup_column = config['dedup_column']

            # Larghezze colonne
            if 'column_widths' in config:
                self._restore_column_widths(trade_table.table, config['column_widths'])

            # Sorting
            if 'sort_column' in config and 'sort_order' in config:
                trade_table.table.sortItems(config['sort_column'], config['sort_order'])

        except Exception as e:
            self.logger.warning(f"Failed to restore trade table: {e}")

    def _restore_preferences(self, dashboard, prefs: Dict[str, Any]):
        """Ripristina preferenze"""
        try:
            if 'paused' in prefs and prefs['paused']:
                dashboard.pause()
                if hasattr(dashboard, 'pause_btn'):
                    dashboard.pause_btn.setText("▶️ Resume")

        except Exception as e:
            self.logger.warning(f"Failed to restore preferences: {e}")

    def _restore_detached_windows(self, dashboard, detached: Dict[str, List[Dict]]):
        """Ripristina finestre detached CON FILTRI"""
        try:
            # Prima chiudi finestre esistenti
            for window in dashboard.detached_pivots + dashboard.detached_charts + dashboard.detached_flows:
                window.close()

            dashboard.detached_pivots.clear()
            dashboard.detached_charts.clear()
            dashboard.detached_flows.clear()

            # 🆕 PIVOT WINDOWS - Con filtri
            for config in detached.get('pivot_windows', []):
                window = dashboard._create_detached_pivot_internal()
                window.restoreGeometry(self._hex_to_qbytearray(config['geometry']))

                # ✅ CARICA DATI STORICI PRIMA DI RESTORE CONFIG
                if not dashboard.all_trades.empty:
                    window.pivot_widget.set_source_data(dashboard.all_trades)

                if config.get('pivot_config'):
                    pivot_cfg = config['pivot_config']

                    # Ripristina filtri avanzati PRIMA della config generale
                    if 'advanced_filter' in pivot_cfg and pivot_cfg['advanced_filter']:
                        filter_group = self._deserialize_filter_group(pivot_cfg['advanced_filter'])
                        if filter_group:
                            window.pivot_widget.active_filter = filter_group
                            # Aggiorna label filtri
                            if hasattr(window.pivot_widget, '_update_filter_label'):
                                window.pivot_widget._update_filter_label()

                    # Poi ripristina configurazione generale
                    if hasattr(window.pivot_widget, 'restore_config'):
                        window.pivot_widget.restore_config(pivot_cfg)

                window.show()

            # 🆕 CHART WINDOWS - Con filtri
            for config in detached.get('chart_windows', []):
                window = dashboard._create_detached_chart_internal()
                window.restoreGeometry(self._hex_to_qbytearray(config['geometry']))

                # ✅ CARICA DATI STORICI PRIMA DI RESTORE CONFIG
                if not dashboard.all_trades.empty:
                    window.chart_widget.set_data(dashboard.all_trades)

                if config.get('chart_config'):
                    chart_cfg = config['chart_config']

                    # Ripristina filtri avanzati PRIMA della config generale
                    if 'advanced_filter' in chart_cfg and chart_cfg['advanced_filter']:
                        filter_group = self._deserialize_filter_group(chart_cfg['advanced_filter'])
                        if filter_group:
                            window.chart_widget.active_filter = filter_group
                            # Aggiorna label filtri
                            if hasattr(window.chart_widget, '_update_filter_label'):
                                window.chart_widget._update_filter_label()

                    # Poi ripristina configurazione generale
                    if hasattr(window.chart_widget, 'restore_config'):
                        window.chart_widget.restore_config(chart_cfg)

                window.show()

            # FLOW WINDOWS
            for config in detached.get('flow_windows', []):
                window = dashboard._create_detached_flow_internal()
                window.restoreGeometry(self._hex_to_qbytearray(config['geometry']))
                window.show()

        except Exception as e:
            self.logger.warning(f"Failed to restore detached windows: {e}")
            import traceback
            traceback.print_exc()

    def _deserialize_filter_group(self, filter_data: Dict[str, Any]):
        """
        🆕 Deserializza un FilterGroup da dict JSON
        """
        try:
            from market_monitor.gui.implementations.PyQt5Dashboard.widgets.filter import (
                FilterOperator, FilterGroup, FilterCondition
            )

            logic = filter_data.get('logic', 'AND')
            
            # Crea FilterGroup SENZA conditions
            filter_group = FilterGroup(logic=logic)

            # Aggiungi conditions dopo
            for cond_dict in filter_data.get('conditions', []):
                try:
                    operator = FilterOperator(cond_dict['operator'])
                    value = self._deserialize_filter_value(cond_dict['value'])
                    value2 = self._deserialize_filter_value(cond_dict.get('value2'))

                    condition = FilterCondition(
                        column=cond_dict['column'],
                        operator=operator,
                        value=value,
                        value2=value2
                    )
                    filter_group.add_condition(condition)

                except Exception as e:
                    self.logger.warning(f"Failed to deserialize filter condition: {e}")
                    continue

            if filter_group.conditions:
                return filter_group

            return None

        except ImportError as ie:
            self.logger.warning(f"Could not import FilterGroup/ColumnFilter for filter restoration: {ie}")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to deserialize filter group: {e}")
            return None

    def _deserialize_filter_value(self, value):
        """Deserializza un valore di filtro (gestisce datetime, etc)"""
        if value is None:
            return None
        if isinstance(value, str):
            # Prova a convertire datetime ISO
            try:
                return pd.to_datetime(value)
            except:
                return value
        if isinstance(value, list):
            return [self._deserialize_filter_value(v) for v in value]
        return value

    # ==================== HELPER METHODS ====================

    def _get_column_widths(self, table) -> List[int]:
        """Ritorna larghezze di tutte le colonne"""
        widths = []
        for i in range(table.columnCount()):
            widths.append(table.columnWidth(i))
        return widths

    def _restore_column_widths(self, table, widths: List[int]):
        """Ripristina larghezze colonne"""
        for i, width in enumerate(widths):
            if i < table.columnCount():
                table.setColumnWidth(i, width)

    # ==================== MANAGEMENT ====================

    def list_dashboards(self) -> List[Dict[str, Any]]:
        """Lista tutte le dashboard salvate"""
        dashboards = []

        for config_file in self.config_dir.glob("*.json"):
            if config_file.stem.endswith('.backup'):
                continue

            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                metadata = state.get('metadata', {})
                metadata['file_path'] = str(config_file)
                metadata['file_size'] = config_file.stat().st_size

                dashboards.append(metadata)

            except Exception as e:
                self.logger.warning(f"Failed to read dashboard {config_file}: {e}")

        dashboards.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return dashboards

    def delete_dashboard(self, name: str) -> bool:
        """Elimina una dashboard"""
        try:
            safe_name = self._sanitize_filename(name)
            config_file = self.config_dir / f"{safe_name}.json"

            if config_file.exists():
                config_file.unlink()
                self.logger.info(f"Dashboard deleted: {name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete dashboard: {e}")
            return False

    def rename_dashboard(self, old_name: str, new_name: str) -> bool:
        """Rinomina una dashboard"""
        try:
            old_safe = self._sanitize_filename(old_name)
            new_safe = self._sanitize_filename(new_name)

            old_file = self.config_dir / f"{old_safe}.json"
            new_file = self.config_dir / f"{new_safe}.json"

            if not old_file.exists():
                return False

            if new_file.exists():
                raise DashboardStateError(f"Dashboard '{new_name}' already exists")

            with open(old_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            state['metadata']['name'] = new_name

            with open(new_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=self._json_serializer, ensure_ascii=False)

            old_file.unlink()

            self.logger.info(f"Dashboard renamed: {old_name} -> {new_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to rename dashboard: {e}")
            return False

    def duplicate_dashboard(self, name: str, new_name: str) -> bool:
        """Duplica una dashboard"""
        try:
            safe_name = self._sanitize_filename(name)
            new_safe = self._sanitize_filename(new_name)

            source_file = self.config_dir / f"{safe_name}.json"
            dest_file = self.config_dir / f"{new_safe}.json"

            if not source_file.exists():
                return False

            if dest_file.exists():
                raise DashboardStateError(f"Dashboard '{new_name}' already exists")

            with open(source_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            state['metadata']['name'] = new_name
            state['metadata']['timestamp'] = datetime.now().isoformat()

            with open(dest_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=self._json_serializer, ensure_ascii=False)

            self.logger.info(f"Dashboard duplicated: {name} -> {new_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to duplicate dashboard: {e}")
            return False

    def export_dashboard(self, name: str, export_path: Path) -> bool:
        """Esporta dashboard"""
        try:
            safe_name = self._sanitize_filename(name)
            source_file = self.config_dir / f"{safe_name}.json"

            if not source_file.exists():
                return False

            import shutil
            shutil.copy2(source_file, export_path)

            self.logger.info(f"Dashboard exported: {name} -> {export_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export dashboard: {e}")
            return False

    def import_dashboard(self, import_path: Path, new_name: Optional[str] = None) -> bool:
        """Importa dashboard"""
        try:
            if not import_path.exists():
                return False

            with open(import_path, 'r', encoding='utf-8') as f:
                state = json.load(f)

            if 'metadata' not in state:
                raise DashboardStateError("Invalid dashboard file format")

            if new_name:
                state['metadata']['name'] = new_name

            name = state['metadata']['name']
            safe_name = self._sanitize_filename(name)
            dest_file = self.config_dir / f"{safe_name}.json"

            if dest_file.exists():
                raise DashboardStateError(f"Dashboard '{name}' already exists")

            with open(dest_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=self._json_serializer, ensure_ascii=False)

            self.logger.info(f"Dashboard imported: {name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to import dashboard: {e}")
            return False

    # ==================== UTILITIES ====================

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer"""
        if isinstance(obj, QByteArray):
            return self._qbytearray_to_hex(obj)
        if isinstance(obj, QColor):
            return obj.name()
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    @staticmethod
    def _qbytearray_to_hex(qba: QByteArray) -> str:
        """Converti QByteArray in hex string"""
        return qba.toHex().data().decode('ascii')

    @staticmethod
    def _hex_to_qbytearray(hex_str: str) -> QByteArray:
        """Converti hex string in QByteArray"""
        return QByteArray.fromHex(hex_str.encode('ascii'))

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitizza nome file"""
        import re
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        return safe[:200]

    def _migrate_config(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Migra configurazione da versione precedente"""
        old_version = state['metadata'].get('version', '0.0')
        self.logger.warning(f"Migrating config from {old_version} to {self.VERSION}")

        # Migrazione 1.0 -> 1.1: aggiungi campi filtri se mancano
        if old_version in ['1.0', '0.0']:
            # Trade table
            if 'trade_table' in state and 'advanced_filter' not in state['trade_table']:
                state['trade_table']['advanced_filter'] = None

            # Detached windows
            if 'detached_windows' in state:
                for pivot in state['detached_windows'].get('pivot_windows', []):
                    if 'pivot_config' in pivot and 'advanced_filter' not in pivot['pivot_config']:
                        pivot['pivot_config']['advanced_filter'] = None

                for chart in state['detached_windows'].get('chart_windows', []):
                    if 'chart_config' in chart and 'advanced_filter' not in chart['chart_config']:
                        chart['chart_config']['advanced_filter'] = None

        state['metadata']['version'] = self.VERSION
        return state