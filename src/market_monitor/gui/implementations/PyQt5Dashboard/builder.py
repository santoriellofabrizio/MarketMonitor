"""
Script semplice per lanciare Trade Dashboard con simulatore.
Usa import assoluti - da eseguire dalla root del progetto.
"""
from ruamel.yaml import YAML
import logging
from PyQt5.QtWidgets import QApplication

from market_monitor.gui.implementations.PyQt5Dashboard.trade_dashboard import TradeDashboard


def build_dashboard(config: str | None = None) -> TradeDashboard:
    """
    Costruisce e restituisce la TradeDashboard.
    NON crea QApplication
    NON chiama exec_()
    """


    if config:
        if not isinstance(config, dict):
            try:
                reader = YAML(typ="safe")
                with open(config, "r") as f:
                    loaded = reader.load(f)
                    if isinstance(loaded, dict):
                        config = loaded
            except Exception as e:
                print(f"⚠️ Config non caricata: {e}")

    logger = logging.getLogger("TradeDashboard")
    logger.propagate = False
    logger.setLevel(logging.INFO)

    dashboard_cfg = config.get("dashboard", {})

    dashboard = TradeDashboard(
        columns=dashboard_cfg.get("columns"),
        mode=dashboard_cfg.get("mode", "redis"),
        redis_config=dashboard_cfg.get("redis", {}),
        logger=logger,
        metrics_config=dashboard_cfg.get("metrics", {}),
    )

    return dashboard



