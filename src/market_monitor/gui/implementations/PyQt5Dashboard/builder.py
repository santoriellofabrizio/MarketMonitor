"""
Script semplice per lanciare Trade Dashboard con simulatore.
Usa import assoluti - da eseguire dalla root del progetto.
"""
import os

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

    log_section = config.get("logging", {})
    logger = setup_custom_logging(log_section)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(_h)

    dashboard_cfg = config.get("dashboard", {})

    dashboard = TradeDashboard(
        columns=dashboard_cfg.get("columns"),
        mode=dashboard_cfg.get("mode", "redis"),
        redis_config=dashboard_cfg.get("redis", {}),
        rabbit_config=dashboard_cfg.get("rabbit", {}),
        logger=logger,
        metrics_config=dashboard_cfg.get("metrics", {}),
    )

    return dashboard


def setup_custom_logging(log_cfg: dict):
    """Configura il logging basandosi sul file YAML."""
    path = log_cfg.get("path", "logs/dashboard.log")
    level_str = log_cfg.get("level", "INFO").upper()
    log_name = log_cfg.get("log_name", "TradeDashboard")

    # Crea la cartella dei log se non esiste
    log_dir = os.path.dirname(path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    level = getattr(logging, level_str, logging.INFO)
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    logger.propagate = False

    # Evitiamo di aggiungere handler duplicati se la funzione viene chiamata più volte
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Handler per Console
        stream_h = logging.StreamHandler()
        stream_h.setFormatter(formatter)
        logger.addHandler(stream_h)

        # Handler per File
        file_h = logging.FileHandler(path)
        file_h.setFormatter(formatter)
        logger.addHandler(file_h)

    return logger
