import logging
import sys
from PyQt5.QtWidgets import QApplication

from market_monitor.gui.implementations.PyQt5Dashboard.builder import build_dashboard


def main():
    # ========== SETUP LOGGING ==========
    logging.basicConfig(
        level=logging.DEBUG,  # Mostra tutto incluso DEBUG
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Output su console
            logging.FileHandler('src/market_monitor/gui/implementations/PyQt5Dashboard/dashboard_debug.log', mode='w')  # Salva su file
        ]
    )

    # Logger per vedere i messaggi importanti
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Starting Trade Dashboard with DEBUG logging")
    logger.info("=" * 80)
    # ===================================

    cfg = sys.argv[1] if len(sys.argv) > 1 else None

    # 1. QApplication PRIMA di qualunque QWidget
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 2. Costruzione dashboard (QWidget)
    dashboard = build_dashboard(cfg)

    # 3. Show
    dashboard.start()

    # 4. Event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()