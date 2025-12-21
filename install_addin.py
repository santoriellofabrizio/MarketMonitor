#!/usr/bin/env python
"""
Script diagnostico per TradeDashboard crash 0xC0000409
"""
import sys
import traceback

print("=" * 60)
print("DIAGNOSTIC SCRIPT - TradeDashboard Crash")
print("=" * 60)

# Step 1: Test imports base
print("\n[1/8] Testing base imports...")
try:
    from PyQt5.QtWidgets import QApplication

    print("✅ PyQt5.QtWidgets OK")
except Exception as e:
    print(f"❌ PyQt5.QtWidgets FAILED: {e}")
    sys.exit(1)

try:
    import pandas as pd

    print("✅ pandas OK")
except Exception as e:
    print(f"❌ pandas FAILED: {e}")
    sys.exit(1)

# Step 2: Test project path
print("\n[2/8] Testing project path...")
try:
    sys.path.insert(0, r"C:\AFMachineLearning\Projects\Trading\MarketMonitorFI\src")
    print("✅ Project path added")
except Exception as e:
    print(f"❌ Path setup FAILED: {e}")
    sys.exit(1)

# Step 3: Test imports progetto
print("\n[3/8] Testing project imports...")
try:
    from market_monitor_fi.GUI.ConcreteGUI.PyQt5Dashboard import trade_dashboard

    print("✅ TradeDashboard import OK")
except Exception as e:
    print(f"❌ TradeDashboard import FAILED:")
    traceback.print_exc()
    sys.exit(1)

try:
    from market_monitor_fi.GUI.ConcreteGUI.PyQt5Dashboard.trade_dashboard import METRIC_DEFINITIONS

    print(f"✅ METRIC_DEFINITIONS OK ({len(METRIC_DEFINITIONS)} metrics)")
except Exception as e:
    print(f"❌ METRIC_DEFINITIONS FAILED:")
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test QApplication creation
print("\n[4/8] Testing QApplication creation...")
try:
    app = QApplication(sys.argv)
    print("✅ QApplication created")
except Exception as e:
    print(f"❌ QApplication FAILED:")
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test Queue creation
print("\n[5/8] Testing Queue and DataSource...")
try:
    from queue import Queue
    from market_monitor_fi.GUI.ThreadGUI.DataSource.QueueDataSource import QueueDataSource

    q = Queue(maxsize=1000)
    datasource = QueueDataSource(q)
    print("✅ Queue and DataSource OK")
except Exception as e:
    print(f"❌ Queue/DataSource FAILED:")
    traceback.print_exc()
    sys.exit(1)

# Step 6: Test Dashboard creation (CRITICAL)
print("\n[6/8] Testing TradeDashboard creation...")
try:
    import logging

    logger = logging.getLogger("Test")
    logger.setLevel(logging.DEBUG)

    # Test con config minima
    dashboard = TradeDashboard.TradeDashboard(
        datasource=datasource,
        mode="queue",
        logger=logger,
        metrics_config={"enabled": False}  # ← Disabilita metriche per test
    )
    print("✅ TradeDashboard created (no metrics)")
except Exception as e:
    print(f"❌ TradeDashboard creation FAILED:")
    traceback.print_exc()
    sys.exit(1)

# Step 7: Test Dashboard creation CON metriche
print("\n[7/8] Testing TradeDashboard with metrics...")
try:
    dashboard2 = TradeDashboard.TradeDashboard(
        datasource=datasource,
        mode="queue",
        logger=logger,
        metrics_config={
            "enabled": True,
            "items": ["total_trades"]  # Solo 1 metrica
        }
    )
    print("✅ TradeDashboard created (with 1 metric)")
except Exception as e:
    print(f"❌ TradeDashboard with metrics FAILED:")
    traceback.print_exc()
    print("\n⚠️  IL CRASH È NELLA CREAZIONE DELLE METRICHE!")
    sys.exit(1)

# Step 8: Test show() - QUESTO POTREBBE CRASHARE
print("\n[8/8] Testing dashboard.show()...")
print("⚠️  This may crash if Qt has issues...")

try:
    dashboard2.show()
    print("✅ dashboard.show() OK")

    # Non eseguiamo app.exec_() per non bloccare
    print("\n✅ ALL TESTS PASSED!")
    print("\nIl crash deve essere in app.exec_() o durante il rendering.")
    print("Questo suggerisce un problema con:")
    print("  - Worker thread initialization")
    print("  - Widget rendering")
    print("  - Signal/slot connections")

except Exception as e:
    print(f"❌ dashboard.show() FAILED:")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETED")
print("=" * 60)