#!/usr/bin/env python
"""Test script per verificare che tutti gli import funzionano correttamente"""

import sys

# Aggiungi src al path esattamente come fa PyInstaller
src_path = r'C:\AFMachineLearning\Projects\Trading\MarketMonitorFI\src'
sys.path.insert(0, src_path)

print("=" * 60)
print("Testing Market Monitor FI Imports")
print("=" * 60)

try:
    print("✓ PyQt5...")
    from PyQt5.QtWidgets import QApplication
    print("  ✓ PyQt5.QtWidgets")
except Exception as e:
    print(f"  ✗ PyQt5 Error: {e}")
    sys.exit(1)

try:
    print("✓ market_monitor_fi.GUI.ConcreteGUI.PyQt5Dashboard.builder...")
    from market_monitor_fi.GUI.ConcreteGUI.PyQt5Dashboard.builder import build_dashboard
    print("  ✓ build_dashboard imported")
except Exception as e:
    print(f"  ✗ Builder Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("✓ market_monitor_fi.GUI.ConcreteGUI.PyQt5Dashboard.widgets...")
    from market_monitor_fi.GUI.ConcreteGUI.PyQt5Dashboard.widgets.dashboard_state import DashboardState
    print("  ✓ DashboardState imported")
except Exception as e:
    print(f"  ✗ Widgets Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("✓ market_monitor_fi.utils.RTData...")
    from market_monitor_fi.live_data_hub.RTData import RTData
    print("  ✓ RTData imported")
except Exception as e:
    print(f"  ✗ RTData Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("✓ External dependencies (numpy, pandas, ruamel.yaml)...")
    import numpy
    import pandas
    from ruamel.yaml import YAML
    print("  ✓ All external dependencies available")
except Exception as e:
    print(f"  ✗ External dependencies Error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All imports successful! Ready for PyInstaller")
print("=" * 60)
