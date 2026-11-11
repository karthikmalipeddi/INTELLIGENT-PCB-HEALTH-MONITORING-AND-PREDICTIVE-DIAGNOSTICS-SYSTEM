"""
main.py
=======
S.P.E.C.S. (Smart Predictive Electronic Component Scanner)
Industry-Grade Edge-Computing Testbench Architecture.

Execution Entry Point.
"""

import os
import sys

# =========================================================================== #
# CRITICAL ARCHITECTURE CONSTRAINT: MOCK_MODE TOGGLE                          #
# =========================================================================== #
# Set MOCK_MODE = True for laptop-based mock demo testing without hardware.
# Set MOCK_MODE = False when deploying on the physical Raspberry Pi 4 testbench.
MOCK_MODE: bool = True
# =========================================================================== #

# Synchronize top-level MOCK_MODE with system configuration
import config
config.MOCK_MODE = MOCK_MODE

# Set Matplotlib backend to Qt5Agg before any other matplotlib import
import matplotlib
matplotlib.use("Qt5Agg")

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from gui_app import SPECSApp


def main():
    """Application entry point for S.P.E.C.S. testbench."""
    print("=" * 70)
    print(" S.P.E.C.S. - Smart Predictive Electronic Component Scanner")
    print(f" Mode: {'[MOCK_MODE - LAPTOP TEST]' if config.MOCK_MODE else '[PRODUCTION - RASPBERRY PI 4]'}")
    print(" Display Aspect: 3.5-inch Touchscreen (480x320)")
    print("=" * 70)

    # Ensure required directories exist
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.CAPTURES_DIR, exist_ok=True)

    # Support command line flag override: python main.py --prod or --mock
    if "--prod" in sys.argv:
        config.MOCK_MODE = False
        print("--> Overridden by command-line flag: PRODUCTION MODE (MOCK_MODE = False)")
    elif "--mock" in sys.argv:
        config.MOCK_MODE = True
        print("--> Overridden by command-line flag: MOCK MODE (MOCK_MODE = True)")

    # Enable high-DPI scaling if supported
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    window = SPECSApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

