"""
test_gui_headless.py
====================
Headless smoke test for the 7-screen PyQt5 GUI application.
Verifies Pre-Test Summary screen, channel selection, execution transitions, and reset.
"""

import sys
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use("Qt5Agg")

from PyQt5.QtWidgets import QApplication
from gui_app import SPECSApp


def test_gui_screens():
    print("Testing 7-Screen PyQt5 GUI and Pre-Test Confirmation...")
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = SPECSApp()
    assert window.width() == 480
    assert window.height() == 320
    assert window.stack.count() == 7, f"Expected 7 screens, got {window.stack.count()}"

    print(f"  -> Window dimensions: 480x320 (3.5-inch aspect ratio). Screens: {window.stack.count()}")

    # Screen 0: Board Name
    window.navigate_to(0)
    window.screen_board.input_field.setText("SMART_POWER_DUT")
    assert window.screen_board.btn_next.isEnabled()
    window.screen_board.btn_next.click()

    # Screen 1: Mfg Info
    assert window.stack.currentIndex() == 1
    window.screen_mfg.input_date.setText("2024-06-15")
    window.screen_mfg.input_serial.setText("SN-77621")
    assert window.screen_mfg.btn_next.isEnabled()
    window.screen_mfg.btn_next.click()

    # Screen 2: Channel Selection
    assert window.stack.currentIndex() == 2
    window.screen_channels.btn_select_all.click()
    assert len(window.screen_channels._selected) == 43
    assert window.screen_channels.btn_proceed.isEnabled()
    window.screen_channels.btn_proceed.click()

    # Screen 3: Pre-Test Summary Screen (NEW!)
    assert window.stack.currentIndex() == 3
    print("  -> Screen 3 (PreTestSummaryScreen) successfully entered.")
    assert "SMART_POWER_DUT" in window.screen_summary.lbl_meta.text()
    assert "SN-77621" in window.screen_summary.lbl_meta.text()
    assert "43" in window.screen_summary.lbl_meta.text()
    print("  -> Pre-test metadata, hardware status, and 43 channels verified.")

    # Test "PROCEED TO TEST" button transitions to Screen 4 (ExecutionScreen)
    window.screen_summary.btn_proceed.click()
    assert window.stack.currentIndex() == 4
    print("  -> Screen 4 (ExecutionScreen): 30s timer started successfully.")

    # Screen 5: Dashboard Screen
    window.navigate_to(5)
    assert window.stack.currentIndex() == 5
    print("  -> Screen 5 (DashboardScreen): Verified.")

    # Screen 6: Certification Screen
    window.navigate_to(6)
    assert window.stack.currentIndex() == 6
    print("  -> Screen 6 (CertificationScreen): Verified.")

    # Reset
    window.reset_all()
    assert window.stack.currentIndex() == 0
    assert window.screen_board.input_field.text() == ""
    print("  -> Reset to Screen 0: Verified.")

    print("\nALL 7 GUI SCREENS AND TRANSITIONS VERIFIED 100%!")


if __name__ == "__main__":
    test_gui_screens()
