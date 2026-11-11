"""
gui_app.py
==========
PyQt5 touch-friendly GUI for the TEST GIG PCB diagnostic testbench.

Designed for a 3.5-inch touchscreen (480x320 pixels) with a dark theme:
Orange (#FF6600), Black (#1A1A1A), and White (#FFFFFF) accents.

Seven Screens managed by QStackedWidget:
    0. BoardNameScreen       — Board name entry with touch keyboard
    1. MfgInfoScreen         — Manufacturing date & serial number
    2. ChannelSelectScreen   — 2-column scrollable grid to select from 43 channels
    3. PreTestSummaryScreen  — Review entered metadata, channels, & hardware before starting
    4. ExecutionScreen       — 30-second countdown timer + background worker
    5. DashboardScreen       — Detailed component failure findings, RUL%, waveforms
    6. CertificationScreen   — Diagnostic Center PDF certificate, QR code, HTTP server
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from typing import Optional

# Matplotlib Qt5Agg backend
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from PyQt5.QtCore import (
    Qt,
    QTimer,
    QThread,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import config
from database_manager import DatabaseManager
from data_acquisition import DataAcquisitionEngine
from diagnostic_engine import DiagnosticEngine
from prognostic_engine import PrognosticEngine
from report_generator import ReportGenerator
from hardware_vision import HardwareManager, ThermalOpticalOverlayEngine


# =========================================================================== #
#  Global Stylesheet                                                          #
# =========================================================================== #

STYLESHEET = f"""
QWidget {{
    background-color: {config.COLOR_BG};
    color:            {config.COLOR_TEXT_PRIMARY};
    font-family:      "Segoe UI", Arial, sans-serif;
}}
QPushButton {{
    background-color: {config.COLOR_PRIMARY};
    color:            {config.COLOR_TEXT_PRIMARY};
    border:           none;
    border-radius:    6px;
    font-size:        13px;
    font-weight:      bold;
    padding:          8px;
    min-height:       38px;
}}
QPushButton:pressed  {{ background-color: {config.COLOR_PRIMARY_PRESSED}; }}
QPushButton:disabled {{ background-color: #383838; color: #777777; }}
QLineEdit {{
    background-color: {config.COLOR_SURFACE};
    color:            {config.COLOR_TEXT_PRIMARY};
    border:           2px solid {config.COLOR_PRIMARY};
    border-radius:    5px;
    padding:          6px;
    font-size:        13px;
}}
QLabel {{ color: {config.COLOR_TEXT_PRIMARY}; }}
QScrollArea {{ border: none; background-color: transparent; }}
QProgressBar {{
    border: 1px solid #444444;
    border-radius: 4px;
    text-align: center;
    background-color: {config.COLOR_SURFACE_INPUT};
    color: {config.COLOR_TEXT_PRIMARY};
    font-weight: bold;
}}
QProgressBar::chunk {{
    background-color: {config.COLOR_PRIMARY};
    border-radius: 3px;
}}
"""


# =========================================================================== #
#  Reusable On-Screen Touch Keyboard                                          #
# =========================================================================== #

class TouchKeyboard(QWidget):
    """QWERTY on-screen touch keyboard designed for fat-finger 3.5-inch displays."""

    key_pressed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(2)

        rows = [
            list("1234567890"),
            list("QWERTYUIOP"),
            list("ASDFGHJKL"),
            list("ZXCVBNM-."),
        ]

        for row_chars in rows:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(2)
            for ch in row_chars:
                btn = self._make_key(ch)
                row_layout.addWidget(btn)
            layout.addLayout(row_layout)

        # Control Row: BACKSPACE | SPACE | ENTER
        bottom = QHBoxLayout()
        bottom.setSpacing(2)

        btn_bs = self._make_key("⌫", tag="BACKSPACE")
        btn_bs.setMinimumWidth(60)
        bottom.addWidget(btn_bs)

        btn_sp = self._make_key("SPACE", tag=" ")
        btn_sp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bottom.addWidget(btn_sp)

        btn_ent = self._make_key("↵", tag="ENTER")
        btn_ent.setMinimumWidth(60)
        bottom.addWidget(btn_ent)

        layout.addLayout(bottom)

    def _make_key(self, label: str, tag: str | None = None) -> QPushButton:
        btn = QPushButton(label)
        btn.setMinimumSize(30, 30)
        btn.setStyleSheet(
            "QPushButton { background-color: #333333; font-size: 11px; "
            "min-height: 28px; border-radius: 4px; padding: 2px; }"
            "QPushButton:pressed { background-color: #FF6600; }"
        )
        emit_value = tag if tag is not None else label
        btn.clicked.connect(lambda checked, v=emit_value: self.key_pressed.emit(v))
        return btn


# =========================================================================== #
#  Screen 0 — Board Name Entry                                                #
# =========================================================================== #

class BoardNameScreen(QWidget):
    """Screen 0: Enter the PCB Board Name using on-screen keyboard."""

    next_pressed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 4)
        layout.setSpacing(4)

        # Header
        title = QLabel(config.APP_NAME)
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setStyleSheet(f"color: {config.COLOR_PRIMARY}; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel(config.APP_SUBTITLE)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Segoe UI", 8))
        subtitle.setStyleSheet(f"color: {config.COLOR_TEXT_MUTED};")
        layout.addWidget(subtitle)

        # Input
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter Board Name (e.g. POWER_DUT_V2)")
        self.input_field.setMaximumHeight(34)
        self.input_field.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.input_field)

        # Keyboard
        self.keyboard = TouchKeyboard()
        self.keyboard.key_pressed.connect(self._handle_key)
        layout.addWidget(self.keyboard)

        # Next Button
        self.btn_next = QPushButton("NEXT →")
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(
            lambda: self.next_pressed.emit(self.input_field.text().strip())
        )
        layout.addWidget(self.btn_next)

    @pyqtSlot(str)
    def _handle_key(self, key: str) -> None:
        if key == "BACKSPACE":
            self.input_field.backspace()
        elif key == "ENTER":
            if self.input_field.text().strip():
                self.next_pressed.emit(self.input_field.text().strip())
        else:
            self.input_field.insert(key)

    @pyqtSlot(str)
    def _on_text_changed(self, text: str) -> None:
        self.btn_next.setEnabled(bool(text.strip()))

    def reset(self) -> None:
        self.input_field.clear()


# =========================================================================== #
#  Screen 1 — Manufacturing Info                                              #
# =========================================================================== #

class MfgInfoScreen(QWidget):
    """Screen 1: Enter Manufacturing Date & Serial Number."""

    back_pressed = pyqtSignal()
    next_pressed = pyqtSignal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._active_field: Optional[QLineEdit] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 4)
        layout.setSpacing(4)

        title = QLabel("Manufacturing Details")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet("color: #FF6600;")
        layout.addWidget(title)

        fields_row = QHBoxLayout()
        self.input_date = QLineEdit()
        self.input_date.setPlaceholderText("Mfg Date: YYYY-MM-DD")
        self.input_date.setMaximumHeight(32)
        self.input_date.textChanged.connect(self._check_fields)
        fields_row.addWidget(self.input_date)

        self.input_serial = QLineEdit()
        self.input_serial.setPlaceholderText("Serial: SN-00000")
        self.input_serial.setMaximumHeight(32)
        self.input_serial.textChanged.connect(self._check_fields)
        fields_row.addWidget(self.input_serial)
        layout.addLayout(fields_row)

        self.input_date.installEventFilter(self)
        self.input_serial.installEventFilter(self)
        self._active_field = self.input_date

        self.keyboard = TouchKeyboard()
        self.keyboard.key_pressed.connect(self._handle_key)
        layout.addWidget(self.keyboard)

        btn_row = QHBoxLayout()
        self.btn_back = QPushButton("← BACK")
        self.btn_back.setStyleSheet("background-color: #383838;")
        self.btn_back.clicked.connect(self.back_pressed.emit)
        btn_row.addWidget(self.btn_back)

        self.btn_next = QPushButton("NEXT →")
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._on_next)
        btn_row.addWidget(self.btn_next)
        layout.addLayout(btn_row)

    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent
        if event.type() == QEvent.FocusIn:
            if obj in (self.input_date, self.input_serial):
                self._active_field = obj
        return super().eventFilter(obj, event)

    @pyqtSlot(str)
    def _handle_key(self, key: str) -> None:
        field = self._active_field or self.input_date
        if key == "BACKSPACE":
            field.backspace()
        elif key == "ENTER":
            if field is self.input_date:
                self.input_serial.setFocus()
            else:
                self._on_next()
        else:
            field.insert(key)

    @pyqtSlot()
    def _check_fields(self) -> None:
        has_date = bool(self.input_date.text().strip())
        has_serial = bool(self.input_serial.text().strip())
        self.btn_next.setEnabled(has_date and has_serial)

    def _on_next(self) -> None:
        d = self.input_date.text().strip()
        s = self.input_serial.text().strip()
        if d and s:
            self.next_pressed.emit(d, s)

    def reset(self) -> None:
        self.input_date.clear()
        self.input_serial.clear()


# =========================================================================== #
#  Screen 2 — Channel Selection (Scrollable 2-Column Grid)                     #
# =========================================================================== #

class ChannelSelectScreen(QWidget):
    """Screen 2: Scrollable 2-column grid to select from 43 electrical channels."""

    back_pressed = pyqtSignal()
    proceed_pressed = pyqtSignal(list)

    ALL_CHANNELS: list[str] = list(config.ELECTRICAL_CHANNELS)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: dict[str, QPushButton] = {}
        self._selected: set[str] = set()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(3)

        total_ch = len(self.ALL_CHANNELS)
        title_row = QHBoxLayout()
        title = QLabel(f"Channel Selection ({total_ch} Ch)")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setStyleSheet(f"color: {config.COLOR_PRIMARY};")
        title_row.addWidget(title)
        title_row.addStretch()

        self.count_label = QLabel(f"(0/{total_ch} selected)")
        self.count_label.setStyleSheet(f"color: {config.COLOR_TEXT_MUTED}; font-size: 11px;")
        title_row.addWidget(self.count_label)
        layout.addLayout(title_row)

        self.btn_select_all = QPushButton("SELECT ALL CHANNELS")
        self.btn_select_all.setMaximumHeight(30)
        self.btn_select_all.setStyleSheet(
            f"QPushButton {{ background-color: {config.COLOR_BORDER}; font-size: 11px; min-height: 28px; border-radius: 4px; }}"
            f"QPushButton:pressed {{ background-color: {config.COLOR_PRIMARY}; }}"
        )
        self.btn_select_all.clicked.connect(self._toggle_all)
        layout.addWidget(self.btn_select_all)

        # Scrollable 2-Column Grid (approx 2x3 visible)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(3)
        grid_layout.setContentsMargins(1, 1, 1, 1)

        for idx, ch_name in enumerate(self.ALL_CHANNELS):
            row = idx // 2
            col = idx % 2
            btn = QPushButton(ch_name)
            btn.setCheckable(True)
            btn.setMinimumHeight(34)
            btn.setStyleSheet(self._toggle_style(False))
            btn.clicked.connect(lambda checked, ch=ch_name: self._on_toggle(ch))
            grid_layout.addWidget(btn, row, col)
            self._buttons[ch_name] = btn

        scroll.setWidget(grid_widget)
        layout.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_back = QPushButton("← BACK")
        self.btn_back.setStyleSheet(f"background-color: {config.COLOR_SURFACE_ALT};")
        self.btn_back.clicked.connect(self.back_pressed.emit)
        btn_row.addWidget(self.btn_back)

        self.btn_proceed = QPushButton("REVIEW & PROCEED →")
        self.btn_proceed.setEnabled(False)
        self.btn_proceed.clicked.connect(
            lambda: self.proceed_pressed.emit(sorted(self._selected))
        )
        btn_row.addWidget(self.btn_proceed)
        layout.addLayout(btn_row)

    @staticmethod
    def _toggle_style(selected: bool) -> str:
        bg = config.COLOR_PRIMARY if selected else config.COLOR_SURFACE_ALT
        pressed = config.COLOR_PRIMARY_PRESSED if selected else config.COLOR_PRIMARY
        return (
            f"QPushButton {{ background-color: {bg}; font-size: 11px; "
            f"min-height: 32px; border-radius: 5px; font-weight: bold; }}"
            f"QPushButton:pressed {{ background-color: {pressed}; }}"
        )

    def _on_toggle(self, ch_name: str) -> None:
        if ch_name in self._selected:
            self._selected.discard(ch_name)
        else:
            self._selected.add(ch_name)
        self._buttons[ch_name].setStyleSheet(self._toggle_style(ch_name in self._selected))
        self._update_count()

    def _toggle_all(self) -> None:
        if len(self._selected) < len(self.ALL_CHANNELS):
            for ch in self.ALL_CHANNELS:
                self._selected.add(ch)
                self._buttons[ch].setStyleSheet(self._toggle_style(True))
        else:
            self._selected.clear()
            for ch in self.ALL_CHANNELS:
                self._buttons[ch].setStyleSheet(self._toggle_style(False))
        self._update_count()

    def _update_count(self) -> None:
        n = len(self._selected)
        total_ch = len(self.ALL_CHANNELS)
        self.count_label.setText(f"({n}/{total_ch} selected)")
        self.btn_proceed.setEnabled(n > 0)

    def reset(self) -> None:
        self._selected.clear()
        for ch in self.ALL_CHANNELS:
            self._buttons[ch].setChecked(False)
            self._buttons[ch].setStyleSheet(self._toggle_style(False))
        self._update_count()


# =========================================================================== #
#  Screen 3 — Pre-Test Inspection Summary (User Confirmation Screen)           #
# =========================================================================== #

class PreTestSummaryScreen(QWidget):
    """Screen 3: Review all entered metadata and selected channels before starting test."""

    back_pressed = pyqtSignal()
    start_test_pressed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._board_name = ""
        self._serial = ""
        self._mfg_date = ""
        self._channels: list[str] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 5)
        layout.setSpacing(4)

        title = QLabel("Pre-Test Inspection Summary")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setStyleSheet("color: #FF6600;")
        layout.addWidget(title)

        subtitle = QLabel("Verify test parameters before executing 30-second acquisition:")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Segoe UI", 8))
        subtitle.setStyleSheet("color: #AAAAAA;")
        layout.addWidget(subtitle)

        # Scrollable Summary Card
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        card = QWidget()
        c_layout = QVBoxLayout(card)
        c_layout.setContentsMargins(6, 4, 6, 4)
        c_layout.setSpacing(4)

        self.lbl_meta = QLabel("Loading metadata...")
        self.lbl_meta.setStyleSheet("font-size: 11px; background-color: #222222; padding: 6px; border-radius: 4px;")
        self.lbl_meta.setWordWrap(True)
        c_layout.addWidget(self.lbl_meta)

        self.lbl_hardware = QLabel("Hardware Status Readiness:")
        self.lbl_hardware.setStyleSheet("font-size: 10px; color: #FF6600; font-weight: bold;")
        c_layout.addWidget(self.lbl_hardware)

        self.lbl_hw_status = QLabel("")
        self.lbl_hw_status.setStyleSheet("font-size: 10px; background-color: #222222; padding: 6px; border-radius: 4px;")
        self.lbl_hw_status.setWordWrap(True)
        c_layout.addWidget(self.lbl_hw_status)

        self.lbl_channels_summary = QLabel("Selected Channels:")
        self.lbl_channels_summary.setStyleSheet("font-size: 10px; color: #FF6600; font-weight: bold;")
        c_layout.addWidget(self.lbl_channels_summary)

        self.lbl_ch_list = QLabel("")
        self.lbl_ch_list.setStyleSheet("font-size: 9px; color: #CCCCCC; background-color: #222222; padding: 6px; border-radius: 4px;")
        self.lbl_ch_list.setWordWrap(True)
        c_layout.addWidget(self.lbl_ch_list)

        scroll.setWidget(card)
        layout.addWidget(scroll, stretch=1)

        # Navigation Buttons
        btn_row = QHBoxLayout()
        self.btn_back = QPushButton("← BACK")
        self.btn_back.setStyleSheet("background-color: #383838;")
        self.btn_back.clicked.connect(self.back_pressed.emit)
        btn_row.addWidget(self.btn_back)

        self.btn_proceed = QPushButton("PROCEED TO TEST (30S) →")
        self.btn_proceed.setStyleSheet(
            "QPushButton { background-color: #FF6600; font-size: 12px; font-weight: bold; min-height: 40px; }"
            "QPushButton:pressed { background-color: #CC5200; }"
        )
        self.btn_proceed.clicked.connect(self.start_test_pressed.emit)
        btn_row.addWidget(self.btn_proceed)
        layout.addLayout(btn_row)

    def populate(self, board_name: str, serial: str, mfg_date: str, channels: list[str]) -> None:
        self._board_name = board_name
        self._serial = serial
        self._mfg_date = mfg_date
        self._channels = channels

        mode_str = "MOCK DEMO TESTBENCH (Laptop Mode)" if config.MOCK_MODE else "PRODUCTION (Raspberry Pi 4)"
        meta_text = (
            f"<b>Device Under Test:</b> {board_name}<br/>"
            f"<b>Serial Number:</b> {serial} &nbsp;|&nbsp; <b>Mfg Date:</b> {mfg_date}<br/>"
            f"<b>System Mode:</b> <font color='#FF6600'>{mode_str}</font><br/>"
            f"<b>Channels Monitored:</b> {len(channels)} of 43 Electrical Channels"
        )
        self.lbl_meta.setText(meta_text)

        # Check connectivity
        if config.MOCK_MODE:
            hw_text = (
                "• <b>Pico Serial Telemetry:</b> MOCK ENGINE (10 Hz, Injected ESR & Drops)<br/>"
                "• <b>ESP32-CAM Stream:</b> FAIL-SAFE MODE ('NO WIFI STREAM')<br/>"
                "• <b>MLX90640 Thermal:</b> FAIL-SAFE MODE ('NA')<br/>"
                "• <b>Ngrok AI Vision:</b> FAIL-SAFE MODE ('NO MODELS ARE WORKING')"
            )
        else:
            pico_port = HardwareManager.detect_pico_serial_port() or config.PICO_SERIAL_PORT
            hw_text = (
                f"• <b>Pico Serial Port:</b> {pico_port} (115200 baud)<br/>"
                f"• <b>ESP32-CAM Optical:</b> {config.ESP32_STREAM_URL}<br/>"
                f"• <b>MLX90640 IR Array:</b> I2C-1 (/dev/i2c-1, 400 kHz)<br/>"
                f"• <b>Ngrok AI Vision API:</b> {config.AI_API_URL}"
            )
        self.lbl_hw_status.setText(hw_text)

        # Channel badges preview
        ch_preview = ", ".join(channels) if channels else "None"
        self.lbl_ch_list.setText(ch_preview)


# =========================================================================== #
#  Background Execution Worker Thread                                         #
# =========================================================================== #

class ExecutionWorker(QThread):
    """Executes the 30-second data acquisition, diagnostics, and prognostics pipeline."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished_with_results = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        board_name: str,
        serial: str,
        mfg_date: str,
        channels: list[str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.board_name = board_name
        self.serial = serial
        self.mfg_date = mfg_date
        self.channels = channels

    def run(self) -> None:  # noqa: C901
        try:
            self.status.emit("Initialising SQLite databases...")
            self.progress.emit(0)
            db = DatabaseManager(data_dir=config.DATA_DIR)
            hw = HardwareManager(mock_mode=config.MOCK_MODE)

            yolo_defects: list[str] = []
            thermal_status = "NA"
            optical_status = "NO WIFI STREAM"
            ai_status = "NO MODELS ARE WORKING"
            thermal_img_path = None
            overlay_img_path = None
            optical_img_path = None
            comp_img_path = None
            defect_img_path = None
            th_matrix: Optional[np.ndarray] = None

            # ----------------------------------------------------------
            # Step 1: Optical & Thermal Acquisition
            # ----------------------------------------------------------
            if config.MOCK_MODE:
                self.status.emit("MOCK_MODE active: Logging visual fail-safes...")
                db.log_thermal(self.board_name, self.serial, status="NA", data="NA")
                db.log_optical(self.board_name, self.serial, status="NO WIFI STREAM", data="NO WIFI STREAM")
                db.log_ai_process(self.board_name, self.serial, status="NO MODELS ARE WORKING", data="NO MODELS ARE WORKING")
                th_matrix = DataAcquisitionEngine.generate_mock_thermal_matrix()
            else:
                self.status.emit("Acquiring Optical & Thermal Streams from hardware...")
                # 1. ESP32-CAM frame
                opt_frame, res_str, opt_status = hw.capture_optical_frame()
                optical_status = opt_status
                if opt_frame is not None:
                    optical_img_path = os.path.join(config.CAPTURES_DIR, f"opt_{self.serial}.jpg")
                    cv2.imwrite(optical_img_path, opt_frame)
                    db.log_optical(self.board_name, self.serial, optical_image_path=optical_img_path, resolution=res_str, status="OK", data="OK")
                else:
                    db.log_optical(self.board_name, self.serial, status=opt_status, data=opt_status)

                # 2. MLX90640 I2C reading
                th_matrix, th_status = hw.read_thermal_sensor()
                thermal_status = th_status
                if th_matrix is not None and opt_frame is not None:
                    fusion = hw.overlay_engine.fuse(th_matrix, opt_frame)
                    thermal_img_path = os.path.join(config.CAPTURES_DIR, f"thermal_{self.serial}.jpg")
                    overlay_img_path = os.path.join(config.CAPTURES_DIR, f"overlay_{self.serial}.jpg")
                    cv2.imwrite(thermal_img_path, fusion["thermal_annotated"])
                    cv2.imwrite(overlay_img_path, fusion["blended_overlay"])

                    db.log_thermal(
                        self.board_name,
                        self.serial,
                        thermal_image_path=thermal_img_path,
                        overlay_image_path=overlay_img_path,
                        top3_hotspot_temps=fusion["top3_temps"],
                        max_temp=fusion["max_temp"],
                        min_temp=fusion["min_temp"],
                        x_offset=fusion["x_offset"],
                        y_offset=fusion["y_offset"],
                        status="OK",
                        data="OK",
                    )
                else:
                    db.log_thermal(self.board_name, self.serial, status=th_status, data=th_status)

                # 3. AI Vision API (SAHI + YOLO)
                self.status.emit("Posting Optical frame to Ngrok AI Vision Engine...")
                comp_img_path, defect_img_path, yolo_defects, ai_status = hw.call_ai_vision_api(opt_frame)
                db.log_ai_process(
                    self.board_name,
                    self.serial,
                    component_image_path=comp_img_path,
                    defect_image_path=defect_img_path,
                    defect_list=yolo_defects,
                    status=ai_status,
                    data=ai_status,
                )

            self.progress.emit(15)

            # ----------------------------------------------------------
            # Step 2: 43-Channel Serial Telemetry Acquisition (30s)
            # ----------------------------------------------------------
            self.status.emit("Executing 30s Telemetry Acquisition Window...")
            if config.MOCK_MODE:
                engine = DataAcquisitionEngine(num_samples=config.NUM_MOCK_SAMPLES, sample_rate_hz=config.TELEMETRY_SAMPLE_RATE_HZ)
                samples = engine.generate_all_data()
                total = len(samples)
                for idx, s in enumerate(samples):
                    db.log_acquisition(self.board_name, self.serial, s)
                    pct = 15 + int(45 * (idx + 1) / total)
                    self.progress.emit(pct)
            else:
                def on_stream_progress(pct: int, msg: str):
                    self.progress.emit(15 + int(pct * 0.45))
                    self.status.emit(msg)

                samples = hw.acquire_live_telemetry_stream(
                    board_name=self.board_name,
                    serial_number=self.serial,
                    db_logger_fn=db.log_acquisition,
                    duration_sec=config.EXECUTION_DURATION_SEC,
                    sample_rate_hz=config.TELEMETRY_SAMPLE_RATE_HZ,
                    progress_callback=on_stream_progress,
                )

            self.progress.emit(65)

            # ----------------------------------------------------------
            # Step 3: Run Diagnostic Engine (7 Component Failure Rules)
            # ----------------------------------------------------------
            self.status.emit("Running Component-Level Failure Diagnostic Rules...")
            diag = DiagnosticEngine()
            diag_results = diag.run_full_diagnostics(
                data=samples,
                thermal_matrix=th_matrix,
                sample_rate=float(config.TELEMETRY_SAMPLE_RATE_HZ),
                yolo_defects=yolo_defects,
            )
            self.progress.emit(80)

            # ----------------------------------------------------------
            # Step 4: Run Prognostic Engine (Coffin-Manson RUL)
            # ----------------------------------------------------------
            self.status.emit("Calculating Prognostic Life Expectancy (Coffin-Manson)...")
            prog = PrognosticEngine()
            temp_data = [s.get("TEMP_AMB", s.get("TMP117_TEMP", 25.0)) for s in samples]

            rul_result = prog.compute_rul(
                temp_data=temp_data,
                mfg_date=self.mfg_date,
                diagnostic_results=diag_results,
            )
            self.progress.emit(90)

            # ----------------------------------------------------------
            # Step 5: Save Processed Results to SQLite
            # ----------------------------------------------------------
            self.status.emit("Logging Examination Results to SQLite...")
            db.log_results(
                board_name=self.board_name,
                serial_number=self.serial,
                mfg_date=self.mfg_date,
                rul_percent=rul_result.get("rul_percent", 0.0),
                faults=diag_results.get("faults", []),
                mahalanobis_scores=diag_results.get("mahalanobis_scores", {}),
                fft_scores=diag_results.get("fft_scores", {}),
                threshold_flags=diag_results.get("threshold_flags", {}),
            )
            self.progress.emit(98)

            results = {
                "board_name": self.board_name,
                "serial_number": self.serial,
                "mfg_date": self.mfg_date,
                "channels": self.channels,
                "samples": samples,
                "faults": diag_results.get("faults", []),
                "categorized_failures": diag_results.get("categorized_failures", {}),
                "mahalanobis_scores": diag_results.get("mahalanobis_scores", {}),
                "fft_scores": diag_results.get("fft_scores", {}),
                "threshold_flags": diag_results.get("threshold_flags", {}),
                "has_electrical_anomaly": diag_results.get("has_electrical_anomaly", False),
                "rul_percent": rul_result.get("rul_percent", 0.0),
                "rul_details": rul_result,
                "thermal_status": thermal_status,
                "optical_status": optical_status,
                "ai_status": ai_status,
                "thermal_img_path": thermal_img_path,
                "overlay_img_path": overlay_img_path,
                "optical_img_path": optical_img_path,
                "comp_img_path": comp_img_path,
                "defect_img_path": defect_img_path,
            }

            self.progress.emit(100)
            self.status.emit("Diagnostic Examination Complete!")
            self.finished_with_results.emit(results)

        except Exception as exc:
            self.error_occurred.emit(f"Execution failed: {exc}\n{traceback.format_exc()}")


# =========================================================================== #
#  Screen 4 — Execution Screen (30-Second Countdown & Worker)                  #
# =========================================================================== #

class ExecutionScreen(QWidget):
    """Screen 4: 30-Second countdown timer with locked UI during high-speed acquisition."""

    execution_complete = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._countdown_seconds = config.EXECUTION_DURATION_SEC
        self._worker_done = False
        self._timer_done = False
        self._results: Optional[dict] = None
        self._worker: Optional[ExecutionWorker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(8)

        layout.addStretch()

        title = QLabel("DIAGNOSTIC TEST IN PROGRESS")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setStyleSheet("color: #AAAAAA;")
        layout.addWidget(title)

        # Massive countdown timer label
        self.timer_label = QLabel("30")
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setFont(QFont("Segoe UI", 60, QFont.Bold))
        self.timer_label.setStyleSheet("color: #FF6600; font-weight: bold;")
        layout.addWidget(self.timer_label)

        self.status_label = QLabel("Initialising Telemetry Stream...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Segoe UI", 9))
        self.status_label.setStyleSheet("color: #FFFFFF;")
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(18)
        layout.addWidget(self.progress_bar)

        lock_notice = QLabel("🔒 Testbench Locked: 43 Channels Logging to SQLite")
        lock_notice.setAlignment(Qt.AlignCenter)
        lock_notice.setFont(QFont("Segoe UI", 8))
        lock_notice.setStyleSheet("color: #888888;")
        layout.addWidget(lock_notice)

        layout.addStretch()

        self._qtimer = QTimer(self)
        self._qtimer.setInterval(1000)
        self._qtimer.timeout.connect(self._tick)

    def start_execution(
        self,
        board_name: str,
        serial: str,
        mfg_date: str,
        channels: list[str],
    ) -> None:
        self._countdown_seconds = config.EXECUTION_DURATION_SEC
        self._worker_done = False
        self._timer_done = False
        self._results = None
        self.timer_label.setText(str(self._countdown_seconds))
        self.progress_bar.setValue(0)
        self.status_label.setText("Acquiring & Processing Data...")

        self._qtimer.start()

        self._worker = ExecutionWorker(board_name, serial, mfg_date, channels)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished_with_results.connect(self._on_worker_done)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    @pyqtSlot()
    def _tick(self) -> None:
        self._countdown_seconds -= 1
        self.timer_label.setText(str(max(self._countdown_seconds, 0)))
        if self._countdown_seconds <= 0:
            self._qtimer.stop()
            self._timer_done = True
            self._try_complete()

    @pyqtSlot(int)
    def _on_progress(self, pct: int) -> None:
        self.progress_bar.setValue(pct)

    @pyqtSlot(str)
    def _on_status(self, text: str) -> None:
        self.status_label.setText(text)

    @pyqtSlot(dict)
    def _on_worker_done(self, results: dict) -> None:
        self._results = results
        self._worker_done = True
        self._try_complete()

    @pyqtSlot(str)
    def _on_error(self, message: str) -> None:
        self._qtimer.stop()
        self.status_label.setText("⚠ Acquisition Error — check console")
        self.timer_label.setText("ERR")
        self.timer_label.setStyleSheet("color: #FF3333;")
        print(f"[ExecutionWorker ERROR] {message}", file=sys.stderr)

    def _try_complete(self) -> None:
        if self._timer_done and self._worker_done and self._results is not None:
            self.execution_complete.emit(self._results)


# =========================================================================== #
#  Screen 5 — Dashboard (Scrollable Detailed Clinical Findings & RUL)          #
# =========================================================================== #

class DashboardScreen(QWidget):
    """Screen 5: Scrollable results dashboard with detailed component failure findings."""

    report_pressed = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._results: dict = {}
        self._plot_path: Optional[str] = None
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        self._layout = QVBoxLayout(content)
        self._layout.setContentsMargins(8, 6, 8, 6)
        self._layout.setSpacing(6)

        # Section 1: Visual Systems Status & Gallery
        sec1_title = QLabel("1. Visual & Computer Vision Systems")
        sec1_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        sec1_title.setStyleSheet("color: #FF6600;")
        self._layout.addWidget(sec1_title)

        status_box = QVBoxLayout()
        self.lbl_thermal = QLabel("🔴 Thermal Camera: NA")
        self.lbl_thermal.setStyleSheet("font-size: 11px; color: #FF4444;")
        status_box.addWidget(self.lbl_thermal)

        self.lbl_optical = QLabel("🔴 Optical Stream: NO WIFI STREAM")
        self.lbl_optical.setStyleSheet("font-size: 11px; color: #FF4444;")
        status_box.addWidget(self.lbl_optical)

        self.lbl_ai = QLabel("🔴 AI Processing: NO MODELS ARE WORKING")
        self.lbl_ai.setStyleSheet("font-size: 11px; color: #FF4444;")
        status_box.addWidget(self.lbl_ai)
        self._layout.addLayout(status_box)

        self.visual_gallery = QGridLayout()
        self._layout.addLayout(self.visual_gallery)

        # Section 2: Waveforms (Matplotlib Canvas)
        sec2_title = QLabel("2. Dynamic Waveform Telemetry")
        sec2_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        sec2_title.setStyleSheet("color: #FF6600;")
        self._layout.addWidget(sec2_title)

        self.figure = Figure(figsize=(4.4, 3.2), dpi=100, facecolor="#1A1A1A")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(240)
        self._layout.addWidget(self.canvas)

        # Section 3: Component-Level Detected Faults
        sec3_title = QLabel("3. Component Failure Detection Flags")
        sec3_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        sec3_title.setStyleSheet("color: #FF6600;")
        self._layout.addWidget(sec3_title)

        self.faults_container = QVBoxLayout()
        self._layout.addLayout(self.faults_container)

        # Section 4: Remaining Useful Life (RUL)
        sec4_title = QLabel("4. Prognostic Health Assessment (RUL)")
        sec4_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        sec4_title.setStyleSheet("color: #FF6600;")
        self._layout.addWidget(sec4_title)

        self.lbl_rul = QLabel("—")
        self.lbl_rul.setAlignment(Qt.AlignCenter)
        self.lbl_rul.setFont(QFont("Segoe UI", 44, QFont.Bold))
        self.lbl_rul.setStyleSheet("color: #FF6600;")
        self._layout.addWidget(self.lbl_rul)

        self.lbl_rul_details = QLabel("")
        self.lbl_rul_details.setAlignment(Qt.AlignCenter)
        self.lbl_rul_details.setWordWrap(True)
        self.lbl_rul_details.setStyleSheet("color: #AAAAAA; font-size: 10px; background-color: #222222; padding: 6px; border-radius: 4px;")
        self._layout.addWidget(self.lbl_rul_details)

        # Button
        self.btn_report = QPushButton("GENERATE DIAGNOSTIC CERTIFICATE →")
        self.btn_report.setStyleSheet(
            "QPushButton { background-color: #FF6600; font-size: 12px; font-weight: bold; min-height: 40px; }"
            "QPushButton:pressed { background-color: #CC5200; }"
        )
        self.btn_report.clicked.connect(lambda: self.report_pressed.emit(self._results))
        self._layout.addWidget(self.btn_report)

        self._layout.addSpacerItem(QSpacerItem(1, 15, QSizePolicy.Minimum, QSizePolicy.Expanding))

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def display_results(self, results: dict) -> None:
        self._results = results

        # Visual status
        th_stat = results.get("thermal_status", "NA")
        opt_stat = results.get("optical_status", "NO WIFI STREAM")
        ai_stat = results.get("ai_status", "NO MODELS ARE WORKING")

        self.lbl_thermal.setText(f"{'🟢' if th_stat == 'OK' else '🔴'} Thermal Camera: {th_stat}")
        self.lbl_optical.setText(f"{'🟢' if opt_stat == 'OK' else '🔴'} Optical Stream: {opt_stat}")
        self.lbl_ai.setText(f"{'🟢' if ai_stat == 'OK' else '🔴'} AI Processing: {ai_stat}")

        # Clear old gallery
        while self.visual_gallery.count():
            item = self.visual_gallery.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        imgs = [
            ("Thermal Overlay", results.get("overlay_img_path")),
            ("Optical Camera", results.get("optical_img_path")),
            ("AI SAHI Components", results.get("comp_img_path")),
            ("AI YOLO Defects", results.get("defect_img_path")),
        ]
        col_idx = 0
        for title_str, path in imgs:
            if path and os.path.isfile(path):
                box = QVBoxLayout()
                t_lbl = QLabel(title_str)
                t_lbl.setStyleSheet("font-size: 9px; color: #FF6600; font-weight: bold;")
                img_lbl = QLabel()
                pix = QPixmap(path).scaled(200, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                img_lbl.setPixmap(pix)
                box.addWidget(t_lbl)
                box.addWidget(img_lbl)
                w = QWidget()
                w.setLayout(box)
                self.visual_gallery.addWidget(w, col_idx // 2, col_idx % 2)
                col_idx += 1

        # Waveforms
        self._draw_plots(results)

        # Faults
        while self.faults_container.count():
            item = self.faults_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        faults = results.get("faults", [])
        if not faults:
            lbl = QLabel("✅ All component failure detection rules passed. No anomalies detected.")
            lbl.setStyleSheet("color: #44FF44; font-size: 11px; background-color: #222222; padding: 6px; border-radius: 4px;")
            self.faults_container.addWidget(lbl)
        else:
            for fault_str in faults:
                lbl = QLabel(f"⚠ {fault_str}")
                lbl.setWordWrap(True)
                lbl.setStyleSheet("color: #FF7722; font-size: 10px; background-color: #262220; padding: 4px; border-radius: 3px;")
                self.faults_container.addWidget(lbl)

        # RUL Score
        rul = float(results.get("rul_percent", 0.0))
        self.lbl_rul.setText(f"{rul:.1f}%")
        if rul > 60:
            self.lbl_rul.setStyleSheet("color: #44FF44; font-size: 44px; font-weight: bold;")
        elif rul > 30:
            self.lbl_rul.setStyleSheet("color: #FF6600; font-size: 44px; font-weight: bold;")
        else:
            self.lbl_rul.setStyleSheet("color: #FF3333; font-size: 44px; font-weight: bold;")

        rul_details = results.get("rul_details", {})
        delta_t = rul_details.get("delta_t", 0.0)
        n_f = rul_details.get("n_f", rul_details.get("n_f_cycles", 0.0))
        damage = rul_details.get("miner_damage", rul_details.get("damage_fraction", 0.0))
        penalty = rul_details.get("penalty_applied", False)

        detail_lines = [
            f"Coffin-Manson Thermal Fatigue: ΔT = {delta_t:.1f}°C  |  N_f = {n_f:.0f} cycles",
            f"Palmgren-Miner Cumulative Damage: D = {damage:.4f}",
            f"Hybrid Electrical Stress Penalty: {'ACTIVE (0.85× MULTIPLIER)' if penalty else 'NONE'}",
        ]
        self.lbl_rul_details.setText("\n".join(detail_lines))

    def _draw_plots(self, results: dict) -> None:
        self.figure.clear()
        samples = results.get("samples", [])
        if not samples:
            return

        n = len(samples)
        t = np.arange(n) / float(config.TELEMETRY_SAMPLE_RATE_HZ)

        ina1_vbus = [s.get("INA1_VBUS", 3.3) for s in samples]
        mux_ch07 = [s.get("MUX_CH07", 3.3) for s in samples]
        temp_amb = [s.get("TMP117_TEMP", s.get("TEMP_AMB", 25.0)) for s in samples]

        label_style = {"color": "#FFFFFF", "fontsize": 7}
        tick_params = {"colors": "#AAAAAA", "labelsize": 6}

        # Subplot 1: INA1_VBUS AC Ripple
        ax1 = self.figure.add_subplot(3, 1, 1)
        ax1.plot(t, ina1_vbus, color="#FF6600", linewidth=1.0)
        ax1.set_title("INA1_VBUS (V) — High-Frequency ESR Ripple", **label_style)
        ax1.set_facecolor("#1A1A1A")
        ax1.tick_params(**tick_params)
        ax1.grid(True, alpha=0.2, color="#666666")

        # Subplot 2: MUX_CH07 Drop
        ax2 = self.figure.add_subplot(3, 1, 2)
        ax2.plot(t, mux_ch07, color="#00AAFF", linewidth=1.0)
        ax2.set_title("MUX_CH07 (V) — Dynamic Step Voltage Drop", **label_style)
        ax2.set_facecolor("#1A1A1A")
        ax2.tick_params(**tick_params)
        ax2.grid(True, alpha=0.2, color="#666666")

        # Subplot 3: Ambient Temp
        ax3 = self.figure.add_subplot(3, 1, 3)
        ax3.plot(t, temp_amb, color="#FF3333", linewidth=1.0)
        ax3.set_title("TMP117 Ambient Temperature (°C) — Overheat Climb", **label_style)
        ax3.set_xlabel("Time (s)", **label_style)
        ax3.set_facecolor("#1A1A1A")
        ax3.tick_params(**tick_params)
        ax3.grid(True, alpha=0.2, color="#666666")

        self.figure.tight_layout(pad=0.8)
        self.canvas.draw()

        try:
            self._plot_path = os.path.join(tempfile.gettempdir(), "specs_electrical_plot.png")
            self.figure.savefig(self._plot_path, dpi=140, facecolor="#1A1A1A", bbox_inches="tight")
            self._results["plot_image_path"] = self._plot_path
        except Exception as exc:
            print(f"[Dashboard] Plot save error: {exc}", file=sys.stderr)


# =========================================================================== #
#  Screen 6 — Certification Screen (Diagnostic Center PDF & QR Download)       #
# =========================================================================== #

class CertificationScreen(QWidget):
    """Screen 6: Official Diagnostic Center Examination Certificate & QR Download."""

    done_pressed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        title = QLabel("Diagnostic Examination Certificate")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setStyleSheet("color: #FF6600;")
        layout.addWidget(title)

        self.status_label = QLabel("Compiling ISO/IEC 17025 Examination Report...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #AAAAAA; font-size: 11px;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.qr_label = QLabel()
        self.qr_label.setAlignment(Qt.AlignCenter)
        self.qr_label.setMinimumSize(140, 140)
        layout.addWidget(self.qr_label)

        self.instructions = QLabel("Scan QR code with your phone to download the official PDF certificate")
        self.instructions.setAlignment(Qt.AlignCenter)
        self.instructions.setWordWrap(True)
        self.instructions.setStyleSheet("color: #FFFFFF; font-size: 10px;")
        layout.addWidget(self.instructions)

        self.url_label = QLabel("")
        self.url_label.setAlignment(Qt.AlignCenter)
        self.url_label.setWordWrap(True)
        self.url_label.setStyleSheet("color: #FF6600; font-size: 9px;")
        self.url_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.url_label)

        layout.addStretch()

        self.btn_done = QPushButton("DONE (SAVE CERTIFICATE & RESET) ✓")
        self.btn_done.setStyleSheet(
            "QPushButton { background-color: #FF6600; font-size: 12px; font-weight: bold; min-height: 38px; }"
            "QPushButton:pressed { background-color: #CC5200; }"
        )
        self.btn_done.clicked.connect(self.done_pressed.emit)
        layout.addWidget(self.btn_done)

    def generate_and_display(self, results: dict) -> None:
        self.status_label.setText("Generating Diagnostic Center Certificate PDF...")
        QApplication.processEvents()

        try:
            rg = ReportGenerator()
            pdf_path = rg.generate_pdf(results)
            self.status_label.setText(f"Certificate Saved: {os.path.basename(pdf_path)}")
            QApplication.processEvents()

            try:
                url = rg.start_file_server(pdf_path, port=config.HTTP_SERVER_PORT)
            except Exception:
                url = f"file:///{pdf_path.replace(os.sep, '/')}"

            self.url_label.setText(url)
            self._show_qr(url)
            self.status_label.setText("Official Certificate Ready for Mobile Download!")

        except Exception as exc:
            self.status_label.setText(f"⚠ Report Generation Error: {exc}")
            print(f"[Certification ERROR] {exc}", file=sys.stderr)
            traceback.print_exc()

    def _show_qr(self, data: str) -> None:
        try:
            import qrcode
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=4,
                border=2,
            )
            qr.add_data(data)
            qr.make(fit=True)

            img = qr.make_image(fill_color="white", back_color="#1A1A1A").convert("RGB")
            img_data = img.tobytes("raw", "RGB")
            w, h = img.size
            qimage = QImage(img_data, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.qr_label.setPixmap(pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:
            self.qr_label.setText("[QR Code Ready]")

    def reset(self) -> None:
        self.qr_label.clear()
        self.url_label.clear()
        self.status_label.setText("Preparing certificate...")


# =========================================================================== #
#  Main Window (SPECSApp) with 7 Screens                                       #
# =========================================================================== #

class SPECSApp(QMainWindow):
    """Main Application Window for S.P.E.C.S. PCB Diagnostic Testbench."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("S.P.E.C.S. - PCB Diagnostic Testbench")
        self.setFixedSize(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.setStyleSheet(STYLESHEET)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self._board_name: str = ""
        self._mfg_date: str = ""
        self._serial: str = ""
        self._channels: list[str] = []

        # 7 Screens in sequence:
        self.screen_board = BoardNameScreen()              # Index 0
        self.screen_mfg = MfgInfoScreen()                  # Index 1
        self.screen_channels = ChannelSelectScreen()        # Index 2
        self.screen_summary = PreTestSummaryScreen()        # Index 3 (NEW Pre-Test Inspection)
        self.screen_exec = ExecutionScreen()                # Index 4
        self.screen_dashboard = DashboardScreen()           # Index 5
        self.screen_cert = CertificationScreen()            # Index 6

        self.stack.addWidget(self.screen_board)     # 0
        self.stack.addWidget(self.screen_mfg)       # 1
        self.stack.addWidget(self.screen_channels)  # 2
        self.stack.addWidget(self.screen_summary)   # 3
        self.stack.addWidget(self.screen_exec)      # 4
        self.stack.addWidget(self.screen_dashboard) # 5
        self.stack.addWidget(self.screen_cert)      # 6

        self._connect_signals()
        self.navigate_to(0)

    def _connect_signals(self) -> None:
        # Screen 0 → Screen 1
        self.screen_board.next_pressed.connect(self._on_board_next)

        # Screen 1 → Screen 0 (back), Screen 2 (next)
        self.screen_mfg.back_pressed.connect(lambda: self.navigate_to(0))
        self.screen_mfg.next_pressed.connect(self._on_mfg_next)

        # Screen 2 → Screen 1 (back), Screen 3 (proceed to pre-test summary)
        self.screen_channels.back_pressed.connect(lambda: self.navigate_to(1))
        self.screen_channels.proceed_pressed.connect(self._on_channels_proceed)

        # Screen 3 (Pre-Test Summary) → Screen 2 (back), Screen 4 (start test!)
        self.screen_summary.back_pressed.connect(lambda: self.navigate_to(2))
        self.screen_summary.start_test_pressed.connect(self._on_start_test_confirmed)

        # Screen 4 → Screen 5 (execution complete)
        self.screen_exec.execution_complete.connect(self._on_execution_complete)

        # Screen 5 → Screen 6 (generate certificate)
        self.screen_dashboard.report_pressed.connect(self._on_report_pressed)

        # Screen 6 → Screen 0 (done / reset)
        self.screen_cert.done_pressed.connect(self.reset_all)

    def navigate_to(self, screen_index: int) -> None:
        if 0 <= screen_index < self.stack.count():
            self.stack.setCurrentIndex(screen_index)

    @pyqtSlot(str)
    def _on_board_next(self, board_name: str) -> None:
        self._board_name = board_name
        self.navigate_to(1)

    @pyqtSlot(str, str)
    def _on_mfg_next(self, mfg_date: str, serial: str) -> None:
        self._mfg_date = mfg_date
        self._serial = serial
        self.navigate_to(2)

    @pyqtSlot(list)
    def _on_channels_proceed(self, channels: list) -> None:
        """User selected channels. Populate pre-test summary screen and show it."""
        self._channels = channels
        self.screen_summary.populate(
            board_name=self._board_name,
            serial=self._serial,
            mfg_date=self._mfg_date,
            channels=self._channels,
        )
        self.navigate_to(3)

    @pyqtSlot()
    def _on_start_test_confirmed(self) -> None:
        """User reviewed summary and tapped 'PROCEED TO TEST' — now start countdown & worker."""
        self.navigate_to(4)
        self.screen_exec.start_execution(
            self._board_name,
            self._serial,
            self._mfg_date,
            self._channels,
        )

    @pyqtSlot(dict)
    def _on_execution_complete(self, results: dict) -> None:
        self.screen_dashboard.display_results(results)
        self.navigate_to(5)

    @pyqtSlot(dict)
    def _on_report_pressed(self, results: dict) -> None:
        self.navigate_to(6)
        self.screen_cert.generate_and_display(results)

    def reset_all(self) -> None:
        self._board_name = ""
        self._mfg_date = ""
        self._serial = ""
        self._channels = []

        self.screen_board.reset()
        self.screen_mfg.reset()
        self.screen_channels.reset()
        self.screen_cert.reset()

        self.navigate_to(0)
