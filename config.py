"""
config.py
=========
TEST GIG - Edge-Computing PCB Diagnostic & Prognostic Testbench.
Centralized Configuration and Environment Settings.

All operational parameters, GUI themes, database file paths, hardware endpoints,
diagnostic failure thresholds, prognostic models, and channel definitions are
controlled centrally from this file.
"""

from __future__ import annotations

import os

# =========================================================================== #
# 1. CRITICAL ARCHITECTURE CONSTRAINT: MOCK_MODE                              #
# =========================================================================== #
# When True (Laptop Testing):
#   - Bypass physical hardware reads (I2C MLX90640, ESP32-CAM stream, Pico serial).
#   - Force visual databases to store fail-safe strings:
#       * thermal_log.db       -> "NA"
#       * optical_log.db       -> "NO WIFI STREAM"
#       * ai_process_log.db    -> "NO MODELS ARE WORKING"
#   - Generate realistic mock NumPy telemetry for the 43 electrical channels.
#
# When False (Production Raspberry Pi 4):
#   - Read 32x24 MLX90640 IR thermal array via I2C (/dev/i2c-1).
#   - Capture optical stream from ESP32-CAM over Wi-Fi (port 81).
#   - Call Ngrok AI vision API (SAHI component + YOLO defect detection).
#   - Read 43-channel telemetry from Raspberry Pi Pico via USB serial (/dev/ttyACM0).
# =========================================================================== #
MOCK_MODE: bool = True

# =========================================================================== #
# 2. BRANDING & SYSTEM IDENTITY                                               #
# =========================================================================== #
APP_NAME: str = "TEST GIG"
APP_TITLE: str = "TEST GIG - PCB Diagnostic Testbench"
APP_SUBTITLE: str = "TEST GIG - Edge Diagnostic & Prognostic Testbench"
LAB_NAME: str = "TEST GIG ADVANCED DIAGNOSTIC & PROGNOSTIC LABORATORY"
CERTIFICATE_PREFIX: str = "CERT-TESTGIG"
REPORT_FILENAME_PREFIX: str = "TESTGIG_Report"
TESTBENCH_STATION_ID: str = "PI4-TESTGIG-01"
AI_SIGNOFF_NAME: str = "TEST GIG AI Engine"

# =========================================================================== #
# 3. GUI SETTINGS (3.5-INCH TOUCH DISPLAY: 480x320)                           #
# =========================================================================== #
SCREEN_WIDTH: int = 480
SCREEN_HEIGHT: int = 320

# Professional Dark Mode Theme (Orange, Black, White Highlights)
COLOR_PRIMARY: str = "#FF6600"          # Signature Testbench Orange
COLOR_PRIMARY_HOVER: str = "#FF8533"    # Lighter Orange for hover/touch
COLOR_PRIMARY_PRESSED: str = "#CC5200"  # Deeper Orange for active button press
COLOR_BG: str = "#1A1A1A"               # Main Dark Background
COLOR_SURFACE: str = "#262626"          # Cards, Panels, and Scroll Containers
COLOR_SURFACE_ALT: str = "#2D2D2D"      # Secondary Button / Unselected State
COLOR_SURFACE_INPUT: str = "#222222"    # Input Fields & Readouts
COLOR_TEXT_PRIMARY: str = "#FFFFFF"     # Crisp White Text
COLOR_TEXT_MUTED: str = "#AAAAAA"       # Subtitles & Labels
COLOR_TEXT_DIM: str = "#888888"         # Inactive / Footnote Text
COLOR_BORDER: str = "#333333"           # Subtle Divider / Card Borders
COLOR_BORDER_FOCUS: str = "#FF6600"     # Focused Input Border
COLOR_DANGER: str = "#FF4444"           # Red Error / Fault Indicators
COLOR_DANGER_BRIGHT: str = "#FF1744"    # Alert Red
COLOR_WARNING: str = "#FFB300"          # Warning Amber
COLOR_SUCCESS: str = "#00E676"          # Certified Green
COLOR_BLACK: str = "#000000"            # Pure Black

# =========================================================================== #
# 4. DIRECTORY PATHS & 5 SQLITE DATABASE STORES                              #
# =========================================================================== #
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
REPORTS_DIR: str = os.path.join(BASE_DIR, "reports")
CAPTURES_DIR: str = os.path.join(BASE_DIR, "captures")

# Ensure required storage directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CAPTURES_DIR, exist_ok=True)

# 5 Separate SQLite Database Filenames (Prevent Database Locking & Contention)
THERMAL_DB_NAME: str = "thermal_log.db"
OPTICAL_DB_NAME: str = "optical_log.db"
AI_PROCESS_DB_NAME: str = "ai_process_log.db"
DATA_ACQ_DB_NAME: str = "data_acq_log.db"
RESULTS_DB_NAME: str = "data_process_results_log.db"

# Absolute File Paths for Databases
THERMAL_DB_PATH: str = os.path.join(DATA_DIR, THERMAL_DB_NAME)
OPTICAL_DB_PATH: str = os.path.join(DATA_DIR, OPTICAL_DB_NAME)
AI_PROCESS_DB_PATH: str = os.path.join(DATA_DIR, AI_PROCESS_DB_NAME)
DATA_ACQ_DB_PATH: str = os.path.join(DATA_DIR, DATA_ACQ_DB_NAME)
RESULTS_DB_PATH: str = os.path.join(DATA_DIR, RESULTS_DB_NAME)

# =========================================================================== #
# 5. HARDWARE ENDPOINTS & PROTOCOLS (Used when MOCK_MODE = False)             #
# =========================================================================== #
# Raspberry Pi Pico USB Serial Configuration
PICO_SERIAL_PORT: str = "/dev/ttyACM0"
PICO_BAUDRATE: int = 115200
PICO_TIMEOUT_SEC: float = 0.5
PICO_SERIAL_CANDIDATES: list[str] = [
    "/dev/ttyACM0",
    "/dev/ttyUSB0",
    "/dev/ttyACM1",
    "/dev/ttyUSB1",
    "COM3",
    "COM4",
    "COM5",
]

# ESP32-CAM Wi-Fi Optical Stream Endpoint
ESP32_IP: str = "192.168.1.50"
ESP32_PORT: int = 81
ESP32_STREAM_URL: str = f"http://{ESP32_IP}:{ESP32_PORT}/stream"

# MLX90640 32x24 Far-Infrared Thermal Sensor (I2C)
MLX90640_I2C_BUS: int = 1               # /dev/i2c-1 on Raspberry Pi 4
MLX90640_I2C_ADDR: int = 0x33           # Default factory I2C slave address
MLX90640_REFRESH_RATE_HZ: int = 4

# Ngrok Remote AI Vision Inference API
AI_API_URL: str = "https://chive-refining-disloyal.ngrok-free.dev/detect"
AI_API_KEY: str = "pcb_mal_123"
AI_API_TIMEOUT_SEC: float = 12.0

# =========================================================================== #
# 6. FAIL-SAFE STRINGS (Mock Mode & Sensor Fallbacks)                         #
# =========================================================================== #
FAILSAFE_THERMAL: str = "NA"
FAILSAFE_OPTICAL: str = "NO WIFI STREAM"
FAILSAFE_AI: str = "NO MODELS ARE WORKING"

# =========================================================================== #
# 7. DIAGNOSTIC MATHEMATICAL FAILURE THRESHOLDS                               #
# =========================================================================== #
# Rule 1: Capacitor Failure (FFT Ripple Analysis on INA Rails)
CAPACITOR_RIPPLE_THRESHOLD_V: float = 0.100     # AC ripple > 100 mV (0.100 V)

# Rule 2: Voltage Regulator / Power Module Stability & Drop
VOLTAGE_DROP_TOLERANCE: float = 0.10            # Rail voltage drop > 10% from nominal
REGULATOR_INSTABILITY_STD: float = 0.045        # Continuous fluctuation std dev > 45 mV

# Rule 3: MOSFET / Transformer Overheating (Thermal Matrix)
MOSFET_MAX_TEMP_C: float = 70.0                 # Absolute max temperature > 70°C
MOSFET_HOTSPOT_DELTA_T: float = 20.0            # Localized hotspot ΔT > 20°C above background

# Rule 4: Mechanical & Connector Integrity (MPU6050 Vibration)
VIBRATION_SPIKE_G: float = 1.6                  # Vibration / acceleration spike > 1.6g

# Rule 5: Environmental & Corrosion Risks (TMP117 & BMP280)
CORROSION_HUMIDITY_THRESHOLD: float = 80.0      # Relative humidity > 80% RH
AMBIENT_OVERHEAT_THRESHOLD_C: float = 50.0      # Ambient chamber temperature > 50°C

# Rule 6: Sensor & Trace Continuity (ADC & MUX Channels)
SENSOR_TRACE_SIGNAL_LOSS_V: float = 0.05        # Sudden signal loss (< 0.05 V)
SENSOR_TRACE_SWING_THRESHOLD: float = 0.40      # Sudden step jump > 40% of baseline

# Rule 7: System-Level Multivariate Anomaly Detection
MAHALANOBIS_THRESHOLD: float = 3.0              # Mahalanobis distance D_M > 3.0

# =========================================================================== #
# 8. PROGNOSTIC RELIABILITY CONSTANTS (Physics-of-Failure RUL)                 #
# =========================================================================== #
# Coffin-Manson Thermal Fatigue Model: N_f = C * (ΔT)^(-n)
COFFIN_MANSON_C: float = 1.0e7                  # Material ductility constant C = 1.0e7
COFFIN_MANSON_N: float = 2.5                    # Fatigue exponent n = 2.5

# Multiplicative Electrical Health Penalty: RUL_adj = RUL_base * penalty
ELECTRICAL_PENALTY_FACTOR: float = 0.85         # 15% life degradation on electrical anomaly
FALLBACK_CYCLES_APPLIED: int = 30               # Fallback cycle estimate if mfg date invalid

# =========================================================================== #
# 9. 43 ELECTRICAL CHANNELS DEFINITIONS (Rename GUI Buttons Directly Here)    #
# =========================================================================== #
ELECTRICAL_CHANNELS: list[str] = [
    # 9x Power Monitor Channels (3x INA226 Voltage/Current/Power Monitors)
    "INA1_VBUS",
    "INA1_ISHUNT",
    "INA1_POWER",
    "INA2_VBUS",
    "INA2_ISHUNT",
    "INA2_POWER",
    "INA3_VBUS",
    "INA3_ISHUNT",
    "INA3_POWER",

    # 2x Direct Microcontroller ADC Channels
    "ADC_CH0",
    "ADC_CH1",

    # 32x Analog Multiplexer (MUX) Test Points
    "MUX_CH00",
    "MUX_CH01",
    "MUX_CH02",
    "MUX_CH03",
    "MUX_CH04",
    "MUX_CH05",
    "MUX_CH06",
    "MUX_CH07",
    "MUX_CH08",
    "MUX_CH09",
    "MUX_CH10",
    "MUX_CH11",
    "MUX_CH12",
    "MUX_CH13",
    "MUX_CH14",
    "MUX_CH15",
    "MUX_CH16",
    "MUX_CH17",
    "MUX_CH18",
    "MUX_CH19",
    "MUX_CH20",
    "MUX_CH21",
    "MUX_CH22",
    "MUX_CH23",
    "MUX_CH24",
    "MUX_CH25",
    "MUX_CH26",
    "MUX_CH27",
    "MUX_CH28",
    "MUX_CH29",
    "MUX_CH30",
    "MUX_CH31",
]

# =========================================================================== #
# 10. OPENCV THERMAL-OPTICAL FUSION PARAMETERS                                #
# =========================================================================== #
X_OFFSET: int = 0                               # Manual affine parallax X shift (pixels)
Y_OFFSET: int = 0                               # Manual affine parallax Y shift (pixels)
ALPHA_TRANSPARENCY: float = 0.45                # 45% thermal overlay transparency
THERMAL_COLORMAP: str = "COLORMAP_JET"          # OpenCV colormap (JET or INFERNO)

# =========================================================================== #
# 11. TIMING, ACQUISITION & SERVER PARAMETERS                                 #
# =========================================================================== #
EXECUTION_DURATION_SEC: int = 30                # 30-second diagnostic window
TELEMETRY_SAMPLE_RATE_HZ: int = 10              # 10 Hz sampling rate
NUM_MOCK_SAMPLES: int = 300                     # 30 s @ 10 Hz = 300 telemetry packets
HTTP_SERVER_PORT: int = 8000                    # Lightweight mobile certificate download server
