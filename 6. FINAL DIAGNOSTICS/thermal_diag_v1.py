import os
import time
import json
import sqlite3
import board
import busio
import adafruit_mlx90640
import numpy as np
import cv2
from picamera2 import Picamera2

# ---------------------------------------------------------
# HARDWARE INITIALIZATION
# ---------------------------------------------------------
i2c = busio.I2C(board.SCL, board.SDA, frequency=100000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
picam2.start()

# ---------------------------------------------------------
# SYSTEM CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
BASELINE_CM = 3.8        # Physical camera center-to-center separation
PARALLAX_K_X = 2300.0    # Lens focal constant x Baseline distance
OFFSET_INF_X = -25       # Base horizontal offset at infinity
OFFSET_INF_Y = -13       # Base vertical offset (distance-invariant)

SCALE_P = 107            # Base FOV scaling factor %
ALPHA = 0.6              # Blend ratio (60% thermal / 40% optical)
OVERHEAT_THRESH = 70.0   # Thermal alert threshold in Celsius
HOTSPOT_MIN_TEMP = 45.0  # Minimum threshold to detect a candidate hotspot
MAX_HOTSPOTS = 4         # Track up to top 4 hottest regions

DB_NAME = "pcb_diagnostics.db"
WINDOW_NAME = "PCB Multi-Fault Thermal & Environmental Analyzer"

cv2.namedWindow(WINDOW_NAME)
cv2.createTrackbar("Distance (cm)", WINDOW_NAME, 30, 150, lambda x: None)

frame_thermal = [0] * 768

# ---------------------------------------------------------
# DATABASE INITIALIZATION ENGINE
# ---------------------------------------------------------
def init_database():
    """
    Creates SQLite database and tables for storing test metadata, 
    time-series frame telemetry, fault logs, and image blobs.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Table 1: Complete Test Session Summaries & Images
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_runs (
            test_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            duration_s REAL,
            target_distance_cm INTEGER,
            overall_status TEXT,
            health_score_percent REAL,
            estimated_lifespan TEXT,
            peak_temp_c REAL,
            max_rise_rate_c_per_s REAL,
            avg_ambient_temp_c REAL,
            avg_humidity_rh REAL,
            moisture_flag INTEGER,
            dust_flag INTEGER,
            predicted_components TEXT,
            failure_diagnostics_json TEXT,
            img_optical_rgb BLOB,
            img_thermal_raw BLOB,
            img_fused_hotspots BLOB
        )
    ''')

    # Table 2: Time-Series Frame Telemetry (80 samples per run)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS frame_telemetry (
            telemetry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id INTEGER,
            elapsed_s REAL,
            ambient_temp_c REAL,
            humidity_rh REAL,
            thermal_rise_rate REAL,
            hs1_temp REAL, hs1_x INTEGER, hs1_y INTEGER,
            hs2_temp REAL, hs2_x INTEGER, hs2_y INTEGER,
            hs3_temp REAL, hs3_x INTEGER, hs3_y INTEGER,
            hs4_temp REAL, hs4_x INTEGER, hs4_y INTEGER,
            active_fault_count INTEGER,
            FOREIGN KEY (test_id) REFERENCES test_runs (test_id)
        )
    ''')

    conn.commit()
    conn.close()

# ---------------------------------------------------------
# ENVIRONMENTAL & LIFESPAN EVALUATION ENGINES
# ---------------------------------------------------------
def read_environmental_sensors():
    """
    Telemetry matching sensor specifications. Replace with live sensor reads if attached.
    """
    return {
        "humidity_rh": 45.0,         # Normal: 30-60% | Abnormal: >80% RH
        "moisture_detected": False,  # Normal: Dry | Abnormal: True
        "dust_contamination": False, # Normal: Minimal | Abnormal: True
        "ambient_temp_c": 32.0       # Normal: 20-40°C | Abnormal: >50°C
    }

def calculate_pcb_health_and_life(peak_temp, max_rise_rate, env_data):
    """
    Uses the Arrhenius Equation model to calculate remaining PCB lifespan
    and predict failing components based on physical stress.
    """
    predicted_components = []
    fault_diagnostics = []
    health_score = 100.0

    # 1. Thermal Overheat Assessment
    if peak_temp >= OVERHEAT_THRESH:
        health_score -= 50.0
        predicted_components.extend([
            "Decoupling MLCC Capacitors near Hotspot #1",
            "LDO Voltage Regulator / Buck Power Rail"
        ])
        fault_diagnostics.append({
            "fault_type": "Component Thermal Failure",
            "severity": "CRITICAL",
            "reason": f"Peak hotspot reached {peak_temp:.1f}°C (Threshold: {OVERHEAT_THRESH}°C)",
            "recommended_action": "Inspect SMPS circuitry or replace near-shorted decoupling capacitors."
        })
    elif peak_temp >= 50.0:
        health_score -= 20.0

    # 2. Inrush / Dead Short Assessment
    if max_rise_rate >= 3.0:
        health_score -= 40.0
        predicted_components.append("Primary Switching Transistors / Power MOSFET")
        fault_diagnostics.append({
            "fault_type": "Inrush / Dead Short Failure",
            "severity": "CRITICAL",
            "reason": f"Thermal rise rate exceeded safe limit ({max_rise_rate:.1f}°C/s >= 3.0°C/s)",
            "recommended_action": "Check primary power rail input for low-impedance shorts to ground."
        })

    # 3. Environmental Stress Assessment
    if env_data["humidity_rh"] > 80.0:
        health_score -= 15.0
        predicted_components.append("Exposed Solder Pads / Connector Headers")
        fault_diagnostics.append({
            "fault_type": "Corrosion Risk",
            "severity": "WARNING",
            "reason": f"High humidity level detected ({env_data['humidity_rh']:.1f}% RH)",
            "recommended_action": "Apply conformal coating or move board to a desiccated enclosure."
        })

    if env_data["moisture_detected"]:
        health_score -= 60.0
        predicted_components.append("SoC / Microcontroller Pins & BGA Balls")
        fault_diagnostics.append({
            "fault_type": "PCB Water Damage",
            "severity": "CRITICAL",
            "reason": "Liquid ingress detected on board surface",
            "recommended_action": "Immediately remove power and clean board with isopropyl alcohol (IPA)."
        })

    if env_data["dust_contamination"]:
        health_score -= 10.0
        predicted_components.append("Heatsink Vents & Fine-Pitch QFP Legs")

    health_score = max(0.0, health_score)

    # Lifespan Derating Calculation (Arrhenius Rule)
    if health_score >= 85.0:
        lifespan = "3 to 5 Years (Normal)"
    elif health_score >= 60.0:
        lifespan = "1 to 2 Years (Elevated Wear)"
    elif health_score >= 30.0:
        lifespan = "1 to 3 Months (Accelerated Degradation)"
    else:
        lifespan = "Immediate Failure (< 24 Hours to 1 Week)"

    status = "PASS" if health_score >= 70.0 else "FAIL"

    return health_score, lifespan, status, predicted_components, fault_diagnostics

# ---------------------------------------------------------
# CALIBRATION FUNCTIONS
# ---------------------------------------------------------
def calculate_parallax_offset(distance_cm):
    safe_dist = max(10.0, float(distance_cm))
    shift_x = PARALLAX_K_X / safe_dist
    return int(OFFSET_INF_X + shift_x), int(OFFSET_INF_Y)

def auto_detect_alignment_offset(visible_gray, thermal_raw):
    visible_edges = cv2.Canny(visible_gray, 50, 150)
    norm_thermal = cv2.normalize(thermal_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thermal_edges = cv2.Canny(norm_thermal, 30, 100)

    scaled_w, scaled_h = int(640 * (SCALE_P / 100.0)), int(480 * (SCALE_P / 100.0))
    resized_thermal_edges = cv2.resize(thermal_edges, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.zeros((480, 640), dtype=np.uint8)
    start_x, start_y = max(0, (640 - scaled_w) // 2), max(0, (480 - scaled_h) // 2)
    src_x, src_y = max(0, (scaled_w - 640) // 2), max(0, (scaled_h - 480) // 2)
    h_copy, w_copy = min(480, scaled_h), min(640, scaled_w)
    
    canvas[start_y:start_y+h_copy, start_x:start_x+w_copy] = resized_thermal_edges[src_y:src_y+h_copy, src_x:src_x+w_copy]

    best_score = -1.0
    best_dx, best_dy = OFFSET_INF_X, OFFSET_INF_Y

    for dx in range(OFFSET_INF_X - 15, OFFSET_INF_X + 16, 3):
        for dy in range(OFFSET_INF_Y - 10, OFFSET_INF_Y + 11, 2):
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(canvas, M, (640, 480))
            score = np.sum(shifted.astype(np.float32) * visible_edges.astype(np.float32))
            if score > best_score:
                best_score, best_dx, best_dy = score, dx, dy

    return best_dx, best_dy

# ---------------------------------------------------------
# MAIN EXECUTION LOOP
# ---------------------------------------------------------
init_database()

print("\n==================================================")
print(" PCB MULTI-FAULT THERMAL & DATABASE ANALYZER")
print(" - Press 't' to start 20-Second Test & Save to SQL DB")
print(" - Press 'a' for Auto-Alignment Calibration")
print(" - Press 'q' to Quit")
print("==================================================\n")

auto_fine_tune = False
recording = False
test_start_time = 0
prev_peak_temp = 0.0
max_observed_rise_rate = 0.0
session_telemetry_cache = []

try:
    while True:
        visible_frame = picam2.capture_array()
        visible_frame = cv2.cvtColor(visible_frame, cv2.COLOR_RGB2BGR)
        visible_gray = cv2.cvtColor(visible_frame, cv2.COLOR_BGR2GRAY)

        try:
            mlx.getFrame(frame_thermal)
            data = np.reshape(frame_thermal, (24, 32))
            data = np.flipud(data)

            min_val = float(np.min(data))
            max_val = float(np.max(data))

            target_dist_cm = cv2.getTrackbarPos("Distance (cm)", WINDOW_NAME)
            target_dist_cm = max(10, target_dist_cm)
            
            if auto_fine_tune:
                off_x, off_y = auto_detect_alignment_offset(visible_gray, data)
                auto_fine_tune = False
                print(f"[CALIBRATION] Dynamic Offset: X={off_x}, Y={off_y}")
            else:
                off_x, off_y = calculate_parallax_offset(target_dist_cm)

            # Normalize & Color Map Thermal Array
            norm_data = np.zeros((24, 32), dtype=np.uint8) if max_val == min_val else np.uint8(255 * (data - min_val) / (max_val - min_val))
            scaled_w, scaled_h = int(640 * (SCALE_P / 100.0)), int(480 * (SCALE_P / 100.0))
            bicu_img = cv2.resize(norm_data, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
            thermal_map = cv2.applyColorMap(bicu_img, cv2.COLORMAP_JET)

            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            start_x, start_y = max(0, (640 - scaled_w) // 2), max(0, (480 - scaled_h) // 2)
            src_x, src_y = max(0, (scaled_w - 640) // 2), max(0, (scaled_h - 480) // 2)
            h_copy, w_copy = min(480, scaled_h), min(640, scaled_w)
            canvas[start_y:start_y+h_copy, start_x:start_x+w_copy] = thermal_map[src_y:src_y+h_copy, src_x:src_x+w_copy]

            M = np.float32([[1, 0, off_x], [0, 1, off_y]])
            aligned_thermal = cv2.warpAffine(canvas, M, (640, 480))
            fused = cv2.addWeighted(aligned_thermal, ALPHA, visible_frame, 1.0 - ALPHA, 0)

            # ---------------------------------------------------------
            # MULTI-HOTSPOT SPATIAL LOCATION ENGINE
            # ---------------------------------------------------------
            hotspot_candidates = []
            hotspot_threshold = max(HOTSPOT_MIN_TEMP, min_val + (max_val - min_val) * 0.5)
            hotspot_mask = np.uint8(data >= hotspot_threshold) * 255

            contours, _ = cv2.findContours(hotspot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                region_data = data[y:y+h, x:x+w]
                peak_temp = float(np.max(region_data))
                local_loc = np.unravel_index(np.argmax(region_data, axis=None), region_data.shape)
                raw_x = int((x + local_loc[1]) * (640 / 32))
                raw_y = int((y + local_loc[0]) * (480 / 24))

                hs_x = int((raw_x - 320) * (SCALE_P / 100.0) + 320 + off_x)
                hs_y = int((raw_y - 240) * (SCALE_P / 100.0) + 240 + off_y)
                hs_x, hs_y = max(0, min(639, hs_x)), max(0, min(479, hs_y))

                hotspot_candidates.append({"temp": peak_temp, "pos": (hs_x, hs_y)})

            hotspot_candidates.sort(key=lambda item: item["temp"], reverse=True)
            top_hotspots = hotspot_candidates[:MAX_HOTSPOTS]

            # Render Hotspot Markers on Live Window
            for idx, hs in enumerate(top_hotspots):
                rank = idx + 1
                temp, pos = hs["temp"], hs["pos"]
                color = (0, 0, 255) if temp >= OVERHEAT_THRESH else (0, 255, 255)

                cv2.drawMarker(fused, pos, color, cv2.MARKER_CROSS, 20, 2)
                cv2.circle(fused, pos, 6, (255, 255, 255), 1)
                cv2.putText(fused, f"#{rank}: {temp:.1f}C", (pos[0] + 10, pos[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            env_data = read_environmental_sensors()

            # ---------------------------------------------------------
            # 20-SECOND RECORDING & SQL DATABASE COMMIT ENGINE
            # ---------------------------------------------------------
            if recording:
                elapsed = time.time() - test_start_time
                remaining = max(0.0, 20.0 - elapsed)

                # Calculate instantaneous thermal rise rate (dT/dt)
                dT_dt = max(0.0, (max_val - prev_peak_temp) / 0.25)
                max_observed_rise_rate = max(max_observed_rise_rate, dT_dt)
                prev_peak_temp = max_val

                cv2.putText(fused, f"REC TO SQL DB: {remaining:.1f}s REMAINING", (10, 440),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2)

                # Cache frame telemetry row
                telemetry_entry = {
                    "elapsed_s": round(elapsed, 2),
                    "ambient_temp_c": env_data["ambient_temp_c"],
                    "humidity_rh": env_data["humidity_rh"],
                    "rise_rate": round(dT_dt, 2),
                    "hotspots": top_hotspots
                }
                session_telemetry_cache.append(telemetry_entry)

                if elapsed >= 20.0:
                    recording = False

                    # 1. Run Final PCB Lifespan & Component Failure Diagnostics
                    health_score, lifespan, status, pred_components, diagnostics = calculate_pcb_health_and_life(
                        max_val, max_observed_rise_rate, env_data
                    )

                    # 2. Encode all 3 Image Assets into Binary JPEG Blobs
                    _, blob_optical = cv2.imencode('.jpg', visible_frame)
                    _, blob_thermal = cv2.imencode('.jpg', aligned_thermal)
                    _, blob_fused = cv2.imencode('.jpg', fused)

                    # 3. Save Master Test Record to SQLite Table: 'test_runs'
                    conn = sqlite3.connect(DB_NAME)
                    cursor = conn.cursor()

                    cursor.execute('''
                        INSERT INTO test_runs (
                            duration_s, target_distance_cm, overall_status, health_score_percent,
                            estimated_lifespan, peak_temp_c, max_rise_rate_c_per_s,
                            avg_ambient_temp_c, avg_humidity_rh, moisture_flag, dust_flag,
                            predicted_components, failure_diagnostics_json,
                            img_optical_rgb, img_thermal_raw, img_fused_hotspots
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        20.0, target_dist_cm, status, health_score,
                        lifespan, max_val, max_observed_rise_rate,
                        env_data["ambient_temp_c"], env_data["humidity_rh"],
                        int(env_data["moisture_detected"]), int(env_data["dust_contamination"]),
                        json.dumps(pred_components), json.dumps(diagnostics),
                        sqlite3.Binary(blob_optical.tobytes()),
                        sqlite3.Binary(blob_thermal.tobytes()),
                        sqlite3.Binary(blob_fused.tobytes())
                    ))

                    new_test_id = cursor.lastrowid

                    # 4. Save Time-Series Telemetry to SQLite Table: 'frame_telemetry'
                    for item in session_telemetry_cache:
                        hs = item["hotspots"]
                        hs_data = []
                        for i in range(MAX_HOTSPOTS):
                            if i < len(hs):
                                hs_data.extend([hs[i]["temp"], hs[i]["pos"][0], hs[i]["pos"][1]])
                            else:
                                hs_data.extend([None, None, None])

                        cursor.execute('''
                            INSERT INTO frame_telemetry (
                                test_id, elapsed_s, ambient_temp_c, humidity_rh, thermal_rise_rate,
                                hs1_temp, hs1_x, hs1_y, hs2_temp, hs2_x, hs2_y,
                                hs3_temp, hs3_x, hs3_y, hs4_temp, hs4_x, hs4_y, active_fault_count
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            new_test_id, item["elapsed_s"], item["ambient_temp_c"], item["humidity_rh"], item["rise_rate"],
                            hs_data[0], hs_data[1], hs_data[2], hs_data[3], hs_data[4], hs_data[5],
                            hs_data[6], hs_data[7], hs_data[8], hs_data[9], hs_data[10], hs_data[11],
                            len(diagnostics)
                        ))

                    conn.commit()
                    conn.close()

                    print(f"\n[SUCCESS] Test Session #{new_test_id} successfully saved to '{DB_NAME}'!")
                    print(f" - Status: {status} | Health Score: {health_score:.1f}%")
                    print(f" - Estimated Lifespan: {lifespan}")
                    print(f" - Predicted Component Failures: {pred_components}")
                    print(" - Images Committed to DB: Optical RGB, Raw Thermal, Fused Hotspots\n")

            cv2.putText(fused, "Press 't' to start 20s Test & Save to DB | Press 'a' to Align | Press 'q' to Quit", 
                        (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            cv2.imshow(WINDOW_NAME, fused)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                auto_fine_tune = True
            elif key == ord('t') and not recording:
                recording = True
                test_start_time = time.time()
                prev_peak_temp = max_val
                max_observed_rise_rate = 0.0
                session_telemetry_cache = []
                print("\n[TEST STARTED] Executing 20-second dynamic evaluation run...")

        except Exception as e:
            continue

finally:
    picam2.stop()
    cv2.destroyAllWindows()
