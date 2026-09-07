import os
import json
import time
import board
import busio
import adafruit_mlx90640
import numpy as np
import cv2
from picamera2 import Picamera2

# ---------------------------------------------------------
# HARDWARE INITIALIZATION
# ---------------------------------------------------------
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
picam2.start()

# ---------------------------------------------------------
# CONFIGURATION & PARAMETER LOADING
# ---------------------------------------------------------
CONFIG_FILE = "calibration_config.json"

# Calibrated baseline settings
default_config = {
    "offset_x": 35,
    "offset_y": -13,
    "scale_percent": 107,
    "alpha": 60,
    "overheat_thresh": 70.0
}

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        print(f"[SUCCESS] Loaded parameters from {CONFIG_FILE}")
    except Exception as e:
        print(f"[WARNING] Failed to parse config file ({e}). Using tuned defaults.")
        config = default_config
else:
    config = default_config
    print("[INFO] 'calibration_config.json' not found. Operating on tuned default parameters.")

OFF_X = config.get("offset_x", 35)
OFF_Y = config.get("offset_y", -13)
SCALE_P = config.get("scale_percent", 107)
ALPHA = config.get("alpha", 60) / 100.0
OVERHEAT_THRESH = config.get("overheat_thresh", 70.0)

WINDOW_NAME = "PCB Thermal Fusion System"
cv2.namedWindow(WINDOW_NAME)

frame_thermal = [0] * 768

print("\n==================================================")
print(" PCB THERMAL FUSION ENGINE")
print(f" Parameters: OFF_X={OFF_X} | OFF_Y={OFF_Y} | Scale={SCALE_P}% | Alpha={int(ALPHA*100)}%")
print(" Press 'q' to terminate application.")
print("==================================================\n")

try:
    while True:
        # 1. Optical Capture
        visible_frame = picam2.capture_array()
        visible_frame = cv2.cvtColor(visible_frame, cv2.COLOR_RGB2BGR)

        try:
            # 2. Thermal Data Capture
            mlx.getFrame(frame_thermal)
            data = np.reshape(frame_thermal, (24, 32))

            # FIXED ORIENTATION: Vertical flip matrix transformation
            data = np.flipud(data)

            min_val = float(np.min(data))
            max_val = float(np.max(data))

            # 3. Min-Max Normalization
            if max_val == min_val:
                norm_data = np.zeros((24, 32), dtype=np.uint8)
            else:
                norm_data = np.uint8(255 * (data - min_val) / (max_val - min_val))

            # 4. Scale & Interpolation (Cubic)
            scaled_w = int(640 * (SCALE_P / 100.0))
            scaled_h = int(480 * (SCALE_P / 100.0))
            bicu_img = cv2.resize(norm_data, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
            thermal_map = cv2.applyColorMap(bicu_img, cv2.COLORMAP_JET)

            # 5. Canvas Fitting
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            start_x = max(0, (640 - scaled_w) // 2)
            start_y = max(0, (480 - scaled_h) // 2)
            
            src_x = max(0, (scaled_w - 640) // 2)
            src_y = max(0, (scaled_h - 480) // 2)
            
            h_copy = min(480, scaled_h)
            w_copy = min(640, scaled_w)
            
            canvas[start_y:start_y+h_copy, start_x:start_x+w_copy] = thermal_map[src_y:src_y+h_copy, src_x:src_x+w_copy]

            # 6. Affine Shift Alignment
            M = np.float32([[1, 0, OFF_X], [0, 1, OFF_Y]])
            aligned_thermal = cv2.warpAffine(canvas, M, (640, 480))

            # 7. Alpha Blend Overlay
            fused = cv2.addWeighted(aligned_thermal, ALPHA, visible_frame, 1.0 - ALPHA, 0)

            # 8. Hotspot Coordinate Tracking
            max_loc = np.unravel_index(np.argmax(data, axis=None), data.shape)
            raw_x = int(max_loc[1] * (640 / 32))
            raw_y = int(max_loc[0] * (480 / 24))
            
            hotspot_x = int((raw_x - 320) * (SCALE_P / 100.0) + 320 + OFF_X)
            hotspot_y = int((raw_y - 240) * (SCALE_P / 100.0) + 240 + OFF_Y)

            hotspot_x = max(0, min(639, hotspot_x))
            hotspot_y = max(0, min(479, hotspot_y))

            # Status alert coloring
            status_color = (0, 0, 255) if max_val >= OVERHEAT_THRESH else (0, 255, 0)

            # Overlay Reticle & Hotspot Crosshair
            cv2.drawMarker(fused, (hotspot_x, hotspot_y), status_color, cv2.MARKER_CROSS, 25, 2)
            cv2.circle(fused, (hotspot_x, hotspot_y), 8, (255, 255, 255), 1)

            # Telemetry UI
            cv2.putText(fused, f"Hotspot Temp: {max_val:.1f}C", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
            cv2.putText(fused, f"OFF_X: {OFF_X} | OFF_Y: {OFF_Y} | Scale: {SCALE_P}%", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            if max_val >= OVERHEAT_THRESH:
                cv2.putText(fused, "[!] HIGH TEMP ALERT", (10, 95), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            cv2.imshow(WINDOW_NAME, fused)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            continue

finally:
    picam2.stop()
    cv2.destroyAllWindows()
