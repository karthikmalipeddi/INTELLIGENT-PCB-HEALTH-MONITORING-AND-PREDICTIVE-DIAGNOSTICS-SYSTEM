import os
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
# SYSTEM CONSTANTS & PARALLAX ENGINE (3.8 cm Baseline)
# ---------------------------------------------------------
BASELINE_CM = 3.8        # Physical camera center-to-center separation
PARALLAX_K_X = 2300.0    # Lens focal constant x Baseline distance
OFFSET_INF_X = -25       # Base horizontal offset at infinity
OFFSET_INF_Y = -13       # Base vertical offset (distance-invariant)

SCALE_P = 107            # Base FOV scaling factor %
ALPHA = 0.6              # Blend ratio (60% thermal / 40% optical)
OVERHEAT_THRESH = 70.0   # Thermal alert threshold in Celsius

WINDOW_NAME = "Auto-Calibrated PCB Thermal Fusion"
cv2.namedWindow(WINDOW_NAME)
cv2.createTrackbar("Distance (cm)", WINDOW_NAME, 30, 150, lambda x: None)  # Interactive distance slider

frame_thermal = [0] * 768

def calculate_parallax_offset(distance_cm):
    """
    Computes horizontal pixel alignment offset based on target distance.
    Formula: Offset_X = Offset_Infinity + (K_X / Distance_cm)
    """
    safe_dist = max(10.0, float(distance_cm))
    shift_x = PARALLAX_K_X / safe_dist
    
    dyn_off_x = int(OFFSET_INF_X + shift_x)
    dyn_off_y = int(OFFSET_INF_Y)
    
    return dyn_off_x, dyn_off_y

def auto_detect_alignment_offset(visible_gray, thermal_raw):
    """
    Performs real-time edge correlation between optical and thermal gradient frames
    to fine-tune alignment automatically.
    """
    visible_edges = cv2.Canny(visible_gray, 50, 150)
    norm_thermal = cv2.normalize(thermal_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thermal_edges = cv2.Canny(norm_thermal, 30, 100)

    scaled_w = int(640 * (SCALE_P / 100.0))
    scaled_h = int(480 * (SCALE_P / 100.0))
    resized_thermal_edges = cv2.resize(thermal_edges, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.zeros((480, 640), dtype=np.uint8)
    start_x, start_y = max(0, (640 - scaled_w) // 2), max(0, (480 - scaled_h) // 2)
    src_x, src_y = max(0, (scaled_w - 640) // 2), max(0, (scaled_h - 480) // 2)
    h_copy, w_copy = min(480, scaled_h), min(640, scaled_w)
    
    canvas[start_y:start_y+h_copy, start_x:start_x+w_copy] = resized_thermal_edges[src_y:src_y+h_copy, src_x:src_x+w_copy]

    best_score = -1.0
    best_dx, best_dy = OFFSET_INF_X, OFFSET_INF_Y

    # Local search matrix around estimated parallax region
    for dx in range(OFFSET_INF_X - 15, OFFSET_INF_X + 16, 3):
        for dy in range(OFFSET_INF_Y - 10, OFFSET_INF_Y + 11, 2):
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(canvas, M, (640, 480))
            score = np.sum(shifted.astype(np.float32) * visible_edges.astype(np.float32))
            
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

    return best_dx, best_dy

print("\n==================================================")
print(" AUTOMATED THERMAL FUSION ENGINE (3.8cm Parallax)")
print(" - Auto-Align: Press 'a' to perform instant edge-matching.")
print(" - Distance Slider: Adjust working target distance in real time.")
print(" - Quit: Press 'q'")
print("==================================================\n")

auto_fine_tune = False

try:
    while True:
        # 1. Capture Optical Frame
        visible_frame = picam2.capture_array()
        visible_frame = cv2.cvtColor(visible_frame, cv2.COLOR_RGB2BGR)
        visible_gray = cv2.cvtColor(visible_frame, cv2.COLOR_BGR2GRAY)

        try:
            # 2. Capture Thermal Frame
            mlx.getFrame(frame_thermal)
            data = np.reshape(frame_thermal, (24, 32))

            # Correct spatial orientation
            data = np.flipud(data)

            min_val = float(np.min(data))
            max_val = float(np.max(data))

            # 3. Read Current Target Distance (cm) & Compute Dynamic Parallax Offset
            target_dist_cm = cv2.getTrackbarPos("Distance (cm)", WINDOW_NAME)
            target_dist_cm = max(10, target_dist_cm)
            
            if auto_fine_tune:
                off_x, off_y = auto_detect_alignment_offset(visible_gray, data)
                auto_fine_tune = False
                print(f"[AUTO-CALIBRATION COMPLETED] Dynamic Offset: X={off_x}, Y={off_y}")
            else:
                off_x, off_y = calculate_parallax_offset(target_dist_cm)

            # 4. Normalize Thermal Array
            if max_val == min_val:
                norm_data = np.zeros((24, 32), dtype=np.uint8)
            else:
                norm_data = np.uint8(255 * (data - min_val) / (max_val - min_val))

            # 5. Cubic Upscaling & Color Mapping
            scaled_w = int(640 * (SCALE_P / 100.0))
            scaled_h = int(480 * (SCALE_P / 100.0))
            bicu_img = cv2.resize(norm_data, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
            thermal_map = cv2.applyColorMap(bicu_img, cv2.COLORMAP_JET)

            # 6. Fit Overlay to Canvas
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            start_x = max(0, (640 - scaled_w) // 2)
            start_y = max(0, (480 - scaled_h) // 2)
            src_x = max(0, (scaled_w - 640) // 2)
            src_y = max(0, (scaled_h - 480) // 2)
            h_copy = min(480, scaled_h)
            w_copy = min(640, scaled_w)
            
            canvas[start_y:start_y+h_copy, start_x:start_x+w_copy] = thermal_map[src_y:src_y+h_copy, src_x:src_x+w_copy]

            # 7. Apply Affine Parallax Shift
            M = np.float32([[1, 0, off_x], [0, 1, off_y]])
            aligned_thermal = cv2.warpAffine(canvas, M, (640, 480))

            # 8. Perform Alpha Blend
            fused = cv2.addWeighted(aligned_thermal, ALPHA, visible_frame, 1.0 - ALPHA, 0)

            # 9. Target Hotspot Calculation
            max_loc = np.unravel_index(np.argmax(data, axis=None), data.shape)
            raw_x = int(max_loc[1] * (640 / 32))
            raw_y = int(max_loc[0] * (480 / 24))
            
            hotspot_x = int((raw_x - 320) * (SCALE_P / 100.0) + 320 + off_x)
            hotspot_y = int((raw_y - 240) * (SCALE_P / 100.0) + 240 + off_y)
            hotspot_x = max(0, min(639, hotspot_x))
            hotspot_y = max(0, min(479, hotspot_y))

            # Visual telemetry display
            status_color = (0, 0, 255) if max_val >= OVERHEAT_THRESH else (0, 255, 0)
            cv2.drawMarker(fused, (hotspot_x, hotspot_y), status_color, cv2.MARKER_CROSS, 25, 2)
            cv2.circle(fused, (hotspot_x, hotspot_y), 8, (255, 255, 255), 1)

            cv2.putText(fused, f"Hotspot Temp: {max_val:.1f}C", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
            cv2.putText(fused, f"Target Dist: {target_dist_cm}cm | Dynamic OFF_X: {off_x} | OFF_Y: {off_y}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
            cv2.putText(fused, "Press 'a' for Edge Auto-Align | Press 'q' to Quit", 
                        (10, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

            if max_val >= OVERHEAT_THRESH:
                cv2.putText(fused, "[!] OVERHEAT WARNING", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            cv2.imshow(WINDOW_NAME, fused)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                auto_fine_tune = True

        except Exception as e:
            continue

finally:
    picam2.stop()
    cv2.destroyAllWindows()
