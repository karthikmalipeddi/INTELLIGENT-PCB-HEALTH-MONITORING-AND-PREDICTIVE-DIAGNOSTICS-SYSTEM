import os
import json
import time
import board
import busio
import adafruit_mlx90640
import numpy as np
import cv2
from picamera2 import Picamera2

# Hardware Initialization
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
picam2.start()

CONFIG_FILE = "calibration_config.json"

default_config = {
    "offset_x": 36,
    "offset_y": 44,
    "scale_percent": 100,
    "alpha": 60,
    "overheat_thresh": 70
}

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
else:
    config = default_config

def nothing(x):
    pass

window_name = "PCB Thermal Fusion Calibration Suite"
cv2.namedWindow(window_name)

cv2.createTrackbar("Offset X", window_name, config["offset_x"] + 100, 200, nothing)
cv2.createTrackbar("Offset Y", window_name, config["offset_y"] + 100, 200, nothing)
cv2.createTrackbar("Scale %", window_name, config["scale_percent"], 150, nothing)
cv2.createTrackbar("Blend Alpha %", window_name, config["alpha"], 100, nothing)

frame_thermal = [0] * 768

print("\n=== THERMAL ALIGNMENT & CALIBRATION MODE ===")
print("Applied 180-degree rotation fix to correct vertical and horizontal inversion.")
print("Press 's' to save configuration | Press 'q' to exit.\n")

try:
    while True:
        off_x = cv2.getTrackbarPos("Offset X", window_name) - 100
        off_y = cv2.getTrackbarPos("Offset Y", window_name) - 100
        scale_p = max(50, cv2.getTrackbarPos("Scale %", window_name))
        alpha_p = cv2.getTrackbarPos("Blend Alpha %", window_name) / 100.0

        visible_frame = picam2.capture_array()
        visible_frame = cv2.cvtColor(visible_frame, cv2.COLOR_RGB2BGR)

        try:
            mlx.getFrame(frame_thermal)
            data = np.reshape(frame_thermal, (24, 32))

            # FIX: Rotate 180 degrees to solve vertical + horizontal flip simultaneously
            data = cv2.rotate(data, cv2.ROTATE_180)

            min_val = float(np.min(data))
            max_val = float(np.max(data))

            if max_val == min_val:
                norm_data = np.zeros((24, 32), dtype=np.uint8)
            else:
                norm_data = np.uint8(255 * (data - min_val) / (max_val - min_val))

            scaled_w = int(640 * (scale_p / 100.0))
            scaled_h = int(480 * (scale_p / 100.0))
            bicu_img = cv2.resize(norm_data, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
            thermal_map = cv2.applyColorMap(bicu_img, cv2.COLORMAP_JET)

            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            start_x = max(0, (640 - scaled_w) // 2)
            start_y = max(0, (480 - scaled_h) // 2)
            
            src_x = max(0, (scaled_w - 640) // 2)
            src_y = max(0, (scaled_h - 480) // 2)
            
            h_copy = min(480, scaled_h)
            w_copy = min(640, scaled_w)
            
            canvas[start_y:start_y+h_copy, start_x:start_x+w_copy] = thermal_map[src_y:src_y+h_copy, src_x:src_x+w_copy]

            M = np.float32([[1, 0, off_x], [0, 1, off_y]])
            aligned_thermal = cv2.warpAffine(canvas, M, (640, 480))

            fused = cv2.addWeighted(aligned_thermal, alpha_p, visible_frame, 1.0 - alpha_p, 0)

            max_loc = np.unravel_index(np.argmax(data, axis=None), data.shape)
            raw_x = int(max_loc[1] * (640 / 32))
            raw_y = int(max_loc[0] * (480 / 24))
            
            hotspot_x = int((raw_x - 320) * (scale_p / 100.0) + 320 + off_x)
            hotspot_y = int((raw_y - 240) * (scale_p / 100.0) + 240 + off_y)

            cv2.drawMarker(fused, (hotspot_x, hotspot_y), (0, 255, 0), cv2.MARKER_CROSS, 25, 2)
            cv2.circle(fused, (hotspot_x, hotspot_y), 8, (255, 255, 255), 1)

            cv2.putText(fused, f"OFF_X: {off_x} | OFF_Y: {off_y} | Scale: {scale_p}%", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
            cv2.putText(fused, f"Hotspot Temp: {max_val:.1f}C", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
            cv2.putText(fused, "Press 's' to Save Config | Press 'q' to Quit", 
                        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow(window_name, fused)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_data = {
                    "offset_x": off_x,
                    "offset_y": off_y,
                    "scale_percent": scale_p,
                    "alpha": int(alpha_p * 100),
                    "overheat_thresh": 70
                }
                with open(CONFIG_FILE, "w") as f:
                    json.dump(save_data, f, indent=4)
                print(f"[CALIBRATION SAVED] Written to {CONFIG_FILE}: {save_data}")

        except Exception as e:
            continue

finally:
    picam2.stop()
    cv2.destroyAllWindows()
