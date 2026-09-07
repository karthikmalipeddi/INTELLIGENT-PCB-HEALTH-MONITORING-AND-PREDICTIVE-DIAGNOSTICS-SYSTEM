import time
import board
import busio
import adafruit_mlx90640
import numpy as np
import cv2
from picamera2 import Picamera2

# 1. Initialize MLX90640 Thermal Sensor
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ

# 2. Initialize Pi Camera via Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
picam2.start()

print("Starting PCB Thermal Fusion System...")

alpha = 0.45  # Blending ratio (0.45 thermal, 0.55 visible)
frame_thermal = [0] * 768

try:
    while True:
        # Capture frame from Pi Camera
        visible_frame = picam2.capture_array()
        visible_frame = cv2.cvtColor(visible_frame, cv2.COLOR_RGB2BGR)

        try:
            # Read frame from MLX90640 sensor
            mlx.getFrame(frame_thermal)
            data = np.reshape(frame_thermal, (24, 32))
            
            min_val = np.min(data)
            max_val = np.max(data)
            
            # Find maximum temperature coordinates
            max_loc = np.unravel_index(np.argmax(data, axis=None), data.shape)
            
            # Map grid coordinates to 640x480 resolution
            scale_x = 640 / 32
            scale_y = 480 / 24
            hotspot_x = int(max_loc[1] * scale_x + scale_x / 2)
            hotspot_y = int(max_loc[0] * scale_y + scale_y / 2)

            # Normalize data and scale thermal heatmap
            if max_val == min_val:
                norm_data = np.zeros((24, 32), dtype=np.uint8)
            else:
                norm_data = np.uint8(255 * (data - min_val) / (max_val - min_val))

            bicu_img = cv2.resize(norm_data, (640, 480), interpolation=cv2.INTER_CUBIC)
            thermal_map = cv2.applyColorMap(bicu_img, cv2.COLORMAP_JET)

            # Perform image fusion
            fused_frame = cv2.addWeighted(thermal_map, alpha, visible_frame, 1 - alpha, 0)

            # Draw target crosshair on hotspot
            cv2.drawMarker(fused_frame, (hotspot_x, hotspot_y), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(fused_frame, f"HOT: {max_val:.1f}C", (hotspot_x + 12, hotspot_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Temperature stats display
            cv2.putText(fused_frame, f"Min: {min_val:.1f}C  Max: {max_val:.1f}C", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("PCB Thermal Fusion System", fused_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            continue

finally:
    picam2.stop()
    cv2.destroyAllWindows()
