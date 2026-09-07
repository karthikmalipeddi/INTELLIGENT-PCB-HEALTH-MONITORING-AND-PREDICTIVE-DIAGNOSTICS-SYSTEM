import time
import board
import busio
import adafruit_mlx90640
import numpy as np
import cv2

# Initialize I2C bus & MLX90640 sensor
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ

print("Starting Enhanced PCB Thermal Tracker...")

frame = [0] * 768

while True:
    try:
        mlx.getFrame(frame)
        
        # Reshape to 24x32 temperature matrix
        data = np.reshape(frame, (24, 32))
        
        min_val = np.min(data)
        max_val = np.max(data)
        
        # Locate coordinates of the hottest pixel (y, x in 24x32 grid)
        max_loc = np.unravel_index(np.argmax(data, axis=None), data.shape)
        
        # Scale coordinates from (24x32) up to (640x480) display space
        scale_x = 640 / 32
        scale_y = 480 / 24
        hotspot_x = int(max_loc[1] * scale_x + scale_x / 2)
        hotspot_y = int(max_loc[0] * scale_y + scale_y / 2)
        
        # Normalize and interpolate image
        if max_val == min_val:
            norm_data = np.zeros((24, 32), dtype=np.uint8)
        else:
            norm_data = np.uint8(255 * (data - min_val) / (max_val - min_val))
            
        bicu_img = cv2.resize(norm_data, (640, 480), interpolation=cv2.INTER_CUBIC)
        color_img = cv2.applyColorMap(bicu_img, cv2.COLORMAP_JET)
        
        # Draw target crosshair over the hottest point
        cv2.drawMarker(color_img, (hotspot_x, hotspot_y), (255, 255, 255), 
                       cv2.MARKER_CROSS, 20, 2)
        cv2.circle(color_img, (hotspot_x, hotspot_y), 10, (255, 255, 255), 1)
        
        # Display max temperature label near the crosshair
        cv2.putText(color_img, f"HOT: {max_val:.1f}C", (hotspot_x + 12, hotspot_y + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        # Top banner stats
        cv2.putText(color_img, f"Min: {min_val:.1f}C  Max: {max_val:.1f}C", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("PCB Thermal Health Monitor", color_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        continue

cv2.destroyAllWindows()
