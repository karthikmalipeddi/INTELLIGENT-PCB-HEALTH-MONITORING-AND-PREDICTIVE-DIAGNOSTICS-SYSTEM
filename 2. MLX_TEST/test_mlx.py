import time
import board
import busio
import adafruit_mlx90640

# Initialize I2C bus using Adafruit Blinka
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)

# Initialize MLX90640 sensor
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ

print("MLX90640 initialized successfully!")

# 768 float values representing 32x24 pixels
frame = [0] * 768

try:
    while True:
        mlx.getFrame(frame)
        min_temp = min(frame)
        max_temp = max(frame)
        avg_temp = sum(frame) / len(frame)

        print(f"Min: {min_temp:.1f}°C | Max: {max_temp:.1f}°C | Avg: {avg_temp:.1f}°C")
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStopped.")
