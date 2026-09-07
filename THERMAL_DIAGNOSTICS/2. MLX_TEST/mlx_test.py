import time
import board
import busio
import adafruit_mlx90640

# Start I2C
i2c = busio.I2C(board.SCL, board.SDA)

# Connect to MLX90640
mlx = adafruit_mlx90640.MLX90640(i2c)

# Set refresh rate
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ

print("MLX90640 connected!")
print("Reading temperatures...\n")

frame = [0] * 768

while True:
    try:
        mlx.getFrame(frame)

        print(
            "Min: {:.2f} °C | Max: {:.2f} °C | Center: {:.2f} °C".format(
                min(frame),
                max(frame),
                frame[12 * 32 + 16]
            )
        )

        time.sleep(0.5)

    except Exception as e:
        print("Error:", e)
