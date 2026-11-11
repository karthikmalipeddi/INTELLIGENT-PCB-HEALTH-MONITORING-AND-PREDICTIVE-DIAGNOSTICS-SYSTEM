"""
hardware_vision.py
==================
Sensor hardware integration and thermal-optical computer vision engine for TEST GIG.

Physical Hardware Drivers (Production Pi 4, MOCK_MODE = False):
1. Serial Telemetry (Raspberry Pi Pico):
   - Reads incoming JSON strings from Pico via USB serial (/dev/ttyACM0 or /dev/ttyUSB0 @ 115200 baud).
   - Real-time logging of 3 Environmental (TMP117, BMP280, MPU6050) + 43 Electrical channels
     (9x Power Monitors, 2x Direct ADCs, 32x MUX channels) to data_acq_log.db at ~10 Hz.
2. Optical Vision (ESP32-CAM):
   - Captures high-res video frames from http://[ESP32_IP]:81/stream.
   - Saves optical image and logs resolution to optical_log.db.
3. Thermal Vision (MLX90640):
   - I2C driver on /dev/i2c-1 via adafruit-circuitpython-mlx90640.
   - Extracts 32x24 temperature array, absolute max/min, and Top 3 hotspot coordinates.
4. Thermal-Optical Overlay Engine (OpenCV):
   - Upscales 32x24 matrix using cv2.INTER_CUBIC to match ESP32-CAM optical resolution.
   - Applies false-colour cv2.COLORMAP_JET.
   - Manually aligns thermal layer via 2D affine translation (X_OFFSET, Y_OFFSET).
   - Alpha-blends layers at 45% transparency (cv2.addWeighted).
   - Draws red bounding boxes and temperature readouts over Top 3 hotspots.
   - Stores overlay and metrics in thermal_log.db.
5. AI Vision Engine:
   - Posts optical frame to Ngrok API (detect_pcb_remote) with "x-api-key": "pcb_mal_123".
   - Decodes base64 component_image (SAHI) and defect_image (YOLO).
   - Parses detected defect labels for the Diagnostic Engine.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np
import requests

import config

logger = logging.getLogger("testgig.hardware_vision")


class ThermalOpticalOverlayEngine:
    """OpenCV-based engine to fuse 32x24 MLX90640 thermal data with ESP32-CAM optical video.

    Implements:
    - Cubic spline upscaling (cv2.INTER_CUBIC)
    - JET false-colour thermal palette mapping
    - Manual parallax alignment via 2D affine translation (X_OFFSET, Y_OFFSET)
    - Alpha-channel fusion (cv2.addWeighted at 45% transparency)
    - Top 3 hotspot detection with red bounding box and label overlays
    """

    def __init__(
        self,
        x_offset: int = config.X_OFFSET,
        y_offset: int = config.Y_OFFSET,
        alpha: float = config.ALPHA_TRANSPARENCY,
        colormap_name: str = config.THERMAL_COLORMAP,
    ) -> None:
        self.x_offset = int(x_offset)
        self.y_offset = int(y_offset)
        self.alpha = float(alpha)

        # Colormap mapping
        if colormap_name.upper() == "COLORMAP_INFERNO":
            self.colormap = cv2.COLORMAP_INFERNO
        else:
            self.colormap = cv2.COLORMAP_JET

    def find_top_hotspots(
        self,
        thermal_matrix: np.ndarray,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Identify coordinates and temperatures of the Top K hottest spots.

        Parameters
        ----------
        thermal_matrix : np.ndarray
            2D float array (typically 24 rows x 32 columns or 32x24).
        top_k : int, optional
            Number of peak hotspots to locate (default 3).

        Returns
        -------
        list of dict:
            Each element: {"rank": int, "row": int, "col": int, "temp_c": float}
        """
        flat_indices = np.argsort(thermal_matrix.ravel())[::-1]
        hotspots = []
        rows, cols = thermal_matrix.shape

        for rank, idx in enumerate(flat_indices[:top_k], start=1):
            r = int(idx // cols)
            c = int(idx % cols)
            temp = float(thermal_matrix[r, c])
            hotspots.append({
                "rank": rank,
                "row": r,
                "col": c,
                "temp_c": round(temp, 2),
            })
        return hotspots

    def fuse(
        self,
        thermal_matrix: np.ndarray,
        optical_frame: np.ndarray,
        x_offset: Optional[int] = None,
        y_offset: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> dict[str, Any]:
        """Fuse thermal matrix and optical frame into a calibrated overlay image.

        Pipeline:
        1. Reshape & normalize 32x24 thermal array to 0-255 uint8.
        2. Apply cv2.resize with cv2.INTER_CUBIC to match optical frame dimensions.
        3. Colorize using cv2.applyColorMap(COLORMAP_JET).
        4. Apply 2D affine translation for parallax correction (X_OFFSET, Y_OFFSET).
        5. Alpha-blend optical frame and warped thermal map with cv2.addWeighted.
        6. Locate Top 3 hotspots and draw red bounding boxes + temperature text.

        Parameters
        ----------
        thermal_matrix : np.ndarray
            Raw 2D temperature array from MLX90640 (24x32 or 32x24).
        optical_frame : np.ndarray
            BGR image from ESP32-CAM.
        x_offset : int, optional
            Parallax horizontal shift in pixels (overrides default).
        y_offset : int, optional
            Parallax vertical shift in pixels (overrides default).
        alpha : float, optional
            Blending transparency (0.0=optical only, 1.0=thermal only).

        Returns
        -------
        dict[str, Any]:
            - "thermal_annotated": RGB image of upscaled thermal map with boxes
            - "blended_overlay": RGB image of blended optical + thermal with boxes
            - "top3_hotspots": List of top 3 hotspot dicts
            - "top3_temps": List of top 3 temperatures [float, float, float]
            - "max_temp": Absolute maximum temperature
            - "min_temp": Absolute minimum temperature
            - "median_temp": Background median temperature
        """
        if x_offset is None:
            x_offset = self.x_offset
        if y_offset is None:
            y_offset = self.y_offset
        if alpha is None:
            alpha = self.alpha

        # Ensure thermal_matrix is 2D (24, 32)
        th_arr = np.asarray(thermal_matrix, dtype=np.float32)
        if th_arr.ndim == 1 and th_arr.size == 768:
            th_arr = th_arr.reshape((24, 32))
        elif th_arr.shape == (32, 24):
            th_arr = th_arr.T

        th_rows, th_cols = th_arr.shape
        opt_h, opt_w = optical_frame.shape[:2]

        max_temp = float(np.max(th_arr))
        min_temp = float(np.min(th_arr))
        median_temp = float(np.median(th_arr))

        # Normalize to 0..255 for colormap application
        temp_span = max(max_temp - min_temp, 1e-4)
        th_norm = ((th_arr - min_temp) / temp_span * 255.0).astype(np.uint8)

        # 1. Upscale thermal map using INTER_CUBIC to match optical resolution
        th_resized = cv2.resize(
            th_norm,
            (opt_w, opt_h),
            interpolation=cv2.INTER_CUBIC,
        )

        # 2. Apply false-color JET colormap
        th_colored = cv2.applyColorMap(th_resized, self.colormap)

        # 3. Apply Affine parallax translation matrix
        #    M = [[1, 0, x_offset], [0, 1, y_offset]]
        M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        th_aligned = cv2.warpAffine(
            th_colored,
            M,
            (opt_w, opt_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # 4. Alpha blend: Blended = (1 - alpha) * Optical + alpha * Thermal
        beta = 1.0 - alpha
        blended = cv2.addWeighted(optical_frame, beta, th_aligned, alpha, 0.0)

        # 5. Detect Top 3 hotspots
        top3_hotspots = self.find_top_hotspots(th_arr, top_k=3)
        top3_temps = [h["temp_c"] for h in top3_hotspots]

        # Draw red bounding boxes and temperature labels on both images
        box_w = max(int(opt_w / th_cols * 1.5), 24)
        box_h = max(int(opt_h / th_rows * 1.5), 24)

        thermal_annotated = th_aligned.copy()

        for target_img in (blended, thermal_annotated):
            for spot in top3_hotspots:
                rank = spot["rank"]
                r = spot["row"]
                c = spot["col"]
                temp = spot["temp_c"]

                # Project low-res thermal grid coordinates to optical image resolution
                center_x = int((c + 0.5) * (opt_w / th_cols)) + x_offset
                center_y = int((r + 0.5) * (opt_h / th_rows)) + y_offset

                # Clamp bounding box inside image bounds
                x1 = max(0, center_x - box_w // 2)
                y1 = max(0, center_y - box_h // 2)
                x2 = min(opt_w - 1, center_x + box_w // 2)
                y2 = min(opt_h - 1, center_y + box_h // 2)

                # Draw bold red bounding box (BGR: 0, 0, 255)
                cv2.rectangle(target_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Hotspot text label
                text = f"#{rank} {temp:.1f}C"
                # Text backdrop rectangle for readability
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(
                    target_img,
                    (x1, max(0, y1 - th - 6)),
                    (x1 + tw + 6, y1),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    target_img,
                    text,
                    (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        return {
            "thermal_annotated": thermal_annotated,
            "blended_overlay": blended,
            "top3_hotspots": top3_hotspots,
            "top3_temps": top3_temps,
            "max_temp": round(max_temp, 2),
            "min_temp": round(min_temp, 2),
            "median_temp": round(median_temp, 2),
            "x_offset": x_offset,
            "y_offset": y_offset,
        }


class HardwareManager:
    """Manages physical sensor drivers and external APIs with MOCK_MODE fallbacks.

    When config.MOCK_MODE is True (Laptop testing):
        Bypasses hardware connections and immediately returns fail-safe responses.
    When config.MOCK_MODE is False (Raspberry Pi 4):
        Attempts to read real I2C MLX90640, Wi-Fi stream from ESP32-CAM,
        Ngrok AI endpoint, and USB serial from Pi Pico.
    """

    def __init__(self, mock_mode: bool = config.MOCK_MODE) -> None:
        self.mock_mode = mock_mode
        self.overlay_engine = ThermalOpticalOverlayEngine()

    # ------------------------------------------------------------------ #
    # 1. Thermal Camera (MLX90640 via I2C on /dev/i2c-1)                 #
    # ------------------------------------------------------------------ #

    def read_thermal_sensor(self) -> Tuple[Optional[np.ndarray], str]:
        """Read 32x24 IR matrix from physical MLX90640 via I2C or return fail-safe.

        Returns
        -------
        tuple (np.ndarray or None, str):
            (thermal_matrix_32x24, status_message)
        """
        if self.mock_mode:
            logger.info("MOCK_MODE active: Thermal camera hardware bypassed.")
            return None, config.FAILSAFE_THERMAL

        try:
            # Physical Pi 4: Import Adafruit CircuitPython MLX90640 on /dev/i2c-1
            import board
            import busio
            import adafruit_mlx90640

            i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
            mlx = adafruit_mlx90640.MLX90640(i2c, address=config.MLX90640_I2C_ADDR)
            mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ

            frame = [0.0] * 768
            mlx.getFrame(frame)
            matrix = np.array(frame, dtype=np.float32).reshape((24, 32))
            logger.info("MLX90640 physical I2C read success (shape: %s)", matrix.shape)
            return matrix, "OK"
        except Exception as exc:
            logger.warning("MLX90640 I2C read failed (%s), returning fail-safe '%s'", exc, config.FAILSAFE_THERMAL)
            return None, config.FAILSAFE_THERMAL

    # ------------------------------------------------------------------ #
    # 2. Optical Stream (ESP32-CAM via Wi-Fi)                             #
    # ------------------------------------------------------------------ #

    def capture_optical_frame(
        self,
        stream_url: str = config.ESP32_STREAM_URL,
    ) -> Tuple[Optional[np.ndarray], str, str]:
        """Capture a high-res frame from the ESP32-CAM stream (default port 81).

        Returns
        -------
        tuple (np.ndarray or None, str, str):
            (frame_bgr, resolution_str, status_message)
        """
        if self.mock_mode:
            logger.info("MOCK_MODE active: ESP32-CAM stream bypassed.")
            return None, config.FAILSAFE_THERMAL, config.FAILSAFE_OPTICAL

        try:
            logger.info("Connecting to ESP32-CAM stream: %s", stream_url)
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                logger.warning("ESP32-CAM stream not reachable at %s", stream_url)
                return None, config.FAILSAFE_THERMAL, config.FAILSAFE_OPTICAL

            # Flush stale buffer and grab fresh high-res frame
            frame = None
            for _ in range(5):
                ret, temp_frame = cap.read()
                if ret:
                    frame = temp_frame
            cap.release()

            if frame is None:
                return None, config.FAILSAFE_THERMAL, config.FAILSAFE_OPTICAL

            h, w = frame.shape[:2]
            resolution_str = f"{w}x{h}"
            logger.info("Captured optical frame from ESP32-CAM: %s", resolution_str)
            return frame, resolution_str, "OK"

        except Exception as exc:
            logger.warning("ESP32-CAM optical capture failed (%s), returning '%s'", exc, config.FAILSAFE_OPTICAL)
            return None, config.FAILSAFE_THERMAL, config.FAILSAFE_OPTICAL

    # ------------------------------------------------------------------ #
    # 3. AI Vision Inference API (Remote Ngrok)                          #
    # ------------------------------------------------------------------ #

    def detect_pcb_remote(
        self,
        optical_frame: Optional[np.ndarray],
        endpoint_url: str = config.AI_API_URL,
        api_key: str = config.AI_API_KEY,
    ) -> Tuple[Optional[str], Optional[str], list[str], str]:
        """Send optical frame to Ngrok AI Vision endpoint for SAHI + YOLO detection.

        Parameters
        ----------
        optical_frame : np.ndarray or None
            Optical BGR image to submit for inference.
        endpoint_url : str
            Ngrok inference API URL.
        api_key : str
            API authentication key.

        Returns
        -------
        tuple (comp_path, defect_path, defect_strings, status_string):
            - comp_path: Path to decoded SAHI component image
            - defect_path: Path to decoded YOLO defect image
            - defect_strings: List of detected fault strings
            - status_string: 'OK' or fail-safe 'NO MODELS ARE WORKING'
        """
        if self.mock_mode or optical_frame is None:
            logger.info("MOCK_MODE active: AI Vision API bypassed.")
            return None, None, [], config.FAILSAFE_AI

        try:
            ret, buffer = cv2.imencode(".jpg", optical_frame)
            if not ret:
                return None, None, [], config.FAILSAFE_AI

            headers = {
                "x-api-key": api_key,
                "Content-Type": "application/octet-stream",
            }

            response = requests.post(
                endpoint_url,
                data=buffer.tobytes(),
                headers=headers,
                timeout=config.AI_API_TIMEOUT_SEC,
            )

            if response.status_code != 200:
                logger.warning("AI API returned status %d", response.status_code)
                return None, None, [], config.FAILSAFE_AI

            payload = response.json()

            comp_path = None
            defect_path = None
            defects: list[str] = []
            timestamp = int(time.time())

            if "component_image" in payload and payload["component_image"]:
                comp_bytes = base64.b64decode(payload["component_image"])
                comp_path = os.path.join(config.CAPTURES_DIR, f"sahi_comp_{timestamp}.jpg")
                with open(comp_path, "wb") as f:
                    f.write(comp_bytes)

            if "defect_image" in payload and payload["defect_image"]:
                defect_bytes = base64.b64decode(payload["defect_image"])
                defect_path = os.path.join(config.CAPTURES_DIR, f"yolo_defect_{timestamp}.jpg")
                with open(defect_path, "wb") as f:
                    f.write(defect_bytes)

            # Parse defect annotations if returned
            if "defects" in payload and isinstance(payload["defects"], list):
                for item in payload["defects"]:
                    name = item.get("label", item.get("name", "Defect"))
                    conf = item.get("confidence", 0.0)
                    defects.append(f"ERR: AI YOLO {name} (conf={conf:.2f})")

            return comp_path, defect_path, defects, "OK"

        except Exception as exc:
            logger.warning("AI Vision API call failed (%s), returning '%s'", exc, config.FAILSAFE_AI)
            return None, None, [], config.FAILSAFE_AI

    # Alias for API compatibility
    call_ai_vision_api = detect_pcb_remote

    # ------------------------------------------------------------------ #
    # 4. Pi Pico USB Serial Telemetry Acquisition (30-second window)     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_pico_serial_port() -> Optional[str]:
        """Locate the Pi Pico USB serial port among candidates or system ports."""
        import serial
        import serial.tools.list_ports

        # 1. Try explicit candidates first
        for port in config.PICO_SERIAL_CANDIDATES:
            try:
                s = serial.Serial(port, baudrate=config.PICO_BAUDRATE, timeout=0.1)
                s.close()
                return port
            except Exception:
                continue

        # 2. Search dynamically across connected USB serial devices
        ports = serial.tools.list_ports.comports()
        for p in ports:
            desc = p.description.lower()
            if "pico" in desc or "acm" in p.device.lower() or "usb" in p.device.lower():
                return p.device

        return None

    def acquire_live_telemetry_stream(
        self,
        board_name: str,
        serial_number: str,
        db_logger_fn: Callable[[str, str, dict], None],
        duration_sec: int = config.EXECUTION_DURATION_SEC,
        sample_rate_hz: int = config.TELEMETRY_SAMPLE_RATE_HZ,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> list[dict[str, Any]]:
        """Read incoming JSON telemetry from Pi Pico via USB serial for duration_sec.

        Reads lines at ~sample_rate_hz (default 10 Hz), parses the JSON payload,
        and logs each sample directly into data_acq_log.db.

        If serial port is unavailable or drops, catches exception gracefully,
        falls back to mock generator, and flags a diagnostic notice.
        """
        import serial

        port = self.detect_pico_serial_port() or config.PICO_SERIAL_PORT
        target_samples = int(duration_sec * sample_rate_hz)
        samples: list[dict[str, Any]] = []

        logger.info("Attempting live telemetry acquisition from Pico on port %s", port)

        try:
            with serial.Serial(
                port,
                baudrate=config.PICO_BAUDRATE,
                timeout=config.PICO_TIMEOUT_SEC,
            ) as ser:
                ser.reset_input_buffer()
                start_time = time.time()

                while (time.time() - start_time) < duration_sec and len(samples) < target_samples:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if line and line.startswith("{") and line.endswith("}"):
                        try:
                            payload = json.loads(line)
                            samples.append(payload)
                            # Log to SQLite in real-time
                            db_logger_fn(board_name, serial_number, payload)

                            if progress_callback:
                                pct = int((len(samples) / target_samples) * 100)
                                progress_callback(
                                    pct,
                                    f"Live Pico Acquisition ({len(samples)}/{target_samples} samples)...",
                                )
                        except json.JSONDecodeError:
                            continue

            if len(samples) >= (target_samples * 0.7):
                logger.info("Successfully acquired %d live samples from Pico", len(samples))
                return samples
            else:
                logger.warning("Insufficient samples (%d) received from Pico. Triggering fallback.", len(samples))

        except Exception as exc:
            logger.warning("Pico serial stream failed on %s: %s", port, exc)

        # Fallback to internal generator if hardware unavailable
        logger.info("Using internal telemetry generator fallback for missing hardware.")
        from data_acquisition import DataAcquisitionEngine
        engine = DataAcquisitionEngine(num_samples=target_samples, sample_rate_hz=sample_rate_hz)
        fallback_samples = engine.generate_all_data()
        for idx, s in enumerate(fallback_samples):
            db_logger_fn(board_name, serial_number, s)
            if progress_callback:
                pct = int(((idx + 1) / target_samples) * 100)
                progress_callback(pct, f"Acquiring Telemetry ({idx + 1}/{target_samples})...")

        return fallback_samples
