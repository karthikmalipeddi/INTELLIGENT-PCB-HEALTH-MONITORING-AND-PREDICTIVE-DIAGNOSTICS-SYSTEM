"""
diagnostic_engine.py
====================
Component-Level Failure Detection Engine for TEST GIG.

Processes a 30-second SQLite telemetry chunk using NumPy and SciPy to detect
specific component and subsystem failures based on industry reliability rules:

1. Capacitor Failure:
   - Run numpy.fft on high-frequency voltage data of INA power monitors.
   - Flag "ERR: Capacitor Degradation" if AC ripple voltage exceeds 100 mV (config.CAPACITOR_RIPPLE_THRESHOLD_V).

2. Voltage Regulator / Power Module Failure:
   - Monitor INA226 rails and 32 MUX channels.
   - Flag "ERR: Regulator Instability" if voltage fluctuates continuously (high std dev > config.REGULATOR_INSTABILITY_STD).
   - Flag "ERR: Voltage Drop" if a rail drops by > 10% (config.VOLTAGE_DROP_TOLERANCE) from its rated baseline.

3. MOSFET / Transformer Failure:
   - Parse MLX90640 thermal matrix.
   - Flag "ERR: MOSFET/Transformer Hotspot" if localized hotspots are 20-30°C higher
     than uniform background (config.MOSFET_HOTSPOT_DELTA_T) or if absolute max temp exceeds 70°C (config.MOSFET_MAX_TEMP_C).

4. Mechanical / Connector Failure:
   - Parse MPU6050 accelerometer and gyroscope data.
   - Flag "ERR: High Vibration / Loose Connector" if vibration amplitudes or acceleration
     spikes exceed standard stable baselines (> 1.6g, config.VIBRATION_SPIKE_G).

5. Environmental / Corrosion Risk:
   - Parse TMP117 and BMP280 data.
   - Flag "ERR: High Corrosion Risk" if ambient humidity exceeds 80% RH (config.CORROSION_HUMIDITY_THRESHOLD).
   - Flag "ERR: Ambient Overheat" if ambient temp exceeds 50°C (config.AMBIENT_OVERHEAT_THRESHOLD_C).

6. Sensor / Trace Failure:
   - Monitor ADC MUX channels.
   - Flag "ERR: Erratic Sensor/Trace" if continuous signals suddenly drop to zero
     (< config.SENSOR_TRACE_SIGNAL_LOSS_V) or exhibit extreme erratic amplitude swings (> config.SENSOR_TRACE_SWING_THRESHOLD).

7. System Level (Mahalanobis Distance):
   - Calculate Mahalanobis distance across voltage and current arrays to catch
     hidden multivariate correlation faults (D_M > config.MAHALANOBIS_THRESHOLD).

All triggered flags are merged with external YOLO vision defects and saved to
data_process_results_log.db.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple
import numpy as np

try:
    from scipy.spatial.distance import mahalanobis as _scipy_mahalanobis
except ImportError:
    def _scipy_mahalanobis(u, v, VI):
        diff = np.asarray(u) - np.asarray(v)
        return float(np.sqrt(np.dot(np.dot(diff, VI), diff.T)))

import config

logger = logging.getLogger("testgig.diagnostic_engine")


class DiagnosticEngine:
    """Component-Level Failure Detection Engine for TEST GIG PCB Testbenches."""

    # INA226 Power Monitor Rails
    INA_PAIRS: list[tuple[str, str, str]] = [
        ("INA1_VBUS", "INA1_ISHUNT", "INA1 (3.3V Rail)"),
        ("INA2_VBUS", "INA2_ISHUNT", "INA2 (5.0V Rail)"),
        ("INA3_VBUS", "INA3_ISHUNT", "INA3 (12.0V Rail)"),
    ]

    # All voltage rails monitored for stability & drop checks (derived from config)
    VOLTAGE_CHANNELS: list[str] = [
        ch for ch in config.ELECTRICAL_CHANNELS
        if "ISHUNT" not in ch and "POWER" not in ch
    ]

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Rule 1: Capacitor Failure (FFT Ripple Analysis)                    #
    # ------------------------------------------------------------------ #

    def check_capacitor_degradation(
        self,
        data: list[dict],
        sample_rate: float = float(config.TELEMETRY_SAMPLE_RATE_HZ),
    ) -> tuple[list[str], dict[str, Any]]:
        """Run numpy.fft on high-frequency voltage data of the INA power monitors.
        Flag 'ERR: Capacitor Degradation' if AC ripple voltage exceeds 100 mV.
        """
        faults: list[str] = []
        details: dict[str, Any] = {}

        for vbus_key, _, label in self.INA_PAIRS:
            if not data or vbus_key not in data[0]:
                continue

            signal = np.array([s[vbus_key] for s in data], dtype=np.float64)
            n_samples = len(signal)
            if n_samples < 4:
                continue

            # Peak-to-peak AC ripple voltage
            vpp = float(np.max(signal) - np.min(signal))

            # FFT analysis
            fft_coeff = np.fft.rfft(signal)
            freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
            magnitudes = np.abs(fft_coeff) / n_samples

            # Exclude DC component (index 0)
            non_dc_mags = magnitudes[1:]
            non_dc_freqs = freqs[1:]

            dom_freq = 0.0
            dom_mag = 0.0
            if len(non_dc_mags) > 0:
                max_idx = int(np.argmax(non_dc_mags))
                dom_freq = float(non_dc_freqs[max_idx])
                dom_mag = float(non_dc_mags[max_idx])

            threshold_exceeded = bool(vpp > config.CAPACITOR_RIPPLE_THRESHOLD_V)
            details[vbus_key] = {
                "vpp_v": vpp,
                "dominant_freq_hz": dom_freq,
                "dominant_mag": dom_mag,
                "exceeds_threshold": threshold_exceeded,
            }

            if threshold_exceeded:
                faults.append(
                    f"ERR: Capacitor Degradation ({label} AC Ripple Vpp={vpp*1000:.1f}mV > 100mV, f={dom_freq:.1f}Hz)"
                )

        return faults, details

    # ------------------------------------------------------------------ #
    # Rule 2: Voltage Regulator / Power Module Failure                   #
    # ------------------------------------------------------------------ #

    def check_voltage_regulators(
        self,
        data: list[dict],
    ) -> tuple[list[str], dict[str, Any]]:
        """Monitor INA226 and MUX arrays.
        Flag 'ERR: Regulator Instability' if voltage fluctuates continuously (std dev > threshold).
        Flag 'ERR: Voltage Drop' if a rail drops by > 10% from its rated baseline.
        """
        faults: list[str] = []
        details: dict[str, Any] = {}

        if not data:
            return faults, details

        for ch in self.VOLTAGE_CHANNELS:
            if ch not in data[0]:
                continue

            values = np.array([s[ch] for s in data], dtype=np.float64)
            if len(values) < 10:
                continue

            # Nominal baseline established from initial 30 samples (or first 20%)
            baseline_n = min(30, len(values) // 2)
            nominal = float(np.median(values[:baseline_n]))
            if abs(nominal) < 0.1:
                # Inactive / zero rail, skip drop checks
                continue

            std_dev = float(np.std(values))
            min_val = float(np.min(values))
            max_drop_pct = float((nominal - min_val) / nominal) * 100.0

            # 1. Check continuous fluctuation / instability
            is_unstable = std_dev > config.REGULATOR_INSTABILITY_STD
            if is_unstable:
                faults.append(
                    f"ERR: Regulator Instability ({ch} Continuous Fluctuation, σ={std_dev*1000:.1f}mV)"
                )

            # 2. Check rail voltage drop > 10%
            is_dropped = max_drop_pct > (config.VOLTAGE_DROP_TOLERANCE * 100.0)
            if is_dropped:
                faults.append(
                    f"ERR: Voltage Drop ({ch} {nominal:.2f}V→{min_val:.2f}V, Drop={max_drop_pct:.1f}% > 10%)"
                )

            details[ch] = {
                "nominal_v": nominal,
                "min_v": min_val,
                "std_dev_v": std_dev,
                "drop_pct": max_drop_pct,
                "is_unstable": is_unstable,
                "is_dropped": is_dropped,
            }

        return faults, details

    # ------------------------------------------------------------------ #
    # Rule 3: MOSFET / Transformer Thermal Hotspots                      #
    # ------------------------------------------------------------------ #

    def check_mosfet_transformer_hotspots(
        self,
        thermal_matrix: Optional[np.ndarray],
    ) -> tuple[list[str], dict[str, Any]]:
        """Parse MLX90640 thermal data.
        Flag 'ERR: MOSFET/Transformer Hotspot' if localized hotspots are 20-30°C higher
        than uniform background or if absolute max temp exceeds 70°C.
        """
        faults: list[str] = []
        details: dict[str, Any] = {}

        if thermal_matrix is None:
            return faults, {"status": "NO_THERMAL_DATA"}

        matrix = np.asarray(thermal_matrix, dtype=np.float32)
        if matrix.ndim == 1 and matrix.size == 768:
            matrix = matrix.reshape((24, 32))

        max_temp = float(np.max(matrix))
        min_temp = float(np.min(matrix))
        median_temp = float(np.median(matrix))
        hotspot_delta_t = max_temp - median_temp

        details = {
            "max_temp_c": max_temp,
            "min_temp_c": min_temp,
            "median_temp_c": median_temp,
            "hotspot_delta_t": hotspot_delta_t,
        }

        # Check conditions
        if max_temp > config.MOSFET_MAX_TEMP_C:
            faults.append(
                f"ERR: MOSFET/Transformer Hotspot (Peak Max Temp={max_temp:.1f}°C > {config.MOSFET_MAX_TEMP_C:.0f}°C)"
            )
        elif hotspot_delta_t >= config.MOSFET_HOTSPOT_DELTA_T:
            faults.append(
                f"ERR: MOSFET/Transformer Hotspot (Hotspot ΔT={hotspot_delta_t:.1f}°C above {median_temp:.1f}°C background)"
            )

        return faults, details

    # ------------------------------------------------------------------ #
    # Rule 4: Mechanical / Connector Failure (MPU6050 Vibration)         #
    # ------------------------------------------------------------------ #

    def check_mechanical_vibration(
        self,
        data: list[dict],
    ) -> tuple[list[str], dict[str, Any]]:
        """Parse MPU6050 accelerometer and gyroscope data.
        Flag 'ERR: High Vibration / Loose Connector' if vibration amplitudes or acceleration
        spikes exceed standard stable baselines (> 1.6g).
        """
        faults: list[str] = []
        details: dict[str, Any] = {}

        if not data or "MPU6050_VIB" not in data[0]:
            return faults, {"status": "NO_MPU6050_DATA"}

        vib_values = np.array([s.get("MPU6050_VIB", 1.0) for s in data], dtype=np.float64)
        peak_vib = float(np.max(vib_values))
        mean_vib = float(np.mean(vib_values))

        details = {
            "peak_vibration_g": peak_vib,
            "mean_vibration_g": mean_vib,
            "threshold_g": config.VIBRATION_SPIKE_G,
        }

        if peak_vib > config.VIBRATION_SPIKE_G:
            faults.append(
                f"ERR: High Vibration / Loose Connector (MPU6050 Peak Accel={peak_vib:.2f}g > {config.VIBRATION_SPIKE_G}g)"
            )

        return faults, details

    # ------------------------------------------------------------------ #
    # Rule 5: Environmental / Corrosion Risk (TMP117 & BMP280)           #
    # ------------------------------------------------------------------ #

    def check_environmental_risks(
        self,
        data: list[dict],
    ) -> tuple[list[str], dict[str, Any]]:
        """Parse TMP117 and BMP280 data.
        Flag 'ERR: High Corrosion Risk' if ambient humidity exceeds 80% RH.
        Flag 'ERR: Ambient Overheat' if ambient temp exceeds 50°C.
        """
        faults: list[str] = []
        details: dict[str, Any] = {}

        if not data:
            return faults, details

        # Ambient Temperature (TMP117 / TEMP_AMB)
        temp_readings = [
            s.get("TMP117_TEMP", s.get("TEMP_AMB", 25.0)) for s in data
        ]
        max_temp = float(np.max(temp_readings))
        min_temp = float(np.min(temp_readings))

        # Relative Humidity
        humidity_readings = [s.get("HUMIDITY", 50.0) for s in data]
        max_humidity = float(np.max(humidity_readings))

        details = {
            "max_ambient_temp_c": max_temp,
            "min_ambient_temp_c": min_temp,
            "max_humidity_pct": max_humidity,
        }

        # 1. High Corrosion Risk
        if max_humidity > config.CORROSION_HUMIDITY_THRESHOLD:
            faults.append(
                f"ERR: High Corrosion Risk (Ambient Humidity={max_humidity:.1f}% RH > {config.CORROSION_HUMIDITY_THRESHOLD:.0f}%)"
            )

        # 2. Ambient Overheat
        if max_temp > config.AMBIENT_OVERHEAT_THRESHOLD_C:
            faults.append(
                f"ERR: Ambient Overheat (TMP117/BMP280 Temp={max_temp:.1f}°C > {config.AMBIENT_OVERHEAT_THRESHOLD_C:.0f}°C)"
            )

        return faults, details

    # ------------------------------------------------------------------ #
    # Rule 6: Sensor / Trace Failure (ADC & MUX Channels)                #
    # ------------------------------------------------------------------ #

    def check_sensor_trace_integrity(
        self,
        data: list[dict],
    ) -> tuple[list[str], dict[str, Any]]:
        """Monitor ADC MUX channels.
        Flag 'ERR: Erratic Sensor/Trace' if continuous signals suddenly drop to zero
        (signal loss) or exhibit extreme erratic amplitude swings.
        """
        faults: list[str] = []
        details: dict[str, Any] = {}

        if not data:
            return faults, details

        # Focus on ADC and MUX trace channels
        target_channels = [
            ch for ch in config.ELECTRICAL_CHANNELS
            if ch.startswith("ADC") or ch.startswith("MUX")
        ]

        for ch in target_channels:
            if ch not in data[0]:
                continue

            values = np.array([s[ch] for s in data], dtype=np.float64)
            if len(values) < 20:
                continue

            baseline = float(np.median(values[:20]))
            if baseline < 0.3:
                # Channel naturally off / GND trace
                continue

            min_val = float(np.min(values))

            # 1. Sudden signal loss to GND (drop to near zero from active rail)
            if baseline >= 0.5 and min_val < config.SENSOR_TRACE_SIGNAL_LOSS_V:
                faults.append(
                    f"ERR: Erratic Sensor/Trace ({ch} Signal Loss: Active {baseline:.2f}V dropped to {min_val:.2f}V)"
                )
                details[ch] = {"status": "SIGNAL_LOSS", "baseline": baseline, "min_val": min_val}
                continue

            # 2. Erratic jump (maximum sample-to-sample delta)
            diffs = np.abs(np.diff(values))
            max_step = float(np.max(diffs))
            if max_step > (baseline * config.SENSOR_TRACE_SWING_THRESHOLD):
                faults.append(
                    f"ERR: Erratic Sensor/Trace ({ch} Erratic Jump: Step Δ={max_step:.2f}V on {baseline:.2f}V rail)"
                )
                details[ch] = {"status": "ERRATIC_SWING", "baseline": baseline, "max_step": max_step}

        return faults, details

    # ------------------------------------------------------------------ #
    # Rule 7: System Level Multivariate Mahalanobis Distance             #
    # ------------------------------------------------------------------ #

    def check_mahalanobis_system_outliers(
        self,
        data: list[dict],
    ) -> tuple[list[str], dict[str, Any]]:
        """Calculate Mahalanobis distance across voltage and current arrays
        to catch hidden multi-variate faults.
        """
        faults: list[str] = []
        scores: dict[str, Any] = {}

        if not data:
            return faults, scores

        for vbus_key, ishunt_key, label in self.INA_PAIRS:
            if vbus_key not in data[0] or ishunt_key not in data[0]:
                continue

            voltage = np.array([s[vbus_key] for s in data], dtype=np.float64)
            current = np.array([s[ishunt_key] for s in data], dtype=np.float64)
            X = np.column_stack((voltage, current))

            mu = np.mean(X, axis=0)
            cov = np.cov(X, rowvar=False) + np.eye(2) * 1e-12
            cov_inv = np.linalg.inv(cov)

            distances = np.array(
                [_scipy_mahalanobis(x, mu, cov_inv) for x in X],
                dtype=np.float64,
            )
            max_dist = float(np.max(distances))

            scores[label] = {
                "max_distance": max_dist,
                "threshold": config.MAHALANOBIS_THRESHOLD,
            }

            if max_dist > config.MAHALANOBIS_THRESHOLD:
                faults.append(
                    f"ERR: {label} Mahalanobis Multivariate Anomaly (D={max_dist:.2f} > {config.MAHALANOBIS_THRESHOLD})"
                )

        return faults, scores

    # ------------------------------------------------------------------ #
    # Master Execution: Run All Diagnostic Rules                         #
    # ------------------------------------------------------------------ #

    def run_full_diagnostics(
        self,
        data: list[dict],
        thermal_matrix: Optional[np.ndarray] = None,
        sample_rate: float = float(config.TELEMETRY_SAMPLE_RATE_HZ),
        yolo_defects: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Execute all 7 component-level failure rules + merge YOLO visual defects.

        Returns
        -------
        dict[str, Any]
            Comprehensive diagnostic results dictionary with categorized reports.
        """
        # Execute each rule
        cap_faults, cap_details = self.check_capacitor_degradation(data, sample_rate)
        reg_faults, reg_details = self.check_voltage_regulators(data)
        th_faults, th_details = self.check_mosfet_transformer_hotspots(thermal_matrix)
        mech_faults, mech_details = self.check_mechanical_vibration(data)
        env_faults, env_details = self.check_environmental_risks(data)
        trace_faults, trace_details = self.check_sensor_trace_integrity(data)
        mah_faults, mah_details = self.check_mahalanobis_system_outliers(data)

        # Merge all triggered component failure faults
        merged_faults: list[str] = []
        merged_faults.extend(cap_faults)
        merged_faults.extend(reg_faults)
        merged_faults.extend(th_faults)
        merged_faults.extend(mech_faults)
        merged_faults.extend(env_faults)
        merged_faults.extend(trace_faults)
        merged_faults.extend(mah_faults)

        # Merge external YOLO / SAHI visual defect detections
        if yolo_defects:
            merged_faults.extend(yolo_defects)

        # Electrical anomaly flag (used by Prognostic Engine for RUL penalty)
        has_electrical_anomaly = (
            len(cap_faults) > 0
            or len(reg_faults) > 0
            or len(mah_faults) > 0
            or len(trace_faults) > 0
        )

        return {
            "faults": merged_faults,
            "has_electrical_anomaly": has_electrical_anomaly,
            "categorized_failures": {
                "capacitor": {"faults": cap_faults, "details": cap_details},
                "voltage_regulator": {"faults": reg_faults, "details": reg_details},
                "mosfet_transformer": {"faults": th_faults, "details": th_details},
                "mechanical_connector": {"faults": mech_faults, "details": mech_details},
                "environmental_corrosion": {"faults": env_faults, "details": env_details},
                "sensor_trace": {"faults": trace_faults, "details": trace_details},
                "system_mahalanobis": {"faults": mah_faults, "details": mah_details},
                "yolo_vision_defects": {"faults": yolo_defects or []},
            },
            "mahalanobis_scores": mah_details,
            "fft_scores": cap_details,
            "threshold_flags": reg_details,
        }
