"""
data_acquisition.py
===================
TEST GIG - 43-Channel Electrical + Environmental Sensor Telemetry Engine.
Generates realistic 43-channel electrical + environmental sensor telemetry
mimicking a 30-second window sampled at 10 Hz (300 samples).

Telemetry Channel Map:
1. Environmental & Inertial Sensors:
   - TMP117: TMP117_TEMP (High precision temp), TEMP_AMB
   - BMP280: BMP280_TEMP, BMP280_PRESS, PRESSURE
   - Humidity: HUMIDITY (% RH)
   - MPU6050: MPU6050_ACC_X, MPU6050_ACC_Y, MPU6050_ACC_Z, MPU6050_VIB (g)
2. 43 Electrical Channels:
   - 9x Power Monitor signals (3x INA226 sensors):
       * INA1 (3.3V Rail): INA1_VBUS, INA1_ISHUNT, INA1_POWER
       * INA2 (5.0V Rail): INA2_VBUS, INA2_ISHUNT, INA2_POWER
       * INA3 (12.0V Rail): INA3_VBUS, INA3_ISHUNT, INA3_POWER
   - 2x Direct ADCs: ADC_CH0, ADC_CH1
   - 32x MUX Channels: MUX_CH00 … MUX_CH31

Injected Diagnostic Anomalies (for Mock Mode testing):
- Capacitor Degradation: AC ripple on INA1_VBUS (Vpp ≈ 190 mV > 100 mV)
- Regulator Instability: High continuous fluctuation on INA1_VBUS
- Voltage Drop: MUX_CH07 drops ~15% (3.30V → 2.77V)
- Ambient Overheat: TMP117_TEMP rises to ~62°C (> 50°C)
- High Corrosion Risk: HUMIDITY climbs to ~84% (> 80%)
- High Vibration / Loose Connector: MPU6050 vibration spikes to ~1.9g (> 1.6g)
- Erratic Sensor / Trace: MUX_CH23 drops to 0.0V (signal loss)
- System Mahalanobis: Correlation outlier on INA1/INA2 pairs
"""

from __future__ import annotations

import numpy as np

import config


class DataAcquisitionEngine:
    """43-Channel Electrical + Environmental Telemetry Engine for TEST GIG."""

    # 43 Electrical Channels loaded directly from centralized config
    ELECTRICAL_CHANNELS: list[str] = list(config.ELECTRICAL_CHANNELS)
    CHANNEL_NAMES: list[str] = ELECTRICAL_CHANNELS  # Backwards compatibility alias

    # Power channels, ADC channels, and MUX channels partitioned for convenience
    POWER_CHANNELS: list[str] = [
        ch for ch in ELECTRICAL_CHANNELS if ch.startswith("INA")
    ]
    ADC_CHANNELS: list[str] = [
        ch for ch in ELECTRICAL_CHANNELS if ch.startswith("ADC")
    ]
    MUX_CHANNELS: list[str] = [
        ch for ch in ELECTRICAL_CHANNELS if ch.startswith("MUX")
    ]

    # Environmental and Inertial channels
    ENV_CHANNELS: list[str] = [
        "TMP117_TEMP", "TEMP_AMB", "BMP280_TEMP", "BMP280_PRESS", "PRESSURE",
        "HUMIDITY", "MPU6050_ACC_X", "MPU6050_ACC_Y", "MPU6050_ACC_Z", "MPU6050_VIB",
    ]

    ALL_TELEMETRY_KEYS: list[str] = ENV_CHANNELS + ELECTRICAL_CHANNELS

    def __init__(
        self,
        num_samples: int = config.NUM_MOCK_SAMPLES,
        sample_rate_hz: int = config.TELEMETRY_SAMPLE_RATE_HZ,
    ) -> None:
        self.num_samples = num_samples
        self.sample_rate_hz = sample_rate_hz
        self._t = np.arange(num_samples) / sample_rate_hz

    def generate_all_data(self) -> list[dict]:
        """Generate 300 samples of mock telemetry with realistic injected anomalies.

        Returns
        -------
        list[dict]
            List of 300 dictionaries, each containing all electrical and environmental keys.
        """
        n = self.num_samples
        t = self._t

        data: dict[str, np.ndarray] = {}

        # --------------------------------------------------------------
        # 1. Environmental Telemetry (TMP117, BMP280, Humidity, MPU6050)
        # --------------------------------------------------------------
        # TMP117: Base 28°C climbing quadratically to 62°C (> 50°C Overheat anomaly)
        ramp = (np.linspace(0, 1, n) ** 2)
        tmp117_temp = 28.0 + 34.0 * ramp + np.random.normal(0, 0.02, n)
        data["TMP117_TEMP"] = tmp117_temp
        data["TEMP_AMB"] = tmp117_temp  # Standard alias

        # BMP280: Temperature and Pressure
        data["BMP280_TEMP"] = tmp117_temp - 0.4 + np.random.normal(0, 0.05, n)
        data["BMP280_PRESS"] = 1013.25 + 0.8 * np.sin(2 * np.pi * 0.05 * t) + np.random.normal(0, 0.1, n)
        data["PRESSURE"] = data["BMP280_PRESS"]

        # Humidity: Base 55% drifting up to 84% near end (> 80% Corrosion Risk anomaly)
        humidity = 55.0 + 29.0 * (np.linspace(0, 1, n) ** 1.5) + np.random.normal(0, 0.3, n)
        data["HUMIDITY"] = humidity

        # MPU6050: Acceleration and Vibration
        # Base stable ~1.0g with an injected shock/vibration spike (> 1.6g) around sample 180
        acc_x = 0.02 + np.random.normal(0, 0.03, n)
        acc_y = 0.01 + np.random.normal(0, 0.03, n)
        acc_z = 1.00 + np.random.normal(0, 0.04, n)

        # Inject high vibration spike around midpoint (Loose Connector / Vibration Anomaly)
        vib_spike = np.exp(-((t - 18.0) ** 2) / 0.8) * 0.95
        acc_z += vib_spike
        vibration_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        data["MPU6050_ACC_X"] = acc_x
        data["MPU6050_ACC_Y"] = acc_y
        data["MPU6050_ACC_Z"] = acc_z
        data["MPU6050_VIB"] = vibration_magnitude

        # --------------------------------------------------------------
        # 2. Power Monitors (9 channels across 3 rails)
        # --------------------------------------------------------------
        # INA1 (3.3V rail): INJECTED 2.5 Hz AC RIPPLE (0.19V pk-pk > 100mV)
        # Simulates capacitor ESR degradation & regulator instability
        ripple_freq_hz = 2.5
        ripple_amplitude = 0.09  # 0.18 V pk-pk ripple
        ripple = ripple_amplitude * np.sin(2 * np.pi * ripple_freq_hz * t)
        instability_noise = np.random.normal(0, 0.015, n)

        ina1_vbus = 3.30 + ripple + instability_noise
        ina1_ishunt = 150.0 + 8.0 * np.sin(2 * np.pi * ripple_freq_hz * t) + np.random.normal(0, 0.5, n)
        ina1_power = ina1_vbus * ina1_ishunt

        data["INA1_VBUS"] = ina1_vbus
        data["INA1_ISHUNT"] = ina1_ishunt
        data["INA1_POWER"] = ina1_power

        # INA2 (5.0V rail — healthy nominal)
        ina2_vbus = 5.00 + np.random.normal(0, 0.005, n)
        ina2_ishunt = 250.0 + np.random.normal(0, 0.8, n)
        ina2_power = ina2_vbus * ina2_ishunt
        data["INA2_VBUS"] = ina2_vbus
        data["INA2_ISHUNT"] = ina2_ishunt
        data["INA2_POWER"] = ina2_power

        # INA3 (12.0V rail — healthy nominal)
        ina3_vbus = 12.00 + np.random.normal(0, 0.01, n)
        ina3_ishunt = 450.0 + np.random.normal(0, 1.2, n)
        ina3_power = ina3_vbus * ina3_ishunt
        data["INA3_VBUS"] = ina3_vbus
        data["INA3_ISHUNT"] = ina3_ishunt
        data["INA3_POWER"] = ina3_power

        # --------------------------------------------------------------
        # 3. Direct ADCs (2 channels)
        # --------------------------------------------------------------
        data["ADC_CH0"] = 2.50 + np.random.normal(0, 0.006, n)
        data["ADC_CH1"] = 1.80 + np.random.normal(0, 0.005, n)

        # --------------------------------------------------------------
        # 4. 32 MUX Channels
        # --------------------------------------------------------------
        base_voltages = [
            0.8, 1.0, 1.2, 1.5, 1.8, 2.5, 3.0, 3.3,
            3.3, 3.3, 5.0, 5.0, 1.1, 1.3, 2.8, 3.3,
            1.2, 1.8, 2.5, 3.3, 5.0, 0.9, 1.5, 2.5,
            3.3, 3.3, 5.0, 1.8, 2.5, 3.3, 4.5, 5.0,
        ]

        for i, ch_name in enumerate(self.MUX_CHANNELS):
            base_v = base_voltages[i % len(base_voltages)]
            rail = np.full(n, base_v, dtype=np.float64) + np.random.normal(0, 0.004, n)

            # ANOMALY: MUX_CH07 step voltage drop (~15% drop from 3.3V to 2.77V at sample 150)
            if i == 7:
                rail[150:] -= 0.53

            # ANOMALY: MUX_CH23 signal loss / erratic trace drop to 0.0V at sample 210
            if i == 23:
                rail[210:] = 0.01 + np.random.normal(0, 0.003, n - 210)

            data[ch_name] = rail

        # Transpose dictionary of arrays into list of dicts (one dict per time sample)
        samples: list[dict] = []
        all_keys = list(data.keys())
        for idx in range(n):
            sample = {k: float(data[k][idx]) for k in all_keys}
            samples.append(sample)

        return samples

    @staticmethod
    def generate_mock_thermal_matrix() -> np.ndarray:
        """Generate a 32x24 mock thermal matrix with an injected MOSFET hotspot.

        Background is ~27.0°C.
        Hotspot at row 10, col 16 is 71.5°C (> 70°C and > 20°C above background),
        triggering the MOSFET/Transformer failure rule.
        """
        matrix = np.full((24, 32), 27.0, dtype=np.float32) + np.random.normal(0, 0.2, (24, 32))
        # Add localized Gaussian hotspot (MOSFET Q3)
        for r in range(24):
            for c in range(32):
                dist = np.sqrt((r - 10)**2 + (c - 16)**2)
                matrix[r, c] += float(44.5 * np.exp(-(dist**2) / 3.5))
        return matrix
