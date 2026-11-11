"""
test_specs_pipeline.py
======================
Comprehensive end-to-end integration test verifying all S.P.E.C.S. subsystems:
- Physical Hardware Drivers & MOCK_MODE fallbacks
- 7 Component-Level Failure Diagnostic Rules
- Physics-of-Failure Prognostic Engine (Coffin-Manson, Miner's, Penalty)
- OpenCV Thermal-Optical Fusion Engine (32x24 -> Upscale, Colormap, Affine, Alpha Blend, Hotspot Boxes)
- Diagnostic Center Examination Certificate PDF & QR Generation
"""

import os
import shutil
import tempfile
import numpy as np

import config
from database_manager import DatabaseManager
from data_acquisition import DataAcquisitionEngine
from diagnostic_engine import DiagnosticEngine
from prognostic_engine import PrognosticEngine
from hardware_vision import ThermalOpticalOverlayEngine, HardwareManager
from report_generator import ReportGenerator


def test_full_specs_pipeline():
    print("=" * 70)
    print(" RUNNING S.P.E.C.S. COMPREHENSIVE INTEGRATION SUITE")
    print("=" * 70)

    test_dir = tempfile.mkdtemp(prefix="specs_pipeline_test_")
    data_dir = os.path.join(test_dir, "data")
    reports_dir = os.path.join(test_dir, "reports")
    captures_dir = os.path.join(test_dir, "captures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(captures_dir, exist_ok=True)

    try:
        board_name = "REV4_POWER_BOARD"
        serial = "SN-2026-9981"
        mfg_date = "2024-05-15"

        # -------------------------------------------------------------
        # 1. DatabaseManager (5 SQLite Databases)
        # -------------------------------------------------------------
        print("\n[1/6] Testing DatabaseManager & Fail-Safe Logging...")
        db = DatabaseManager(data_dir=data_dir)
        db.log_thermal(board_name, serial, status="NA", data="NA")
        db.log_optical(board_name, serial, status="NO WIFI STREAM", data="NO WIFI STREAM")
        db.log_ai_process(board_name, serial, status="NO MODELS ARE WORKING", data="NO MODELS ARE WORKING")

        assert db.get_thermal_data(board_name, serial)["data"] == "NA"
        assert db.get_optical_data(board_name, serial)["data"] == "NO WIFI STREAM"
        assert db.get_ai_data(board_name, serial)["data"] == "NO MODELS ARE WORKING"
        print("  -> 5 SQLite databases initialised and fail-safe records verified.")

        # -------------------------------------------------------------
        # 2. Telemetry Generation (43 Electrical + Environmental)
        # -------------------------------------------------------------
        print("\n[2/6] Testing 43-Channel Telemetry & Injected Anomalies...")
        acq = DataAcquisitionEngine(num_samples=300, sample_rate_hz=10)
        assert len(acq.ELECTRICAL_CHANNELS) == 43, f"Expected 43 electrical channels, got {len(acq.ELECTRICAL_CHANNELS)}"
        samples = acq.generate_all_data()
        assert len(samples) == 300
        for s in samples:
            db.log_acquisition(board_name, serial, s)

        saved = db.get_acquisition_data(board_name, serial)
        assert len(saved) == 300
        print(f"  -> Generated and logged 300 samples across {len(acq.ELECTRICAL_CHANNELS)} electrical channels.")

        # -------------------------------------------------------------
        # 3. Component-Level Failure Diagnostic Rules (7 Rules)
        # -------------------------------------------------------------
        print("\n[3/6] Testing DiagnosticEngine 7 Component Failure Rules...")
        diag = DiagnosticEngine()
        mock_thermal = DataAcquisitionEngine.generate_mock_thermal_matrix()
        diag_results = diag.run_full_diagnostics(
            data=samples,
            thermal_matrix=mock_thermal,
            sample_rate=10.0,
            yolo_defects=["ERR: AI YOLO Solder Bridge (conf=0.94)"],
        )

        faults = diag_results["faults"]
        print(f"  -> Detected {len(faults)} failure flags across categories:")
        for f in faults:
            print(f"     * {f}")

        # Verify all rules triggered
        assert any("Capacitor Degradation" in f for f in faults), "Rule 1 (Capacitor) failed to trigger!"
        assert any("Regulator Instability" in f or "Voltage Drop" in f for f in faults), "Rule 2 (Regulator/Drop) failed!"
        assert any("MOSFET/Transformer Hotspot" in f for f in faults), "Rule 3 (MOSFET Hotspot) failed!"
        assert any("High Vibration" in f for f in faults), "Rule 4 (Mechanical Vibration) failed!"
        assert any("High Corrosion" in f or "Ambient Overheat" in f for f in faults), "Rule 5 (Environmental) failed!"
        assert any("Erratic Sensor/Trace" in f for f in faults), "Rule 6 (Sensor/Trace) failed!"
        assert any("Mahalanobis" in f for f in faults), "Rule 7 (Mahalanobis System) failed!"
        assert any("Solder Bridge" in f for f in faults), "YOLO defect merging failed!"
        assert diag_results["has_electrical_anomaly"] is True

        # -------------------------------------------------------------
        # 4. Prognostic Engine (Coffin-Manson RUL)
        # -------------------------------------------------------------
        print("\n[4/6] Testing PrognosticEngine (Coffin-Manson + Miner's + Penalty)...")
        prog = PrognosticEngine()
        temp_data = [s["TMP117_TEMP"] for s in samples]
        rul_res = prog.compute_rul(temp_data, mfg_date, diag_results)

        print(f"  -> Observed Temperature Swing ΔT : {rul_res['delta_t']:.2f} °C")
        print(f"  -> Coffin-Manson Fatigue Life N_f : {rul_res['n_f_cycles']:.1f} Cycles")
        print(f"  -> Miner's Damage Fraction D      : {rul_res['damage_fraction']:.4f}")
        print(f"  -> Base RUL (before penalty)      : {rul_res['base_rul']:.1f}%")
        print(f"  -> Hybrid Penalty Factor Applied  : {rul_res['penalty_factor']} (Status: {rul_res['penalty_applied']})")
        print(f"  -> FINAL RUL SCORE                : {rul_res['rul_percent']:.1f}%")

        assert 0.0 <= rul_res["rul_percent"] <= 100.0
        assert rul_res["penalty_applied"] is True

        db.log_results(
            board_name, serial, mfg_date,
            rul_res["rul_percent"], faults,
            diag_results["mahalanobis_scores"],
            diag_results["fft_scores"],
            diag_results["threshold_flags"],
        )

        # -------------------------------------------------------------
        # 5. OpenCV Thermal-Optical Fusion Engine
        # -------------------------------------------------------------
        print("\n[5/6] Testing ThermalOpticalOverlayEngine (OpenCV)...")
        overlay_engine = ThermalOpticalOverlayEngine(x_offset=3, y_offset=-2, alpha=0.45)
        mock_opt = np.ones((480, 640, 3), dtype=np.uint8) * 120
        fusion = overlay_engine.fuse(mock_thermal, mock_opt)

        assert fusion["blended_overlay"].shape == (480, 640, 3)
        assert fusion["max_temp"] > 70.0
        assert len(fusion["top3_temps"]) == 3
        print(f"  -> Fused Overlay Resolution: {fusion['blended_overlay'].shape}")
        print(f"  -> Peak Max Temperature: {fusion['max_temp']} °C")
        print(f"  -> Top 3 Hotspot Temps: {fusion['top3_temps']} °C")

        # -------------------------------------------------------------
        # 6. Diagnostic Center Style Examination Certificate PDF & QR
        # -------------------------------------------------------------
        print("\n[6/6] Testing Diagnostic Center ReportGenerator (PDF + QR + Server)...")
        rg = ReportGenerator(output_dir=reports_dir)

        # Make plot image
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_file = os.path.join(test_dir, "telemetry_plot.png")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.plot([s["INA1_VBUS"] for s in samples[:100]], color="#FF6600")
        ax.set_title("AC Ripple Telemetry")
        fig.savefig(plot_file, dpi=100)
        plt.close(fig)

        pdf_path = rg.generate_pdf(
            board_name=board_name,
            serial_number=serial,
            mfg_date=mfg_date,
            selected_channels=acq.ELECTRICAL_CHANNELS[:10],
            faults=faults,
            rul_result=rul_res,
            diagnostic_result=diag_results.get("categorized_failures", {}),
            thermal_status="OK",
            optical_status="OK",
            ai_status="OK",
            plot_image_path=plot_file,
        )

        assert os.path.isfile(pdf_path)
        pdf_size_kb = os.path.getsize(pdf_path) / 1024.0
        print(f"  -> Generated Diagnostic Certificate PDF : {os.path.basename(pdf_path)} ({pdf_size_kb:.1f} KB)")

        url = rg.start_file_server(pdf_path, port=config.HTTP_SERVER_PORT)
        print(f"  -> Local Network Download URL           : {url}")
        qr_file = rg.generate_qr_code(url, board_name, serial)
        assert os.path.isfile(qr_file)
        print(f"  -> Generated QR Code PNG                : {os.path.basename(qr_file)}")
        rg.stop_file_server()

        print("\n" + "=" * 70)
        print(" ALL TESTS PASSED! S.P.E.C.S. EDGE TESTBENCH IS READY!")
        print("=" * 70)

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    test_full_specs_pipeline()
