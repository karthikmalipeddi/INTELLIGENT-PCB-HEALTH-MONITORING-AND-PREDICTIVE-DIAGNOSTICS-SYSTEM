"""
database_manager.py
====================
Manages five separate SQLite databases for the TEST GIG system:
    1. thermal_log.db       — Thermal camera logs (Fail-safe: "NA" or full 4-metric metadata)
    2. optical_log.db       — Optical camera logs (Fail-safe: "NO WIFI STREAM" or image path/res)
    3. ai_process_log.db    — AI processing logs (Fail-safe: "NO MODELS ARE WORKING" or YOLO/SAHI)
    4. data_acq_log.db      — Raw 43-channel acquisition data (JSON per sample)
    5. data_process_results_log.db — Post-processing results (RUL, faults, scores)

Five distinct SQLite databases prevent thread contention and cross-locking
during high-speed acquisition and background analytics.
"""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
from typing import Any, Optional

import config


class DatabaseManager:
    """Central gateway to every persistent data store used by TEST GIG."""

    _THERMAL_DB = config.THERMAL_DB_NAME
    _OPTICAL_DB = config.OPTICAL_DB_NAME
    _AI_PROCESS_DB = config.AI_PROCESS_DB_NAME
    _DATA_ACQ_DB = config.DATA_ACQ_DB_NAME
    _RESULTS_DB = config.RESULTS_DB_NAME

    def __init__(self, data_dir: str = config.DATA_DIR) -> None:
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        self._init_thermal_db()
        self._init_optical_db()
        self._init_ai_process_db()
        self._init_acquisition_db()
        self._init_results_db()

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _db_path(self, filename: str) -> str:
        """Return the full path to a database file inside data_dir."""
        return os.path.join(self.data_dir, filename)

    @staticmethod
    def _now() -> str:
        """ISO-8601 timestamp for the current instant."""
        return datetime.datetime.now().isoformat()

    # ------------------------------------------------------------------ #
    #  Table initialisation                                                #
    # ------------------------------------------------------------------ #

    def _init_thermal_db(self) -> None:
        """Initialise thermal_log table.
        Must store:
          1) The thermal image with 3 red bounding boxes + text for Top 3 hotspots
          2) The Blended Overlay image
          3) The Top 3 hotspot temperatures
          4) The absolute highest and lowest temp values
        Fail-safe fallback: config.FAILSAFE_THERMAL ("NA")
        """
        with sqlite3.connect(self._db_path(self._THERMAL_DB)) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS thermal_log (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp           TEXT    NOT NULL,
                    board_name          TEXT    NOT NULL,
                    serial_number       TEXT    NOT NULL,
                    status              TEXT    DEFAULT '{config.FAILSAFE_THERMAL}',
                    thermal_image_path  TEXT,
                    overlay_image_path  TEXT,
                    top3_hotspot_temps  TEXT,
                    max_temp            REAL,
                    min_temp            REAL,
                    x_offset            INTEGER DEFAULT 0,
                    y_offset            INTEGER DEFAULT 0,
                    data                TEXT    DEFAULT '{config.FAILSAFE_THERMAL}'
                );
                """
            )
            conn.commit()

    def _init_optical_db(self) -> None:
        """Initialise optical_log table.
        Stores raw optical image path and resolution.
        Fail-safe fallback: config.FAILSAFE_OPTICAL ("NO WIFI STREAM")
        """
        with sqlite3.connect(self._db_path(self._OPTICAL_DB)) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS optical_log (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp           TEXT    NOT NULL,
                    board_name          TEXT    NOT NULL,
                    serial_number       TEXT    NOT NULL,
                    status              TEXT    DEFAULT '{config.FAILSAFE_OPTICAL}',
                    optical_image_path  TEXT,
                    resolution          TEXT,
                    data                TEXT    DEFAULT '{config.FAILSAFE_OPTICAL}'
                );
                """
            )
            conn.commit()

    def _init_ai_process_db(self) -> None:
        """Initialise ai_process_log table.
        Stores YOLO/SAHI API responses and decoded image paths.
        Fail-safe fallback: config.FAILSAFE_AI ("NO MODELS ARE WORKING")
        """
        with sqlite3.connect(self._db_path(self._AI_PROCESS_DB)) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS ai_process_log (
                    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp            TEXT    NOT NULL,
                    board_name           TEXT    NOT NULL,
                    serial_number        TEXT    NOT NULL,
                    status               TEXT    DEFAULT '{config.FAILSAFE_AI}',
                    component_image_path TEXT,
                    defect_image_path    TEXT,
                    api_response_json    TEXT,
                    defect_list          TEXT,
                    data                 TEXT    DEFAULT '{config.FAILSAFE_AI}'
                );
                """
            )
            conn.commit()

    def _init_acquisition_db(self) -> None:
        """Create the acquisition_log table if it does not exist."""
        with sqlite3.connect(self._db_path(self._DATA_ACQ_DB)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS acquisition_log (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT    NOT NULL,
                    board_name    TEXT    NOT NULL,
                    serial_number TEXT    NOT NULL,
                    channel_data  TEXT    NOT NULL
                );
                """
            )
            conn.commit()

    def _init_results_db(self) -> None:
        """Create the results_log table if it does not exist."""
        with sqlite3.connect(self._db_path(self._RESULTS_DB)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS results_log (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp          TEXT    NOT NULL,
                    board_name         TEXT    NOT NULL,
                    serial_number      TEXT    NOT NULL,
                    mfg_date           TEXT,
                    rul_percent        REAL,
                    faults             TEXT,
                    mahalanobis_scores TEXT,
                    fft_scores         TEXT,
                    threshold_flags    TEXT
                );
                """
            )
            conn.commit()

    # ================================================================== #
    #  Thermal Log Operations                                              #
    # ================================================================== #

    def log_thermal(
        self,
        board_name: str,
        serial_number: str,
        thermal_image_path: Optional[str] = None,
        overlay_image_path: Optional[str] = None,
        top3_hotspot_temps: Optional[list[float]] = None,
        max_temp: Optional[float] = None,
        min_temp: Optional[float] = None,
        x_offset: int = config.X_OFFSET,
        y_offset: int = config.Y_OFFSET,
        status: str = config.FAILSAFE_THERMAL,
        data: str = config.FAILSAFE_THERMAL,
    ) -> None:
        """Insert thermal camera log record.

        In MOCK_MODE or hardware fail-safe, stores data='NA'.
        In production, stores annotated thermal image, blended overlay image,
        top 3 hotspot temperatures, and absolute highest/lowest temperatures.
        """
        top3_json = json.dumps(top3_hotspot_temps) if top3_hotspot_temps else "[]"
        with sqlite3.connect(self._db_path(self._THERMAL_DB)) as conn:
            conn.execute(
                """
                INSERT INTO thermal_log (
                    timestamp, board_name, serial_number, status,
                    thermal_image_path, overlay_image_path, top3_hotspot_temps,
                    max_temp, min_temp, x_offset, y_offset, data
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    self._now(),
                    board_name,
                    serial_number,
                    status,
                    thermal_image_path,
                    overlay_image_path,
                    top3_json,
                    max_temp,
                    min_temp,
                    x_offset,
                    y_offset,
                    data,
                ),
            )
            conn.commit()

    def get_thermal_data(self, board_name: str, serial_number: str) -> dict:
        """Retrieve most recent thermal log for a given board."""
        with sqlite3.connect(self._db_path(self._THERMAL_DB)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM thermal_log WHERE board_name = ? AND serial_number = ? ORDER BY id DESC LIMIT 1;",
                (board_name, serial_number),
            )
            row = cursor.fetchone()
        if not row:
            return {"status": config.FAILSAFE_THERMAL, "data": config.FAILSAFE_THERMAL}
        return dict(row)

    # ================================================================== #
    #  Optical Log Operations                                              #
    # ================================================================== #

    def log_optical(
        self,
        board_name: str,
        serial_number: str,
        optical_image_path: Optional[str] = None,
        resolution: Optional[str] = None,
        status: str = config.FAILSAFE_OPTICAL,
        data: str = config.FAILSAFE_OPTICAL,
    ) -> None:
        """Insert optical camera log record."""
        with sqlite3.connect(self._db_path(self._OPTICAL_DB)) as conn:
            conn.execute(
                """
                INSERT INTO optical_log (
                    timestamp, board_name, serial_number, status,
                    optical_image_path, resolution, data
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    self._now(),
                    board_name,
                    serial_number,
                    status,
                    optical_image_path,
                    resolution,
                    data,
                ),
            )
            conn.commit()

    def get_optical_data(self, board_name: str, serial_number: str) -> dict:
        """Retrieve most recent optical log for a given board."""
        with sqlite3.connect(self._db_path(self._OPTICAL_DB)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM optical_log WHERE board_name = ? AND serial_number = ? ORDER BY id DESC LIMIT 1;",
                (board_name, serial_number),
            )
            row = cursor.fetchone()
        if not row:
            return {"status": config.FAILSAFE_OPTICAL, "data": config.FAILSAFE_OPTICAL}
        return dict(row)

    # ================================================================== #
    #  AI Process Log Operations                                           #
    # ================================================================== #

    def log_ai_process(
        self,
        board_name: str,
        serial_number: str,
        component_image_path: Optional[str] = None,
        defect_image_path: Optional[str] = None,
        api_response: Optional[dict] = None,
        defect_list: Optional[list[str]] = None,
        status: str = config.FAILSAFE_AI,
        data: str = config.FAILSAFE_AI,
    ) -> None:
        """Insert AI processing log record."""
        api_json = json.dumps(api_response) if api_response else "{}"
        defects_json = json.dumps(defect_list) if defect_list else "[]"
        with sqlite3.connect(self._db_path(self._AI_PROCESS_DB)) as conn:
            conn.execute(
                """
                INSERT INTO ai_process_log (
                    timestamp, board_name, serial_number, status,
                    component_image_path, defect_image_path, api_response_json, defect_list, data
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    self._now(),
                    board_name,
                    serial_number,
                    status,
                    component_image_path,
                    defect_image_path,
                    api_json,
                    defects_json,
                    data,
                ),
            )
            conn.commit()

    def get_ai_data(self, board_name: str, serial_number: str) -> dict:
        """Retrieve most recent AI process log for a given board."""
        with sqlite3.connect(self._db_path(self._AI_PROCESS_DB)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM ai_process_log WHERE board_name = ? AND serial_number = ? ORDER BY id DESC LIMIT 1;",
                (board_name, serial_number),
            )
            row = cursor.fetchone()
        if not row:
            return {"status": config.FAILSAFE_AI, "data": config.FAILSAFE_AI}
        return dict(row)

    # ================================================================== #
    #  Data Acquisition Log Operations                                     #
    # ================================================================== #

    def log_acquisition(
        self,
        board_name: str,
        serial_number: str,
        channel_data: dict[str, Any],
    ) -> None:
        """Log a single telemetry time sample (JSON encoded)."""
        payload = json.dumps(channel_data)
        with sqlite3.connect(self._db_path(self._DATA_ACQ_DB)) as conn:
            conn.execute(
                """
                INSERT INTO acquisition_log (timestamp, board_name, serial_number, channel_data)
                VALUES (?, ?, ?, ?);
                """,
                (self._now(), board_name, serial_number, payload),
            )
            conn.commit()

    def log_acquisition_batch(
        self,
        board_name: str,
        serial_number: str,
        batch: list[dict[str, Any]],
    ) -> None:
        """Bulk insert an entire 30-second run (300 samples) in one transaction."""
        now_ts = self._now()
        rows = [
            (now_ts, board_name, serial_number, json.dumps(sample))
            for sample in batch
        ]
        with sqlite3.connect(self._db_path(self._DATA_ACQ_DB)) as conn:
            conn.executemany(
                """
                INSERT INTO acquisition_log (timestamp, board_name, serial_number, channel_data)
                VALUES (?, ?, ?, ?);
                """,
                rows,
            )
            conn.commit()

    def get_acquisition_data(
        self,
        board_name: str,
        serial_number: str,
    ) -> list[dict[str, Any]]:
        """Retrieve all telemetry samples for a board in chronological order."""
        with sqlite3.connect(self._db_path(self._DATA_ACQ_DB)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT channel_data FROM acquisition_log
                WHERE board_name = ? AND serial_number = ?
                ORDER BY id ASC;
                """,
                (board_name, serial_number),
            )
            rows = cursor.fetchall()

        result = []
        for row in rows:
            try:
                result.append(json.loads(row["channel_data"]))
            except Exception:
                pass
        return result

    # ================================================================== #
    #  Data Process Results Log Operations                                 #
    # ================================================================== #

    def log_results(
        self,
        board_name: str,
        serial_number: str,
        mfg_date: str,
        rul_percent: float,
        faults: list[str],
        mahalanobis_scores: dict[str, Any] | None = None,
        fft_scores: dict[str, Any] | None = None,
        threshold_flags: dict[str, Any] | None = None,
    ) -> None:
        """Store post-processing computation results."""
        faults_json = json.dumps(faults)
        mah_json = json.dumps(mahalanobis_scores or {})
        fft_json = json.dumps(fft_scores or {})
        thresh_json = json.dumps(threshold_flags or {})

        with sqlite3.connect(self._db_path(self._RESULTS_DB)) as conn:
            conn.execute(
                """
                INSERT INTO results_log (
                    timestamp, board_name, serial_number, mfg_date,
                    rul_percent, faults, mahalanobis_scores, fft_scores, threshold_flags
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    self._now(),
                    board_name,
                    serial_number,
                    mfg_date,
                    float(rul_percent),
                    faults_json,
                    mah_json,
                    fft_json,
                    thresh_json,
                ),
            )
            conn.commit()

    def get_results(
        self,
        board_name: str,
        serial_number: str,
    ) -> Optional[dict[str, Any]]:
        """Retrieve the latest processed diagnostic results for a board."""
        with sqlite3.connect(self._db_path(self._RESULTS_DB)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM results_log
                WHERE board_name = ? AND serial_number = ?
                ORDER BY id DESC LIMIT 1;
                """,
                (board_name, serial_number),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        record = dict(row)
        for key in ("faults", "mahalanobis_scores", "fft_scores", "threshold_flags"):
            if record.get(key):
                try:
                    record[key] = json.loads(record[key])
                except Exception:
                    pass
        return record
