"""
prognostic_engine.py
====================
Remaining Useful Life (RUL) estimator for TEST GIG PCB Testbenches.

This module implements a physics-of-failure prognostic pipeline that fuses
thermal-fatigue modelling with real-time electrical diagnostics:

    1. **Coffin–Manson Thermal Fatigue Model** – estimates the number of
       thermal cycles to failure based on the observed temperature swing:
           N_f = C * (ΔT)^(-n)
       where C and n are configured in config.py.
    2. **Miner's Rule (Palmgren–Miner Linear Damage Accumulation)** – converts
       accumulated thermal cycles into a cumulative damage fraction:
           D = n_i / N_f
    3. **Hybrid Electrical Penalty** – degrades the RUL estimate when the
       companion diagnostic engine has flagged an electrical anomaly:
           RUL_adj = RUL_base * penalty_factor

The master method ``compute_rul`` chains these three stages and returns a
comprehensive result dictionary.

Dependencies
------------
* numpy    – min / max helpers
* datetime – manufacturing-date parsing and elapsed-day calculation

References
----------
* Coffin, L. F. (1954). "A study of the effects of cyclic thermal stresses
  on a ductile metal." *Trans. ASME*, 76, 931–950.
* Miner, M. A. (1945). "Cumulative damage in fatigue." *J. Appl. Mech.*,
  12(3), A159–A164.
"""

from __future__ import annotations

import numpy as np
from datetime import datetime, date

import config


class PrognosticEngine:
    """Stateless RUL computation engine for TEST GIG.

    All methods are pure functions of their arguments; the class carries no
    mutable state and a single instance can safely be reused across boards or
    acquisition windows.
    """

    # ──────────────────────────────────────────────────────────────────
    # Method 1 – Coffin–Manson Thermal Fatigue
    # ──────────────────────────────────────────────────────────────────

    def coffin_manson(
        self,
        delta_t: float,
        C: float = config.COFFIN_MANSON_C,
        n: float = config.COFFIN_MANSON_N,
    ) -> float:
        """Estimate the number of thermal cycles to failure using the
        Coffin–Manson relation.

        .. math::

            N_f = C \\cdot (\\Delta T)^{-n}

        where:
        * **C** – material / geometry constant (config.COFFIN_MANSON_C = 1.0e7).
        * **n** – fatigue exponent (config.COFFIN_MANSON_N = 2.5).
        * **ΔT** – peak-to-peak temperature swing observed over the
          acquisition window (°C).

        Parameters
        ----------
        delta_t : float
            Temperature swing ΔT = T_max − T_min (°C). Must be > 0.
        C : float, optional
            Material constant (default from config.COFFIN_MANSON_C).
        n : float, optional
            Fatigue exponent (default from config.COFFIN_MANSON_N).

        Returns
        -------
        float
            ``N_f`` – estimated number of cycles to failure.
        """
        if delta_t <= 0:
            raise ValueError(
                f"delta_t must be > 0, got {delta_t}. "
                "A zero or negative temperature swing implies infinite life."
            )

        # N_f = C · (ΔT)^(-n)
        n_f: float = C * (delta_t ** (-n))
        return n_f

    # ──────────────────────────────────────────────────────────────────
    # Method 2 – Miner's Rule (Palmgren–Miner)
    # ──────────────────────────────────────────────────────────────────

    def miners_rule(
        self,
        cycles_applied: int,
        cycles_to_failure: float,
    ) -> float:
        """Compute the cumulative fatigue damage fraction using Miner's Rule.

        .. math::

            D = \\frac{n_i}{N_{f,i}}

        where:
        * **n_i** – number of thermal cycles the board has actually experienced.
        * **N_{f,i}** – number of cycles to failure at the measured stress amplitude.

        Parameters
        ----------
        cycles_applied : int
            Number of thermal cycles the board has undergone (n_i).
        cycles_to_failure : float
            Predicted cycles to failure at the current stress level (N_f).

        Returns
        -------
        float
            Damage fraction ``D`` in the range [0.0, ∞).
        """
        if cycles_to_failure <= 0:
            raise ValueError(
                f"cycles_to_failure must be > 0, got {cycles_to_failure}."
            )

        damage: float = cycles_applied / cycles_to_failure
        return damage

    # ──────────────────────────────────────────────────────────────────
    # Method 3 – Hybrid Electrical Penalty
    # ──────────────────────────────────────────────────────────────────

    def apply_electrical_penalty(
        self,
        base_rul: float,
        has_electrical_anomaly: bool,
        penalty_factor: float = config.ELECTRICAL_PENALTY_FACTOR,
    ) -> float:
        """Degrade the base RUL when electrical anomalies exist.

        .. math::

            RUL_{adj} = RUL_{base} \\times \\text{penalty\\_factor}

        Parameters
        ----------
        base_rul : float
            RUL percentage before penalty (0–100).
        has_electrical_anomaly : bool
            True if the diagnostic engine flagged at least one electrical anomaly.
        penalty_factor : float, optional
            Multiplicative factor (config.ELECTRICAL_PENALTY_FACTOR = 0.85).

        Returns
        -------
        float
            Adjusted RUL percentage.
        """
        if has_electrical_anomaly:
            return base_rul * penalty_factor
        return base_rul

    # ──────────────────────────────────────────────────────────────────
    # Master Method – Full RUL Pipeline
    # ──────────────────────────────────────────────────────────────────

    def compute_rul(
        self,
        temp_data: list[float],
        mfg_date: str,
        diagnostic_results: dict | None = None,
        has_electrical_anomaly: bool | None = None,
    ) -> dict:
        """End-to-end Remaining Useful Life computation for TEST GIG."""
        # ---- Step 1: Temperature swing ΔT ----
        temp_arr = np.array(temp_data, dtype=np.float64)
        delta_t: float = float(np.max(temp_arr) - np.min(temp_arr))

        if delta_t < 1e-6:
            delta_t = 1e-6

        # ---- Step 2: Coffin–Manson → N_f ----
        n_f: float = self.coffin_manson(
            delta_t,
            C=config.COFFIN_MANSON_C,
            n=config.COFFIN_MANSON_N,
        )

        # ---- Step 3: Cycles applied ----
        try:
            mfg_dt: date = datetime.strptime(mfg_date, "%Y-%m-%d").date()
            today: date = date.today()
            days_elapsed: int = (today - mfg_dt).days
            cycles_applied: int = max(0, days_elapsed)
        except Exception:
            cycles_applied = config.FALLBACK_CYCLES_APPLIED

        # ---- Step 4: Miner's Rule → damage D ----
        damage: float = self.miners_rule(cycles_applied, n_f)

        # ---- Step 5: Base RUL ----
        base_rul: float = max(0.0, (1.0 - damage)) * 100.0

        # ---- Step 6: Electrical penalty ----
        if has_electrical_anomaly is None:
            if diagnostic_results and isinstance(diagnostic_results, dict):
                has_anomaly = bool(diagnostic_results.get("has_electrical_anomaly", False))
            else:
                has_anomaly = False
        else:
            has_anomaly = bool(has_electrical_anomaly)

        penalty_factor: float = config.ELECTRICAL_PENALTY_FACTOR
        adjusted_rul: float = self.apply_electrical_penalty(
            base_rul, has_anomaly, penalty_factor
        )

        # ---- Step 7: Clamp to [0, 100] % ----
        final_rul: float = float(np.clip(adjusted_rul, 0.0, 100.0))

        return {
            "rul_percent": final_rul,
            "delta_t": delta_t,
            "n_f": n_f,
            "n_f_cycles": n_f,
            "cycles_applied": cycles_applied,
            "damage_fraction": damage,
            "miner_damage": damage,
            "base_rul": base_rul,
            "penalty_applied": has_anomaly,
            "penalty_factor": penalty_factor,
        }
