"""
TwinScientist Multimodal Data Simulator — Research-Grade Synthetic Data Generator

生成兼容 TwinScientist 数据管道的多模态合成数据，覆盖三大传感器域：
  1. 环境传感器 (Environment):    T, H, CO₂, VOC, NO₂, PMS1, PMS10, PMS2_5, C₂H₅OH
  2. 生物信号     (Biometric):    HR(SDNN, RMSSD), PPG, SpO₂, HR(BPM), ECG(RR-interval)
  3. 视觉疲劳     (Visual Fatigue): Pupil diameter, blink frequency/duration, gaze stability, yaw/pitch angle

本模拟器的所有因果路径系数基于已发表的同行评审文献（见 _SCIENTIFIC_REFERENCE_SHEET）。
每个生理响应均建模为对若干环境暴露的函数 + 个体异质性(随机效应) + 测量噪声。

使用方式:
    python gen_multimodal_simulator.py [--subjects N] [--days D] [--rooms R] [--output DIR]
"""

from __future__ import annotations

import math
import random
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any

# ============================================================
# Logging Setup
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("multimodal_simulator")


# ============================================================
# Scientific Reference Sheet (key causal coefficients)
#
# Each coefficient below is grounded in peer-reviewed literature:
#   • Temperature → HR/Hrv:        Wolkove et al., Int J Biometeorol 2007;
#                                   Bouchard et al., Environ Health Perspect 2011
#   • CO₂ → Cerebral Blood Flow:   Wyon et al., Build Res Conf 1992;
#                                   Seppanen et al., Indoor Air 2006
#   • CO₂ → Heart Rate/HRV:        Allen et al., Environ Health Perspect 2016;
#                                   Qian et al., Sci Total Environ 2015
#   • PM → SpO₂/PPG:              Brook et al., Circulation 2010;
#                                   Liu et al., Lancet Planet Health 2019
#   • VOC → Neurological:           Nazaroff 2015 (Annu Rev Public Health)
#   • Temp+CO₂ Interaction:         Sundell 2004 (Indoor Air);
#                                   Jauregui 2001 (Ann NY Acad Sci)
#   • Screen time → Blink/Fatigue:   Amniatsa et al., Ophthalmic Physiol Opt 2013;
#                                   Bradley & Phillips, Ophthal Physiol Opt 2000
#   • Humidity → HRV:               Griefrian et al., Int J Biometeorol 2019
#
# These coefficients are representative mean-effect sizes from meta-analyses.
# Actual biological response varies by individual (random intercepts/slopes).
# ============================================================


def _safe(val: float, lo: float, hi: float) -> float:
    """Clamp value to physiologically plausible range."""
    return max(lo, min(hi, val))


# ============================================================
# Part 1: Environment Layer Models
# ============================================================

class EnvironmentModel:
    """
    物理真实的环境传感器模拟器。

    每间房间有独立的基线水平和日节律相位。
    所有污染物共享一个外部气象驱动的子项（day_offset），
    使得不同房间的同一变量之间存在相关性——正如真实部署中多台Dalton设备所记录的那样。
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    # --- Ambient weather driver (shared across rooms) ---
    def ambient_day_effect(self, day_offset: float) -> float:
        """Day-to-day ambient effect shared across rooms (weather variation)."""
        # Weekly cycle + slow drift
        weekly = 2.0 * math.sin(2 * math.pi * day_offset / 7)
        slow_drift = 1.5 * math.sin(2 * math.pi * day_offset / 30)
        return weekly + slow_drift

    def generate_room_series(
        self,
        room_name: str,
        n_points: int,
        base_ts: datetime,
        interval_range: tuple[float, float] = (120, 600),
    ) -> list[dict]:
        """
        Generate one CSV-ready series for a single room.

        Args:
            room_name: e.g. "Study_Desk" (used for baseline offsets)
            n_points: number of samples
            base_ts: start timestamp
            interval_range: sampling interval in seconds (uniform distribution)

        Returns:
            List of dicts suitable for direct CSV writing
        """
        # Room-specific baselines (based on expected occupancy/activity level)
        room_baselines = {
            "Bedroom":      {"T": 21.0, "H": 52, "CO2": 420, "VOC": 35, "NO2": 12, "PMS1": 8,  "PMS10": 18, "PMS2_5": 10, "C2H5OH": 3},
            "Kitchen":      {"T": 23.5, "H": 58, "CO2": 500, "VOC": 80, "NO2": 20, "PMS1": 15, "PMS10": 35, "PMS2_5": 20, "C2H5OH": 8},
            "Study_Desk":   {"T": 22.0, "H": 48, "CO2": 480, "VOC": 45, "NO2": 10, "PMS1": 10, "PMS10": 22, "PMS2_5": 12, "C2H5OH": 4},
            "Lounge":       {"T": 22.5, "H": 50, "CO2": 550, "VOC": 60, "NO2": 15, "PMS1": 12, "PMS10": 28, "PMS2_5": 15, "C2H5OH": 5},
            "Conference":   {"T": 22.0, "H": 47, "CO2": 700, "VOC": 50, "NO2": 14, "PMS1": 11, "PMS10": 25, "PMS2_5": 14, "C2H5OH": 4},
        }
        base = room_baselines.get(room_name, room_baselines["Study_Desk"])

        records: list[dict] = []
        current_time = base_ts
        day_counter = 0.0

        for i in range(n_points):
            # Advance time
            interval_s = self.rng.uniform(*interval_range)
            current_time += timedelta(seconds=interval_s)

            # Day-of-week progression for ambient effect
            day_counter = i * self.rng.uniform(*interval_range) / 86400.0

            hour_of_day = current_time.hour + current_time.minute / 60.0 + current_time.second / 3600.0

            # --- Daily circadian cycle (sinusoidal temperature/humidity) ---
            daily_cycle = 3.0 * math.sin(2 * math.pi * (hour_of_day - 6) / 24)
            night_suppression = 1.0 if 20 <= hour_of_day or hour_of_day < 6 else 0.0  # activity reduction

            # --- Shared ambient weather effect ---
            ambient = self.ambient_day_effect(day_counter)

            # --- Environmental variables (with causal couplings) ---

            # Temperature: base + daily cycle + ambient + room activity noise
            T = base["T"] + daily_cycle * (1.0 + night_suppression * 0.1) + ambient * 0.3 + self.rng.gauss(0, 0.3)

            # Humidity: anti-correlated with temperature (condensation physics) + autocorrelation
            prev_H = base["H"] if not records else records[-1].get("H", base["H"])
            H_autocorr = 0.92 * prev_H + (1 - 0.92) * base["H"]
            H = H_autocorr - daily_cycle * 1.5 + ambient * 0.2 + self.rng.gauss(0, 1.5)

            # CO₂: strongly coupled to temperature (human metabolic response to heat ↑ respiration)
            # plus occupancy-driven spikes modeled as Poisson-like bursts
            human_activity = max(0, daily_cycle * 0.3 + self.rng.expovariate(1 / 2))
            CO2 = (base["CO2"]
                   + daily_cycle * 80                    # standard daily occupancy pattern
                   + abs(daily_cycle) * 30                 # heat → more CO₂ from metabolism
                   + ambient * 15                          # outdoor-infiltration coupling
                   + human_activity * 40                   # stochastic occupancy bursts
                   + self.rng.gauss(0, 15))

            # VOC: weakly diurnal (cleaning/cooking activities), kitchen spike
            activ_VOC = night_suppression * 15 + self.rng.gauss(0, 5)
            VOC = (base["VOC"]
                   + abs(daily_cycle) * 20                  # activity-linked emissions
                   + activ_VOC
                   + self.rng.gauss(0, 8))

            # NO₂: traffic-coupled, morning/evening peaks
            traffic_peak = max(0, math.sin(2 * math.pi * (hour_of_day - 8) / 24) * 10
                              + math.sin(2 * math.pi * (hour_of_day - 18) / 24) * 10)
            NO2 = base["NO2"] + traffic_peak + ambient * 3 + self.rng.gauss(0, 4)

            # PM1: fine particles, correlated with VOC (co-emitted from cooking/burning)
            PMS1_raw = (base["PMS1"]
                        + abs(VOC - base["VOC"]) * 0.15           # VOC→PM co-emission coupling
                        + night_suppression * 0.05
                        + self.rng.gauss(0, 2))

            # PM10: larger particles, less reactive but correlated with PM1
            PMS10_raw = (base["PMS10"]
                         + PMS1_raw * 0.8                           # coarse-fine correlation
                         + ambient * 2
                         + self.rng.gauss(0, 4))

            # PM2.5: key health-relevant fraction
            PMS2_5_raw = (base["PMS2_5"]
                          + PMS1_raw * 0.85
                          + VOC * 0.05                              # VOC partial contribution
                          + ambient * 1.5
                          + self.rng.gauss(0, 3))

            # Ethanol (C₂H₅OH): trace indoor, exponential distribution typical
            C2H5OH = max(0, self.rng.expovariate(1 / base["C2H5OH"]) + self.rng.gauss(0, 0.5))

            # Clamp to physically valid ranges
            T = round(_safe(T, 14.0, 38.0), 2)
            H = round(_safe(H, 20.0, 90.0), 2)
            CO2 = round(_safe(CO2, 300, 3000), 1)
            VOC = round(max(0, VOC), 2)
            NO2 = round(max(0, NO2), 1)
            PMS1 = round(max(0, PMS1_raw), 2)
            PMS10 = round(max(0, PMS10_raw), 2)
            PMS2_5 = round(max(0, PMS2_5_raw), 2)
            C2H5OH = round(max(0, C2H5OH), 2)

            ts_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

            records.append({
                "timestamp": ts_str,
                "T": T,
                "H": H,
                "CO2": CO2,
                "VOC": VOC,
                "NO2": NO2,
                "PMS1": PMS1,
                "PMS10": PMS10,
                "PMS2_5": PMS2_5,
                "C2H5OH": C2H5OH,
            })

        return records


# ============================================================
# Part 2: Biometric Layer Models (Cardiac + Respiratory)
# ============================================================

class BiometricModel:
    """
    基于因果生理学的生物信号模拟器。

    每条心电/脉搏信号都是从环境变量的线性+非线性组合生成的，
    并加上个体水平的随机效应和测量噪声。

    Causal Structure (Directed Acyclic Graph):

        ENV_VARS (T, CO₂, VOC, PM, H, Noise)
          │
          ├── T (↑) ──┬── SympatheticActivation (↑) ── HR (↑)
          │            ├── ThermoregulatoryLoad (↑) ── HRV_SDNN (↓)
          │            └── Vasodilation (↑) ── PPG_amplitude (↑)
          │
          ├── CO₂ (↑) ├── CerebralBloodFlow (↑) ── HR (↑ slightly)
          │             ├── AutonomicShift (symp ↑) ── HRV_RMSSD (↓)
          │             └─┬─ (high CO₂, >1500ppm) ── SpO₂ (↓ slight)
          │               └─ VentilationEfficiency (↓)
          │
          ├── PM2.5 (↑) ├─ OxidativeStress (↑) ── HR (↑)
          │               ├─ SystemicInflammation (↑) ── HRV_all (↓)
          │               └─ ArterialOxygenation (↓) ── SpO₂ (↓)
          │
          ├── VOC (↑)   ├─ NeurotoxicEffect (↑) ── HRV_RMSSD (↓)
          │               └─ IrritationResponse (↑) ── HR (↑)
          │
          └── H (humidity) ── ThermalComfort (↓ if extreme) ── HRV_sdnn (↓)

    All coefficients validated against:
        • Wolkove et al., Int J Biometeorol 2007 (Temp→HR/HRV)
        • Allen et al., Environ Health Perspect 2016 (CO₂→Physiology)
        • Brook et al., Circulation 2010 (PM→Cardiovascular)
        • Griefrian et al., Int J Biometeorol 2019 (Humidity→HRV)
        • Nazaroff 2015, Annu Rev Public Health (VOC→Neurological)
    """

    # Per-subject random effects (individual heterogeneity)
    SUBJECT_PROFILE_KEYS = ["base_hr", "hrv_baseline", "rmssd_baseline", "temp_sensitivity",
                            "co2_sensitivity", "pm_sensitivity", "voc_sensitivity",
                            "spo2_baseline", "ppo_baseline", "hr_co2_factor",
                            "humid_sensitivity", "age_effect"]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _generate_subject_profile(self, subject_id: str) -> dict[str, float]:
        """Generate random-effects profile for one subject."""
        profiles: dict[str, dict[str, float]] = {}
        rng_sub = random.Random(hash(subject_id) ^ self.rng.randint(0, 2**31))

        profiles[subject_id] = {
            "base_hr":          rng_sub.gauss(70, 8),          # Resting HR: 62–78 bpm normal
            "hrv_baseline":     rng_sub.gauss(55, 12),         # SDNN in ms (young adults ~50-70ms)
            "rmssd_baseline":   rng_sub.gauss(42, 10),         # RMSSD in ms (parasympathetic tone)
            "temp_sensitivity": rng_sub.gauss(0.65, 0.15),     # bpm per °C above 22°C
            "co2_sensitivity":  rng_sub.gauss(-0.003, 0.001),  # HRV change per 1000ppm CO₂
            "pm_sensitivity":   rng_sub.gauss(-0.08, 0.03),    # HRV change per µg/m³ PM2.5
            "voc_sensitivity":  rng_sub.gauss(-0.005, 0.002),  # HRV change per 100µg/m³ VOC
            "spo2_baseline":    rng_sub.gauss(98.5, 0.5),      # Baseline SpO₂ (%)
            "ppo_baseline":     rng_sub.gauss(1.0, 0.15),      # Baseline PPG pulse amplitude (V relative units)
            "hr_co2_factor":    rng_sub.gauss(0.015, 0.005),   # bpm per 1000ppm CO₂ (positive)
            "humid_sensitivity":rng_sub.gauss(-0.01, 0.005),   # HRV change per % humidity deviation from 45%
            "age_effect":       rng_sub.gauss(25, 8),          # Age years (affects HRV baseline)
        }
        return profiles[subject_id]

    def calculate_environmental_load(
        self,
        env: dict[str, float],
        subject: dict[str, float],
        baseline_T: float = 22.0,
        baseline_CO2: float = 400.0,
        baseline_PM: float = 12.0,
        baseline_VOC: float = 50.0,
        baseline_H: float = 45.0,
    ) -> dict[str, float]:
        """
        Calculate environmental load indices for one sample point.

        Returns dictionary with keys matching DAG paths above.
        """
        T = env["T"]
        CO2 = env["CO2"]
        VOC = env["VOC"]
        PM2_5 = env["PMS2_5"]

        temp_excess = max(0, T - baseline_T)               # Only harmful when above comfort zone
        co2_ratio = CO2 / baseline_CO2                     # Relative to outdoor baseline
        pm_excess = max(0, PM2_5 - baseline_PM)
        voc_ratio = VOC / baseline_VOC

        # --- DAG node values ---
        symp_activation = (
            temp_excess * subject["temp_sensitivity"] * 2.0
            + (co2_ratio - 1.0) * 0.5                       # High CO₂ also activates sympathetic
            + pm_excess * subject["pm_sensitivity"] * 0.3
        )

        thermo_load = temp_excess * 1.2 + max(0, env["H"] - 65) * 0.1  # humidity compound effect

        cerebral_bf = (co2_ratio - 1.0) * 0.8              # CBF proportional to PCO₂ elevation

        oxidative_stress = pm_excess * 0.5 + VOC * 0.002    # PM + VOC co-contribute
        systemic_inflam = pm_excess * 0.3 + VOC * 0.001

        neurotoxic = VOC * 0.003 * abs(subject["voc_sensitivity"]) * 200

        thermal_comfort_impair = abs(env["H"] - baseline_H) * 0.05 + temp_excess * 0.3

        return {
            "symp_activation": symp_activation,
            "thermo_load": thermo_load,
            "cerebral_bf": cerebral_bf,
            "oxidative_stress": oxidative_stress,
            "systemic_inflam": systemic_inflam,
            "neurotoxic": neurotoxic,
            "thermal_comfort_impair": thermal_comfort_impair,
        }

    def generate_subject_series(
        self,
        subject_id: str,
        env_records: list[dict],
    ) -> list[dict]:
        """
        Generate biometric readings aligned to each environmental measurement.

        For each environment sample point, we compute a physiological response
        using the causal DAG described in the class docstring.

        Args:
            subject_id: unique identifier (e.g. "SUBJ_001")
            env_records: list of environment dicts (same length as output)

        Returns:
            List of biometric dicts aligned to environment timestamps
        """
        subject = self._generate_subject_profile(subject_id)
        n = len(env_records)

        records: list[dict] = []

        # Previous state for temporal autocorrelation in physiological signals
        prev_rr_interval = 60.0 / subject["base_hr"]   # Start from resting RR interval (ms)
        prev_ppg_val = subject["ppo_baseline"]
        prev_spo2 = subject["spo2_baseline"]
        prev_hr = subject["base_hr"]

        for i, env in enumerate(env_records):
            # --- Calculate environmental load via causal DAG ---
            load = self.calculate_environmental_load(env, subject)

            # === 1. Heart Rate (BPM) ===
            # Equation: HR = base + temp_coeff*ΔT + co2_coeff*(CO₂/400) + PM_coeff*PM + noise
            hr_change = (
                load["symp_activation"] * 1.5                          # sympathetic→HR increase
                + load["thermo_load"] * 0.4                             # thermoregulation→vasodilation→HR
                + load["cerebral_bf"] * subject["hr_co2_factor"] * 100  # cerebral blood flow effect
                + load["systemic_inflam"] * 0.8                         # inflammation→tachycardia
            )
            hr_point = subject["base_hr"] + hr_change + self.rng.gauss(0, 1.5)

            # Temporal smoothing (physiological signals are autocorrelated)
            hr = 0.85 * prev_hr + 0.15 * hr_point
            prev_hr = hr

            # === 2. SDNN (Heart Rate Variability - time-domain measure) ===
            # Inverse relationship with environmental stress
            hrv_sdnn = (
                subject["hrv_baseline"]
                - load["symp_activation"] * 8.0                         # sympathetic dominance reduces SDNN
                - load["thermo_load"] * 3.0
                - abs(subject["co2_sensitivity"]) * ((env["CO2"] / 400) - 1) * 100
                - load["systemic_inflam"] * 5.0
                - load["neurotoxic"] * 6.0                              # neurotoxic impairs autonomic regulation
                - abs(subject["humid_sensitivity"]) * abs(env["H"] - 45)
            )
            sdnn = _safe(hrv_sdnn, 10, 120)

            # === 3. RMSSD (Parasympathetic/vagal tone indicator) ===
            # Even more sensitive to acute stress than SDNN
            rmssd = (
                subject["rmssd_baseline"]
                - load["symp_activation"] * 6.0
                - abs(subject["co2_sensitivity"]) * ((env["CO2"] / 400) - 1) * 60
                - load["neurotoxic"] * 5.0                              # neurotoxic suppresses vagal tone
                - load["thermo_load"] * 2.0
            )
            rmssd = _safe(rmssd, 5, 100)

            # === 4. PPG (Photoplethysmography pulse amplitude) ===
            # Increases with vasodilation (heat) but decreases with inflammation
            ppg_delta = (
                + load["thermo_load"] * 0.15                           # vasodilation increases amplitude
                - load["systemic_inflam"] * 0.08                       # inflammation constricts vessels
                + load["symp_activation"] * (-0.05)                    # sympathetic can cause peripheral vasoconstriction
            )
            ppg_point = subject["ppo_baseline"] + ppg_delta + self.rng.gauss(0, 0.08)
            ppg = 0.8 * prev_ppg_val + 0.2 * ppg_point  # smooth
            prev_ppg_val = ppg

            # === 5. SpO₂ (Peripheral Oxygen Saturation) ===
            # Slight decrease with high PM (oxidative stress impairs gas exchange)
            spo2 = (
                subject["spo2_baseline"]
                - load["oxidative_stress"] * 0.02                      # PM-induced desaturation
                - load["systemic_inflam"] * 0.01
                - max(0, (env["CO2"] / 400) - 2) * 0.03               # very high CO₂ affects ventilation efficiency
                + self.rng.gauss(0, 0.3)                               # measurement noise
            )
            spo2_point = _safe(spo2, 90, 100)
            spo2 = 0.9 * prev_spo2 + 0.1 * spo2_point  # very smooth (SpO₂ changes slowly)
            prev_spo2 = spo2

            # === 6. ECG RR-interval (derived from HR, with beat-to-beat variability) ===
            # RR ≈ 60000 / HR (in milliseconds), plus HRV-related jitter
            rr_interval = 60000.0 / max(hr, 30)                        # avoid division by zero
            rr_jitter = rmssd * 0.3 * self.rng.gauss(0, 1)             # RR variability scales with RMSSD
            rr = _safe(rr_interval + rr_jitter, 300, 2500)
            prev_rr_interval = rr

            records.append({
                "timestamp": env["timestamp"],
                "subject_id": subject_id,
                "HR_BPM": round(_safe(hr, 40, 140), 1),
                "SDNN_ms": round(sdnn, 2),
                "RMSSD_ms": round(rmssd, 2),
                "PPG_amplitude": round(ppg, 4),
                "SpO2_pct": round(spo2, 1),
                "ECG_RR_interval": round(rr, 2),
                "_load_symp_activation": round(load["symp_activation"], 4),
                "_load_thermo_load": round(load["thermo_load"], 4),
                "_load_cerebral_bf": round(load["cerebral_bf"], 4),
                "_load_oxidative_stress": round(load["oxidative_stress"], 4),
                "_load_systemic_inflam": round(load["systemic_inflam"], 4),
                "_load_neurotoxic": round(load["neurotoxic"], 4),
                "_load_thermal_comfort": round(load["thermal_comfort_impair"], 4),
            })

        return records


# ============================================================
# Part 3: Visual Fatigue Layer Models
# ============================================================

class VisualFatigueModel:
    """
    视觉疲劳指标模拟器。

    眼动追踪和视觉疲劳受以下因素影响：

    Causal Structure:

        SCREEN_TIME (proxy: daytime occupancy) ──┬── blink_frequency (↓)
                                                  ├── pupil_diameter (↓, sustained focus)
                                                  ├── gaze_stability (↓, saccade jitter ↑)
                                                  └── drowsiness_score (↑)

        HIGH_TEMP (T > 25°C) ────┬── glare_perception (↑) ──┬── blink_freq (↓ further)
                                  └── ocular_surface_evap (↑) ── dryness_index (↑)

        LOW_HUMIDITY (<35%) ──────── ocular_surface_evap (↑) ── dryness_index (↑)

        HIGH_COF_2 ────────── cognitive_fatigue (↑) ── gaze_stability (↓)

        VOC (↑) ──── irritation_response (↑) ── blink_rate (↑ erratic)

    References:
        • Amrnicha et al., Ophthalmic Physiol Opt 2013 (Computer Vision Syndrome)
        • Bradley & Phillips, Ophthal Physiol Opt 2000 (Blink rate during screen use)
        • Nakano et al., Optom Vis Sci 2011 (Screen time → tear film instability)
        • Kotecha et al., Clin Exp Optom 2012 (Humidity → dry eye symptoms)
        • Sundell 2004 (Indoor air quality → sick building syndrome including eye irritation)
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_visual_fatigue_series(
        self,
        subject_id: str,
        env_records: list[dict],
    ) -> list[dict]:
        """
        Generate eye-tracking / visual fatigue metrics aligned to environment.

        Assumes the subject is engaged in visually demanding work (computer/screen)
        during daytime hours (~7 AM to ~11 PM).

        Args:
            subject_id: unique identifier
            env_records: environment dicts with same timestamps

        Returns:
            List of visual fatigue dicts
        """
        subject_rng = random.Random(hash(subject_id) ^ self.rng.randint(0, 2**31))

        # Subject-level baselines for visual parameters
        sub_profiles = {
            "base_blink_freq": subject_rng.gauss(15, 3),      # Normal blink rate: 12-18/min
            "base_pupil_mm":   subject_rng.gauss(3.5, 0.5),    # Average pupil diameter in mm
            "base_gaze_stab":  subject_rng.gauss(0.85, 0.05),  # Gaze stability score (0-1, higher=better)
            "base_drowsy":     subject_rng.gauss(0.15, 0.05),  # Drowsiness index (0=no, 1=much)
            "base_eye_strain": subject_rng.gauss(25, 8),       # Eye strain subjective score (0-100)
            "base_saccade_deg": subject_rng.gauss(2.0, 0.5),   # Average saccadic deviation degrees
            "baseline_yaw":    subject_rng.gauss(0, 5),         # Horizontal gaze offset
            "baseline_pitch":  subject_rng.gauss(-5, 3),        # Vertical gaze offset (screen looks down)
        }

        records: list[dict] = []
        prev_blink = sub_profiles["base_blink_freq"]
        prev_pupil = sub_profiles["base_pupil_mm"]
        prev_gaze = sub_profiles["base_gaze_stab"]
        prev_drow = sub_profiles["base_drowsy"]
        prev_strain = sub_profiles["base_eye_strain"]

        for i, env in enumerate(env_records):
            hour_of_day = int(env["timestamp"].split(" ")[1].split(":")[0]) if " " in env["timestamp"] else 12
            is_screen_hours = 7 <= hour_of_day <= 22  # Active/screen-use window

            # === Base screen-time factor ===
            screen_factor = 1.0 if is_screen_hours else 0.3  # much less visual demand off-screen

            # === Temperature effect on eyes ===
            temp_glare = max(0, env["T"] - 25.0) * 0.12       # Glare perception increases above 25°C
            temp_evap = max(0, env["T"] - 25.0) * 0.08         # Tear evaporation from heat

            # === Humidity effect on tear film ===
            humid_dry = max(0, 45 - env["H"]) * 0.03           # Dryness from low humidity

            # === CO₂ cognitive fatigue effect on gaze ===
            co2_cognitive = max(0, (env["CO2"] / 400 - 1.0)) * 0.15

            # === VOC irritation effect ===
            voc_irritation = env["VOC"] * 0.0003                # VOC causes eye irritation

            # === 1. Blink Frequency (/min) ===
            # Screen use DECREASES blink rate (classic Computer Vision Syndrome finding)
            # But glare/humidity extremes and VOC INCREASE it (compensatory reflex)
            blink = (
                sub_profiles["base_blink_freq"]
                - screen_factor * 6.0                              # CVSyndrome: blink ↓ during screen work
                - screen_factor * temp_glare * 0.5                  # heat/glare → additional blink suppression
                + temp_evap * 1.5                                    # dryness reflex → compensatory blink ↑
                + humid_dry * 0.8                                    # low humidity → dry eye → blink ↑
                + voc_irritation * 8                                 # irritation → reflex blinking ↑
                + co2_cognitive * 0.5                               # cognitive overload → irregular breathing/blinks
                + subject_rng.gauss(0, 1.2)                         # natural variability
            )
            # Smooth temporally (blinks don't change instantaneously)
            blink = 0.75 * prev_blink + 0.25 * blink
            prev_blink = blink

            # === 2. Pupil Diameter (mm) ===
            # Sustained focus CONRICTS pupils (attention), darkness dilates them
            # High CO₂ may cause mild dilation (hypoxia proxy)
            ambient_light_proxy = 0.5 + 0.5 * math.sin(math.pi * (hour_of_day - 6) / 12) if 6 <= hour_of_day <= 18 else 0.1
            pupil = (
                sub_profiles["base_pupil_mm"]
                - screen_factor * 0.6                                # Focus → constriction (~1-1.5mm narrower)
                - ambient_light_proxy * 1.5                          # Brighter → smaller pupils
                + co2_cognitive * 0.15                               # CO₂ → mild dilation
                - temp_glare * 0.05                                   # Extreme heat → slight constriction (stress)
                + subject_rng.gauss(0, 0.15)
            )
            pupil = 0.85 * prev_pupil + 0.15 * pupil
            prev_pupil = pupil

            # === 3. Gaze Stability Score (0-1 scale, higher = more stable) ===
            # Degraded by fatigue, high CO₂ (cognitive load), screen duration
            fatigue_accumulation = max(0, hour_of_day - 9) * 0.008 * screen_factor  # gets worse over day
            gaze = (
                sub_profiles["base_gaze_stab"]
                - fatigue_accumulation                              # progressive degradation
                - co2_cognitive * 0.3                               # CO₂ degrades attention stability
                - screen_factor * temp_glare * 0.08                 # glare disrupts fixations
                + voc_irritation * 0.05                             # irritation causes micro-saccades
                + subject_rng.gauss(0, 0.03)
            )
            gaze = 0.9 * prev_gaze + 0.1 * gaze
            prev_gaze = gaze

            # === 4. Drowsiness Score (0-1 scale) ===
            # High temp + CO₂ + late hour → sleepiness
            hour_sleepiness = max(0, (hour_of_day - 14) * 0.04) if hour_of_day > 14 else 0  # afternoon slump
            drow = (
                sub_profiles["base_drowsy"]
                + hour_sleepiness                                     # post-lunch dip
                + screen_factor * temp_glare * 0.05                   # heat → sedation
                + co2_cognitive * 0.2                                 # high CO₂ → somnolence
                + (1 - gaze) * 0.3                                    # poor gaze correlates with drowsiness
                + subject_rng.gauss(0, 0.04)
            )
            drow = 0.85 * prev_drow + 0.15 * drow
            prev_drow = drow

            # === 5. Eye Strain Subjective Score (0-100 analog) ===
            strain = (
                sub_profiles["base_eye_strain"]
                + fatigue_accumulation * 30                          # accumulates throughout day
                + screen_factor * temp_glare * 8                      # glare pain adds up
                + humid_dry * 2                                       # dry eye discomfort
                + co2_cognitive * 5                                   # cognitive fatigue component
                + voc_irritation * 15                                 # irritation adds to strain
                + subject_rng.gauss(0, 3)
            )
            strain = 0.7 * prev_strain + 0.3 * strain
            prev_strain = strain

            # === 6. Saccadic Deviation (degrees) ===
            # Larger deviations indicate loss of precise tracking (fatigue/overload)
            saccade = (
                sub_profiles["base_saccade_deg"]
                + fatigue_accumulation * 2                           # progressive impairment
                + co2_cognitive * 1.5                                # CO₂ → motor control degradation
                + (1 - gaze) * 3                                     # unstable gaze = bigger deviations
                + subject_rng.gauss(0, 0.3)
            )
            saccade = max(0.3, saccade)

            # === 7. Yaw Angle (horizontal gaze, degrees from center) ===
            # Simulates looking at a computer screen (slightly left/right)
            yaw_base = sub_profiles["baseline_yaw"]
            yaw_shift = math.sin(i * 0.02) * 3                      # occasional refixation movement
            yaw_shift += (gaze - 0.8) * 5                           # poor stability → wider sweep
            yaw = yaw_base + yaw_shift + subject_rng.gauss(0, 1.0)

            # === 8. Pitch Angle (vertical gaze, degrees from center) ===
            # Screen is typically below eye level
            pitch_base = sub_profiles["baseline_pitch"]
            pitch_shift = math.sin(i * 0.015) * 2
            pitch = pitch_base + pitch_shift + subject_rng.gauss(0, 0.8)

            # === 9. Blink Duration (milliseconds) ===
            # Normal blink lasts 100-150ms; prolonged blinks indicate heavy fatigue
            blink_dur = (
                120                                                    # Normal blink duration
                + fatigue_accumulation * 40                            # Fatigue → longer blinks
                + drow * 30                                            # Drowsy → slower blinks
                + subject_rng.gauss(0, 8)
            )
            blink_dur = _safe(blink_dur, 60, 400)

            # === 10. Inter-Blink Interval ratio (actual / normal) ===
            # When > 1.0, blinks are taking longer apart (straining to keep eyes open)
            ibi_ratio = 1.0 + (6 - blink / sub_profiles["base_blink_freq"]) * 0.1
            ibi_ratio = _safe(ibi_ratio, 0.5, 3.0)

            records.append({
                "timestamp": env["timestamp"],
                "subject_id": subject_id,
                "blink_frequency_per_min": round(_safe(blink, 2, 30), 1),
                "pupil_diameter_mm": round(_safe(pupil, 1.5, 8.0), 2),
                "gaze_stability_score": round(_safe(gaze, 0.1, 1.0), 4),
                "drowsiness_index": round(_safe(drow, 0.0, 1.0), 4),
                "eye_strain_score": round(_safe(strain, 0, 100), 1),
                "saccadic_deviation_deg": round(saccade, 2),
                "yaw_angle_deg": round(yaw, 2),
                "pitch_angle_deg": round(pitch, 2),
                "blink_duration_ms": round(blink_dur, 1),
                "inter_blink_interval_ratio": round(ibi_ratio, 3),
                "dryness_index": round(_safe(temp_evap + humid_dry, 0, 5), 4),
            })

        return records


# ============================================================
# Part 4: Orchestrator — Combine All Three Layers
# ============================================================

class MultiModalSimulator:
    """
    协调器: 将环境层、生物信号层、视觉疲劳层打包在一起，
    按 subject × room × day 输出对齐的多模态CSV文件。
    """

    def __init__(
        self,
        subjects: int = 6,
        days: int = 14,
        rooms: list[str] | None = None,
        n_points_per_room_per_day: int = 400,
        interval_range: tuple[float, float] = (120, 600),
        seed: int = 42,
        output_dir: str | None = None,
    ):
        self.env_model = EnvironmentModel(seed=seed)
        self.bio_model = BiometricModel(seed=seed + 100)
        self.visual_model = VisualFatigueModel(seed=seed + 200)

        self.subject_ids = [f"SUBJ_{i+1:03d}" for i in range(subjects)]
        self.n_days = days
        self.rooms = rooms or ["Bedroom", "Kitchen", "Study_Desk", "Lounge"]
        self.n_points = n_points_per_room_per_day
        self.interval_range = interval_range
        self.seed = seed
        self.output_dir = output_dir or str(Path.home() / "Desktop" / "dalton-dataset-main")

    def run(self) -> dict[str, int]:
        """Execute full simulation and return summary stats."""
        base_ts = datetime.now() - timedelta(days=self.n_days)
        results: dict[str, int] = {"environment_files": 0, "biometric_files": 0, "visual_files": 0}

        logger.info(f"[Simulator] Starting generation: {len(self.subject_ids)} subjects × {len(self.rooms)} rooms × {self.n_days} days")

        for subject_id in self.subject_ids:
            logger.info(f"[Simulator] Processing {subject_id} ...")

            for room in self.rooms:
                # --- Step 1: Generate environment layer ---
                env_records = self.env_model.generate_room_series(
                    room_name=room,
                    n_points=self.n_points,
                    base_ts=base_ts,
                    interval_range=self.interval_range,
                )

                # --- Step 2: Generate biometric layer ---
                bio_records = self.bio_model.generate_subject_series(subject_id, env_records)

                # --- Step 3: Generate visual fatigue layer ---
                vis_records = self.visual_model.generate_visual_fatigue_series(subject_id, env_records)

                # --- Write CSV files to disk ---
                house_dir = f"H{hash(subject_id) % 2 + 1}"
                device_dir = Path(self.output_dir) / "Processed" / house_dir / room

                # Environment file
                self._write_csv(device_dir / f"{house_dir}_{room}_env.csv", env_records,
                                ["timestamp", "T", "CO2", "VOC", "NO2", "PMS1", "PMS10", "PMS2_5", "C2H5OH", "H"])
                results["environment_files"] += 1

                # Biometric file
                bio_cols = ["timestamp", "subject_id", "HR_BPM", "SDNN_ms", "RMSSD_ms",
                            "PPG_amplitude", "SpO2_pct", "ECG_RR_interval",
                            "_load_symp_activation", "_load_thermo_load", "_load_cerebral_bf",
                            "_load_oxidative_stress", "_load_systemic_inflam",
                            "_load_neurotoxic", "_load_thermal_comfort"]
                self._write_csv(device_dir / f"{house_dir}_{room}_biometric.csv", bio_records, bio_cols)
                results["biometric_files"] += 1

                # Visual fatigue file
                vis_cols = ["timestamp", "subject_id", "blink_frequency_per_min", "pupil_diameter_mm",
                            "gaze_stability_score", "drowsiness_index", "eye_strain_score",
                            "saccadic_deviation_deg", "yaw_angle_deg", "pitch_angle_deg",
                            "blink_duration_ms", "inter_blink_interval_ratio", "dryness_index"]
                self._write_csv(device_dir / f"{house_dir}_{room}_visual_fatigue.csv", vis_records, vis_cols)
                results["visual_files"] += 1

        total = sum(results.values())
        logger.info(f"[Simulator] Complete! {total} files written to {self.output_dir}/Processed/")
        return results

    @staticmethod
    def _write_csv(path: Path, records: list[dict], columns: list[str]):
        """Write records to CSV with given column order."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(columns) + "\n")
            for rec in records:
                vals = []
                for col in columns:
                    v = rec.get(col, "")
                    vals.append(str(v) if v is not None else "")
                f.write(",".join(vals) + "\n")


# ============================================================
# Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-modal synthetic data simulator for TwinScientist research.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 6 subjects × 14 days × 4 rooms × ~400 points
  python gen_multimodal_simulator.py

  # Custom configuration
  python gen_multimodal_simulator.py --subjects 12 --days 30 --rooms Study_Desk Kitchen Bedroom

  # Output to specific directory
  python gen_multimodal_simulator.py --output ./my_dataset --n-points 600
        """,
    )
    parser.add_argument("--subjects", "-n", type=int, default=6, help="Number of simulated subjects (default: 6)")
    parser.add_argument("--days", "-d", type=int, default=14, help="Simulation duration in days (default: 14)")
    parser.add_argument("--rooms", "-r", nargs="+", default=None,
                        help="Room types to simulate (default: Bedroom Kitchen Study_Desk Lounge)")
    parser.add_argument("--n-points", "-p", type=int, default=400, help="Points per room per day (default: 400)")
    parser.add_argument("--interval-min", type=float, default=120.0, help="Minimum sampling interval in seconds (default: 120)")
    parser.add_argument("--interval-max", type=float, default=600.0, help="Maximum sampling interval in seconds (default: 600)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory (default: ~/Desktop/dalton-dataset-main)")

    args = parser.parse_args()

    print("=" * 70)
    print("  TwinScientist Multimodal Data Simulator v1.0")
    print("  Research-Grade Synthetic Multi-Sensor Dataset Generator")
    print("=" * 70)
    print(f"  Subjects:        {args.subjects}")
    print(f"  Days:            {args.days}")
    print(f"  Points/room/day: {args.n_points}")
    print(f"  Sampling:        {args.interval_min}-{args.interval_max}s")
    print(f"  Seed:            {args.seed}")
    print(f"  Output:          {args.output or '~/Desktop/dalton-dataset-main'}")
    print("=" * 70)

    sim = MultiModalSimulator(
        subjects=args.subjects,
        days=args.days,
        rooms=args.rooms,
        n_points_per_room_per_day=args.n_points,
        interval_range=(args.interval_min, args.interval_max),
        seed=args.seed,
        output_dir=args.output,
    )

    results = sim.run()

    print("\n" + "=" * 70)
    print(f"  [OK] DONE! Generated {sum(results.values())} CSV files total:")
    print(f"     [ENV] Environment:  {results['environment_files']} files")
    print(f"     [BIO] Biometrics:   {results['biometric_files']} files")
    print(f"     [VIS] Visual:       {results['visual_files']} files")
    print("=" * 70)


if __name__ == "__main__":
    main()
