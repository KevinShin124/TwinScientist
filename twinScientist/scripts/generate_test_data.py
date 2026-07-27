#!/usr/bin/env python3
"""Generate synthetic environment-health CSV datasets for TwinScientist testing.

Theme: Air Quality + Heart Rate Variability (HRV) — N-of-1 Personal Study
Generates 90 days of data with realistic correlations:
  - Higher PM2.5 → lower HRV (SDNN)
  - Higher temperature → higher resting heart rate
  - Higher humidity → worse sleep quality
  - O3 peaks → increased stress markers
"""

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

OUTPUT_DIR = Path.home() / "Desktop"
DAYS = 90
START_DATE = date(2026, 4, 28)

# ── Helpers ──────────────────────────────────────────────────────────────────

def seasonal_trend(day_idx, amplitude=1.0, phase=0, period=90):
    """Smooth sinusoidal seasonal trend."""
    return amplitude * math.sin(2 * math.pi * (day_idx + phase) / period)

def weekly_pattern(day_idx, amplitude=1.0):
    """Weekly cycle — weekends differ from weekdays."""
    dow = (START_DATE + timedelta(days=day_idx)).weekday()  # 0=Mon, 6=Sun
    return amplitude * (1.0 if dow >= 5 else -0.3)

def noise(sigma=1.0):
    return random.gauss(0, sigma)

def clamp(val, lo, hi):
    return max(lo, min(hi, round(val, 2)))


# ── 1. Environmental Data: env_air_quality.csv ───────────────────────────────

def generate_env():
    rows = []
    for i in range(DAYS):
        d = START_DATE + timedelta(days=i)

        # PM2.5: baseline 25, seasonal (higher in winter), weekly (worse on weekdays)
        pm25 = clamp(
            25 + seasonal_trend(i, 8, 10, 90) + weekly_pattern(i, 5) + noise(6),
            5, 80
        )
        # PM10: correlated with PM2.5
        pm10 = clamp(pm25 * 1.6 + noise(8), 8, 120)

        # O3: inversely correlated with PM2.5 (ozone is higher in summer)
        o3 = clamp(45 + seasonal_trend(i, 12, 50, 90) + noise(8), 10, 100)

        # NO2: traffic-related, higher on weekdays
        no2 = clamp(20 + weekly_pattern(i, 8) + noise(5), 5, 60)

        # Temperature (°C): seasonal, spring→summer warming
        temp = clamp(18 + i * 0.15 + seasonal_trend(i, 4, 20, 90) + noise(3), 8, 36)

        # Humidity (%): seasonal
        humidity = clamp(60 - seasonal_trend(i, 8, 80, 90) + weekly_pattern(i, -3) + noise(5), 30, 95)

        # Wind speed (m/s)
        wind = clamp(3.0 + noise(1.5), 0.2, 10)

        # Air pressure (hPa)
        pressure = clamp(1013 + seasonal_trend(i, 5, 30, 90) + noise(3), 995, 1030)

        rows.append({
            "date": d.isoformat(),
            "pm25_ugm3": pm25,
            "pm10_ugm3": pm10,
            "o3_ugm3": o3,
            "no2_ugm3": no2,
            "temperature_c": temp,
            "humidity_pct": humidity,
            "wind_speed_ms": wind,
            "pressure_hpa": pressure,
        })
    return rows


# ── 2. Health Data: health_metrics.csv ───────────────────────────────────────

def generate_health(env_rows):
    rows = []
    for i, env in enumerate(env_rows):
        d = START_DATE + timedelta(days=i)

        pm25 = env["pm25_ugm3"]
        temp = env["temperature_c"]
        humidity = env["humidity_pct"]

        # HRV (SDNN ms): baseline 55, negatively affected by PM2.5, high temp, humidity
        sdnn = clamp(55 - pm25 * 0.15 - abs(temp - 22) * 0.8 - (humidity - 50) * 0.1 + noise(5), 20, 80)

        # RMSSD (ms): similar pattern
        rmssd = clamp(sdnn * 0.75 + noise(3), 15, 60)

        # Resting heart rate (bpm): higher with temp, PM2.5
        rhr = clamp(62 + pm25 * 0.08 + (temp - 18) * 0.5 + noise(3), 50, 85)

        # Sleep quality score (0-100): worse with PM2.5, high humidity, high temp
        sleep = clamp(72 - pm25 * 0.2 - (humidity - 50) * 0.15 - abs(temp - 20) * 0.5 + noise(5), 30, 95)

        # Stress level (0-100): PM2.5 + O3 + work stress
        stress = clamp(40 + pm25 * 0.3 + env["o3_ugm3"] * 0.1 + weekly_pattern(i, -8) + noise(6), 10, 90)

        # Blood pressure systolic (mmHg)
        bp_sys = clamp(115 + pm25 * 0.1 + (temp - 18) * 0.3 + stress * 0.08 + noise(3), 100, 140)

        # Blood pressure diastolic (mmHg)
        bp_dia = clamp(75 + pm25 * 0.05 + stress * 0.04 + noise(2), 60, 95)

        # Step count
        steps = clamp(8500 + weekly_pattern(i, -2000) + noise(1500), 2000, 15000)

        # Weight (kg) — slight trend
        weight = clamp(70.0 + i * 0.005 + noise(0.3), 68, 72)

        rows.append({
            "date": d.isoformat(),
            "hrv_sdnn_ms": sdnn,
            "hrv_rmssd_ms": rmssd,
            "resting_hr_bpm": rhr,
            "sleep_quality": int(sleep),
            "stress_level": int(stress),
            "bp_systolic_mmhg": bp_sys,
            "bp_diastolic_mmhg": bp_dia,
            "steps": int(steps),
            "weight_kg": weight,
        })
    return rows


# ── 3. Merged Dataset: merged_env_health.csv ─────────────────────────────────

def merge(env_rows, health_rows):
    rows = []
    for env, health in zip(env_rows, health_rows):
        merged = {**env, **{k: v for k, v in health.items() if k != "date"}}
        rows.append(merged)
    return rows


# ── 4. Event Log: health_events.csv ──────────────────────────────────────────

def generate_events():
    """Sparse health events — headaches, allergy flare-ups, etc."""
    events = []
    for i in range(DAYS):
        d = START_DATE + timedelta(days=i)
        # Random chance of an event
        if random.random() < 0.12:  # ~11 events
            event_types = [
                ("headache", "mild", "Woke up with mild headache, resolved by noon"),
                ("headache", "moderate", "Moderate headache in afternoon, took ibuprofen"),
                ("allergy", "mild", "Mild sneezing and itchy eyes"),
                ("allergy", "moderate", "Moderate allergy symptoms, used antihistamine"),
                ("fatigue", "mild", "Felt unusually tired in the morning"),
                ("fatigue", "moderate", "Significant fatigue, low energy all day"),
                ("insomnia", "mild", "Trouble falling asleep, ~30 min delay"),
                ("insomnia", "moderate", "Woke up at 3am, couldn't fall back asleep"),
                ("exercise", "high", "Intense workout: 5K run + strength training"),
                ("exercise", "moderate", "Moderate exercise: 30 min cycling"),
                ("medication", "n/a", "Took vitamin D supplement"),
                ("stress_event", "high", "Important work deadline, high stress"),
            ]
            ev = random.choice(event_types)
            events.append({
                "date": d.isoformat(),
                "event_type": ev[0],
                "severity": ev[1],
                "notes": ev[2],
            })
    return events


# ── 5. Intervention Log: interventions.csv ───────────────────────────────────

def generate_interventions():
    """Track when the person used air purifier, changed habits, etc."""
    interventions = [
        {"start_date": "2026-05-10", "end_date": "2026-07-27", "type": "air_purifier",
         "description": "HEPA air purifier running in bedroom, 8h/night"},
        {"start_date": "2026-06-01", "end_date": "2026-07-27", "type": "supplement",
         "description": "Omega-3 supplement, 1000mg daily"},
        {"start_date": "2026-05-20", "end_date": "2026-05-27", "type": "travel",
         "description": "Business trip to Beijing (higher pollution exposure)"},
        {"start_date": "2026-06-15", "end_date": "2026-06-22", "type": "vacation",
         "description": "Vacation in rural area (clean air, low stress)"},
    ]
    return interventions


# ── Main ─────────────────────────────────────────────────────────────────────

def write_csv(filename, rows, fieldnames):
    path = OUTPUT_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  [OK] {filename}  ({len(rows)} rows) -> {path}")


def main():
    print("Generating TwinScientist test datasets…\n")

    env_rows = generate_env()
    write_csv("env_air_quality.csv", env_rows, list(env_rows[0].keys()))

    health_rows = generate_health(env_rows)
    write_csv("health_metrics.csv", health_rows, list(health_rows[0].keys()))

    merged_rows = merge(env_rows, health_rows)
    write_csv("merged_env_health.csv", merged_rows, list(merged_rows[0].keys()))

    events = generate_events()
    write_csv("health_events.csv", events, ["date", "event_type", "severity", "notes"])

    interventions = generate_interventions()
    write_csv("interventions.csv", interventions, ["start_date", "end_date", "type", "description"])

    print("\n[Done] 5 CSV files ready on your Desktop.")
    print("   Suggested test query:")
    print('   python -m main --question "How does PM2.5 exposure affect HRV and sleep quality?" --data merged_env_health.csv --iterations 3')


if __name__ == "__main__":
    main()