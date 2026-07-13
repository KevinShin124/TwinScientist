"""Dalton 合成数据生成器 — 创建可直接使用的测试数据"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import math
import random

def generate_dalton_data(
    n_points_per_device: int = 2000,
    houses: list | None = None,
    rooms: list | None = None,
    date_range_days: int = 7,
    seed: int = 42,
    output_dir: str | None = None,
):
    """Generate Dalton-format sensor data. One CSV per device."""

    rng = random.Random(seed)

    if houses is None:
        houses = ["H1", "H2"]
    if rooms is None:
        rooms = ["Study_Desk", "Kitchen", "Bedroom"]

    if output_dir is None:
        output_dir = str(Path.home() / "Desktop" / "dalton-dataset-main")

    output_path = Path(output_dir)
    total_files = 0

    for house in houses:
        for room in rooms:
            base_ts = datetime.now() - timedelta(days=date_range_days)
            timestamps = []
            current_time = base_ts

            for i in range(n_points_per_device):
                offset = rng.uniform(300, 900)
                current_time += timedelta(seconds=offset)
                timestamps.append(current_time)

            # Now write ONE CSV with all points
            csv_filename = f"{house}_{room}.csv"
            device_dir = output_path / "Processed" / house
            device_dir.mkdir(parents=True, exist_ok=True)

            csv_path = device_dir / csv_filename

            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write("timestamp,T,CO2,VOC,NO2,PMS1,PMS10,PMS2_5,C2H5OH,H\n")

                for ts in timestamps:
                    hour_of_day = ts.hour + ts.minute / 60
                    daily_cycle = 3.0 * math.sin(2 * math.pi * (hour_of_day - 6) / 24)

                    T = round(22.0 + daily_cycle + rng.gauss(0, 0.5), 2)
                    CO2 = round(400 + daily_cycle * 80 + rng.gauss(0, 30), 1)
                    VOC = round(50 + abs(daily_cycle) * 20 + rng.gauss(0, 10), 2)
                    NO2 = round(rng.uniform(5, 40), 1)
                    PM1 = round(max(0, 10 + rng.gauss(0, 3)), 2)
                    PM10 = round(max(0, 25 + rng.gauss(0, 5)), 2)
                    PM2_5 = round(max(0, 15 + rng.gauss(0, 4)), 2)
                    C2H5OH = round(max(0, rng.expovariate(1/5)), 2)
                    H = round(50 + daily_cycle * 2 + rng.gauss(0, 2), 2)

                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{ts_str},{T},{CO2},{VOC},{NO2},{PM1},{PM10},{PM2_5},{C2H5OH},{H}\n")

            total_files += 1

    return str(output_path), total_files


if __name__ == "__main__":
    output_dir, count = generate_dalton_data()
    print("")
    print("="*60)
    print(f"  DONE! Generated {count} CSV files")
    print(f"  Location: {output_dir}/Processed/")
    print("="*60)
    print("")
    print("Next steps:")
    print("  1. Copy to twinScientist data/sensors/:")
    print(f'     cp -r "{output_dir}\\Processed\\*" "twinScientist\\data\\sensors\\"')
    print("  2. Or run `python -m main` then type [数据上传]")
    print("")
