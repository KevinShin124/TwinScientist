"""DALTON 室内空气质量数据集 — 适配 twinScientist 的数据管道

功能：
- 自动识别 DALTON 原始格式 / 预处理格式 / 合并格式
- 提取时间戳列并为每个污染物生成独立记录流
- 如果没有真实数据，使用 Daltons 标准列名生成合成测试数据
- 输出兼容 TimeSeriesChannel.ingest_csv() 的格式
"""

from __future__ import annotations

import os
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Daltons 定义的污染物列名（从 library/constants.py）
DALTON_POLLUTANTS = ['C2H5OH', 'CO', 'CO2', 'NO2', 'PMS1', 'PMS10', 'PMS2_5', 'VoC']
DALTON_ENV_VARS = ['T', 'H']  # Temperature, Humidity
ALL_SENSORS = DALTON_POLLUTANTS + DALTON_ENV_VARS


def detect_dalton_format(records: list[dict]) -> str:
    """自动检测 DALTON CSV 格式类型"""
    if not records:
        return "unknown"
    sample = records[0]
    keys = set(sample.keys())

    if 'ts' in keys and 'ID' in keys and 'Loc' in keys:
        return "merged"  # Merged/data_H1.csv 格式
    elif 'timestamp' in keys or 'date_time' in keys:
        return "processed"  # Processed/房间/日期/device.csv 格式
    elif any(k in keys for k in ALL_SENSORS):
        return "raw_sensor"  # Raw sensor output (one file per pollutant)
    else:
        return "unknown"


def parse_timestamp(ts_str: str) -> str:
    """标准化时间戳为 ISO 格式"""
    try:
        # Daltons ts 格式: "YYYY-MM-DD HH:MM:SS"
        dt = datetime.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S")
        return dt.isoformat() + "+00:00"
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(ts_str.strip())
        return dt.isoformat()
    except ValueError:
        return ts_str.strip()


def extract_records_from_dalton(
    records: list[dict],
    format_type: str,
) -> list[dict]:
    """
    将 Daltons 格式的原始记录转换为 twinScientist ingest 期望的标准格式。

    返回每条记录的列表：[{sensor_type, pollutant_name, timestamp, value, reading}]
    """
    extracted = []

    if format_type == "merged":
        # 每行包含多个污染物，需要展开
        for row in records:
            ts_raw = row.get('ts', '')
            ts = parse_timestamp(ts_raw)
            device_id = row.get('ID', '')
            location = row.get('Loc', '')

            for col in ALL_SENSORS:
                val = row.get(col)
                if val is not None and val != '' and val != 'NA':
                    try:
                        numeric_val = float(val)
                        if not np.isnan(numeric_val):
                            extracted.append({
                                "sensor_type": "dalton_iot",
                                "pollutant_name": col,
                                "device_id": device_id,
                                "location": location,
                                "timestamp": ts,
                                "value": numeric_val,
                                "reading": f"{col}={numeric_val}",
                            })
                    except (ValueError, TypeError):
                        pass

    elif format_type == "processed":
        # Processed 文件夹的 CSV（通常只包含一个设备/一天的数据）
        ts_key = 'timestamp' if 'timestamp' in records[0] else 'date_time'
        for row in records:
            ts = parse_timestamp(str(row.get(ts_key, '')))
            # 其他列为传感器读数
            for key, val in row.items():
                if key == ts_key:
                    continue
                try:
                    numeric_val = float(val)
                    if not np.isnan(numeric_val):
                        extracted.append({
                            "sensor_type": "dalton_processed",
                            "pollutant_name": key,
                            "timestamp": ts,
                            "value": numeric_val,
                            "reading": f"{key}={numeric_val}",
                        })
                except (ValueError, TypeError):
                    pass

    elif format_type == "raw_sensor":
        # Raw sensor output（单列数据，文件名决定传感器类型）
        sensor_cols = [k for k in records[0].keys() if k not in ('ts', 'timestamp', 'date_time')]
        for col in sensor_cols:
            for row in records:
                ts_raw = row.get('ts') or row.get('timestamp') or row.get('date_time')
                if ts_raw is not None:
                    ts = parse_timestamp(str(ts_raw))
                    val = row.get(col)
                    try:
                        numeric_val = float(val)
                        if not np.isnan(numeric_val):
                            extracted.append({
                                "sensor_type": "dalton_raw",
                                "pollutant_name": col,
                                "timestamp": ts,
                                "value": numeric_val,
                                "reading": f"{col}={numeric_val}",
                            })
                    except (ValueError, TypeError):
                        pass

    return extracted


def generate_synthetic_dalton_data(
    n_points_per_device: int = 500,
    houses: list[str] | None = None,
    rooms: list[str] | None = None,
    date_range_days: int = 7,
) -> dict[str, list[list[dict]]]:
    """
    生成符合 Daltons 格式的合成时序数据用于 pipeline 测试。

    模拟信号关系（真实世界近似值）：
    - 温度(T) ↑ → CO2 ↑ (人体活动增加)
    - VOC ↑ → PM2.5 ↑ (厨房烹饪场景)
    - 湿度(H) ↑ → PMS1/PMS10 轻微影响

    Args:
        n_points_per_device: 每台设备的采样点数
        houses: 房屋编号列表
        rooms: 房间名称列表
        date_range_days: 数据覆盖天数

    Returns:
        {house/room_path: [[{...}] for each device]} — 嵌套列表格式匹配 Daltons 目录结构
    """
    if houses is None:
        houses = ["H1"]  # 最小化样本
    if rooms is None:
        rooms = ["Kitchen"]

    rng = np.random.default_rng(42)

    all_output = {}

    for house in houses:
        room_data = {}
        for room in rooms:
            # 创建多条时间序列（不同传感器位置）
            devices = [f"{house}_{room}_sensor1", f"{house}_{room}_sensor2"]
            room_files = []

            for device_idx, device_id in enumerate(devices):
                base_ts = datetime.now() - timedelta(days=date_range_days)
                intervals = rng.uniform(5 * 60, 15 * 60, n_points_per_device).cumsum()

                timestamps = []
                records = []

                t_base = 22.0  # base temperature (°C)
                co2_base = 400  # base CO2 (ppm)
                voc_base = 50   # base VOC (µg/m³)
                humidity_base = 50  # base humidity (%)

                for i, offset_s in enumerate(intervals):
                    ts = base_ts + timedelta(seconds=int(offset_s))
                    timestamps.append(ts.isoformat())

                    # Simulate realistic signal relationships
                    # Daily cycle: temperature varies sinusoidally
                    hour_of_day = ts.hour + ts.minute / 60
                    daily_cycle = 3.0 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)

                    T = round(t_base + daily_cycle + rng.normal(0, 0.5), 2)
                    CO2 = round(co2_base + daily_cycle * 80 + rng.normal(0, 30), 1)  # temp→co2 causal signal
                    VOC = round(voc_base + abs(daily_cycle) * 20 + rng.normal(0, 10), 2)
                    NO2 = round(rng.uniform(5, 40), 1)
                    PM1 = round(max(0, 10 + rng.normal(0, 3)), 2)
                    PM10 = round(max(0, 25 + rng.normal(0, 5)), 2)
                    PM2_5 = round(max(0, 15 + rng.normal(0, 4)), 2)
                    C2H5OH = round(max(0, rng.exponential(5)), 2)
                    H = round(humidity_base + daily_cycle * 2 + rng.normal(0, 2), 2)

                    record = {
                        "ts": timestamps[-1],
                        "ID": device_id,
                        "Loc": f"{house}/{room}",
                        "T": T,
                        "CO2": CO2,
                        "VOC": VOC,
                        "NO2": NO2,
                        "PMS1": PM1,
                        "PMS10": PM10,
                        "PMS2_5": PM2_5,
                        "C2H5OH": C2H5OH,
                        "H": H,
                    }
                    records.append(record)

                room_files.append((records, f"{device_id}.csv"))

            room_data[f"{house}/{room}"] = room_files

        all_output[house] = list(room_data.values())

    return all_output


def write_dalton_csv(data: list[dict], output_path: str) -> str:
    """将提取后的记录写入标准 CSV 文件"""
    from pathlib import Path
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not data:
        logger.warning(f"[DaltonIngest] No data for {output_path}, skipping")
        return ""

    columns = list(data[0].keys())
    with open(out, 'w', encoding='utf-8') as f:
        f.write(",".join(columns) + "\n")
        for row in data:
            values = []
            for col in columns:
                val = row.get(col, "")
                if isinstance(val, (list, dict)):
                    val = str(val)
                values.append(str(val))
            f.write(",".join(values) + "\n")

    logger.info(f"[DaltonIngest] Wrote {len(data)} records to {output_path}")
    return str(out)


def ingest_dalton_dataset(data_dir: str) -> dict[str, list[dict]]:
    """
    主入口函数：读取 Daltons 数据集并返回标准化数据结构。

    Args:
        data_dir: Daltons 数据根目录路径

    Returns:
        {source_label: [records]} — source_label 为 "house_room_date" 格式
    """
    from pathlib import Path
    import pandas as pd

    root = Path(data_dir)
    result = {}

    if not root.exists():
        logger.warning(f"[DaltonIngest] Data directory not found: {data_dir}")
        return {}

    # 遍历所有 CSV 文件
    csv_files = list(root.glob("**/*.csv"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            records = df.to_dict(orient="records")
            fmt = detect_dalton_format(records)
            parsed = extract_records_from_dalton(records, fmt)

            if parsed:
                source_label = str(csv_file.relative_to(root)).replace(os.sep, "_")
                result[source_label] = parsed

        except Exception as e:
            logger.error(f"[DaltonIngest] Failed to process {csv_file}: {e}")
            continue

    return result
