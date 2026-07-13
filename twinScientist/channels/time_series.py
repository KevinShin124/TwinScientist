"""
Layer 5 - Item 18: Multi-Source Time-Series Engine

支持环境传感器 + PPG + 血氧多源异步接入、时间对齐（互相关）、信号质量评估。

当前为半功能实现：数据加载框架已就绪，算法部分使用 numpy/scipy 实现基础版本。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================
# Daltons IoT Dataset Helpers — from NeurIPS 2024 DALTON paper
# ============================================================
DALTON_POLLUTANTS = ['C2H5OH', 'CO', 'CO2', 'NO2', 'PMS1', 'PMS10', 'PMS2_5', 'VoC']
DALTON_ENV_VARS = ['T', 'H']
DALTON_ALL_COLS = DALTON_POLLUTANTS + DALTON_ENV_VARS


def _detect_daltons_format(records: list[dict]) -> str:
    """自动检测 Daltons CSV 格式类型"""
    if not records:
        return "unknown"
    keys = set(records[0].keys())
    if 'ts' in keys and 'ID' in keys and 'Loc' in keys:
        return "merged"  # Merged/data_H1.csv
    elif 'timestamp' in keys or 'date_time' in keys:
        return "processed"  # Processed/room_date/device.csv
    elif any(k in keys for k in DALTON_ALL_COLS):
        return "raw_sensor"
    return "unknown"


def _parse_daltons_records(
    records: list[dict],
    single_sensor_file: bool = False,
) -> list[dict]:
    """将 Daltons 格式的原始记录转换为 twinScientist ingest 期望的标准格式"""
    import numpy as np
    extracted = []
    fmt = _detect_daltons_format(records)

    if fmt == "merged" and not single_sensor_file:
        for row in records:
            ts_raw = row.get('ts', '')
            device_id = row.get('ID', '')
            location = row.get('Loc', '')
            for col in DALTON_ALL_COLS:
                val = row.get(col)
                if val is not None and val != '' and val != 'NA':
                    try:
                        numeric_val = float(val)
                        if not np.isnan(numeric_val):
                            extracted.append({
                                "timestamp": ts_raw,
                                "value": numeric_val,
                                "pollutant_name": col,
                                "device_id": device_id,
                                "location": location,
                                "reading": f"{col}={numeric_val}",
                            })
                    except (ValueError, TypeError):
                        pass
    else:
        # Default / processed / raw: first valid column is the sensor value
        for row in records:
            ts_raw = row.get('ts') or row.get('timestamp') or row.get('date_time', '')
            for key, val in row.items():
                if key.lower() in ('ts', 'timestamp', 'date_time', 'id', 'loc', 'device'):
                    continue
                try:
                    numeric_val = float(val)
                    if not np.isnan(numeric_val):
                        extracted.append({
                            "timestamp": str(ts_raw),
                            "value": numeric_val,
                            "pollutant_name": key,
                            "reading": f"{key}={numeric_val}",
                        })
                        break  # One reading per record
                except (ValueError, TypeError):
                    pass

    return extracted


# ============================================================
# Core Classes
# ============================================================


class SignalQualityEvaluator:
    """信号质量评估器"""

    @staticmethod
    def assess_ppg(signal: list[float]) -> dict:
        """
        PPG 信号质量评估

        Returns: {quality_score, issues, recommendations}
        """
        import numpy as np
        if len(signal) < 10:
            return {"quality_score": 0.0, "issues": ["样本数不足"], "recommendations": ["增加采样时长"]}

        arr = np.array(signal)
        snr_ratio = (np.max(arr) - np.min(arr)) / (np.std(arr) + 1e-10)
        # NaN check using numpy which correctly detects nan values
        nan_mask = np.isnan(arr)
        missing_count = int(nan_mask.sum())
        missing_ratio = missing_count / len(signal)

        quality = min(1.0, max(0.0, (snr_ratio / 100.0) * (1.0 - missing_ratio)))

        issues = []
        if missing_ratio > 0.05:
            issues.append(f"缺失值比例 {missing_ratio:.1%}")
        if snr_ratio < 10:
            issues.append("信噪比过低")
        if np.abs(np.diff(arr)).mean() < 1e-6:
            issues.append("信号几乎无变化（可能设备故障）")

        return {
            "quality_score": round(quality, 3),
            "signal_length": len(signal),
            "snr_ratio": round(snr_ratio, 2),
            "missing_ratio": round(missing_ratio, 4),
            "issues": issues,
            "recommendations": [f"建议检查{issue}" for issue in issues] if issues else ["信号质量良好"],
        }

    @staticmethod
    def assess_env_sensor(signal: list[dict], expected_cols: list[str]) -> dict:
        """
        环境传感器数据质量评估
        """
        if not signal:
            return {"quality_score": 0.0, "issues": ["空数据集"]}

        import numpy as np
        valid_rows = sum(1 for row in signal if all(k in row and row[k] is not None for k in expected_cols))
        quality = valid_rows / len(signal)

        missing_cols = [c for c in expected_cols if c not in signal[0]]
        issues = []
        if quality < 0.9:
            issues.append(f"完整率仅 {quality:.1%}")
        if missing_cols:
            issues.append(f"缺失列: {missing_cols}")

        return {
            "quality_score": round(quality, 3),
            "total_records": len(signal),
            "valid_records": valid_rows,
            "missing_columns": missing_cols,
            "issues": issues,
        }


class TimeSeriesChannel:
    """多源时序数据通道 — 接入、对齐、质量评估"""

    def __init__(self, sensor_dir: str, biometric_dir: str, visual_dir: str):
        self.sensor_dir = Path(sensor_dir)
        self.biometric_dir = Path(biometric_dir)
        self.visual_dir = Path(visual_dir)
        self.evaluator = SignalQualityEvaluator()

    async def connect(self) -> None:
        for d in [self.sensor_dir, self.biometric_dir, self.visual_dir]:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"[TimeSeriesChannel] Ready — dirs: {self.sensor_dir}, {self.biometric_dir}, {self.visual_dir}")

    async def ingest_csv(self, file_path: str, data_type: str = "sensor") -> list[dict]:
        """
        从 CSV 文件加载数据（通用读取器 + Daltons IoT 传感器格式适配）

        Args:
            file_path: CSV 文件路径
            data_type: 数据类型 ("sensor", "dalton_merged", "dalton_processed")

        Returns:
            - 普通模式: [{timestamp, value}] 标准时间序列记录
            - dalton_merged: 自动展开为多污染物的列表，每个记录包含 pollutant_name
            - dalton_processed: 解析 Processed/ 目录的单一设备格式
        """
        import pandas as pd
        from channels.time_series import _detect_daltons_format, _parse_daltons_records

        filepath = Path(file_path)
        if not filepath.exists():
            logger.warning(f"[TimeSeriesChannel] File not found: {file_path}")
            return []

        try:
            df = pd.read_csv(filepath)
            records = df.to_dict(orient="records")
            logger.info(f"[TimeSeriesChannel] Loaded {len(records)} records from {filepath.name}")

            # Daltons IoT merged format (Merged/data_H1.csv): one row has all pollutants
            if data_type == "dalton_merged":
                return _parse_daltons_records(records)

            # Daltons processed format (Processed/House/Date/device.csv): single sensor per file
            if data_type == "dalton_processed":
                return _parse_daltons_records(records, single_sensor_file=True)

            # Default: standard time series with ts/value columns
            standard = []
            for rec in records:
                ts_raw = rec.get('ts') or rec.get('timestamp') or rec.get('date_time', '')
                val_raw = rec.get('value') or rec.get('reading', '')
                if ts_raw and val_raw != '':
                    try:
                        standard.append({
                            "timestamp": ts_raw,
                            "value": float(val_raw),
                            "reading": str(rec),
                        })
                    except (ValueError, TypeError):
                        pass

            logger.info(f"[TimeSeriesChannel] Standardized to {len(standard)} records")
            return standard

        except Exception as e:
            logger.error(f"[TimeSeriesChannel] Failed to read {file_path}: {e}")
            return []

    async def align_by_timestamp(
        self,
        datasets: dict[str, list[dict]],
        time_key: str = "timestamp",
        tolerance_ms: int = 1000,
    ) -> dict[str, list[dict]]:
        """
        多源时序数据时间对齐

        核心方法：基于时间戳的最近邻对齐（nearest-neighbor alignment）
        对每个数据集的主时间戳，找到其他数据集在容忍范围内的最近点。

        Args:
            datasets: {source_name: [{time_key, value}, ...]}
            time_key: 时间字段名
            tolerance_ms: 对齐容忍窗口（毫秒）

        Returns: 对齐后的数据集（每条记录都包含所有源的最近值）
        """
        if not datasets:
            return {}

        # Find common time range across all datasets
        all_times = []
        for source, records in datasets.items():
            for rec in records:
                t = rec.get(time_key)
                if t is not None:
                    all_times.append(t)

        if not all_times:
            return datasets

        # Use numpy/scipy-based nearest neighbor alignment
        try:
            import numpy as np
            from bisect import bisect_left

            master_time = sorted(set(all_times))[:1000]  # cap for performance
            aligned = {}

            # Pre-extract and sort timestamps for binary search per source
            ds_index = {}
            for source, records in datasets.items():
                ts_data = [(float(r.get(time_key, 0)), r) for r in records if r.get(time_key) is not None]
                ts_data.sort(key=lambda x: x[0])
                ds_index[source] = [t for t, _ in ts_data], [r for _, r in ts_data]

            for master_t in master_time:
                combined = {"master_timestamp": master_t}
                for source in datasets:
                    timestamps, recs = ds_index.get(source, ([], []))
                    if not timestamps:
                        combined[f"{source}_value"] = None
                        continue

                    lo = bisect_left(timestamps, master_t)

                    best_rec = None
                    best_dist = tolerance_ms + 1

                    candidates = []
                    if lo < len(timestamps):
                        candidates.append(lo)
                    if lo > 0:
                        candidates.append(lo - 1)
                    if lo + 1 < len(timestamps):
                        candidates.append(lo + 1)

                    for idx in candidates:
                        dist = abs(timestamps[idx] - master_t)
                        if dist <= tolerance_ms and dist < best_dist:
                            best_dist = dist
                            best_rec = recs[idx]

                    combined[f"{source}_value"] = best_rec.get("value", best_rec) if best_rec else None

                aligned.setdefault(str(master_t), {}).update(combined)

            result = list(aligned.values())
            logger.info(f"[TimeSeriesChannel] Aligned to {len(result)} points")
            return {k: result for k in datasets.keys()}

        except ImportError:
            logger.warning("[TimeSeriesChannel] numpy not available — using simple alignment")
            return datasets  # fallback: return unchanged

    async def cross_correlation(self, series_a: list[float], series_b: list[float]) -> dict:
        """
        互相关函数 — 检测两个时间序列之间的延迟和方向性关系

        Returns: {max_correlation, optimal_lag, direction, significant}
        """
        import numpy as np

        n = min(len(series_a), len(series_b))
        a = np.array(series_a[:n])
        b = np.array(series_b[:n])

        # Remove mean
        a = a - a.mean()
        b = b - b.mean()

        norm = np.sqrt(np.sum(a**2) * np.sum(b**2)) + 1e-10
        correlations = np.correlate(a, b, mode='full') / norm

        optimal_lag = np.argmax(correlations) - (n - 1)
        max_corr = correlations[optimal_lag]

        # Direction: positive lag means a leads b (b's peak comes after a's)
        if optimal_lag > 0:
            direction = "a→b"   # a changes first, b follows
        elif optimal_lag < 0:
            direction = "b→a"   # b changes first, a follows
        else:
            direction = "simultaneous"
        significant = abs(max_corr) > 0.3  # threshold for significance

        return {
            "max_correlation": round(float(max_corr), 4),
            "optimal_lag": int(optimal_lag),
            "direction": direction,
            "significant": significant,
        }

    async def assess_signal_quality(self, signal_name: str, signal_data: list[dict]) -> dict:
        """统一信号质量评估入口"""
        if not signal_data:
            return {"signal": signal_name, "quality_score": 0.0, "issues": ["空数据"]}

        if isinstance(signal_data[0], dict):
            values = []
            for item in signal_data:
                v = item.get("value") or item.get("reading") or item.get("measurement")
                if v is not None:
                    try:
                        values.append(float(v))
                    except (ValueError, TypeError):
                        pass
            return self.evaluator.assess_ppg(values) if values else {"signal": signal_name, "quality_score": 0.0}
        else:
            return self.evaluator.assess_ppg(signal_data)
