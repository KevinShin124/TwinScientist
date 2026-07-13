"""数据自动整理模块

功能：
- 用户上传任意 CSV 文件，模块自动检测内容并分类到对应目录
- 支持 Daltons 传感器格式、生物信号（PPG/HRV）、视觉疲劳（眼动）等
- 自动跳过无法识别的文件并给出提示

用法：
    python -m data.organizer                   # 扫描 data/upload/ 下的所有文件并自动分类
    python -m data.organizer --dir ./my_data   # 指定待整理目录
"""

from __future__ import annotations

import csv
import shutil
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================
# 基础目录定义
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
UPLOAD_DIR = BASE_DIR / "data" / "upload"          # 用户放文件的入口目录
DATA_DIR = BASE_DIR / "data"                       # 目标数据根目录

SENSOR_DIR = DATA_DIR / "sensors"
BIOMETRIC_DIR = DATA_DIR / "biometric"
VISUAL_DIR = DATA_DIR / "visual_fatigue"

# ============================================================
# 列名匹配规则 —— 用于检测数据类型
# ============================================================

# 环境传感器数据（Daltons 标准列名 + 常见中文别名）
ENV_SENSOR_COLS = {
    "T", "H", "CO2", "CO2_ppm", "co2", "carbon_dioxide",
    "NO2", "no2", "nitrogen_dioxide", "NOx",
    "PMS1", "PMS10", "PMS2_5", "pm2_5", "pm25", "pm1", "pm10",
    "VOC", "voc", "tvoc", "vocs", "VoC",
    "C2H5OH", "ethanol",
    "pressure", "气压", "温度", "湿度", "空气质量",
}

# 生物信号数据
BIOMETRIC_COLS = {
    "ppg", "PPG", "photoplethysmography", "光电容积脉搏波",
    "hrv", "sdnn", "rmssd", "心率变异性",
    "heart_rate", "hr", "bpm", "心率",
    "spo2", "spO2", "血氧", "oxygen_saturation",
    "ecg", "eeg", "eog", "血压", "blood_pressure",
    "respiratory", "呼吸", "temperature_skin", "皮肤温度",
}

# 视觉疲劳数据
VISUAL_COLS = {
    "eye", "眼球", "gaze", "注视", "pupil", "瞳孔", "blink", "眨眼",
    "yaw", "pitch", "roll", "head_pose", "头部姿态",
    "facial", "面部", "expression", "表情", "smile", "微笑",
    "drowsiness", "困倦", "fatigue", "疲劳", "yawn", "打哈欠",
    "video", "摄像头", "图像", "image", "frame",
}

# 时间戳通用列名
TIMESTAMP_COLS = {"ts", "timestamp", "date_time", "time", "datetime", "日期", "时间"}


def detect_data_type(filepath: Path) -> tuple[str, float]:
    """
    读取 CSV 前几行，根据列名判断数据类型。

    Returns:
        (category, confidence)
        category: "sensor" | "biometric" | "visual" | "unknown"
        confidence: 0~1，越接近 1 表示判断越确信
    """
    if not filepath.suffix.lower() in (".csv", ".tsv"):
        return "unknown", 0.0

    try:
        sample_rows = []
        dialect = None

        # Try reading with csv.Sniffer for proper delimiter detection
        raw_text = filepath.read_text(encoding="utf-8-sig")[:4096]
        try:
            dialect = csv.Sniffer().sniff(raw_text)
        except csv.Error:
            dialect = csv.excel  # fallback

        with open(filepath, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f, dialect=dialect)
            for i, row in enumerate(reader):
                if i >= 3:  # Read at most first 3 rows
                    break
                if row:
                    sample_rows.append([cell.strip() for cell in row])

        if len(sample_rows) < 2:
            return "unknown", 0.0

        header = {cell.upper() for cell in sample_rows[0]}
        header_lower = {cell.lower() for cell in sample_rows[0]}
        header_all = header | header_lower | {cell.replace(" ", "") for cell in sample_rows[0]}

        # Count matching columns
        env_hits = sum(1 for c in header_all if c in ENV_SENSOR_COLS)
        bio_hits = sum(1 for c in header_all if c in BIOMETRIC_COLS)
        visual_hits = sum(1 for c in header_all if c in VISUAL_COLS)

        total_numeric_cols = sum(1 for cell in sample_rows[0][1:] if _is_numeric(cell))

        # Confidence scoring
        has_timestamp = any(c in TIMESTAMP_COLS for c in header_all)

        scores = {}
        if env_hits > 0 and has_timestamp:
            scores["sensor"] = min(env_hits * 0.3 + 0.5, 1.0)
        elif env_hits > 0:
            scores["sensor"] = min(env_hits * 0.2, 0.8)

        if bio_hits > 0 and has_timestamp:
            scores["biometric"] = min(bio_hits * 0.3 + 0.5, 1.0)
        elif bio_hits > 0:
            scores["biometric"] = min(bio_hits * 0.2, 0.8)

        if visual_hits > 0 and has_timestamp:
            scores["visual"] = min(visual_hits * 0.3 + 0.5, 1.0)
        elif visual_hits > 0:
            scores["visual"] = min(visual_hits * 0.2, 0.8)

        # Also check by numeric content (for sensor data with numeric col names)
        numeric_in_header = [c for c in sample_rows[0] if c.isdigit()]
        dalton_pollutants = {"T", "H", "CO2", "CO", "NO2", "VOC", "PMS1", "PMS10", "PMS2_5", "C2H5OH"}
        if set(numeric_in_header) & dalton_pollutants and has_timestamp:
            scores["sensor"] = max(scores.get("sensor", 0), 0.9)

        if not scores:
            return "unknown", 0.0

        best_cat = max(scores, key=scores.get)
        best_score = scores[best_cat]

        return best_cat, best_score

    except Exception as e:
        logger.warning(f"[DataOrganizer] Failed to read {filepath}: {e}")
        return "unknown", 0.0


def _is_numeric(value: str) -> bool:
    """Check if string is a numeric value."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def organize_file(filepath: Path, force_dest: str | None = None) -> dict:
    """
    整理单个文件。

    Args:
        filepath: 待整理的文件路径
        force_dest: 强制指定目标目录名 ("sensor"|"biometric"|"visual")

    Returns:
        {
            "original": path_str,
            "filename": name,
            "category": detected_category or "unknown",
            "dest_dir": target_directory,
            "dest_path": final_filepath,
            "moved": bool,  # whether file was actually moved
            "skipped": bool,  # whether file was skipped
            "reason": explanation message
        }
    """
    original = filepath.resolve()
    filename = original.name

    if force_dest:
        dest_map = {
            "sensor": SENSOR_DIR,
            "biometric": BIOMETRIC_DIR,
            "visual": VISUAL_DIR,
        }
        dest_dir = dest_map.get(force_dest.lower(), DATA_DIR)
        category = force_dest.lower()
        reason = f"用户指定移动到 {force_dest} 目录"
    else:
        category, confidence = detect_data_type(original)
        if category == "unknown":
            return {
                "original": str(original),
                "filename": filename,
                "category": "unknown",
                "dest_dir": "",
                "dest_path": "",
                "moved": False,
                "skipped": True,
                "reason": "无法识别的数据类型（请确保 CSV 第一行为列名，且包含相关指标）",
            }

        dest_map = {
            "sensor": SENSOR_DIR,
            "biometric": BIOMETRIC_DIR,
            "visual": VISUAL_DIR,
        }
        dest_dir = dest_map[category]
        reason = f"自动检测为{get_chinese_name(category)}数据 (置信度={confidence:.1%})"

    # Move file
    dest_path = dest_dir / filename

    # Handle duplicate names
    if dest_path.exists():
        stem = original.stem
        suffix = original.suffix
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(original), str(dest_path))

    return {
        "original": str(original),
        "filename": filename,
        "category": category,
        "dest_dir": str(dest_dir),
        "dest_path": str(dest_path),
        "moved": True,
        "skipped": False,
        "reason": reason,
    }


def get_chinese_name(category: str) -> str:
    """返回类别的中文描述"""
    return {
        "sensor": "环境传感器",
        "biometric": "生物信号",
        "visual": "视觉疲劳",
        "unknown": "未知",
    }.get(category, "未知")


def organize_directory(dir_path: str | Path = None, verbose: bool = True) -> list[dict]:
    """
    批量整理指定目录下的所有文件。

    Args:
        dir_path: 待整理的根目录，默认为 data/upload/
        verbose: 是否打印详细日志

    Returns:
        所有处理结果的列表
    """
    src = Path(dir_path) if dir_path else UPLOAD_DIR

    if not src.exists():
        src.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"\n[data-upload] 上传目录已创建: {src}")
            print(f"[data-upload] 请将您的数据文件放入此目录后重新运行整理命令\n")
        return []

    results = []

    # Find all CSV/TSV files recursively
    files = sorted(src.glob("**/*"), recursive=True)
    data_files = [f for f in files if f.is_file() and f.suffix.lower() in (".csv", ".tsv")]

    if not data_files:
        if verbose:
            print(f"\n[data-upload] {src} 下没有找到 CSV/TSV 文件\n")
        return []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  [data-upload] 数据自动整理器 v1.0")
        print(f"  待整理文件数: {len(data_files)}")
        print(f"  源目录: {src}")
        print(f"{'='*60}\n")

    moved_count = 0
    unknown_count = 0

    for filepath in data_files:
        result = organize_file(filepath)
        results.append(result)

        if result["skipped"]:
            unknown_count += 1
            if verbose:
                print(f"  ⚠️  跳过: {result['filename']}")
                print(f"     原因: {result['reason']}")
        else:
            moved_count += 1
            if verbose:
                chinese = get_chinese_name(result["category"])
                dest_name = {
                    "sensor": "环境传感器",
                    "biometric": "生物信号",
                    "visual": "视觉疲劳",
                }.get(result["category"], "")
                print(f"  ✅  移至 data/{dest_name}/ ({chinese}): {result['filename']}")

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  完成！共处理 {len(data_files)} 个文件")
        print(f"  ✅ 成功移动: {moved_count}")
        print(f"  ⚠️  未识别: {unknown_count}")
        print(f"{'─'*60}\n")

    return results


if __name__ == "__main__":
    import sys

    # Ensure directories exist
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    SENSOR_DIR.mkdir(parents=True, exist_ok=True)
    BIOMETRIC_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

    args = sys.argv[1:]
    if "--verbose" in args:
        VERBOSE = True
    else:
        VERBOSE = True  # Always verbose for CLI

    organize_directory(verbose=VERBOSE)
