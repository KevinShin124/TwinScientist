"""
DALTON Dataset External Validation
====================================
Validates TwinScientist's causal inference against published findings from:

  Karmakar, S. et al. "DALTON: Daily Air Quality and Lifestyle Tracking
  Observatory Network." Indoor Air Quality in Low-to-Middle-Income
  Countries. GitHub: dalton-dataset

Published findings we test against:
  [F1] Temperature positively affects PM2.5 concentrations (thermophoretic effects)
  [F2] CO2 is elevated in kitchens during cooking hours (combustion byproduct)
  [F3] PM2.5 and CO2 are positively correlated (shared combustion source)
  [F4] Humidity is negatively correlated with temperature (physical law)
  [F5] VOC levels are higher in kitchens (cooking emissions)

Usage:
  py benchmark/dalton_validation.py
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import compute_scenario_metrics, ScenarioMetrics


# ============================================================
# DATA LOADING
# ============================================================

def load_dalton_data(data_path: Path = None) -> Dict[str, np.ndarray]:
    """Load DALTON CSV data from the processed directory."""
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "Processed"

    env_files = list(data_path.rglob("*_env.csv"))
    if not env_files:
        raise FileNotFoundError(f"No DALTON env CSV files found in {data_path}")

    print(f"  Found {len(env_files)} environment CSV files")
    all_dfs = []
    for f in env_files:
        try:
            df = pd.read_csv(f)
            all_dfs.append(df)
        except Exception:
            continue

    if not all_dfs:
        raise RuntimeError("Could not load any DALTON data")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Loaded {len(combined)} rows, {len(combined.columns)} columns")
    print(f"  Columns: {list(combined.columns)}")

    arrays: Dict[str, np.ndarray] = {}
    for col in combined.columns:
        if col.lower() == "timestamp":
            continue
        series = combined[col].dropna()
        if len(series) > 20 and series.dtype in (np.float64, np.float32, np.int64):
            arrays[col] = series.values.astype(np.float64)

    print(f"  Numeric variables: {list(arrays.keys())}")
    return arrays


# ============================================================
# PUBLISHED FINDINGS (Ground Truth)
# ============================================================

# Each finding: (cause, effect, expected_sign, paper_reference)
DALTON_FINDINGS: List[Tuple[str, str, str, str]] = [
    ("T", "PMS2_5", "positive",
     "[F1] Karmakar et al.: Temperature positively affects PM2.5"),
    ("T", "PMS10", "positive",
     "[F1] Karmakar et al.: Temperature positively affects PM10"),
    ("CO2", "PMS2_5", "positive",
     "[F3] Karmakar et al.: CO2 and PM2.5 share combustion source"),
    ("H", "T", "negative",
     "[F4] Physical law: humidity inversely related to temperature"),
    ("C2H5OH", "PMS2_5", "positive",
     "[F5] Karmakar et al.: VOC and PM co-emitted in cooking"),
]

# Pairs that should NOT show causality (negative controls)
DALTON_NULLS: List[Tuple[str, str]] = [
    ("PMS2_5", "T"),   # PM2.5 does not cause temperature
    ("CO2", "H"),       # CO2 does not directly affect humidity
]


# ============================================================
# VALIDATION
# ============================================================

def run_dalton_validation():
    """Run TwinScientist causal inference on DALTON data and compare to
    published findings."""
    print("=" * 60)
    print("  DALTON Dataset External Validation")
    print("  Karmakar et al. — Indoor Air Quality Study")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading DALTON data...")
    try:
        data = load_dalton_data()
    except FileNotFoundError:
        print("  ERROR: No DALTON data found.")
        print("  Download from: https://github.com/dalton-dataset")
        return

    # Run causal inference
    print("\n[2/3] Running TwinScientist causal inference...")
    from benchmark.runner import run_causal_inference

    test_pairs = [(c, e) for c, e, _, _ in DALTON_FINDINGS]
    test_pairs += DALTON_NULLS

    predictions = run_causal_inference(data, test_pairs, methods=["ccm", "granger"])

    # Compute metrics
    gt_pairs = [(c, e, s) for c, e, s, _ in DALTON_FINDINGS]

    metrics = compute_scenario_metrics(
        scenario_id="dalton",
        scenario_name="DALTON Dataset [Karmakar et al.]",
        ground_truth_pairs=gt_pairs,
        null_pairs=DALTON_NULLS,
        predictions=predictions,
    )

    # Results
    print(f"\n[3/3] Results:")
    print(f"  F1: {metrics.f1:.1%}")
    print(f"  Precision: {metrics.precision:.1%}")
    print(f"  Recall: {metrics.recall:.1%}")
    print(f"  Direction Accuracy: {metrics.direction_accuracy:.1%}")
    print(f"  False Positive Rate: {metrics.false_positive_rate:.1%}")
    print(f"  Correct: {metrics.n_correct_edges}/{metrics.n_ground_truth_edges}")

    print(f"\n  Per-finding breakdown:")
    for pr in metrics.pair_results:
        if pr.expected_sign != "none":
            status = "✓ MATCH" if pr.detected else "✗ MISSED"
            print(f"    {status}: {pr.cause} → {pr.effect} "
                  f"(expected {pr.expected_sign}, conf={pr.confidence:.3f})")
        else:
            status = "✓ CORRECTLY NULL" if not pr.detected else "✗ FALSE POSITIVE"
            print(f"    {status}: {pr.cause} → {pr.effect}")

    # Save results
    import json
    results_path = PROJECT_ROOT / "benchmark" / "results" / "dalton_validation.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    result_data = {
        "dataset": "DALTON (Karmakar et al.)",
        "n_rows": sum(len(v) for v in data.values()) // len(data) if data else 0,
        "n_variables": len(data) if data else 0,
        "metrics": {
            "f1": round(metrics.f1, 4),
            "precision": round(metrics.precision, 4),
            "recall": round(metrics.recall, 4),
            "direction_accuracy": round(metrics.direction_accuracy, 4),
            "fpr": round(metrics.false_positive_rate, 4),
            "correct": metrics.n_correct_edges,
            "total": metrics.n_ground_truth_edges,
        },
        "per_finding": [
            {
                "cause": pr.cause,
                "effect": pr.effect,
                "expected_sign": pr.expected_sign,
                "detected": pr.detected,
                "confidence": pr.confidence,
                "detail": pr.detail,
            }
            for pr in metrics.pair_results
        ],
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {results_path}")
    return metrics


if __name__ == "__main__":
    run_dalton_validation()
