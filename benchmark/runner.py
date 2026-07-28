"""
Benchmark Runner — Main Execution Engine
=========================================
Generates synthetic data with known ground truth, runs TwinScientist's
causal inference engine on it, and computes quantitative accuracy metrics.

Usage:
    py benchmark/runner.py              # Run all scenarios
    py benchmark/runner.py --method ccm # Test only CCM
    py benchmark/runner.py --scenario s1_temp_hr  # Single scenario
"""

from __future__ import annotations
import sys
import os
import json
import subprocess
import time
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.scenarios import (
    get_all_scenarios, BenchmarkScenario, CausalPair,
    GROUND_TRUTH_EDGES, NULL_EFFECT_PAIRS,
    ENV_COLUMNS, BIO_COLUMNS,
)
from benchmark.metrics import (
    compute_scenario_metrics, ScenarioMetrics, aggregate_metrics,
)
from benchmark.baselines import (
    correlation_baseline, granger_baseline, random_baseline,
)


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = PROJECT_ROOT / "benchmark" / "data"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
CACHE_DIR = DATA_DIR / "cache"  # Cached generated datasets


# ============================================================
# DATA GENERATION
# ============================================================

def generate_synthetic_data(
    n_subjects: int = 3,
    n_days: int = 21,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """
    Generate synthetic data using gen_multimodal_simulator.py.
    Returns path to the output directory.
    """
    output_dir = CACHE_DIR / f"s{n_subjects}_d{n_days}_seed{seed}"

    if output_dir.exists() and not force:
        csv_count = len(list(output_dir.glob("*.csv")))
        if csv_count > 0:
            print(f"  [cache] Using existing data: {output_dir} ({csv_count} CSVs)")
            return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [generate] Running gen_multimodal_simulator "
          f"(subjects={n_subjects}, days={n_days}, seed={seed})...")

    result = subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "gen_multimodal_simulator.py"),
            "--subjects", str(n_subjects),
            "--days", str(n_days),
            "--rooms", "Bedroom",
            "--n-points", str(n_days * 144),
            "--seed", str(seed),
            "--output", str(output_dir),
        ],
        capture_output=True, text=True,
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print(f"  [WARN] Simulator stderr: {result.stderr[:300]}")
        # Continue anyway — maybe data already exists

    csv_count = len(list(output_dir.rglob("*.csv")))
    print(f"  [generate] Done: {csv_count} CSV files generated")
    return output_dir


def load_and_merge_data(data_dir: Path, n_days: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Load all env + biometric CSVs, merge on timestamp, return arrays per variable.

    Args:
        data_dir: Path to CSV files
        n_days: If set, use only the first N days (for weak-effect scenarios)

    Returns:
        Dict mapping variable name → 1D numpy array
    """
    # Simulator writes to: output_dir/Processed/{subject}/{room}/*_env.csv
    env_files = sorted(data_dir.rglob("*_env.csv"))
    bio_files = sorted(data_dir.rglob("*_biometric.csv"))

    if not env_files or not bio_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    all_env_dfs = []
    all_bio_dfs = []

    for env_f, bio_f in zip(env_files, bio_files):
        try:
            env_df = pd.read_csv(env_f)
            bio_df = pd.read_csv(bio_f)

            # Normalize column names
            env_df.columns = [c.strip() for c in env_df.columns]
            bio_df.columns = [c.strip() for c in bio_df.columns]

            # Drop timestamp columns from one side to avoid duplication
            ts_cols = [c for c in bio_df.columns if 'timestamp' in c.lower() or c.lower() == 'time']
            bio_clean = bio_df.drop(columns=ts_cols, errors='ignore')

            # Row-wise merge: env and bio rows are aligned (same subject, same timestamps)
            merged = pd.concat([env_df.reset_index(drop=True),
                                bio_clean.reset_index(drop=True)], axis=1)
            all_env_dfs.append(merged)
        except Exception as e:
            print(f"  [WARN] Could not load {env_f}: {e}")
            continue

    if not all_env_dfs:
        raise RuntimeError("No data could be loaded")

    combined = pd.concat(all_env_dfs, ignore_index=True)

    # Optional: subset to first N days
    if n_days is not None:
        ts_col = "timestamp" if "timestamp" in combined.columns else combined.columns[0]
        combined[ts_col] = pd.to_datetime(combined[ts_col], errors="coerce")
        if combined[ts_col].notna().any():
            min_date = combined[ts_col].min()
            max_days = min_date + pd.Timedelta(days=n_days)
            combined = combined[combined[ts_col] <= max_days]

    # Extract arrays
    arrays: Dict[str, np.ndarray] = {}
    for col in ENV_COLUMNS + BIO_COLUMNS:
        if col in combined.columns:
            series = combined[col].dropna()
            arrays[col] = series.values.astype(np.float64)
        else:
            # Try case-insensitive match
            for c in combined.columns:
                if c.upper() == col.upper() or c.lower() == col.lower():
                    series = combined[c].dropna()
                    arrays[col] = series.values.astype(np.float64)
                    break

    print(f"  [load] Loaded {len(combined)} rows, {len(arrays)} variables")
    return arrays


# ============================================================
# CAUSAL INFERENCE
# ============================================================

def run_causal_inference(
    data: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str]],
    methods: List[str] = None,
) -> List[Tuple[str, str, Optional[str], Optional[float]]]:
    """
    Run TwinScientist's causal inference engine on specified variable pairs.

    The engine is async; each pair's methods are dispatched via asyncio.

    Returns:
        List of (cause, effect, detected_sign_or_None, confidence)
    """
    import asyncio

    if methods is None:
        methods = ["ccm", "granger"]

    GRANGER_MIN_CONF = 0.85  # Granger: 1-p >= 0.85 => p <= 0.15

    results = []
    for cause, effect in pairs:
        if cause not in data or effect not in data:
            results.append((cause, effect, None, None))
            continue

        x_arr = data[cause]
        y_arr = data[effect]

        n = min(len(x_arr), len(y_arr))
        if n < 20:
            results.append((cause, effect, None, None))
            continue

        x_list = x_arr[:n].tolist()
        y_list = y_arr[:n].tolist()
        corr = float(np.corrcoef(x_arr[:n], y_arr[:n])[0, 1])

        best_sign = None
        best_conf = 0.0
        detected = False

        async def _run_one(method: str):
            from tools.causal_inference import CausalInferenceEngine
            engine = CausalInferenceEngine()
            return await engine.run(method, x=x_list, y=y_list)

        for method in methods:
            try:
                if method == "ccm":
                    out = asyncio.run(_run_one("ccm"))
                    if isinstance(out, dict):
                        direction = out.get("causal_direction", "unclear")
                        # CCM standalone: only accept if clear direction X→Y
                        if direction in ("X→Y", "bidirectional"):
                            rho_xy = out.get("ccm_rho_x_to_y", 0)
                            rho_yx = out.get("ccm_rho_y_to_x", 0)
                            max_rho = max(rho_xy, rho_yx)
                            if max_rho > best_conf:
                                best_conf = max_rho
                                best_sign = "positive" if corr >= 0 else "negative"
                                detected = True

                elif method == "granger":
                    out = asyncio.run(_run_one("granger"))
                    if isinstance(out, dict):
                        gc = out.get("overall_granger_causality", False)
                        p_val = out.get("min_p_value", 1.0)
                        granger_conf = 1.0 - min(p_val, 0.999)
                        if gc and granger_conf >= GRANGER_MIN_CONF:
                            if granger_conf > best_conf:
                                best_conf = granger_conf
                                best_sign = "positive" if corr >= 0 else "negative"
                                detected = True

            except Exception as e:
                print(f"    [WARN] {method}({cause}, {effect}): {e}")
                continue

        if detected and best_sign is not None:
            results.append((cause, effect, best_sign, best_conf))
        else:
            results.append((cause, effect, None, 0.0))

    return results
                        


def run_pc_fci_all(
    data: Dict[str, np.ndarray],
    variables: List[str],
) -> Dict[str, Dict[str, int]]:
    """
    Run PC-FCI on all variables at once. Returns adjacency dict: {var_i: {var_j: 0|1}}
    """
    import asyncio

    var_arrays = {}
    min_len = float("inf")
    for v in variables:
        if v in data:
            var_arrays[v] = data[v]
            min_len = min(min_len, len(data[v]))

    if min_len < 20 or len(var_arrays) < 2:
        return {}

    trimmed = {}
    for v, arr in var_arrays.items():
        trimmed[v] = arr[:min_len].tolist()

    # Build data matrix: rows=samples, cols=variables
    var_list = list(trimmed.keys())
    data_matrix = []
    for i in range(min_len):
        row = [trimmed[v][i] for v in var_list]
        data_matrix.append(row)

    adj: Dict[str, Dict[str, int]] = {}
    for v in var_list:
        adj.setdefault(v, {})

    try:
        from tools.causal_inference import CausalInferenceEngine
        engine = CausalInferenceEngine()
        out = asyncio.run(engine.run("pc_fci", data=data_matrix, alpha=0.05))

        if isinstance(out, dict):
            edges = out.get("edges", [])
            for edge in edges:
                if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    src, dst = str(edge[0]), str(edge[1])
                    adj.setdefault(src, {})
                    adj[src][dst] = 1

            adj_mat = out.get("adjacency_matrix", None)
            if adj_mat is not None and isinstance(adj_mat, list):
                for i, vi in enumerate(var_list):
                    adj.setdefault(vi, {})
                    row = adj_mat[i] if i < len(adj_mat) else []
                    for j, vj in enumerate(var_list):
                        if j < len(row) and row[j] != 0:
                            adj[vi][vj] = 1
    except Exception as e:
        print(f"  [WARN] PC-FCI failed: {e}")

    return adj


def _infer_sign(x: np.ndarray, y: np.ndarray, corr: float = None) -> str:
    """Infer positive/negative direction from data."""
    if corr is None:
        if len(x) < 3:
            return "positive"
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else 0.0
    return "positive" if corr >= 0 else "negative"


# ============================================================
# MAIN RUNNER
# ============================================================

@dataclass
class BenchmarkResult:
    """Complete benchmark run results."""
    timestamp: str
    method: str
    n_scenarios: int
    scenario_metrics: List
    aggregate: Dict[str, Any]
    baselines: Dict[str, Any]


def run_benchmark(
    method: str = "ccm",
    scenario_ids: Optional[List[str]] = None,
    force_regenerate: bool = False,
):
    """
    Run the full benchmark.

    Args:
        method: Causal inference method ("ccm", "granger", "auto_select", "all")
        scenario_ids: Specific scenarios to run (None = all 10)
        force_regenerate: Re-generate synthetic data

    Returns:
        BenchmarkResult with all metrics
    """
    methods_list = [method] if method != "all" else ["ccm", "granger"]
    all_scenarios = get_all_scenarios()

    if scenario_ids:
        all_scenarios = [s for s in all_scenarios if s.id in scenario_ids]

    print(f"\n{'='*60}")
    print(f"  TwinScientist Benchmark — Method: {method}")
    print(f"  Scenarios: {len(all_scenarios)}")
    print(f"{'='*60}\n")

    # Step 1: Generate comprehensive dataset
    max_subjects = max(s.n_subjects for s in all_scenarios)
    max_days = max(s.n_days for s in all_scenarios)
    base_seed = all_scenarios[0].seed

    print(f"[1/4] Generating synthetic data...")
    data_dir = generate_synthetic_data(
        n_subjects=max_subjects,
        n_days=max_days,
        seed=base_seed,
        force=force_regenerate,
    )

    # Step 2: Load data
    print(f"\n[2/4] Loading data...")
    all_data = load_and_merge_data(data_dir)

    # For weak-effect scenarios, reload with fewer days
    weak_data = None
    for s in all_scenarios:
        if s.weak_effect:
            print(f"  [weak] Loading subset for {s.id} ({s.n_days} days)...")
            weak_data = load_and_merge_data(data_dir, n_days=s.n_days)
            break

    # Step 3: Run inference per scenario
    print(f"\n[3/4] Running causal inference...")
    scenario_metrics = []

    for i, sc in enumerate(all_scenarios):
        print(f"\n  Scenario {i+1}/{len(all_scenarios)}: {sc.id} — {sc.name}")

        data = weak_data if sc.weak_effect else all_data

        test_pairs = [(cp.cause, cp.effect) for cp in sc.causal_pairs]
        test_pairs += sc.null_pairs

        print(f"    Testing {len(test_pairs)} pairs...")

        start = time.time()
        predictions = run_causal_inference(data, test_pairs, methods=methods_list)
        elapsed = time.time() - start
        print(f"    Done in {elapsed:.1f}s ({len(predictions)} results)")

        gt_pairs = [(cp.cause, cp.effect, cp.expected_sign) for cp in sc.causal_pairs]

        sm = compute_scenario_metrics(
            scenario_id=sc.id,
            scenario_name=sc.name,
            ground_truth_pairs=gt_pairs,
            null_pairs=sc.null_pairs,
            predictions=predictions,
        )

        print(f"    F1={sm.f1:.3f}  AUC={sm.auc:.3f}  P={sm.precision:.3f}  R={sm.recall:.3f}  "
              f"DirAcc={sm.direction_accuracy:.3f}  FPR={sm.false_positive_rate:.3f}")
        scenario_metrics.append(sm)

    # Step 3b: PC-FCI on full DAG
    s10 = [s for s in all_scenarios if s.id == "s10_full_dag"]
    if s10:
        print(f"\n  [PC-FCI] Running full-DAG structure discovery...")
        available_vars = [v for v in (ENV_COLUMNS + BIO_COLUMNS) if v in all_data]
        adj = run_pc_fci_all(all_data, available_vars)

        pc_predictions = []
        for vi in adj:
            for vj, val in adj[vi].items():
                if val == 1 and vi in ENV_COLUMNS and vj in BIO_COLUMNS:
                    sign = _infer_sign(
                        all_data.get(vi, np.array([0])),
                        all_data.get(vj, np.array([0])),
                    )
                    pc_predictions.append((vi, vj, sign, 0.5))

        gt_s10 = [(cp.cause, cp.effect, cp.expected_sign) for cp in s10[0].causal_pairs]
        sm_pc = compute_scenario_metrics(
            scenario_id="s10_pc_fci",
            scenario_name="全因果图 — PC-FCI",
            ground_truth_pairs=gt_s10,
            null_pairs=s10[0].null_pairs,
            predictions=pc_predictions,
        )
        print(f"    PC-FCI F1={sm_pc.f1:.3f}  P={sm_pc.precision:.3f}  "
              f"R={sm_pc.recall:.3f}")
        scenario_metrics.append(sm_pc)

    # Step 4: Baselines
    print(f"\n[4/4] Computing baselines...")
    agg = aggregate_metrics(scenario_metrics)

    all_causal = list(set(
        (cp.cause, cp.effect) for s in all_scenarios for cp in s.causal_pairs
    ))
    all_null = list(set(p for s in all_scenarios for p in s.null_pairs))
    all_test = all_causal + all_null

    corr_preds = correlation_baseline(all_data, all_test)
    granger_preds = granger_baseline(all_data, all_test)
    rand_preds = random_baseline(all_test)

    gt_for_bl = [(cp.cause, cp.effect, cp.expected_sign)
                 for s in all_scenarios for cp in s.causal_pairs]

    corr_m = compute_scenario_metrics(
        "baseline_corr", "Pearson相关性基线", gt_for_bl, all_null, corr_preds)
    granger_m = compute_scenario_metrics(
        "baseline_granger", "纯Granger基线", gt_for_bl, all_null, granger_preds)
    rand_m = compute_scenario_metrics(
        "baseline_random", "随机基线", gt_for_bl, all_null, rand_preds)

    print(f"\n  Baseline comparison:")
    print(f"    Correlation: F1={corr_m.f1:.3f}  FPR={corr_m.false_positive_rate:.3f}")
    print(f"    Granger:     F1={granger_m.f1:.3f}  FPR={granger_m.false_positive_rate:.3f}")
    print(f"    Random:      F1={rand_m.f1:.3f}  FPR={rand_m.false_positive_rate:.3f}")

    baselines = {
        "correlation": {"f1": round(corr_m.f1, 4), "precision": round(corr_m.precision, 4),
                        "recall": round(corr_m.recall, 4), "fpr": round(corr_m.false_positive_rate, 4)},
        "granger": {"f1": round(granger_m.f1, 4), "precision": round(granger_m.precision, 4),
                    "recall": round(granger_m.recall, 4), "fpr": round(granger_m.false_positive_rate, 4)},
        "random": {"f1": round(rand_m.f1, 4), "precision": round(rand_m.precision, 4),
                   "recall": round(rand_m.recall, 4), "fpr": round(rand_m.false_positive_rate, 4)},
    }

    result = BenchmarkResult(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        method=method,
        n_scenarios=len(all_scenarios),
        scenario_metrics=scenario_metrics,
        aggregate=agg,
        baselines=baselines,
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "metrics.json"

    serializable = {
        "timestamp": result.timestamp,
        "method": result.method,
        "n_scenarios": result.n_scenarios,
        "aggregate": result.aggregate,
        "baselines": result.baselines,
        "per_scenario": [
            {
                "id": m.scenario_id,
                "name": m.scenario_name,
                "f1": round(m.f1, 4),
                "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "direction_accuracy": round(m.direction_accuracy, 4),
                "sign_accuracy": round(m.sign_accuracy, 4),
                "false_positive_rate": round(m.false_positive_rate, 4),
                "tp": m.true_positives, "fp": m.false_positives,
                "fn": m.false_negatives,
                "correct": m.n_correct_edges,
                "total_gt": m.n_ground_truth_edges,
                "detected": m.n_detected_edges,
            }
            for m in scenario_metrics
        ],
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {results_path}")
    return result


def run_external_benchmarks(method: str = "all"):
    """
    Run external academic benchmarks (Sugihara 2012, Granger 1969, Runge 2019).
    These are canonical tests from peer-reviewed literature with known ground truth.
    """
    from benchmark.external_benchmarks import get_external_benchmarks

    methods_list = [method] if method != "all" else ["ccm", "granger"]
    benchmarks = get_external_benchmarks()

    print(f"\n{'='*60}")
    print(f"  TwinScientist — External Academic Benchmarks")
    print(f"  Benchmark sources: Sugihara 2012 (Science), Granger 1969 (Econometrica),")
    print(f"                     Runge 2019 (Science Advances)")
    print(f"  Method: {method}")
    print(f"{'='*60}\n")

    scenario_metrics = []

    for i, bm in enumerate(benchmarks):
        print(f"\n  Benchmark {i+1}/{len(benchmarks)}: {bm.name}")
        data = bm.generate()

        test_pairs = [(c, e) for c, e, _, _ in bm.ground_truth]
        test_pairs += bm.null_pairs

        print(f"    Testing {len(test_pairs)} pairs...")
        predictions = run_causal_inference(data, test_pairs, methods=methods_list)

        gt_pairs = [(c, e, s) for c, e, s, _ in bm.ground_truth]

        sm = compute_scenario_metrics(
            scenario_id=f"ext_{i+1}",
            scenario_name=bm.name,
            ground_truth_pairs=gt_pairs,
            null_pairs=bm.null_pairs,
            predictions=predictions,
        )
        print(f"    F1={sm.f1:.3f}  AUC={sm.auc:.3f}  P={sm.precision:.3f}  "
              f"R={sm.recall:.3f}  DirAcc={sm.direction_accuracy:.3f}")
        scenario_metrics.append(sm)

    agg = aggregate_metrics(scenario_metrics)

    result = BenchmarkResult(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        method=method,
        n_scenarios=len(benchmarks),
        scenario_metrics=scenario_metrics,
        aggregate=agg,
        baselines={},
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "metrics_external.json"

    serializable = {
        "timestamp": result.timestamp,
        "method": result.method,
        "n_scenarios": result.n_scenarios,
        "aggregate": result.aggregate,
        "per_scenario": [
            {
                "id": m.scenario_id, "name": m.scenario_name,
                "f1": round(m.f1, 4), "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "direction_accuracy": round(m.direction_accuracy, 4),
                "auc": round(m.auc, 4),
                "correct": m.n_correct_edges, "total_gt": m.n_ground_truth_edges,
            }
            for m in scenario_metrics
        ],
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"\n  External benchmark results saved to: {results_path}")
    return result


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TwinScientist Benchmark Runner")
    parser.add_argument("--method", default="ccm",
                        choices=["ccm", "granger", "auto_select", "all"],
                        help="Causal inference method to test")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run a single scenario (e.g., s1_temp_hr)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-generation of synthetic data")
    parser.add_argument("--report", action="store_true",
                        help="Generate markdown report after benchmark")
    parser.add_argument("--external", action="store_true",
                        help="Run external academic benchmarks (Sugihara/Granger/Runge)")

    args = parser.parse_args()

    if args.external:
        result = run_external_benchmarks(method=args.method)
        if args.report:
            from benchmark.report import generate_report
            generate_report(result)
    else:
        scenario_ids = [args.scenario] if args.scenario else None

        result = run_benchmark(
            method=args.method,
            scenario_ids=scenario_ids,
            force_regenerate=args.force,
        )

    if args.report:
        from benchmark.report import generate_report
        generate_report(result)
