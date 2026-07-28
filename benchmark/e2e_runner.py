"""
End-to-End Pipeline Benchmark
Tests the COMPLETE TwinScientist pipeline (L1-L5).
Usage: py benchmark/e2e_runner.py --scenario s1_temp_hr
"""

from __future__ import annotations
import sys, os, json, asyncio, time, shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.scenarios import get_all_scenarios, BenchmarkScenario
from benchmark.metrics import compute_scenario_metrics, aggregate_metrics
from benchmark.runner import generate_synthetic_data

RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"


def run_e2e_pipeline(scenario, data_dir):
    from config.settings import settings
    from core.graph import cognitive_graph
    import uuid

    causal_pairs = scenario.causal_pairs
    if not causal_pairs:
        return {"status": "no_causal_pairs", "evidence_chains": []}

    first_pair = causal_pairs[0]
    question = (f"Does {first_pair.cause} causally affect {first_pair.effect} "
                f"in indoor environments?")

    sensors_dir = PROJECT_ROOT / "data" / "sensors"
    backup_dir = sensors_dir.parent / "sensors_backup"
    original_files = set()

    if sensors_dir.exists():
        backup_dir.mkdir(exist_ok=True)
        for f in sensors_dir.glob("*"):
            if f.is_file():
                shutil.copy2(f, backup_dir / f.name)
                original_files.add(f.name)

    sensors_dir.mkdir(parents=True, exist_ok=True)
    env_files = list(data_dir.rglob("*_env.csv"))
    bio_files = list(data_dir.rglob("*_biometric.csv"))

    # Merge env + bio CSVs only (skip visual fatigue)
    import pandas as pd
    copied = 0
    for env_f, bio_f in zip(env_files, bio_files):
        if 'visual' in str(env_f).lower() or 'visual' in str(bio_f).lower():
            continue
        env_df = pd.read_csv(env_f)
        bio_df = pd.read_csv(bio_f)
        ts_cols = [c for c in bio_df.columns if 'timestamp' in c.lower() or c == 'time']
        bio_clean = bio_df.drop(columns=ts_cols, errors='ignore')
        merged = pd.concat([env_df.reset_index(drop=True),
                            bio_clean.reset_index(drop=True)], axis=1)
        dest = sensors_dir / f"benchmark_{env_f.stem.replace('_env', '')}.csv"
        merged.to_csv(dest, index=False)
        copied += 1

    print(f"    Copied {copied} data files")

    settings.max_iterations = 3
    initial_state = {
        "query": question,
        "domain": "environment-human health",
        "_max_iterations_": 3,
        "auto_confirm": True,
        "iteration": 1,
    }

    thread_id = f"benchmark-{scenario.id}-{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 500}

    # Pre-convert any numpy types in initial state
    import numpy as np
    for k, v in list(initial_state.items()):
        if isinstance(v, (np.integer, np.floating)):
            initial_state[k] = v.item()
        elif isinstance(v, np.ndarray):
            initial_state[k] = v.tolist()

    result = {}
    try:
        async def _run():
            import traceback
            last_state = {}
            try:
                async for event in cognitive_graph.astream_events(
                    initial_state, config, version="v2"
                ):
                    kind = event.get("event", "")
                    name = event.get("name", "")
                    if kind == "on_chain_end":
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            last_state.update(output)
                return last_state
            except Exception as inner_e:
                print(f"    [TRACEBACK] {traceback.format_exc()[-500:]}")
                raise
        result = asyncio.run(_run())
    except Exception as e:
        print(f"    [ERROR] Pipeline failed: {e}")
        result = {"status": "error", "error": str(e)}
    finally:
        for f in sensors_dir.glob("benchmark_*"):
            if f.exists():
                f.unlink()
        for name in original_files:
            src = backup_dir / name
            if src.exists():
                shutil.copy2(src, sensors_dir / name)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

    return {
        "status": "ok" if "error" not in result else "error",
        "evidence_chains": result.get("evidence_chains", []),
        "iterations": result.get("iteration", 0),
        "question": question,
        "error": result.get("error", ""),
    }


def parse_causal_conclusions(evidence_chains):
    predictions = []
    for chain in evidence_chains:
        if not isinstance(chain, dict):
            continue
        direction = chain.get("causal_direction", "")
        strength = chain.get("strength", 0.0)
        cause = effect = None
        for sep in ("->", "→"):
            if sep in direction:
                parts = direction.split(sep)
                if len(parts) == 2:
                    cause, effect = parts[0].strip(), parts[1].strip()
                    break
        if cause and effect:
            confidence = float(strength) if isinstance(strength, (int, float)) else 0.5
            predictions.append((cause, effect, "positive", confidence))
    return predictions


def run_e2e_benchmark(scenario_ids=None):
    all_scenarios = get_all_scenarios()
    if scenario_ids:
        all_scenarios = [s for s in all_scenarios if s.id in scenario_ids]

    print()
    print("=" * 60)
    print(f"  TwinScientist End-to-End Pipeline Benchmark")
    print(f"  Scenarios: {len(all_scenarios)}")
    print("=" * 60)
    print()

    scenario_metrics = []

    for i, sc in enumerate(all_scenarios):
        print(f"  Scenario {i+1}/{len(all_scenarios)}: {sc.id} - {sc.name}")
        print(f"    Generating data...")

        data_dir = generate_synthetic_data(
            n_subjects=sc.n_subjects,
            n_days=sc.n_days,
            seed=sc.seed,
            force=True,
        )

        print(f"    Running full pipeline...")
        start = time.time()
        result = run_e2e_pipeline(sc, data_dir)
        elapsed = time.time() - start
        print(f"    Pipeline completed in {elapsed:.0f}s")

        if result.get("status") != "ok":
            print(f"    [SKIP] Pipeline failed: {result.get('error', 'unknown')}")
            continue

        predictions = parse_causal_conclusions(result["evidence_chains"])
        print(f"    Evidence chains: {len(result['evidence_chains'])}")
        print(f"    Parsed predictions: {len(predictions)}")

        if not predictions:
            print(f"    [WARN] No causal conclusions extracted")
            continue

        gt_pairs = [(cp.cause, cp.effect, cp.expected_sign) for cp in sc.causal_pairs]
        sm = compute_scenario_metrics(
            scenario_id=sc.id,
            scenario_name=sc.name,
            ground_truth_pairs=gt_pairs,
            null_pairs=sc.null_pairs,
            predictions=predictions,
        )

        print(f"    F1={sm.f1:.3f}  P={sm.precision:.3f}  R={sm.recall:.3f}  "
              f"DirAcc={sm.direction_accuracy:.3f}")
        scenario_metrics.append(sm)

    agg = aggregate_metrics(scenario_metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "e2e_metrics.json"

    serializable = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_scenarios": len(scenario_metrics),
        "aggregate": agg,
        "per_scenario": [
            {
                "id": m.scenario_id, "name": m.scenario_name,
                "f1": round(m.f1, 4), "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "direction_accuracy": round(m.direction_accuracy, 4),
                "correct": m.n_correct_edges, "total_gt": m.n_ground_truth_edges,
            }
            for m in scenario_metrics
        ],
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"  Results saved to: {results_path}")
    return serializable


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TwinScientist E2E Pipeline Benchmark")
    parser.add_argument("--scenario", type=str, default=None)
    args = parser.parse_args()
    scenario_ids = [args.scenario] if args.scenario else None
    run_e2e_benchmark(scenario_ids=scenario_ids)
