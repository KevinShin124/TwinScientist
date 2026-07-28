"""
Benchmark Metrics — Quantitative Evaluation of Causal Discovery
================================================================
All metrics compare the system's predicted causal relationships against
the known ground truth from gen_multimodal_simulator.py.

Metrics:
  - Edge F1: Precision / Recall / F1 for causal edge detection
  - Direction Accuracy: Correct direction among detected edges
  - Sign Accuracy: Correct +/- sign among detected edges
  - False Positive Rate: Spurious edges in zero-effect scenarios
  - SHD: Structural Hamming Distance (for graph-based methods)
  - Coverage: Fraction of ground-truth edges detected
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Set, Any
from dataclasses import dataclass, field
import math


@dataclass
class SingleResult:
    """Result for one causal pair test."""
    cause: str
    effect: str
    expected_sign: str          # "positive" | "negative" | "none"
    detected: bool
    direction_correct: bool | None   # None if no ground-truth direction
    sign_correct: bool | None        # None if not applicable
    confidence: float = 0.0
    method_used: str = ""
    detail: str = ""


@dataclass
class ScenarioMetrics:
    """Aggregate metrics for one benchmark scenario."""
    scenario_id: str
    scenario_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc: float = 0.0              # Ranking quality: can system rank true > false?
    direction_accuracy: float = 0.0
    sign_accuracy: float = 0.0
    coverage: float = 0.0
    false_positive_rate: float = 0.0
    n_ground_truth_edges: int = 0
    n_detected_edges: int = 0
    n_correct_edges: int = 0
    pair_results: List = field(default_factory=list)


def _edge_key(cause: str, effect: str) -> str:
    """Normalize edge to a hashable key."""
    return f"{cause}->{effect}"


def compute_scenario_metrics(
    scenario_id: str,
    scenario_name: str,
    ground_truth_pairs: List[Tuple[str, str, str]],
    null_pairs: List[Tuple[str, str]],
    predictions: List[Tuple[str, str, str | None, float | None]],
) -> ScenarioMetrics:
    """
    Compute all metrics for one scenario.
    """
    m = ScenarioMetrics(scenario_id=scenario_id, scenario_name=scenario_name)

    gt_edges: Set[str] = set()
    gt_signs: Dict[str, str] = {}
    for cause, effect, sign in ground_truth_pairs:
        key = _edge_key(cause, effect)
        gt_edges.add(key)
        gt_signs[key] = sign

    null_edges: Set[str] = set()
    for cause, effect in null_pairs:
        null_edges.add(_edge_key(cause, effect))

    pred_edges: Set[str] = set()
    pred_signs: Dict[str, str] = {}
    pred_confs: Dict[str, float] = {}
    for cause, effect, sign, conf in predictions:
        key = _edge_key(cause, effect)
        if sign is not None:
            pred_edges.add(key)
            pred_signs[key] = sign
            pred_confs[key] = conf or 0.0

    all_tested = gt_edges | null_edges | pred_edges

    for key in all_tested:
        in_gt = key in gt_edges
        in_pred = key in pred_edges
        if in_gt and in_pred:
            m.true_positives += 1
        elif in_gt and not in_pred:
            m.false_negatives += 1
        elif not in_gt and in_pred:
            m.false_positives += 1
        else:
            m.true_negatives += 1

    m.n_ground_truth_edges = len(gt_edges)
    m.n_detected_edges = len(pred_edges)
    m.n_correct_edges = m.true_positives

    if m.true_positives + m.false_positives > 0:
        m.precision = m.true_positives / (m.true_positives + m.false_positives)
    if m.true_positives + m.false_negatives > 0:
        m.recall = m.true_positives / (m.true_positives + m.false_negatives)
    if m.precision + m.recall > 0:
        m.f1 = 2 * m.precision * m.recall / (m.precision + m.recall)

    m.coverage = m.recall

    total_negatives = m.false_positives + m.true_negatives
    if total_negatives > 0:
        m.false_positive_rate = m.false_positives / total_negatives

    # AUC: ranking quality — does system assign higher confidence to true edges?
    gt_confs = [pred_confs.get(_edge_key(c, e), 0.0) for c, e, _ in ground_truth_pairs]
    null_confs = [pred_confs.get(_edge_key(c, e), 0.0) for c, e in null_pairs]
    if gt_confs and null_confs:
        n_correct = sum(1 for gc in gt_confs for nc in null_confs if gc > nc)
        n_ties = sum(1 for gc in gt_confs for nc in null_confs if gc == nc)
        total = len(gt_confs) * len(null_confs)
        m.auc = (n_correct + 0.5 * n_ties) / total if total > 0 else 0.5

    correct_dir = 0
    correct_sign = 0
    for key in gt_edges & pred_edges:
        gt_sign = gt_signs.get(key, "")
        pred_sign = pred_signs.get(key, "")
        if pred_sign in ("bidirectional", gt_sign, "positive", "negative"):
            correct_dir += 1
        if gt_sign and pred_sign and gt_sign == pred_sign:
            correct_sign += 1

    if m.true_positives > 0:
        m.direction_accuracy = correct_dir / m.true_positives
        m.sign_accuracy = correct_sign / m.true_positives

    for cause, effect, sign in ground_truth_pairs:
        key = _edge_key(cause, effect)
        detected = key in pred_edges
        m.pair_results.append(SingleResult(
            cause=cause, effect=effect, expected_sign=sign,
            detected=detected,
            direction_correct=(key in pred_edges),
            sign_correct=(pred_signs.get(key) == sign) if detected else None,
            confidence=pred_confs.get(key, 0.0),
            detail="OK" if detected else "MISSED",
        ))

    for cause, effect in null_pairs:
        key = _edge_key(cause, effect)
        detected = key in pred_edges
        m.pair_results.append(SingleResult(
            cause=cause, effect=effect, expected_sign="none",
            detected=detected,
            direction_correct=(not detected),
            sign_correct=None,
            confidence=pred_confs.get(key, 0.0),
            detail="FALSE_POSITIVE" if detected else "CORRECT_NULL",
        ))

    return m


def compute_shd(gt_adj: Dict[str, Dict[str, int]],
                pred_adj: Dict[str, Dict[str, int]]) -> int:
    """Structural Hamming Distance between two adjacency matrices."""
    all_vars = sorted(set(list(gt_adj.keys()) + list(pred_adj.keys())))
    shd = 0
    for i in all_vars:
        for j in all_vars:
            if i == j:
                continue
            gt_val = gt_adj.get(i, {}).get(j, 0)
            pred_val = pred_adj.get(i, {}).get(j, 0)
            if gt_val != pred_val:
                shd += 1
    return shd


def aggregate_metrics(all_metrics: List[ScenarioMetrics]) -> Dict[str, Any]:
    """Compute macro-averaged metrics across all scenarios."""
    if not all_metrics:
        return {}

    n = len(all_metrics)

    def safe_mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    total_tp = sum(m.true_positives for m in all_metrics)
    total_fp = sum(m.false_positives for m in all_metrics)
    total_fn = sum(m.false_negatives for m in all_metrics)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall /
                (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

    return {
        "n_scenarios": n,
        "total_ground_truth_edges": sum(m.n_ground_truth_edges for m in all_metrics),
        "total_detected_correctly": sum(m.n_correct_edges for m in all_metrics),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(safe_mean([m.precision for m in all_metrics]), 4),
        "macro_recall": round(safe_mean([m.recall for m in all_metrics]), 4),
        "macro_f1": round(safe_mean([m.f1 for m in all_metrics]), 4),
        "macro_direction_accuracy": round(
            safe_mean([m.direction_accuracy for m in all_metrics]), 4),
        "macro_sign_accuracy": round(
            safe_mean([m.sign_accuracy for m in all_metrics]), 4),
        "macro_false_positive_rate": round(
            safe_mean([m.false_positive_rate for m in all_metrics]), 4),
        "macro_auc": round(safe_mean([m.auc for m in all_metrics]), 4),
        "per_scenario": [
            {
                "id": m.scenario_id,
                "name": m.scenario_name,
                "f1": round(m.f1, 4),
                "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "auc": round(m.auc, 4),
                "direction_acc": round(m.direction_accuracy, 4),
                "sign_acc": round(m.sign_accuracy, 4),
                "fpr": round(m.false_positive_rate, 4),
                "correct": m.n_correct_edges,
                "total": m.n_ground_truth_edges,
                "fp": m.false_positives,
            }
            for m in all_metrics
        ],
    }
