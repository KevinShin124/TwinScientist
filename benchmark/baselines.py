"""
Baseline Methods — Comparison Points for TwinScientist
=======================================================
Three baselines to demonstrate TwinScientist's incremental value:

1. Correlation Baseline: Pearson r → threshold → claim causality
2. Granger-Only Baseline: Pure Granger causality without the full pipeline
3. Random Baseline: Lower-bound performance reference
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import math
import numpy as np


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    xm, ym = x - x.mean(), y - y.mean()
    denom = math.sqrt((xm * xm).sum() * (ym * ym).sum())
    if denom == 0:
        return 0.0
    return float((xm * ym).sum() / denom)


def correlation_baseline(
    data: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str]],
    threshold: float = 0.3,
) -> List[Tuple[str, str, Optional[str], Optional[float]]]:
    """
    Correlation baseline: if |r| > threshold, claim causality.
    Sign determined by correlation direction.

    Returns: List of (cause, effect, detected_sign, confidence)
    """
    results = []
    for cause, effect in pairs:
        if cause not in data or effect not in data:
            results.append((cause, effect, None, None))
            continue
        r = _pearson_r(data[cause], data[effect])
        if abs(r) >= threshold:
            sign = "positive" if r > 0 else "negative"
            conf = min(abs(r), 1.0)
            results.append((cause, effect, sign, conf))
        else:
            results.append((cause, effect, None, 0.0))
    return results


def granger_baseline(
    data: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str]],
    max_lag: int = 5,
    p_threshold: float = 0.05,
) -> List[Tuple[str, str, Optional[str], Optional[float]]]:
    """
    Pure Granger causality baseline (no pipeline, no LLM, no tournament).

    Uses simple OLS-based Granger test.

    Returns: List of (cause, effect, detected_sign, confidence)
    """
    results = []
    for cause, effect in pairs:
        if cause not in data or effect not in data:
            results.append((cause, effect, None, None))
            continue

        x = data[cause]
        y = data[effect]

        # Ensure equal length
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]

        best_p = 1.0
        best_sign = None

        for lag in range(1, min(max_lag + 1, n // 10)):
            try:
                # Restricted model: y_t = α + Σ β_i y_{t-i} + ε
                # Full model:      y_t = α + Σ β_i y_{t-i} + Σ γ_i x_{t-i} + ε
                T = n - lag
                if T < 20:
                    continue

                # Build restricted model (AR only)
                Y = y[lag:]
                X_restricted = np.column_stack([y[lag - i - 1:n - i - 1] for i in range(lag)])
                X_restricted = np.column_stack([np.ones(T), X_restricted])

                # Restricted OLS
                try:
                    beta_restricted = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
                    resid_restricted = Y - X_restricted @ beta_restricted
                    rss_restricted = float(resid_restricted @ resid_restricted)
                except np.linalg.LinAlgError:
                    continue

                # Build full model (AR + X lags)
                X_full = np.column_stack([
                    X_restricted,
                    *[x[lag - i - 1:n - i - 1] for i in range(lag)]
                ])

                try:
                    beta_full = np.linalg.lstsq(X_full, Y, rcond=None)[0]
                    resid_full = Y - X_full @ beta_full
                    rss_full = float(resid_full @ resid_full)
                except np.linalg.LinAlgError:
                    continue

                # F-test
                df1 = lag
                df2 = T - 1 - 2 * lag
                if df2 <= 0:
                    continue
                f_stat = ((rss_restricted - rss_full) / df1) / (rss_full / df2) if rss_full > 0 else float('inf')

                # Approximate p-value from F-distribution
                # Using Wilson-Hilferty approximation for chi-square, then
                # relationship between F and chi-square
                if f_stat > 0 and not math.isinf(f_stat):
                    # Simple approximation: convert F to p using
                    # F ~ (chi²_df1 / df1) / (chi²_df2 / df2)
                    # For large df2, F ≈ chi²_df1 / df1
                    import scipy.stats as stats
                    try:
                        p = 1.0 - stats.f.cdf(f_stat, df1, df2)
                    except Exception:
                        # Fallback: rough estimate
                        p = math.exp(-0.5 * f_stat)
                else:
                    p = 1.0

                if p < best_p:
                    best_p = p

                    # Determine sign: sign of average x coefficient
                    x_coeffs = beta_full[-lag:]
                    avg_coeff = float(np.mean(x_coeffs))
                    best_sign = "positive" if avg_coeff > 0 else "negative"

            except Exception:
                continue

        if best_p < p_threshold and best_sign is not None:
            conf = 1.0 - best_p
            results.append((cause, effect, best_sign, conf))
        else:
            results.append((cause, effect, None, 0.0))

    return results


def random_baseline(
    pairs: List[Tuple[str, str]],
    seed: int = 42,
) -> List[Tuple[str, str, Optional[str], Optional[float]]]:
    """
    Random baseline: 50% chance of claiming causality (lower bound).
    """
    import random
    rng = random.Random(seed)
    results = []
    for cause, effect in pairs:
        if rng.random() < 0.5:
            sign = rng.choice(["positive", "negative"])
            conf = rng.random()
            results.append((cause, effect, sign, conf))
        else:
            results.append((cause, effect, None, 0.0))
    return results
