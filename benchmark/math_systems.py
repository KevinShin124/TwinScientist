"""
Standard Mathematical Validation Systems
=========================================
Two mathematically-defined dynamical systems with analytically
verifiable ground-truth causal structures. These are NOT "self-generated"
— any researcher can verify the ground truth from the equations.

1. Lorenz Attractor — the gold standard for CCM validation
2. VAR(2) Model — the gold standard for Granger causality validation
"""

import numpy as np
from typing import Dict, List, Tuple


# ============================================================
# LORENZ SYSTEM
# ============================================================
# dx/dt = σ(y - x)
# dy/dt = x(ρ - z) - y
# dz/dt = xy - βz
#
# Standard parameters: σ=10, ρ=28, β=8/3 (chaotic regime)
#
# GROUND TRUTH CAUSAL EDGES (derived from equation structure):
#   x → y  (because x appears in dy/dt)
#   y → x  (because y appears in dx/dt)
#   x → z  (because x appears in dz/dt via xy term)
#   y → z  (because y appears in dz/dt via xy term)
#   z → y  (because z appears in dy/dt via -xz term)
#
# These are mathematical facts, not assumptions.

LORENZ_GROUND_TRUTH: List[Tuple[str, str, str, str]] = [
    ("x", "y", "positive", "x 出现在 dy/dt 方程中"),
    ("y", "x", "positive", "y 出现在 dx/dt 方程中"),
    ("x", "z", "positive", "x 出现在 dz/dt 方程中"),
    ("y", "z", "positive", "y 出现在 dz/dt 方程中"),
    ("z", "y", "negative", "z 出现在 dy/dt 方程中（-xz 项）"),
]


def generate_lorenz(
    n_points: int = 2000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    seed: int = 42,
    noise_std: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    Generate Lorenz attractor time series.

    Args:
        n_points: Number of time steps (after discarding transient)
        dt: Integration time step
        sigma, rho, beta: Lorenz system parameters
        seed: Random seed
        noise_std: Observational noise standard deviation

    Returns:
        Dict with keys "x", "y", "z" → 1D numpy arrays
    """
    rng = np.random.default_rng(seed)

    # Discard transient (first 1000 steps)
    n_transient = 1000
    total_steps = n_transient + n_points

    x, y, z = 1.0, 1.0, 1.0
    xs = np.zeros(total_steps)
    ys = np.zeros(total_steps)
    zs = np.zeros(total_steps)

    for i in range(total_steps):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x += dx
        y += dy
        z += dz
        xs[i] = x
        ys[i] = y
        zs[i] = z

    # Discard transient, add noise
    xs = xs[n_transient:] + rng.normal(0, noise_std, n_points)
    ys = ys[n_transient:] + rng.normal(0, noise_std, n_points)
    zs = zs[n_transient:] + rng.normal(0, noise_std, n_points)

    return {"x": xs, "y": ys, "z": zs}


# ============================================================
# VAR(2) MODEL
# ============================================================
# A 5-variable Vector Autoregressive model of order 2 with known
# coefficient matrices. The Granger-causal structure is determined
# by which coefficients are non-zero.
#
# Variables: v0, v1, v2, v3, v4
#
# Coefficient matrices (lag 1 and lag 2):
#
# Lag-1:  A1[i,j] ≠ 0 means vj Granger-causes vi at lag 1
#          v0   v1   v2   v3   v4
#   v0  [ 0.5  0.3  0    0    0   ]   v0(t) = 0.5*v0(t-1) + 0.3*v1(t-1)
#   v1  [ 0    0.4  0.2  0    0   ]   v1(t) = 0.4*v1(t-1) + 0.2*v2(t-1)
#   v2  [ 0    0    0.6  0.3  0   ]   v2(t) = 0.6*v2(t-1) + 0.3*v3(t-1)
#   v3  [ 0    0    0    0.5  0.2 ]   v3(t) = 0.5*v3(t-1) + 0.2*v4(t-1)
#   v4  [ 0.1  0    0    0    0.5 ]   v4(t) = 0.1*v0(t-1) + 0.5*v4(t-1)
#
# Lag-2:  A2[i,j] ≠ 0 means vj Granger-causes vi at lag 2
#          v0   v1   v2   v3   v4
#   v0  [ 0.2  0    0    0    0   ]
#   v1  [ 0    0.2  0    0    0   ]
#   v2  [ 0    0    0.2  0    0   ]
#   v3  [ 0    0    0    0.2  0   ]
#   v4  [ 0    0    0    0    0.2 ]
#
# GROUND TRUTH GRANGER-CAUSAL EDGES (vj → vi if A1[i,j]≠0 or A2[i,j]≠0):
#   v1 → v0  (A1[0,1]=0.3)
#   v2 → v1  (A1[1,2]=0.2)
#   v3 → v2  (A1[2,3]=0.3)
#   v4 → v3  (A1[3,4]=0.2)
#   v0 → v4  (A1[4,0]=0.1)
# Plus self-loops (autoregressive, not causal between variables):
#   v0→v0, v1→v1, v2→v2, v3→v3, v4→v4 (at lag 1 and lag 2)

VAR_GROUND_TRUTH: List[Tuple[str, str, str, str]] = [
    ("v1", "v0", "positive", "A1[0,1]=0.3: v1(t-1) → v0(t)"),
    ("v2", "v1", "positive", "A1[1,2]=0.2: v2(t-1) → v1(t)"),
    ("v3", "v2", "positive", "A1[2,3]=0.3: v3(t-1) → v2(t)"),
    ("v4", "v3", "positive", "A1[3,4]=0.2: v4(t-1) → v3(t)"),
    ("v0", "v4", "positive", "A1[4,0]=0.1: v0(t-1) → v4(t)"),
]

# NULL pairs for VAR: variables that are NOT causally connected
VAR_NULL_PAIRS: List[Tuple[str, str]] = [
    ("v0", "v2"),  # v0 does NOT Granger-cause v2
    ("v2", "v4"),  # v2 does NOT Granger-cause v4
    ("v4", "v1"),  # v4 does NOT Granger-cause v1
    ("v3", "v0"),  # v3 does NOT Granger-cause v0
    ("v1", "v3"),  # v1 does NOT Granger-cause v3
]


def generate_var2(
    n_points: int = 2000,
    seed: int = 42,
    noise_std: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Generate VAR(2) time series with known Granger-causal structure.

    Args:
        n_points: Number of time steps
        seed: Random seed
        noise_std: Innovation noise standard deviation

    Returns:
        Dict with keys "v0".."v4" → 1D numpy arrays
    """
    rng = np.random.default_rng(seed)
    n_vars = 5

    # Lag-1 coefficient matrix
    A1 = np.array([
        [0.5, 0.3, 0.0, 0.0, 0.0],
        [0.0, 0.4, 0.2, 0.0, 0.0],
        [0.0, 0.0, 0.6, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.2],
        [0.1, 0.0, 0.0, 0.0, 0.5],
    ])

    # Lag-2 coefficient matrix (diagonal only — self-feedback)
    A2 = np.diag([0.2, 0.2, 0.2, 0.2, 0.2])

    # Burn-in
    n_burn = 500
    total = n_burn + n_points
    data = np.zeros((total, n_vars))

    for t in range(2, total):
        innovation = rng.normal(0, noise_std, n_vars)
        data[t] = A1 @ data[t - 1] + A2 @ data[t - 2] + innovation

    result = {}
    for i in range(n_vars):
        result[f"v{i}"] = data[n_burn:, i]

    return result
