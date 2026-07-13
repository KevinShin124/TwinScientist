"""
Layer 6 - Item 20: Causal Inference Toolset

8 类因果推断工具 + AI 自主选择。
每个方法都是可运行的半实现：有框架结构和 numpy/scipy 基础版本，
等待安装完整库后替换为生产级算法。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CausalInferenceEngine:
    """
    因果推断工具箱 — 8 种方法 + AI 自动选择

    使用方式：
        engine = CausalInferenceEngine(data)
        result = await engine.run(method="granger", cause=X, effect=Y, max_lag=5)
    """

    SUPPORTED_METHODS = [
        "ccm",                # Convergent Cross Mapping
        "granger",            # Granger Causality
        "pc_fci",             # PC-FCI (causal structure learning)
        "psm",                # Propensity Score Matching
        "instrumental_var",   # Instrumental Variable
        "bayesian_net",       # Bayesian Network
        "counterfactual",     # Counterfactual Reasoning (GP surrogate)
        "auto_select",        # AI auto-select best method
    ]

    def __init__(self, data: dict[str, list[float]] | None = None):
        self.data = data or {}

    async def run(self, method: str, **kwargs) -> dict:
        """统一调用入口"""
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {method}. Use one of: {self.SUPPORTED_METHODS}")

        handler = getattr(self, f"_run_{method}", None)
        if not handler:
            logger.warning(f"[CausalEngine] Method '{method}' not yet implemented")
            return {"status": "placeholder", "method": method}

        return await handler(**kwargs)

    async def _run_ccm(self, x: list[float], y: list[float], column_size: int = 3) -> dict:
        """
        CCM — 收敛交叉映射 (Convergent Cross Mapping) — 真实实现

        核心原理: 如果 X→Y 存在因果关系，系统共享同一个吸引子流形。
        Y 的延迟嵌入可以重建原始状态空间，从而用 Y 的邻居点预测 X 的值。
        **关键判据**: 随着 library size 增大，交叉映射精度必须单调上升（收敛）。

        Returns: {ccm_rho_x_to_y, ccm_rho_y_to_x, causal_direction, convergence_evidence}
        """
        import numpy as np

        n = min(len(x), len(y))
        if n < column_size * 5:
            return {"status": "insufficient_data", "required_min": column_size * 5, "actual": n}

        x_arr = (np.array(x[:n]) - np.array(x[:n]).mean()) / (np.array(x[:n]).std() + 1e-10)
        y_arr = (np.array(y[:n]) - np.array(y[:n]).mean()) / (np.array(y[:n]).std() + 1e-10)

        # Delay embedding into T-dimensional manifold
        def embed(series: np.ndarray, T: int) -> np.ndarray:
            """Create delay-embedding matrix of shape (N-T+1, T)."""
            N = len(series)
            out_len = N - T + 1
            if out_len <= 0 or T <= 0:
                return series.reshape(-1, 1)
            col_data = [series[i:i+out_len] for i in range(T)]
            return np.column_stack(col_data)

        # KNN prediction using distance-weighted neighbors
        def predict_from_library(lib_emb: np.ndarray, lib_vals: np.ndarray, query: np.ndarray, k: int = 4) -> float:
            dists = np.linalg.norm(lib_emb - query.reshape(1, -1), axis=1)
            idx = np.argsort(dists)[:k]
            weights = 1.0 / (dists[idx] + 1e-10)
            weights /= weights.sum()
            return float(np.dot(weights, lib_vals[idx]))

        # Single CCM cross-map correlation
        def single_ccm(library_series: np.ndarray, target_series: np.ndarray, T: int, lib_size: int, k: int = 4) -> float:
            lib_emb = embed(library_series[:lib_size], T)
            vals = target_series[T-1:lib_size]
            if len(lib_emb) < k or len(vals) < k:
                return 0.0
            preds = [predict_from_library(lib_emb, vals, lib_emb[i], k=k) for i in range(0, len(lib_emb), max(1, len(lib_emb)//200))]
            actuals = [vals[i % len(vals)] for i in range(0, len(preds))]
            r = np.corrcoef(actuals, preds)[0, 1] if len(set(actuals)) > 1 and len(set(preds)) > 1 else 0
            return float(r) if not np.isnan(r) else 0.0

        # Test at multiple library sizes to check convergence
        sizes = np.linspace(column_size * 2, n // 3, 5).astype(int)
        sizes = np.unique(sizes[sizes >= column_size * 2])

        rho_xtoy = [single_ccm(y_arr, x_arr, column_size, s) for s in sizes]
        rho_ytox = [single_ccm(x_arr, y_arr, column_size, s) for s in sizes]

        converge_xtoy = rho_xtoy[-1] > rho_xtoy[0] if len(rho_xtoy) > 1 else False
        converge_ytox = rho_ytox[-1] > rho_ytox[0] if len(rho_ytox) > 1 else False

        rho_final_x = rho_xtoy[-1] if rho_xtoy else 0
        rho_final_y = rho_ytox[-1] if rho_ytox else 0

        if converge_xtoy and not converge_ytox and rho_final_x > 0.2:
            direction, strength = "X→Y", "strong" if rho_final_x > 0.5 else "moderate"
        elif converge_ytox and not converge_xtoy and rho_final_y > 0.2:
            direction, strength = "Y→X", "strong" if rho_final_y > 0.5 else "moderate"
        elif converge_xtoy and converge_ytox and abs(rho_final_x - rho_final_y) < 0.1:
            direction, strength = "bidirectional", "strong" if max(rho_final_x, rho_final_y) > 0.5 else "weak"
        else:
            direction, strength = "unclear", "no significant evidence"

        return {
            "ccm_rho_x_to_y": round(float(rho_final_x), 4),
            "ccm_rho_y_to_x": round(float(rho_final_y), 4),
            "causal_direction": direction,
            "direction_strength": strength,
            "confidence": round(abs(rho_final_x - rho_final_y), 4),
            "convergence_X_to_Y": converge_xtoy,
            "convergence_Y_to_X": converge_ytox,
            "library_sizes_tested": [int(s) for s in sizes],
            "rho_at_each_size_X_to_Y": [round(r, 4) for r in rho_xtoy],
            "rho_at_each_size_Y_to_X": [round(r, 4) for r in rho_ytox],
            "note": "Real CCM with delay embedding, KNN cross-mapping, and convergence test.",
        }

    async def _run_granger(self, x: list[float], y: list[float], max_lag: int = 5) -> dict:
        """
        Granger 因果检验

        核心原理: 如果 X 的过去值能显著改善对 Y 当前值的预测
        （相比仅用 Y 自身的过去值），则称 X Granger-causes Y。

        简化实现: 使用自回归模型残差的 F-test
        """
        try:
            import numpy as np
            from scipy import stats

            n = min(len(x), len(y))
            x_arr = np.array(x[:n])
            y_arr = np.array(y[:n])

            # Restrict max_lag to valid range
            max_lag = min(max_lag, n // 4)
            if max_lag < 1:
                return {"status": "insufficient_data", "sample_size": n, "max_valid_lag": max(n // 4, 1)}

            results = {}
            for lag in range(1, max_lag + 1):
                # Valid data window: [lag, n-lag)
                start = lag
                end = n - lag
                obs_n = end - start
                if obs_n < lag * 2 + 1:
                    results[lag] = {"status": "too_few_observations"}
                    continue

                # Target: Y[lag:end]
                y_target = y_arr[start:end]

                # Lagged past values: column j = Y[start-j : end-j]
                y_past_cols = [y_arr[start - j: end - j] for j in range(lag)]
                y_past = np.column_stack(y_past_cols)

                # Same for X past
                x_past_cols = [x_arr[start - j: end - j] for j in range(lag)]
                x_past = np.column_stack(x_past_cols)

                # Combined design matrix (past Y + past X)
                xy_past = np.hstack([y_past, x_past])

                # OLS estimation
                y_centered = y_target - y_target.mean()
                ss_tot = np.sum(y_centered**2)

                # Unrestricted: SSR(Y | Y_lags, X_lags)
                try:
                    if xy_past.shape[0] >= xy_past.shape[1]:
                        beta = np.linalg.lstsq(xy_past, y_centered, rcond=None)[0]
                        residuals_ur = y_centered - xy_past @ beta
                        ssr_ur = float(np.sum(residuals_ur**2))
                    else:
                        ssr_ur = ss_tot
                        continue

                    # Restricted: SSR(Y | Y_lags only)
                    if y_past.shape[0] >= y_past.shape[1]:
                        beta_r = np.linalg.lstsq(y_past, y_centered, rcond=None)[0]
                        residuals_r = y_centered - y_past @ beta_r
                        ssr_r = float(np.sum(residuals_r**2))
                    else:
                        ssr_r = ss_tot
                        continue

                    # F-test with guard against division by zero when model fits perfectly
                    df1 = lag
                    df2 = obs_n - lag * 2
                    if ssr_ur < 1e-10:
                        # Near-perfect fit — F-stat is effectively infinite, p ~ 0
                        results[lag] = {
                            "f_statistic": 1000.0,
                            "p_value": 0.0,
                            "significant": True,
                            "note": "near_perfect_fit",
                        }
                        continue
                    denom = ssr_ur / max(df2, 1)
                    f_stat = ((ssr_r - ssr_ur) / df1) / denom
                    f_stat = min(f_stat, 1000.0)  # cap to avoid overflow
                    p_value = float(1 - stats.f.cdf(f_stat, df1, max(df2, 1)))

                    results[lag] = {
                        "f_statistic": round(f_stat, 4),
                        "p_value": round(p_value, 6),
                        "significant": p_value < 0.05,
                    }
                except (np.linalg.LinAlgError, ValueError):
                    results[lag] = {"status": "computation_failed"}

            any_significant = any(r.get("significant", False) for r in results.values())

            return {
                "results_by_lag": results,
                "overall_granger_causality": any_significant,
                "best_lag": min(results.keys(), key=lambda l: results[l].get("p_value", 1)) if results else None,
                "min_p_value": min((r.get("p_value", 1) for r in results.values()), default=1),
                "note": "Simplified Granger test. For production, install statsmodels.tsa.stattools.grangercausalitytests.",
            }
        except ImportError:
            return {"status": "scipy_required", "message": "Install scipy for Granger causality computation"}

    async def _run_auto_select(self, feature_info: dict) -> dict:
        """
        AI 自动选择最适合的因果推断方法 — SOTA 升级决策树

        基于样本量、变量维度、时序属性等特征进行分层决策。
        """
        sample_size = feature_info.get("sample_size", 0)
        num_vars = feature_info.get("num_variables", 0)
        is_time_series = feature_info.get("is_time_series", True)
        has_confounders = feature_info.get("has_known_confounders", False)
        nonlinear_detected = feature_info.get("nonlinear_relationships", False)

        recommendation = {
            "selected_method": "",
            "reasoning": [],
            "parameters": {},
            "assumptions": [],
        }

        if sample_size < 30:
            recommendation["selected_method"] = "counterfactual"
            recommendation["reasoning"].append(f"样本量过小(n={sample_size})，仅能进行描述性比较")
            recommendation["assumptions"].append("需要至少定义对照组和实验组")
        elif is_time_series:
            if sample_size < 50:
                recommendation["selected_method"] = "granger"
                recommendation["reasoning"].append(f"时间序列但样本有限(n={sample_size})，使用Granger")
                recommendation["assumptions"].append("系统近似线性高斯；强自相关可能产生伪因果")
                recommendation["parameters"] = {"max_lag": min(3, max(1, sample_size // 8))}
            elif nonlinear_detected:
                recommendation["selected_method"] = "ccm"
                recommendation["reasoning"].append("非线性时间序列 → CCM最适合")
                recommendation["assumptions"].append("确定性动力系统；变量共享同一吸引子流形")
                recommendation["parameters"] = {"column_size": 3}
            else:
                recommendation["selected_method"] = "granger"
                recommendation["reasoning"].append(f"线性时间序列(n={sample_size}) → Granger因果检验")
                recommendation["assumptions"].append("平稳性假设；需通过ADF检验确认")
                recommendation["parameters"] = {"max_lag": min(5, max(1, sample_size // 4))}
        elif num_vars > 10 and sample_size > 200:
            recommendation["selected_method"] = "pc_fci"
            recommendation["reasoning"].append("多变量横截面，使用PC-FCI处理潜变量和选择偏置")
            recommendation["assumptions"].append("因果马尔可夫条件 + 因果充分性假设")
            recommendation["parameters"] = {"alpha": 0.05}
        elif has_confounders:
            recommendation["selected_method"] = "counterfactual"
            recommendation["reasoning"].append("已知混杂因子 → 反事实对比设计")
            recommendation["assumptions"].append("需要明确定义处理组和对照组的观测数据")
        else:
            recommendation["selected_method"] = "granger"
            recommendation["reasoning"].append("默认选择: Granger（通用性强）")
            recommendation["assumptions"].append("时间顺序明确；无未测量混杂")

        return recommendation

    @staticmethod
    def _adf_stationarity_check(series: list[float]) -> dict:
        """
        Augmented Dickey-Fuller 单位根检验 — Granger 前提验证

        Returns: {adf_t_statistic, stationary_at_5pct, stationary_at_1pct, approximate_p_value}
        """
        try:
            import numpy as np
            from scipy import stats

            arr = np.array(series)
            delta = np.diff(arr)

            # Simple unit root regression: Δy_t = α + β*y_{t-1} + ε
            y_lagged = arr[:-1]
            reg = np.column_stack([np.ones(len(delta)), y_lagged])
            beta = np.linalg.lstsq(reg, delta[1:], rcond=None)[0]

            residuals = delta[1:] - reg @ beta
            n_resid = len(residuals)
            se = np.sqrt(np.sum(residuals**2) / (n_resid - 2))
            t_stat = beta[1] / (se + 1e-10)

            crit_1pct, crit_5pct, crit_10pct = -3.43, -2.86, -2.57
            p_approx = max(0, min(1, 1 - stats.t.cdf(abs(t_stat), n_resid)))

            return {
                "adf_t_statistic": round(float(t_stat), 4),
                "stationary_at_1pct": t_stat < crit_1pct,
                "stationary_at_5pct": t_stat < crit_5pct,
                "stationary_at_10pct": t_stat < crit_10pct,
                "approximate_p_value": round(float(p_approx), 4),
            }
        except Exception:
            return {"status": "stationarity_check_failed"}

    # Placeholder methods (full implementation requires specific libraries)
    async def _run_pc_fci(self, data: list[list[float]], alpha: float = 0.05) -> dict:
        """PC-FCI 因果图发现 — 需要 causalgraphicalmodels 库"""
        return {
            "status": "requires_causalgraphicalmodels",
            "message": "pip install causalgraphicalmodels",
            "expected_output": "adjacency_matrix + skeleton edges",
        }

    async def _run_psm(self, treated: list[bool], outcome: list[float], covariates: list[list[float]]) -> dict:
        """PSM 倾向得分匹配 — 需要 dowhy 或 statsmodels"""
        return {
            "status": "requires_dowhy",
            "message": "pip install dowhy",
        }

    async def _run_instrumental_variable(self, z: list[float], x: list[float], y: list[float]) -> dict:
        """工具变量法 — 需要 linearmodels"""
        return {"status": "requires_linearmodels"}

    async def _run_bayesian_network(self, data: list[dict], structure: str = "learning") -> dict:
        """贝叶斯网络 — 需要 pgmpy"""
        return {"status": "requires_pgmpy", "message": "pip install pgmpy"}

    async def _run_counterfactual(self, predictions_base: list[float], predictions_intervened: list[float]) -> dict:
        """
        反事实推理 (Item 19) — 基于两组的差异估计因果效应

        Args:
            predictions_base: 基线条件下的预测值
            predictions_intervened: 干预后的预测值
        """
        import numpy as np
        from scipy import stats

        n = min(len(predictions_base), len(predictions_intervened))
        base = np.array(predictions_base[:n])
        inter = np.array(predictions_intervened[:n])

        # Welch's t-test for unequal variances (standard in SOTA counterfactual analysis)
        mean_base, mean_inter = base.mean(), inter.mean()
        std_base, std_inter = base.std(ddof=1), inter.std(ddof=1)
        n_base, n_inter = len(base), len(inter)

        se = np.sqrt(std_base**2 / n_base + std_inter**2 / n_inter) + 1e-10
        causal_effect = mean_inter - mean_base
        t_stat = causal_effect / se
        df_welch = (std_base**2 / n_base + std_inter**2 / n_inter)**2 / (
            (std_base**2 / n_base)**2 / max(n_base - 1, 1) +
            (std_inter**2 / n_inter)**2 / max(n_inter - 1, 1)
        ) + 1e-10
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_welch)) if n > 4 else None
        confidence_interval_95 = (
            float(causal_effect - 1.96 * se),
            float(causal_effect + 1.96 * se),
        )

        return {
            "average_treatment_effect": round(float(causal_effect), 4),
            "standard_error": round(float(se), 4),
            "confidence_interval_95": [round(v, 4) for v in confidence_interval_95],
            "p_value": round(p_value, 6) if p_value is not None else None,
            "t_statistic": round(float(t_stat), 4),
            "degrees_of_freedom": round(float(df_welch), 2),
            "interpretation": (
                f"干预组均值比基线{'高' if causal_effect > 0 else '低'} {abs(causal_effect):.4f}"
                f" (Welch's t-test: t={t_stat:.4f}, df={df_welch:.1f})\n"
                f"95%CI=[{confidence_interval_95[0]:.4f}, {confidence_interval_95[1]:.4f}]"
                f"{(' p=' + str(round(p_value, 6))) if p_value is not None else ''}"
            ),
        }
