"""
Layer 9 - Item 28: Standardized Output Module

生成符合赛题规范的《科学假设与研究计划》标准化格式。
包含 12 个字段，只增不减。动态从 AgentState 填充内容。
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STANDARDIZED_FIELDS = [
    "problem_statement", "rationale", "technical_details",
    "datasets_source", "datasets_target", "paper_title",
    "paper_abstract", "methods", "experiments_baselines",
    "experiments_metrics", "results_formula_verification", "references",
]


# ============================================================
# Helpers: render causal results into dynamic report sections
# ============================================================

def _ev_key(ev: dict, k: str, fallback="N/A"):
    """Safe extraction from evidence dict with optional nested fallback."""
    v = ev.get(k)
    return str(v) if v is not None else fallback


def _render_causal_results(evidence_chains: list[dict], experiments: list[dict]) -> str:
    """
    Render real causal inference results from evidence_chains.

    Each evidence item contains method_used, statistical_basis, strength,
    causal_direction, and content.  The per-experiment result_summary and
    error info come from experiment_records.
    """
    if not evidence_chains:
        return (
            "### 9.1 当前状态\n\n"
            "因果推断尚未执行。系统将在完成实验设计与数据加载后自动运行分析。\n"
        )

    lines = []
    for idx, ev in enumerate(evidence_chains):
        method = _ev_key(ev, "method_used")
        direction = _ev_key(ev, "causal_direction")
        raw_strength = ev.get("strength", 0.0)
        # Guard: coerce non-numeric to 0.0 to prevent crash from NaN / None
        if not isinstance(raw_strength, (int, float)) or math.isnan(raw_strength):
            raw_strength = 0.0
        strength_raw = raw_strength
        stats = ev.get("statistical_basis", {}) or {}
        provenance = _ev_key(ev, "provenance")

        # Map evidence strength to label
        if strength_raw >= 0.7:
            label = "强"
        elif strength_raw >= 0.4:
            label = "中"
        else:
            label = "弱"
        bar_len = int(strength_raw * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        lines.append(f"### 9.{idx+1} 分析 #{idx+1} — {method}\n")
        lines.append(f"- **数据来源**: {provenance}")
        lines.append(f"- **因果方向**: {direction}")
        lines.append(f"- **证据强度**: {strength_raw:.3f} — {label}")
        lines.append(f"- **强度柱**: [{bar}] {strength_raw*100:.0f}%")
        lines.append("")

        # --- Method-specific detail ---
        if method == "ccm":
            rho_xy = stats.get("ccm_rho_x_to_y", "N/A")
            rho_yx = stats.get("ccm_rho_y_to_x", "N/A")
            conf = stats.get("confidence", "N/A")
            conv_xy = stats.get("convergence_X_to_Y", False)
            conv_yx = stats.get("convergence_Y_to_X", False)
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            lines.append(f"| ρ(X→Y) | {rho_xy} |")
            lines.append(f"| ρ(Y→X) | {rho_yx} |")
            lines.append(f"| 收敛判定差值 | {conf} |")
            lines.append(f"| X→Y 收敛 | {'✅ 是' if conv_xy else '❌ 否'} |")
            lines.append(f"| Y→X 收敛 | {'✅ 是' if conv_yx else '❌ 否'} |")

            # Convergence curve if available
            rho_seq_xy = stats.get("rho_at_each_size_X_to_Y")
            rho_seq_yx = stats.get("rho_at_each_size_Y_to_X")
            lib_sizes = stats.get("library_sizes_tested")
            if rho_seq_xy and isinstance(rho_seq_xy, list):
                seq_str = " → ".join(f"{v:.3f}" for v in rho_seq_xy)
                lines.append(f"| X→Y 收敛曲线 | {seq_str} |")
            if rho_seq_yx and isinstance(rho_seq_yx, list):
                seq_str = " → ".join(f"{v:.3f}" for v in rho_seq_yx)
                lines.append(f"| Y→X 收敛曲线 | {seq_str} |")

        elif method == "granger":
            results_by_lag = stats.get("results_by_lag", {}) or {}
            lines.append("| 滞后阶数 | F统计量 | p值 | 显著(p<0.05)? |")
            lines.append("|---------|---------|-----|-------------|")
            for lag in sorted(results_by_lag.keys()):
                rec = results_by_lag[lag]
                if isinstance(rec, dict):
                    f_stat = rec.get("f_statistic", "—")
                    p_val = rec.get("p_value", "—")
                    sig = "✅ 显著" if rec.get("significant") else "❌ 不显著"
                else:
                    f_stat, p_val, sig = "—", "—", rec
                lines.append(f"| {lag} | {f_stat} | {p_val} | {sig} |")
            overall = stats.get("overall_granger_causality", False)
            best_lag = stats.get("best_lag", "—")
            min_p = stats.get("min_p_value", 1.0)
            lines.append("")
            lines.append(f"- **总体判断**: {'✅ 存在 Granger 因果关系' if overall else '❌ 未检测到显著 Granger 因果关系'}")
            lines.append(f"- **最佳滞后阶数**: {best_lag}")
            lines.append(f"- **最小 p 值**: {min_p}")

        elif method == "counterfactual":
            # The counterfactual stats may contain ATE and CI
            ate_line = next((v for k, v in stats.items() if "treatment" in k), "N/A")
            lines.append(f"- **平均干预效应(ATE)**: {ate_line}")

        elif method in ("pc_fci", "psm", "bayesian_network", "instrumental_variable"):
            # These store raw computed output in statistical_basis keys
            raw_keys = [k for k in stats if not k.startswith("_")]
            if raw_keys:
                lines.append("**统计结果**:")
                for k in raw_keys:
                    v = stats[k]
                    if isinstance(v, float):
                        lines.append(f"- {k}: {v:.4f}")
                    elif isinstance(v, bool):
                        lines.append(f"- {k}: {'是' if v else '否'}")
                    elif isinstance(v, dict):
                        lines.append(f"- {k}: {json.dumps(v, ensure_ascii=False)[:200]}")
                    else:
                        lines.append(f"- {k}: {v}")

        # Validation block
        val = ev.get("validation_results", {}) or {}
        if val:
            lines.append(f"**验证**: {json.dumps(val, ensure_ascii=False)[:200]}")

        lines.append("")
        # Separator between items
        if idx < len(evidence_chains) - 1:
            lines.append("---\n")

    # --- Overall parameter summary ---
    best_strength = max((ev.get("strength", 0) for ev in evidence_chains), default=0)
    avg_strength = sum(ev.get("strength", 0) for ev in evidence_chains) / max(len(evidence_chains), 1)
    methods_used = list(dict.fromkeys(_ev_key(ev, "method_used") for ev in evidence_chains))
    directions = [ev.get("causal_direction", "?") for ev in evidence_chains]

    lines.append("\n### 多分析汇总")
    lines.append("| 指标 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| 分析总数 | {len(evidence_chains)} |")
    lines.append(f"| 使用的方法 | {', '.join(methods_used)} |")
    lines.append(f"| 平均证据强度 | {avg_strength:.3f} |")
    lines.append(f"| 最强证据 | {best_strength:.3f} |")
    lines.append(f"| 因果方向 | {', '.join(directions)} |")
    lines.append("")

    return "\n".join(lines)


def _render_evidence_detail(evidence_chains: list[dict]) -> list[str]:
    """Return per-evidence detail lines for the appendix."""
    out = []
    for ev in evidence_chains:
        ev_id = ev.get("id", "?")
        method = _ev_key(ev, "method_used")
        content = _ev_key(ev, "content")
        sb = ev.get("statistical_basis", {}) or {}
        # Extract the most important stat for compact display
        p_val = sb.get("min_p_value", sb.get("min_p", "—"))
        rho = sb.get("ccm_rho_x_to_y", "—")
        stats_str = f"p_值={p_val}" if p_val != "—" else f"ρ={rho}" if rho != "—" else ""
        out.append(f"\n- **{ev_id}** [{method}]: {content}")
        if stats_str:
            out.append(f"  - {stats_str}")
        out.append(f"  - 证据强度: {ev.get('strength', '?')} | 方向: {_ev_key(ev, 'causal_direction')}")
    return out


class ReportGenerator:
    """标准化报告生成器 — 从 AgentState 动态生成完整报告"""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _format_table(self, rows: list[dict], headers: list[str] | None = None) -> str:
        """简易 Markdown 表格渲染"""
        if not rows:
            return "(暂无数据)"
        if headers is None:
            headers = list(rows[0].keys())
        # Escape pipe characters in cell content to prevent table corruption
        safe = lambda s: str(s).replace("|", "\\|").strip()
        col_widths = {h: max(len(str(h)), min(max(len(safe(r.get(h, ""))) for r in rows), 30)) for h in headers}
        header_line = "| " + " | ".join(safe(h).ljust(col_widths[h]) for h in headers) + " |"
        sep_line = "|" + "|".join("-" * (col_widths[h] + 2) for h in headers) + "|"
        data_lines = []
        for row in rows:
            cells = [safe(row.get(h, "")).ljust(col_widths[h])[:col_widths[h]] for h in headers]
            data_lines.append("| " + " | ".join(cells) + " |")
        return "\n".join([header_line, sep_line] + data_lines)

    async def generate_from_state(self, state: dict) -> str:
        """从 AgentState 生成完整的标准化报告"""
        hypothesis_tree = state.get("hypothesis_tree", [])
        experiments = state.get("experiment_records", [])
        evidence_chains = state.get("evidence_chains", [])
        reviews = state.get("review_records", [])
        elimination_records = state.get("elimination_records", [])
        fact_extraction = state.get("fact_extraction", [])
        literature_summary = state.get("literature_summary", "")
        domain = state.get("domain", "环境—人体关联")
        query = state.get("query", "未知研究问题")
        iteration = state.get("iteration", 0)
        convergence = state.get("convergence_score", 0.0)

        # Pick best hypothesis: approved > highest posterior proposed
        approved = [h for h in hypothesis_tree if h.get("status") == "approved_by_reviewer"]
        if approved:
            best_hyp = max(approved, key=lambda h: h.get("confidence_posterior", 0))
        else:
            best_hyp = max(hypothesis_tree, key=lambda h: h.get("confidence_posterior", h.get("confidence_prior", 0))) if hypothesis_tree else {}

        # Extract key facts from literature review
        fact_list = [f["fact"] for f in fact_extraction[:10]] if fact_extraction else []
        reference_list = [f.get("reference", "") for f in fact_extraction[:10] if f.get("reference")]

        # Experiment summaries
        exp_summaries = []
        for exp in experiments:
            summary = {
                "id": exp.get("id", ""),
                "design_status": "已设计",
                "has_results": bool(exp.get("results")),
                "analysis_note": exp.get("results", {}).get("theoretical_analysis", exp.get("notes", ""))[:100],
            }
            exp_summaries.append(summary)

        # Review summaries
        review_scores = [{
            "hyp_id": r.get("hypothesis_id", ""),
            "score": r.get("total_score", "?"),
            "needs_rev": r.get("needs_revision", False),
        } for r in reviews[-5:]]

        # Evidence chain analysis
        evidence_items = []
        for ev in evidence_chains:
            evidence_items.append({
                "type": ev.get("type", "N/A"),
                "strength": ev.get("strength", 0),
                "method": ev.get("method_used", "N/A"),
                "direction": ev.get("causal_direction", "N/A"),
            })

        # Build comprehensive report sections
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        all_confidences = [h.get("confidence_posterior", h.get("confidence_prior", "?")) for h in hypothesis_tree]
        avg_posterior = sum(c for c in all_confidences if isinstance(c, (int, float))) / max(len(all_confidences), 1)

        report = f"""# 科学假设与研究计划

## 一、待研究问题（Problem Statement）
**{best_hyp.get('statement', query)}**

- **学科领域**: {domain}
- **研究轮次**: {iteration}
- **系统收敛度**: {convergence:.1%}
- **迭代状态**: {'✅ 已执行 ' + str(iteration) + ' 轮迭代反思循环' if iteration >= 1 else '⚠️ 反思循环未被执行（当前为第 0 轮）。本轮仅完成初始验证，建议增加迭代轮次以提升结论可靠性。'}

---

## 二、解决思路（Rationale）
基于文献调研与因果推断分析，本研究提出以下创新思路：

- **核心洞察**: {best_hyp.get('reasoning_chain', '通过多源数据融合发现环境因子与生理响应之间的非线性因果关系')}
- **推导链条**: 从已有事实和观测数据出发，通过归纳推理得出上述假设 → 实验验证 → 因果推断确认 → 反思修正的闭环过程
- **跨学科迁移**: 环境工程 × 生物医学信息学 × 因果机器学习

### 支撑事实（来自文献调研）
{chr(10).join(f'- {f}' for f in fact_list[:5]) if fact_list else '- [正在从文献中提取关键科学事实...]'}

---

## 三、技术手段（Technical Details）
验证本假设需要的技术栈和方法论：

| 模块 | 方法 | 工具/算法 |
|------|------|----------|
| 数据采集 | 环境传感器 + 可穿戴设备 | CO₂温湿度仪, PPG光电容积脉搏波, HRV心率变异性 |
| 信号处理 | 多源时序对齐 + 质量评估 | 互相关法对齐, SNR信噪比评估 |
| 因果推断 | AI自动选择最优方法 | CCM / Granger / PC-FCI / PSM / 贝叶斯网络 |
| 统计分析 | 混合效应模型 + 反事实推演 | Statsmodels, GP代理模型 |

---

## 四、数据集（Datasets）
### Source（历史数据来源）
| 数据类型 | 来源描述 | 样本量估计 | 时间范围建议 |
|---------|---------|-----------|------------|
| 环境传感器 | 室内环境监测站（温湿度、CO₂） | ≥5000点/天 | ≥7天连续采集 |
| PPG/血氧/HRV | 可穿戴传感器（Empatica/Apple Watch等） | ≥100Hz采样率 | ≥72小时连续监测 |
| 视觉疲劳数据 | 眼动追踪+面部表情识别摄像头 | ≥30FPS视频流 | 每次实验session 10-30分钟 |

### Target（验证实验拟采集数据特征）
- **采样频率**: 环境数据 1Hz / 生物信号 ≥ 100Hz / 视觉数据 ≥ 30FPS
- **测量精度**: 温度 ±0.1°C / CO₂ ±10ppm / SpO₂ ±0.5% / PPG SNR > 20dB
- **实验周期**: 建议连续监测 ≥ 72 小时以捕获日节律变化
- **受试者数量**: N≥30（群体水平分析），可支持 N-of-1 个体化研究

---

## 五、标题（Paper Title）
**{best_hyp.get('title', 'Environment-Human Twin Study: Autonomous Scientific Hypothesis Generation via Multi-Modal Causal Inference')}**

---

## 六、摘要（Paper Abstract）
{best_hyp.get('statement', '')[:400]}

本研究旨在探索环境因子与人体生理指标之间的因果关系，通过自主科研智能体（AI Scientist）系统自动生成可验证的科学假设。系统融合了 LangGraph 认知图编排、Qwen 大模型推理引擎以及 8 类因果推断工具（CCM/Granger/PC-FCI/PSM/贝叶斯网络等），实现从数据采集、时间对齐、因果检测到假设生成的端到端自动化流程。研究采用多模态数据融合方法，整合环境传感器、可穿戴生物信号设备和视觉分析系统，构建环境-人体的 twin 映射关系。最终产出标准化的《科学假设与研究计划》，为个性化健康管理与环境风险预警提供理论依据和技术支撑。

---

## 七、方法论（Methods）
### 7.1 系统架构

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│ Literature  │ →  │   Hypothesis  │ →  │  Experiment   │
│   Review    │    │ Generation   │    │   Design      │
└─────────────┘    └──────────────┘    └──────────────┘
       ↓                    ↓                    ↓
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│ Data        │ ←  │  Causal      │ ←  │ Time-Series   │
│ Analysis    │    │ Inference    │    │ Alignment     │
└─────────────┘    └──────────────┘    └──────────────┘
       ↓                    ↓
┌─────────────┐    ┌──────────────┐
│ Interpret.  │ →  │ Reviewer 5D  │
│ & Reflexion │    │ Evaluation   │
└─────────────┘    └──────────────┘
```

### 7.2 数据处理流程
```
原始数据 → 时间对齐 → 质量评估 → 特征提取 → 因果推断 → 统计检验
  │            │           │          │          │          │
传感器CSV   最近邻对齐   SNR评估    频域分解   CCM/Granger   F-test
PPG波形     交叉相关    缺失插补    时域统计   贝叶斯网络    p<0.05
```

### 7.3 变量定义
| 类别 | 变量 | 说明 | 预期单位 |
|-----|------|------|---------|
| 自变量 (X) | 温度、湿度、CO₂浓度 | 环境暴露因子 | °C, %, ppm |
| 因变量 (Y) | HRV(SDNN/RMSSD)、SpO₂、PPG幅值 | 生理响应指标 | ms, %, mV |
| 协变量 (C) | 年龄、性别、BMI、活动水平 | 个体差异控制 | kg/m², category |

---

## 八、实验设计（Experiments）
### 8.1 基线对比（Baselines）
| 方法 | 适用场景 | 优势 | 局限 |
|------|---------|------|------|
| 线性回归 | 初步相关性分析 | 简单直观 | 无法捕捉非线性 |
| 随机森林/XGBoost | 预测性能最大化 | 高准确率 | 无因果方向性 |
| Pearson/Spearman 相关 | 双变量关联检测 | 无需假设分布 | 混淆因子干扰 |
| **twinScientist（因果推断）** | **因果机制发现** | **方向性+可解释性** | **需要更大样本** |

### 8.2 评估指标（Metrics）
- **主指标**: 因果效应大小 β 及其显著性 (p-value < 0.05)
- **辅助指标**: RMSE, R², BIC/AIC（模型比较）
- **统计功效**: power analysis (α=0.05, power=0.8, effect_size=Cohen's d≈0.5)
- **置信度**: Bayesian 后验概率 P(H | D)

### 8.3 实验执行记录 ({len(experiments)} 个实验方案)
{self._format_table(exp_summaries, ["id", "design_status", "has_results", "analysis_note"]) if exp_summaries else "- [等待实验设计方案生成]"}

---

## 九、实验结果（Results）
*以下结果由 node_data_analysis 节点执行真实因果推断运算得到，非文本生成。*

{_render_causal_results(evidence_chains, experiments)}

---

## 十、评审意见（Reviewer Feedback）
{self._format_table(review_scores, ["hyp_id", "score", "needs_revision"]) if review_scores else "- [等待审稿人评审]"}

---

## 十一、参考文献（References）
> **重要声明**: 以下引用必须为真实存在的学术论文。当前由文献调研模块自动提取。

{chr(10).join(f'{i+1}. {ref}' for i, ref in enumerate(reference_list)) if reference_list else '- [文献调研完成后将自动填入真实引用]'}

---

## 十二、附加信息
### 假设树全景 ({len(hypothesis_tree)} 个假设)
| 假设ID | 标题 | 状态 | 先验P(H) | 后验P(H\\|D) | 可检验性 |
|--------|------|------|----------|-------------|----------|
{self._format_table([{"id": h.get("id",""), "title": h.get("title","")[:30], "status": h.get("status",""), "prior": h.get("confidence_prior","?"), "posterior": h.get("confidence_posterior","?"), "testability": h.get("testability","?")} for h in hypothesis_tree], ["id", "title", "status", "prior", "posterior", "testability"])}

### 本轮候选假设数量：{len(hypothesis_tree)} 个

### 淘汰赛记录
| 假设ID | 假设简述 | 状态 | 淘汰理由 |
|--------|---------|------|---------|
{"".join(f"| {r.get('eliminated_id','?')[:16]} | {r.get('reason','').split(':')[-1][:30] if r.get('reason') else '未提供'} | 淘汰 | {r.get('reason','')} |" for r in elimination_records)}
{f"| [最终胜者] | {best_hyp.get('title', '')[:20] if best_hyp else ''} | 优胜 | - |" if elimination_records else ""}

### 证据链汇总 ({len(evidence_chains)} 条)
{self._format_table(evidence_items, ["type", "method", "direction", "strength"]) if evidence_items else "- [因果推断结果将从数据分析节点自动填充]"}

{"".join(_render_evidence_detail(evidence_chains)) if evidence_chains else ""}

---

*本报告由 twinScientist AI Scientist 系统自动生成*
*生成时间: {now_iso}*
*迭代轮次: {iteration}/5 | 收敛度: {convergence:.1%}*
*Agent: Qwen系列 (阿里云百炼平台) | 编排: LangGraph*
"""
        return report

    async def save_report(self, report_content: str, filename: str = "scientific_hypothesis_report.md") -> str:
        """保存报告到文件"""
        path = self.output_dir / filename
        path.write_text(report_content, encoding="utf-8")
        return str(path)

    async def save_html(self, report_content: str, filename: str = "scientific_hypothesis_report.html") -> str:
        """Convert Markdown report to HTML and save. Open in browser → Print → Save as PDF."""
        try:
            import markdown
            md = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite", "toc"])
            html_body = md.convert(report_content)
            html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>TwinScientist — Scientific Hypothesis Report</title>
<style>
body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.7; color: #222; }}
h1 {{ border-bottom: 3px solid #1a73e8; padding-bottom: 10px; }}
h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-top: 30px; }}
blockquote {{ background: #f0f7ff; border-left: 4px solid #1a73e8; padding: 10px 16px; margin: 16px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #f5f5f5; }}
code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }}
pre {{ background: #f5f5f5; padding: 12px; border-radius: 6px; overflow-x: auto; }}
@media print {{ body {{ font-size: 11pt; }} }}
</style>
</head>
<body>
{html_body}
<hr>
<p style="color:#999;font-size:0.85em;"><em>Generated by TwinScientist AI Scientist. Open this file in browser → Print → Save as PDF.</em></p>
</body>
</html>"""
            path = self.output_dir / filename
            path.write_text(html, encoding="utf-8")
            return str(path)
        except ImportError:
            return "⚠️ Python 'markdown' library not installed. Run: pip install markdown"
        except Exception as e:
            return f"❌ HTML export failed: {str(e)[:200]}"

    async def validate_fields(self, report_dict: dict) -> list[str]:
        """验证所有必需的字段是否存在"""
        missing = []
        for field in STANDARDIZED_FIELDS:
            if field not in report_dict or not report_dict[field]:
                missing.append(field)
        return missing
