"""
Multi-Modal Output — ASCII charts, Mermaid diagrams, and formatted tables.

Apple/Google-grade visual output without external dependencies.
Generates publication-quality visualizations from causal inference results.

Usage:
    from core.visualization import render_causal_chart, render_hypothesis_tree
    chart = render_causal_chart(evidence_chains)
    diagram = render_hypothesis_tree(hypotheses)
"""

from __future__ import annotations

import math
from typing import Any


def render_causal_chart(evidence_chains: list[dict], width: int = 60) -> str:
    """
    Render a causal inference results summary as an ASCII chart.

    Shows method, strength, direction, and key statistics for each evidence chain.
    """
    if not evidence_chains:
        return "(No causal inference results available)"

    causal = [e for e in evidence_chains if e.get("type") == "causal_inference"]
    if not causal:
        return "(No causal inference results yet — run data_analysis first)"

    lines = ["", "### Causal Inference Results", "", "```"]
    lines.append(f"{'Method':<12} {'Strength':>8} {'Direction':<16} {'Key Stat':<20}")
    lines.append("-" * width)

    for ev in causal:
        method = ev.get("method_used", "?")[:10]
        strength = ev.get("strength", 0)
        direction = str(ev.get("causal_direction", "N/A"))[:14]
        sb = ev.get("statistical_basis", {})

        # Extract key stat
        key_stat = ""
        if method == "granger":
            min_p = sb.get("min_p_value", "?")
            key_stat = f"min p={min_p}"
        elif method == "ccm":
            rho = sb.get("ccm_rho_x_to_y", "?")
            key_stat = f"rho={rho}"
        elif method == "counterfactual":
            ate = sb.get("average_treatment_effect", "?")
            key_stat = f"ATE={ate}"
        else:
            key_stat = str(list(sb.keys())[:1])[:18]

        # Strength bar
        bar_len = int(strength * 10)
        bar = "#" * bar_len + "-" * (10 - bar_len)

        lines.append(f"{method:<12} {strength:.3f} [{bar}] {direction:<16} {key_stat:<20}")

    lines.append("```")
    lines.append("")
    lines.append(f"*Evidence strength: 0-1 scale. >0.7 = strong, 0.4-0.7 = moderate, <0.4 = weak.*")
    return "\n".join(lines)


def render_hypothesis_tree(hypotheses: list[dict], max_display: int = 15) -> str:
    """
    Render hypothesis tree as a Mermaid mindmap diagram.
    """
    active = [h for h in hypotheses if h.get("status") not in ("pruned", "refuted", "refuted_in_tournament")]
    if not active:
        return "(No active hypotheses)"

    lines = ["", "### Hypothesis Tree", "", "```mermaid", "mindmap"]
    lines.append(f"  root((Research Question))")

    # Group by status
    approved = [h for h in active if h.get("status") == "approved_by_reviewer"]
    proposed = [h for h in active if h.get("status") == "proposed"]
    active_hyps = [h for h in active if h.get("status") == "active"]

    if approved:
        lines.append("    Approved")
        for h in approved[:max_display]:
            title = h.get("title", "?")[:40]
            conf = h.get("confidence_posterior", h.get("confidence_prior", 0))
            lines.append(f"      [{title}]")
            lines.append(f"        P(H|D)={conf:.0%}")

    if active_hyps:
        lines.append("    Active")
        for h in active_hyps[:max_display]:
            title = h.get("title", "?")[:40]
            conf = h.get("confidence_posterior", h.get("confidence_prior", 0))
            lines.append(f"      [{title}]")
            lines.append(f"        P(H|D)={conf:.0%}")

    if proposed:
        lines.append("    Proposed")
        for h in proposed[:max_display]:
            title = h.get("title", "?")[:40]
            lines.append(f"      [{title}]")

    lines.append("```")
    lines.append("")
    lines.append(f"*{len(active)} active hypotheses total. Displayed top {max_display}.*")
    return "\n".join(lines)


def render_evidence_timeline(evidence_chains: list[dict], experiments: list[dict]) -> str:
    """
    Render a timeline showing how evidence was built through experiments.
    """
    if not evidence_chains:
        return "(No evidence chains)"

    lines = ["", "### Evidence Timeline", "", "```"]
    lines.append("Experiment -> Method -> Evidence Strength -> Conclusion")
    lines.append("")

    for i, ev in enumerate(evidence_chains):
        method = ev.get("method_used", "?")
        strength = ev.get("strength", 0)
        content = ev.get("content", "")[:80]
        # Find matching experiment
        exp_id = ev.get("linked_experiments", ["?"])[0] if ev.get("linked_experiments") else "?"
        exp_id_short = exp_id[:10] if isinstance(exp_id, str) else str(exp_id)[:10]

        bar = "#" * int(strength * 10) + "-" * (10 - int(strength * 10))
        lines.append(f"  {exp_id_short} -> {method:<10} -> [{bar}] {strength:.3f} -> {content}")

    lines.append("```")
    return "\n".join(lines)


def render_data_quality_report(data_files: list[str], sample_sizes: dict[str, int] | None = None) -> str:
    """
    Render a data quality overview table.
    """
    if not data_files:
        return "(No data files available)"

    lines = ["", "### Data Quality Overview", ""]
    lines.append("| File | Type | Records | Quality |")
    lines.append("|------|------|---------|---------|")

    for f in data_files:
        fname = f.replace("\\", "/").split("/")[-1][:30]
        if "env" in fname.lower():
            dtype = "Environmental"
        elif "biometric" in fname.lower():
            dtype = "Biometric"
        else:
            dtype = "Sensor"

        size = sample_sizes.get(f, "?") if sample_sizes else "?"
        quality = "Good" if size != "?" and (isinstance(size, int) and size > 100) else "Check"

        lines.append(f"| {fname} | {dtype} | {size} | {quality} |")

    return "\n".join(lines)


def render_convergence_chart(convergence_history: list[float]) -> str:
    """
    Render convergence over iterations as an ASCII line chart.
    """
    if not convergence_history:
        return "(No convergence data)"

    lines = ["", "### Convergence Over Iterations", "", "```"]
    max_width = 40
    max_val = max(max(convergence_history), 0.01)

    for i, val in enumerate(convergence_history):
        bar_len = int(val / max_val * max_width)
        bar = "#" * bar_len + "-" * (max_width - bar_len)
        lines.append(f"  Round {i+1:2d}: [{bar}] {val:.1%}")

    lines.append("```")
    lines.append("")
    if len(convergence_history) >= 2:
        last_two = convergence_history[-2:]
        delta = abs(last_two[0] - last_two[1])
        status = "Converged" if delta < 0.05 else "Still evolving"
        lines.append(f"*Status: {status} (delta={delta:.1%})*")

    return "\n".join(lines)


def inject_visualizations(report: str, state: dict) -> str:
    """
    Inject visualizations into the report at appropriate locations.
    Called after report generation to enhance the output.
    """
    evidence_chains = state.get("evidence_chains", [])
    hypotheses = state.get("hypothesis_tree", [])
    experiments = state.get("experiment_records", [])
    convergence_history = state.get("convergence_history", [])

    # Inject causal chart after Section 9 (Results)
    causal_chart = render_causal_chart(evidence_chains)
    if "### Causal Inference Results" in causal_chart:
        report = report.replace(
            "## 十、评审意见",
            f"{causal_chart}\n\n---\n\n## 十、评审意见"
        )

    # Inject hypothesis tree after Section 12
    hyp_tree = render_hypothesis_tree(hypotheses)
    if "### Hypothesis Tree" in hyp_tree:
        report = report.replace(
            "*本报告由 twinScientist",
            f"{hyp_tree}\n\n---\n\n*本报告由 twinScientist"
        )

    # Inject convergence chart if available
    if convergence_history and len(convergence_history) >= 2:
        conv_chart = render_convergence_chart(convergence_history)
        if "### Convergence Over Iterations" in conv_chart:
            report = report.replace(
                "## 十二、附加信息",
                f"{conv_chart}\n\n---\n\n## 十二、附加信息"
            )

    return report