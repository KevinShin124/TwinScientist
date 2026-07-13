"""
Patched node_report_writing for twinScientist
"""

# This function replaces node_report_writing in core/nodes.py
# It restores the original REPORT_WRITING_TEMPLATE-based approach
# but adds Section 9 pre-population from real data

def build_section9(state):
    """Build Section 9 markdown from actual experiment results."""
    evidence_chains = list(state.get("evidence_chains", []))
    experiments = state.get("experiment_records", [])

    has_real_analysis = any(e.get("type") == "causal_inference" for e in evidence_chains)

    if not has_real_analysis:
        return ""  # Use default template section

    parts = ["\n### 九、实验结果（Results）\n*(以下基于真实数据分析)*\n\n"]

    for ev in evidence_chains:
        if ev.get("type") != "causal_inference":
            continue

        method = ev.get("method_used", "?")
        strength = ev.get("strength", 0)
        content_ev = ev.get("content", "")
        stat_basis = ev.get("statistical_basis", {})
        causal_dir = ev.get("causal_direction", None)

        parts.append(f"#### 数据方法: {method}\n")
        parts.append(f"**因果推断摘要**: {content_ev}\n\n")
        parts.append("| 指标 | 值 | 说明 |\n|------|-----|------|\n")
        parts.append(f"| 证据强度 | {strength:.4f} | 0-1 置信度分数 |\n")
        parts.append(f"| 分析方法 | {method} | AI自动选择的最优方法 |\n")
        parts.append(f"| 实验数量 | {len(experiments)} | 执行的实验数 |\n")
        if causal_dir and causal_dir != "None":
            parts.append(f"| 因果方向 | {causal_dir} | 因果推断的方向性 |\n")
        sb_str = "; ".join(f"{k}: {v}" for k, v in list(stat_basis.items())[:6])
        if sb_str:
            parts.append(f"| 统计依据 | {sb_str} |\n")
        parts.append("\n")

    return "".join(parts)


if __name__ == "__main__":
    # Test: create mock state and verify section generation
    mock_state = {
        "evidence_chains": [
            {
                "type": "causal_inference",
                "method_used": "granger",
                "strength": 1.0,
                "content": "Test result summary",
                "statistical_basis": {"max_lag": 3, "p_value": 0.001},
                "causal_direction": "T -> CO2"
            }
        ],
        "experiment_records": [{"id": "exp_test"}],
    }
    section = build_section9(mock_state)
    print(section)
