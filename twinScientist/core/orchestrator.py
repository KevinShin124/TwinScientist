"""
Layer 2: Orchestrator — LLM驱动的动态路由引擎

替代硬编码Python if/else路由，让Qwen模型基于当前State上下文
自主决策下一步认知操作。这是国际AI Scientist的主流模式（AI_Scientist, AutoDevin）。

架构模式：
- Supervisor/Coordinator Pattern：Orchestrator 作为调度中枢，综合评估证据、
  不确定性、异常图谱后做出全局最优决策
- Conditionally Dynamic：某些确定性流程（ethics_check → literature_review）
  仍保持直接边；需要LLM判断的分支通过 new_orchestrator_router() 路由
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.state import AgentState
from core.llm_client import QwenClient
from config.settings import settings
from core.prompts import ORCHESTRATOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ============================================================
# Semantic Similarity Calculation
# ============================================================

def _hypothesis_statement_similarity(stmt_a: str, stmt_b: str) -> float:
    """
    计算两个假设陈述的语义相似度 (0-1)。
    使用字符级 bigram Jaccard 相似度。
    """
    if not stmt_a or not stmt_b:
        return 0.0
    # Normalize whitespace
    a = ' '.join(stmt_a.strip().split())
    b = ' '.join(stmt_b.strip().split())
    # Character bigrams for Chinese-friendly tokenization
    def bigrams(s):
        return set(s[i:i+2] for i in range(len(s)-1)) if len(s) > 1 else {s}
    a_set = bigrams(a)
    b_set = bigrams(b)
    if not a_set or not b_set:
        return 0.0
    intersection = len(a_set & b_set)
    union = len(a_set | b_set)
    return round(intersection / union, 4)


def _check_orchestrator_stop_conditions(state: AgentState) -> dict:
    """
    Orchestrator 每轮结束时的停止/继续决策检查。

    Returns: {
        "stop": bool,               # 是否应该停止迭代
        "reason": str,              # 停止原因
        "max_round_reached": bool,  # 是否达到最大轮次
        "evidence_strong": bool,    # 证据强度 + 评审分数是否达标
        "converged": bool,          # 假设语义是否已收敛
        "similarity_score": float,  # 当前假设与上一轮的相似度
    }
    """
    result = {
        "stop": False,
        "reason": "",
        "max_round_reached": False,
        "evidence_strong": False,
        "converged": False,
        "similarity_score": 0.0,
    }

    iteration = state.get("iteration", 0)
    max_iterations = min(state.get("_max_iterations_", 200), 200)  # 200轮硬上限

    # --- Condition 1: 检查当前轮次是否达到最大轮次（200轮）---
    if iteration >= max_iterations:
        result["max_round_reached"] = True
        result["stop"] = True
        result["reason"] = f"已达到最大轮次上限 ({max_iterations}/{max_iterations})"
        logger.info(f"[OrchestratorStop] MAX_ROUNDS_REACHED: {result['reason']}")
        return result

    # --- Gather evidence strength and latest review score ---
    evidence_chains = state.get("evidence_chains", [])
    reviews = state.get("review_records", [])

    # Evidence strength: average of all chain strengths
    if evidence_chains:
        avg_evidence_strength = sum(
            e.get("strength", 0.5) for e in evidence_chains
        ) / len(evidence_chains)
    else:
        # Fallback to approved hypothesis posterior confidence
        approved_hyps = [
            h for h in state.get("hypothesis_tree", [])
            if h.get("status") == "approved_by_reviewer"
        ]
        if approved_hyps:
            avg_evidence_strength = max(
                h.get("confidence_posterior", 0.5) for h in approved_hyps
            )
        else:
            avg_evidence_strength = 0.0

    # Latest review score
    if reviews:
        latest_review = reviews[-1]
        latest_review_score = latest_review.get("total_score", 0)
    else:
        latest_review_score = 0

    # --- Condition 2: 检查本轮证据强度 > 0.85 且评审得分 > 80 ---
    if avg_evidence_strength > 0.85 and latest_review_score > 80:
        result["evidence_strong"] = True
        result["stop"] = True
        result["reason"] = (
            f"证据强度充足 (avg_strength={avg_evidence_strength:.3f}, "
            f"latest_score={latest_review_score}/100)，结论明确可终止"
        )
        logger.info(f"[OrchestratorStop] EVIDENCE_STRONG: {result['reason']}")
        return result

    # --- Condition 3: 检查当前假设与上一轮假设的语义相似度 ---
    # Get all approved hypotheses from the tree
    approved_hyps = [
        h for h in state.get("hypothesis_tree", [])
        if h.get("status") == "approved_by_reviewer"
    ]

    if len(approved_hyps) >= 2:
        # Sort by created_at timestamp to get chronological order
        sorted_hyps = sorted(approved_hyps, key=lambda h: h.get("created_at", ""))
        prev_stmt = sorted_hyps[-2].get("statement", "")
        curr_stmt = sorted_hyps[-1].get("statement", "")
        similarity = _hypothesis_statement_similarity(prev_stmt, curr_stmt)
        result["similarity_score"] = similarity

        if similarity > 0.95:
            result["converged"] = True
            result["stop"] = True
            result["reason"] = (
                f"假设已收敛：当前假设与上一轮语义相似度 = {similarity:.4f} (> 0.95)"
            )
            logger.info(f"[OrchestratorStop] CONVERGED: {result['reason']}")
            return result
    elif len(approved_hyps) == 1:
        # Only one approved — compute similarity against the original query as baseline
        query = state.get("query", "")
        if query and approved_hyps[0].get("statement"):
            result["similarity_score"] = _hypothesis_statement_similarity(query, approved_hyps[0]["statement"])

    # --- None of the stop conditions met → continue to next round ---
    result["stop"] = False
    result["reason"] = "未满足任何停止条件，进入下一轮反思"
    logger.info("[OrchestratorStop] CONTINUE: No stop condition met")
    return result


def _get_latest_hypothesis_text(state: AgentState) -> str | None:
    """Return the statement text of the most recently approved hypothesis."""
    hypotheses = state.get("hypothesis_tree", [])
    approved = [
        h for h in hypotheses if h.get("status") == "approved_by_reviewer"
    ]
    if not approved:
        # Fallback to latest active/proposed hypothesis
        candidates = [
            h for h in hypotheses if h.get("status") in ("active", "proposed")
        ]
        if not candidates:
            return None
    else:
        candidates = approved
    # Sort by created_at to get the newest
    sorted_candidates = sorted(candidates, key=lambda h: h.get("created_at", ""))
    return sorted_candidates[-1].get("statement", "") if sorted_candidates else None


# ============================================================
# State Evaluation Metrics
# ============================================================

def evaluate_state(state: AgentState) -> dict:
    """
    计算当前状态的量化指标，用于LLM决策和确定性路由。
    返回一个包含所有关键信号的结构化字典。
    """
    hypotheses = state.get("hypothesis_tree", [])
    reviews = state.get("review_records", [])
    evidence_chains = state.get("evidence_chains", [])
    anomaly_graph = state.get("anomaly_graph", [])
    experiments = state.get("experiment_records", [])

    # --- Evidence Strength (0-1) ---
    if evidence_chains:
        strengths = [e.get("strength", 0.5) for e in evidence_chains]
        avg_evidence = sum(strengths) / len(strengths)
        max_evidence = max(strengths)
        min_evidence = min(strengths)
    elif reviews:
        # 用评审分数间接估计证据强度
        avg_score = sum(r.get("total_score", 50) for r in reviews) / len(reviews)
        avg_evidence = avg_score / 100.0
        max_evidence = max(r.get("total_score", 50) for r in reviews) / 100.0
        min_evidence = min(r.get("total_score", 50) for r in reviews) / 100.0
    else:
        avg_evidence = 0.0
        max_evidence = 0.0
        min_evidence = 0.0

    # --- Uncertainty Level (0-1, 越高越不确定) ---
    confidences = [
        h.get("confidence_posterior", h.get("confidence_prior", 0.5))
        for h in hypotheses if h.get("status") != "pruned"
    ]
    if len(confidences) >= 2:
        confidence_spread = max(confidences) - min(confidences)
        # 假设数量多但分歧大 = 高不确定性
        uncertainty = confidence_spread * 0.6 + min(len(hypotheses) * 0.1, 0.4)
    elif confidences:
        # 只有一个活跃假设 = 中等确定性
        uncertainty = 0.3
    else:
        uncertainty = 1.0  # 没有假设 = 完全不确定

    # --- Anomaly Risk (low/medium/high/none) ---
    risk_levels = [a.get("severity", "none") for a in anomaly_graph]
    high_count = risk_levels.count("high")
    medium_count = risk_levels.count("medium")
    if high_count > 0:
        anomaly_risk = "high"
    elif medium_count > 0:
        anomaly_risk = "medium"
    elif len(anomaly_graph) > 0:
        anomaly_risk = "low"
    else:
        anomaly_risk = "none"

    # --- Hypothesis Status Distribution ---
    status_counts = {}
    for h in hypotheses:
        s = h.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    approved_count = status_counts.get("approved_by_reviewer", 0)
    proposed_count = status_counts.get("proposed", 0)
    refuted_count = status_counts.get("refuted", 0)
    needs_rev_count = status_counts.get("needs_revision", 0)

    # --- Iteration Context ---
    iteration = state.get("iteration", 0)
    max_iter = min(state.get("_max_iterations_", 200), 200)  # 200轮硬上限
    consecutive_failures = state.get("consecutive_failures", 0)
    convergence = state.get("convergence_score", 0.0)

    return {
        "avg_evidence_strength": round(avg_evidence, 3),
        "max_evidence_strength": round(max_evidence, 3),
        "uncertainty_level": round(min(uncertainty, 1.0), 3),
        "anomaly_risk": anomaly_risk,
        "num_hypotheses": len(hypotheses),
        "status_distribution": status_counts,
        "approved_count": approved_count,
        "refuted_count": refuted_count,
        "consecutive_failures": consecutive_failures,
        "iteration": iteration,
        "max_iterations": max_iter,
        "remaining_budget": max_iter - iteration,
        "convergence_score": convergence,
        "num_experiments": len(experiments),
        "num_reviews": len(reviews),
        # --- Orchestrator convergence info for next round comparison ---
        "latest_hypothesis_statement": _get_latest_hypothesis_text(state) if hypotheses else "",
    }


def format_eval_report(ev: dict) -> str:
    """将评估指标格式化为 LLM 可读的诊断报告"""
    lines = [
        f"## 当前状态诊断\n\n"
        f"**迭代轮次**: {ev['iteration']}/{ev['max_iterations']} (剩余{ev['remaining_budget']}轮)\n"
        f"**收敛度**: {ev['convergence_score']:.1%}\n\n"
        f"### 假设池 ({ev['num_hypotheses']} 个)\n",
    ]
    for status, count in ev['status_distribution'].items():
        icon = {"approved_by_reviewer": "✅", "proposed": "📝",
                "needs_revision": "🔄", "refuted": "❌",
                "pruned": "✂️"}.get(status, "❓")
        lines.append(f"- {icon} [{status}]: {count}")

    lines += [
        f"\n### 证据质量\n"
        f"- 平均强度: {ev['avg_evidence_strength']:.1%} | 最高: {ev['max_evidence_strength']:.1%}\n"
        f"- 不确定性水平: {ev['uncertainty_level']:.1%}\n",
        f"### 风险与异常\n"
        f"- 异常风险等级: {ev['anomaly_risk']}\n"
        f"- 连续失败次数: {ev['consecutive_failures']}\n\n"
        f"### 实验进度\n"
        f"- 已设计实验: {ev['num_experiments']}\n"
        f"- 已完成评审: {ev['num_reviews']}",
    ]
    return "\n".join(lines)


# ============================================================
# LLM-Driven Decision Router
# ============================================================

AVAILABLE_ACTIONS = {
    "literature_review": "继续深入文献调研，挖掘更多事实",
    "hypothesis_generation": "生成新的科学假设或进化现有假设",
    "experiment_design": "为选定假设设计完整实验方案",
    "data_analysis": "执行数据分析与因果推断",
    "interpretation": "解读分析结果，更新假设置信度",
    "reviewer_agent": "提交假设进行五维评审",
    "reflection": "反思当前方向，修正失败假设",
    "termination_eval": "评估是否达到终止条件",
    "report_writing": "生成最终标准化研究报告",
    "pi_agent_meeting": "PI Agent 整合成果并汇报",
    "evolution_manager": "自我进化，提取meta-insights",
}

DECISION_PROMPT_TEMPLATE = """## 任务：科研编排决策

你是 twinScientist 系统的 Orchestrator。根据以下状态诊断，选择**最合适的一个**认知节点继续推进。

**⚠️ 多轮循环硬约束（最多 200 轮）：**
- 当前轮次超过 200 轮时，必须终止并进入 report_writing
- 每一轮结束后都必须回答：①本轮实验有哪些漏洞或局限？②如果修正这些漏洞，假设应该怎么改？③修正后的假设是否值得再验证一次？

{state_diagnosis}

**上一步刚执行的操作**: {last_executed_action}

## 候选认知节点及其含义
""" + "\n".join(f"- **{k}**: {v}" for k, v in AVAILABLE_ACTIONS.items()) + """

## 决策规则
1. **优先处理高质量候选假设**：如果有 approved 假设但未进行实验设计，选 experiment_design
2. **高不确定性时探索新方向**：uncertainty > 0.6 且还有迭代预算 → hypothesis_generation
3. **有证据时验证假设**：avg_evidence > 0.4 → interpretation → reviewer_agent 路径
4. **遇到异常时反思修正**：anomaly_risk 在 medium/high → reflection
5. **接近极限时收口**：remaining ≤ 2 → termination_eval → report_writing
6. **每个假设都应该被验证**：不要跳过 experiment_design 直接进入 report_writing
7. **禁止无效循环**：如果连续生成→评审→拒绝→重新生成的循环超过 3 轮，换策略（改变搜索方向或缩小范围）
8. **禁止重复执行**：`last_executed_action` 是上一步刚执行完的节点，不要再次选择它（termination_eval 除外）
9. **二百轮上限**：无论研究进展如何，第 200 轮结束后必须终止

## 输出格式（严格遵守）
```
<DECISION>
action: <从上面列表中选一个唯一的行动名称>
reason: <简短理由，为什么这个行动最紧迫>
</DECISION>
"""


async def llm_orchestrator_decision(state: AgentState) -> str:
    """
    LLM驱动的动态路由核心函数。

    调用 Qwen 模型基于当前 State 的诊断报告做出最优下一步决策。
    返回值是下一个认知节点的名称字符串。
    """
    ev = evaluate_state(state)
    diagnosis = format_eval_report(ev)

    llm = QwenClient(
        base_url=settings.bailian_base_url,
        api_key=settings.bailian_api_key,
        model=settings.model_name,
    )

    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
        {"role": "user", "content": DECISION_PROMPT_TEMPLATE.format(state_diagnosis=diagnosis, last_executed_action=state.get("current_action", "none"))},
    ]

    try:
        result = await llm.chat_complete(messages=messages, temperature=0.3, max_tokens=1024)
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Parse the <DECISION> block (matches DECISION_PROMPT_TEMPLATE output format)
        match = re.search(r'<DECISION>\s*\naction:\s*(\S+)', content, re.DOTALL)
        if match:
            chosen_action = match.group(1).lower().strip()
            if chosen_action in AVAILABLE_ACTIONS:
                logger.info(f"[Orchestrator] LLM chose: {chosen_action}")
                return chosen_action

        # Fallback to deterministic heuristic
        logger.warning(f"[Orchestrator] Failed to parse LLM decision: {content[:100]}")
        return _deterministic_fallback(state)

    except Exception as e:
        logger.error(f"[Orchestrator] LLM call failed: {e}")
        return _deterministic_fallback(state)


def _deterministic_fallback(state: AgentState) -> str:
    """
    LLM 不可用时的确定性降级路由。
    保证系统即使在无 LLM 连接时也能基本运行。
    """
    hypotheses = state.get("hypothesis_tree", [])
    status_counts = {}
    for h in hypotheses:
        s = h.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    # Priority order: pending review → approved without experiment → reflection → termination
    if status_counts.get("approved_by_reviewer", 0) > 0:
        return "experiment_design"
    if status_counts.get("proposed", 0) > 0:
        return "experiment_design"
    if status_counts.get("needs_revision", 0) > 0:
        return "reflection"

    if state.get("iteration", 0) >= 200:  # 200轮硬上限
        return "termination_eval"

    return "hypothesis_generation"


# ============================================================
# Updated Routing Functions (combining LLM + heuristics)
# ============================================================

def route_after_literature(state: AgentState) -> str:
    """文献完成后：确定性的过渡到假设生成阶段"""
    facts = state.get("fact_extraction", [])
    if not facts or len(facts) < 2:
        return "literature_review"  # 继续补充
    return "hypothesis_generation"


def route_after_hypothesis(state: AgentState) -> str:
    """假设生成后：让LLM决定是否进入实验设计还是先补充其他环节"""
    # New proposals can always go to experiment design
    proposed = [h for h in state.get("hypothesis_tree", []) if h.get("status") == "proposed"]
    if proposed:
        return "experiment_design"

    approved = [h for h in state.get("hypothesis_tree", []) if h.get("status") == "approved_by_reviewer"]
    if approved:
        return "experiment_design"

    return "reflection"  # No viable hypotheses to explore


def route_after_experiment(state: AgentState) -> str:
    """实验设计完成后进入数据分析"""
    return "data_analysis"


def route_after_analysis(state: AgentState) -> str:
    """数据分析后进入解读"""
    return "interpretation"


def route_after_reviewer(state: AgentState) -> str:
    """Reviewer后：低分回reflection，高分进报告"""
    reviews = state.get("review_records", [])
    if not reviews:
        return "reflection"

    latest = reviews[-1]
    if latest.get("total_score", 0) >= 75:
        return "report_writing"
    return "reflection"


def route_after_reflection(state: AgentState) -> str:
    """反思后：检查预算，决定再生成还是终止"""
    if state.get("iteration", 0) >= 200 or state.get("consecutive_failures", 0) >= 3:  # 200轮硬上限
        return "terminating"
    return "hypothesis_generation"


def route_after_termination(state: AgentState) -> str:
    """
    终止决策路由 —— 严格优先级顺序

    P1: 迭代次数硬上限 (绝对天花板)
    P2: 读取 node_termination_eval 的计算结论 (优先复用，不复算)
    P3: 独立综合评分回退 (原始行为的最后兜底)
    """
    max_iters = state.get("_max_iterations_", 200)
    iteration = state.get("iteration", 0)

    # === P1: 绝对天花板 ===
    if iteration >= max_iters:
        return "report_writing"

    # === P2: 读取 TerminationEval 已经算好的结论 ===
    term_result = state.get("_termination_result")
    if term_result:
        should_term = term_result.get("should_terminate", False)
        logger.info(
            f"[RouteAfterTerm] Taking TerminationEval verdict "
            f"(should_terminate={should_term}): {term_result.get('stop_reason', '?')}"
        )
        if should_term:
            return "report_writing"
        return "hypothesis_generation"

    # === P3: 回退到原始独立计算 ===
    convergence = state.get("convergence_score", 0.0)
    evidence_chains = state.get("evidence_chains", [])
    exploration_done = state.get("exploration_exhausted", False)
    evidence_str = sum(e.get("strength", 0.5) for e in evidence_chains) / max(len(evidence_chains), 1)
    combined = convergence * 0.4 + evidence_str * 0.3 + (0.8 if exploration_done else 0.0) * 0.3

    if combined >= 0.85:
        return "report_writing"
    return "hypothesis_generation"
