"""
Layer 2: Cognitive Graph — LLM驱动的科研编排系统

关键变化（从固定流程图升级为 AI Scientist 主流模式）：
- Supervisor Pattern: Orchestrator 节点作为调度中枢，每轮做出全局最优决策
- Dynamic Routing: 大多数流程通过动态路由而非硬编码边连接
- Deterministic Guardrails: ethics_check → literature_review 保持确定性
- Fallback Path: LLM 决策失败时自动降级到确定性启发式路由
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from config.settings import settings
from core.state import AgentState
from core.nodes import (
    node_literature_review,
    node_hypothesis_generation,
    node_tournament_eval,
    node_experiment_design,
    node_data_analysis,
    node_interpretation,
    node_reflection,
    node_report_writing,
    node_pi_agent_meeting,
    node_reviewer_agent,
    node_ethics_check,
    node_termination_eval,
    node_human_approval,
    node_evolution_manager,
)
from core.orchestrator import (
    _check_orchestrator_stop_conditions,
    route_after_literature,
    route_after_experiment,
    route_after_analysis,
    route_after_reflection,
    route_after_termination,
    _mc_log_and_recommend,
)

logger = logging.getLogger(__name__)


def _after_reviewer_route(state: AgentState) -> str:
    """
    After reviewer_agent finishes, delegate to the orchestrator stop-check.

    Routing logic:
      0) Minimum rounds guardrail — force at least min_rounds (default 2) of
         iteration before ANY early termination. Prevents single-pass false positives.
      1) Max rounds reached → termination_eval
      2) Evidence >0.85 AND score >85 AND rounds>=min_rounds → report_writing (conclusion clear)
      3) Hypothesis similarity >0.95 → termination_eval (converged)
      4) Otherwise → reflection (continue next round)
    """
    # === Guardrail: minimum rounds before any termination ===
    min_rounds = int(os.getenv("MIN_REFLECTION_ROUNDS", "2"))
    current_iter = state.get("iteration", 0)
    if current_iter < min_rounds:
        logger.info(
            f"[AfterReviewer] MIN_ROUNDS NOT MET: iteration={current_iter}<{min_rounds}, "
            f"forcing reflection instead of stopping"
        )
        _mc_log_and_recommend(state, "reflection")
        return "reflection"

    checks = _check_orchestrator_stop_conditions(state)

    if checks["stop"]:
        if checks["max_round_reached"] or checks["converged"]:
            logger.info(f"[AfterReviewer] ORCHESTRATOR STOPPED (terminate): {checks['reason']}")
            action = "termination_eval"
        else:
            # Evidence strong AND passed min rounds — proceed to report writing
            logger.info(f"[AfterReviewer] EVIDENCE_STRONG (after {min_rounds}+ rounds): {checks['reason']}")
            action = "report_writing"
    else:
        logger.info(f"[AfterReviewer] CONTINUE: {checks['reason']}")
        action = "reflection"

    # Log this routing decision to MC experience store
    _mc_log_and_recommend(state, action)
    return action


def build_cognitive_graph() -> "CompiledGraph":
    """
    构建认知图（DAG）。

    架构概览（Supervisor/Coordinator Pattern）:

        START
          │
          ▼
     ┌──────────────┐  ← 确定性伦理审查（必须通过）
     │ ethics_check │──┬── blocked → termination_eval
     └──────┬───────┘  ├── human_review → human_approval (HITL)
            │           └── approved → literature_review
            ▼
     ┌──────────────┐
     │literature_   │  ← 确定性地启动文献调研
     │ review       │
     └──────┬───────┘
            ▼ (如果facts>=2)
     ┌──────────────────┐
     │hypothesis_gen    │  ← 生成5-10个候选假设
     └──────┬───────────┘
            ▼ (确定性的下一步)
     ┌──────────────────┐
     │tournament_eval   │  ← 淘汰赛：两两比较选出优胜者
     └──────┬───────────┘
            ▼ (确定性的下一步)
     ┌──────────────────┐
     │experiment_design │  ← 为获胜假设设计实验
     └──────┬───────────┘
            ▼ (确定性的下一步)
     ┌──────────────────┐
     │data_analysis     │  ← 真实数据因果推断
     └──────┬───────────┘
            ▼
     ┌──────────────────┐
     │interpretation    │  ← 结果解读
     └──────┬───────────┘
            ▼
     ┌──────────────────┐
     │reviewer_agent    │  ← 五维评审
     └──────┬───────────┘
            │
       score≥75  │  <75
       ▼        ▼
    report    reflection ──▶ back to hypothesis_generation loop
    writing
       │
       ▼
    pi_agent_meeting ──▶ human_approval ──▶ evolution_manager ──▶ END
    """

    workflow = StateGraph(AgentState)

    # ----------------------------------------------------------
    # Register all nodes
    # ----------------------------------------------------------
    workflow.add_node("ethics_check", node_ethics_check)
    workflow.add_node("literature_review", node_literature_review)
    workflow.add_node("hypothesis_generation", node_hypothesis_generation)
    workflow.add_node("tournament_eval", node_tournament_eval)
    workflow.add_node("experiment_design", node_experiment_design)
    workflow.add_node("data_analysis", node_data_analysis)
    workflow.add_node("interpretation", node_interpretation)
    workflow.add_node("reviewer_agent", node_reviewer_agent)
    workflow.add_node("reflection", node_reflection)
    workflow.add_node("report_writing", node_report_writing)
    workflow.add_node("pi_agent_meeting", node_pi_agent_meeting)
    workflow.add_node("human_approval", node_human_approval)
    workflow.add_node("evolution_manager", node_evolution_manager)
    workflow.add_node("termination_eval", node_termination_eval)

    def _route_ethics(state: AgentState) -> str:
        """Route based on ethics check result — Item 15"""
        status = state.get("ethics_status", "")
        if status == "blocked":
            return "ethics_blocked"  # terminate early
        if status == "human_review_required":
            return "human_approval"   # HITL checkpoint
        return "literature_review"   # approved

    # ----------------------------------------------------------
    # Edges — Deterministic entry path with ethics guardrail
    # ----------------------------------------------------------
    workflow.set_entry_point("ethics_check")
    workflow.add_conditional_edges(
        "ethics_check",
        _route_ethics,
        {
            "ethics_blocked": "termination_eval",  # blocked → evaluate & write report explaining why
            "human_approval": "human_approval",
            "literature_review": "literature_review",
        },
    )

    # Literature → deterministic route (prevents orchestrator from re-selecting literature_review)
    workflow.add_conditional_edges(
        "literature_review",
        route_after_literature,
        {
            "literature_review": "literature_review",
            "hypothesis_generation": "hypothesis_generation",
        },
    )

    # Hypothesis generation → Tournament Eval (elimination bracket)
    workflow.add_conditional_edges(
        "hypothesis_generation",
        lambda s: "tournament_eval" if any(
            h.get("status") in ("proposed", "active", "approved_by_reviewer")
            for h in s.get("hypothesis_tree", [])
        ) else "reflection",
        {
            "tournament_eval": "tournament_eval",
            "reflection": "reflection",
        },
    )

    # Tournament eval → Experiment design (winner gets experiments designed)
    workflow.add_edge("tournament_eval", "experiment_design")

    # Experiment design → data analysis (deterministic)
    workflow.add_edge("experiment_design", "data_analysis")

    # Analysis → interpretation (deterministic)
    workflow.add_edge("data_analysis", "interpretation")

    # Interpretation → Reviewer (deterministic)
    workflow.add_edge("interpretation", "reviewer_agent")

    # Reviewer → Orchestrator stop/continue check:
    #   1) Max rounds (5)? → termination_eval
    #   2) Evidence >0.85 AND review >80? → report_writing
    #   3) Hypothesis similarity >0.95? → termination_eval
    #   4) Otherwise → reflection (next round)
    workflow.add_conditional_edges(
        "reviewer_agent",
        _after_reviewer_route,
        {
            "reflection": "reflection",
            "report_writing": "report_writing",
            "termination_eval": "termination_eval",
        },
    )

    # Reflection → back to hypothesis_generation (for revision loop)
    def _route_reflection(state: AgentState) -> str:
        """Reflection → hypothesis_generation 或终止 + MC 日志"""
        max_iter = state.get("_max_iterations_", 200)
        if state.get("iteration", 0) >= max_iter or state.get("consecutive_failures", 0) >= 3:
            action = "terminating"
        else:
            action = "hypothesis_generation"
        _mc_log_and_recommend(state, action)
        return action

    workflow.add_conditional_edges(
        "reflection",
        _route_reflection,
        {
            "hypothesis_generation": "hypothesis_generation",
            "terminating": "termination_eval",
        },
    )

    # Termination evaluation → either write report or explore more
    workflow.add_conditional_edges(
        "termination_eval",
        route_after_termination,
        {
            "report_writing": "report_writing",
            "hypothesis_generation": "hypothesis_generation",
        },
    )

    # Report → final stages
    workflow.add_edge("report_writing", "pi_agent_meeting")
    workflow.add_edge("pi_agent_meeting", "human_approval")
    workflow.add_edge("human_approval", "evolution_manager")
    workflow.add_edge("evolution_manager", END)

    checkpointer = MemorySaver()

    compiled = workflow.compile(
        interrupt_before=["human_approval"],  # HITL: 等待人类确认后再继续
        checkpointer=checkpointer,  # 多轮会话恢复 + 迭代历史
    )

    return compiled


# Export for main entry
cognitive_graph = build_cognitive_graph()
