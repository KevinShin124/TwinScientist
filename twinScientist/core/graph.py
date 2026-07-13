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
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from config.settings import settings
from core.state import AgentState
from core.nodes import (
    node_literature_review,
    node_hypothesis_generation,
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
    llm_orchestrator_decision,
    route_after_literature,
    route_after_experiment,
    route_after_analysis,
    route_after_reviewer,
    route_after_reflection,
    route_after_termination,
)

logger = logging.getLogger(__name__)


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
     │hypothesis_gen    │  ← 生成假设
     └──────┬───────────┘
            ▼ (确定性的下一步)
     ┌──────────────────┐
     │experiment_design │  ← 为假设设计实验
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
    workflow.add_node("orchestrator_plan", _node_orchestrator_plan)

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

    # Hypothesis generation → deterministic route to experiment design
    # This ensures every proposed hypothesis gets an experiment designed for it
    workflow.add_conditional_edges(
        "hypothesis_generation",
        lambda s: "experiment_design" if any(
            h.get("status") in ("proposed", "active", "approved_by_reviewer")
            for h in s.get("hypothesis_tree", [])
        ) else "reflection",
        {
            "experiment_design": "experiment_design",
            "reflection": "reflection",
        },
    )

    # Experiment design → data analysis (deterministic)
    workflow.add_edge("experiment_design", "data_analysis")

    # Analysis → interpretation (deterministic)
    workflow.add_edge("data_analysis", "interpretation")

    # Interpretation → Reviewer (deterministic)
    workflow.add_edge("interpretation", "reviewer_agent")

    # Reviewer → conditional based on score
    workflow.add_conditional_edges(
        "reviewer_agent",
        route_after_reviewer,
        {
            "reflection": "reflection",
            "report_writing": "report_writing",
        },
    )

    # Reflection → back to hypothesis_generation (for revision loop)
    workflow.add_conditional_edges(
        "reflection",
        lambda s: "terminating" if s.get("iteration", 0) >= s.get("_max_iterations_", 15) or s.get("consecutive_failures", 0) >= 3 else "hypothesis_generation",
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


async def _node_orchestrator_plan(state: AgentState) -> dict:
    """
    Orchestrator 规划节点 — LLM 驱动的决策引擎。

    Each iteration开始时调用一次，评估当前状态并输出
    下一个要执行的操作。将决策结果存储在 state['next_step']
    供后续路由使用。
    """
    chosen_action = await llm_orchestrator_decision(state)
    logger.info(f"[OrchestratorPlan] Selected next action: {chosen_action}")

    return {
        "next_step": chosen_action,
        "current_action": chosen_action,
    }


# Export for main entry
cognitive_graph = build_cognitive_graph()
