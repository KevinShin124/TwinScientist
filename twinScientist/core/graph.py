"""
Layer 2: Cognitive Graph — LLM-driven research orchestration with Debate & HITL

Key enhancements over original:
- Supervisor Pattern: Orchestrator decides next cognitive operation dynamically
- Dynamic Routing: Most flow through decision nodes rather than hardcoded edges
- Deterministic Guardrails: ethics_check → literature_review remain deterministic
- Fallback Path: Auto-degrade to heuristic routing if LLM fails
- **NEW**: Multi-Agent Debate after reviewer_agent (Pro/Con/Judge)
- **NEW**: Enhanced HITL at multiple interrupt points
- **NEW**: User chat messages injected into subsequent LLM calls

Graph topology:
    START
      │
      ▼
 ┌──────────────┐  ← Ethics check (blocked → terminate / human_review → HITL / approved → lit_review)
 │ ethics_check │──┬── blocked → termination_eval
 │              │  ├── human_review → human_approval (HITL checkpoint)
 │              │  └── approved → literature_review
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │literature_   │  ← Search Crossref/arXiv, extract verified facts
 │ review       │  ← Knowledge graph auto-built from extracted entities
 └──────┬───────┘
        ▼ (if facts ≥ 2)
 ┌──────────────────┐
 │hypothesis_gen    │  ← LogicEngine + LLM generate candidates
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │tournament_eval   │  ← Bracket elimination of N candidates
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │experiment_design │  ← Design experiments for winning hypothesis
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │data_analysis     │  ← Causal inference: CCM/Granger/Counterfactual
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │interpretation    │  ← Update hypothesis confidence
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │reviewer_agent    │  ← Five-dimension peer review
 └──────┬───────────┘
        │ score≥60           <60 or needs_revision
        ▼            ┌──────────────┐
  debate_orchestrator│reflection──▶back to hypothesis_generation
        ▼
 ┌──────────────────┐
 │termination_eval   │  ← Multi-dimensional convergence assessment
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │report_writing    │  ← Assemble final report with all evidence
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │pi_agent_meeting  │  ← PI integrates findings
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │human_approval    │  ← HITL gate (approve/revise/chat/halt)
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │evolution_manager │  ← Extract meta-insights
 └──────────────────┘
      END
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
    node_termination_eval,  # ← uses nodes.py (NOT nodes_term_patch.py)
    node_human_approval,
    node_evolution_manager,
)
from core.nodes_post_chat import _node_post_report_chat
from core.debate import DebateOrchestrator
from core.orchestrator import (
    _check_orchestrator_stop_conditions,
    set_orch_check_in_state,
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
    After reviewer_agent finishes, delegate to orchestrator stop-check.

    Routing logic:
      0) Minimum rounds guardrail — force at least min_rounds iterations
         before ANY early termination. Prevents single-pass false positives.
      1) Stop conditions met (any reason) → debate_orchestrator first,
         then termination_eval → report_writing
      2) Otherwise → reflection (next round)

    Key design: ALL stop paths funnel through debate first when we have
    hypotheses to stress-test. This ensures adversarial review before
    proceeding to report writing or termination.
    """
    # === Guardrail: minimum rounds before any termination ===
    min_rounds = int(os.getenv("MIN_REFLECTION_ROUNDS", "0"))
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
        # Stop condition met → run debate first (adversarial review), then terminate
        logger.info(f"[AfterReviewer] STOP CONDITION MET: {checks['reason']}")
        action = "debate_then_terminate"
    else:
        # Not stopping → direct path to reflection (no detour through debate)
        logger.info(f"[AfterReviewer] CONTINUE: {checks['reason']}")
        action = "reflection"

    # Log this routing decision to MC experience store
    _mc_log_and_recommend(state, action)
    return action


def build_cognitive_graph() -> "CompiledGraph":
    """
    Build the cognitive graph with enhanced debate and HITL integration.

    Key changes from base graph:
    - Added debate_then_terminate intermediate node for adversarial review
    - Human approval now supports chat-based interaction via ChatAgent
    - More interrupt_before checkpoints for continuous HITL engagement
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
    workflow.add_node("debate_then_terminate", _node_debate_then_terminate)
    workflow.add_node("report_writing", node_report_writing)
    workflow.add_node("pi_agent_meeting", node_pi_agent_meeting)
    workflow.add_node("human_approval", node_human_approval)
    workflow.add_node("evolution_manager", node_evolution_manager)
    workflow.add_node("post_report_chat", _node_post_report_chat)
    workflow.add_node("termination_eval", node_termination_eval)

    def _route_ethics(state: AgentState) -> str:
        """Route based on ethics check result."""
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
            "ethics_blocked": "termination_eval",
            "human_approval": "human_approval",
            "literature_review": "literature_review",
        },
    )

    # Literature → deterministic route
    workflow.add_conditional_edges(
        "literature_review",
        route_after_literature,
        {
            "literature_review": "literature_review",
            "hypothesis_generation": "hypothesis_generation",
        },
    )

    # Hypothesis generation → Tournament Eval
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

    # Tournament eval → Experiment design (deterministic)
    workflow.add_edge("tournament_eval", "experiment_design")

    # Experiment design → data analysis (deterministic)
    workflow.add_edge("experiment_design", "data_analysis")

    # Analysis → interpretation (deterministic)
    workflow.add_edge("data_analysis", "interpretation")

    # Interpretation → Reviewer (deterministic)
    workflow.add_edge("interpretation", "reviewer_agent")

    # Reviewer → Orchestrator: debate_first_if_stopping, reflection_if_continuing
    workflow.add_conditional_edges(
        "reviewer_agent",
        _after_reviewer_route,
        {
            "reflection": "reflection",
            "report_writing": "report_writing",  # Legacy shortcut (score high enough)
            "debate_then_terminate": "debate_then_terminate",
        },
    )

    # Reflection → back to hypothesis_generation or terminating
    def _route_reflection(state: AgentState) -> str:
        """Reflection → hypothesis_generation or termination + MC log."""
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

    # Debate → termination evaluation (mandatory — fixes missing edge bug)
    workflow.add_edge("debate_then_terminate", "termination_eval")

    # Report → final stages
    workflow.add_edge("report_writing", "pi_agent_meeting")
    workflow.add_edge("pi_agent_meeting", "human_approval")
    workflow.add_edge("human_approval", "evolution_manager")
    workflow.add_edge("evolution_manager", "post_report_chat")

    checkpointer = MemorySaver()

    compiled = workflow.compile(
        interrupt_before=[],  # CLI mode: no interrupts. UI mode handles HITL via its own flow.
        interrupt_after=[],   # CLI mode: no interrupts.
        checkpointer=checkpointer,
    )

    return compiled


# ──────────────────────────────────────────────
# Helpers for post-report chat routing (used by UI)
# ──────────────────────────────────────────────

def _route_post_report_chat(state: AgentState) -> str:
    """Route based on user intent after research concludes."""
    signal = state.get("_routing_signal", "continue_chatting")

    # Direct route to previous stages when user requests re-analysis
    if signal.startswith("loop_back_to_"):
        target = signal.replace("loop_back_to_", "")
        logger.info(f"[PostReportChatRouter] Looping back to {target}")
        return target

    if signal == "end_session":
        return "END"

    # If still chatting with no explicit end signal, stay at current node
    # The next interrupt will pick up the user's message
    if signal == "end_session" or "accept_and_end" in signal:
        return "END"

    # Default: let the flow continue (eventually hits END via human path)
    return "_next_node_or_check"


# ──────────────────────────────────────────────
# Debate node (Multi-Agent adversarial review)
# ──────────────────────────────────────────────

async def _node_debate_then_terminate(state: AgentState) -> dict:
    """
    【智能体思辨】Pro/Con/Judge 单轮辩论 — 比赛要求核心功能。

    在终止前对最优假设进行对抗性辩论，体现"智能体思辨"能力。
    精简为 1 轮：LogicEngine 已在 hypothesis_generation 阶段完成确定性推理，
    辩论阶段只做最后一轮 LLM 对抗论证 + 裁判。
    """
    from core.llm_client import get_global_client

    hypotheses = state.get("hypothesis_tree", [])
    active_hyps = [
        h for h in hypotheses
        if h.get("status") not in ("pruned", "refused", "refuted_in_tournament")
    ]

    if not active_hyps:
        logger.info("[Debate] No active hypotheses, skipping debate")
        return {
            "current_action": "debate_then_terminate",
            "educational_annotations": [
                _edu_annotation("debate",
                    "无活跃假设可辩论，跳过辩论阶段。"
                    "在完整的研究流程中，辩论阶段会对最优假设进行 Pro/Con/Judge 三方对抗论证。")
            ],
        }

    evidence_chains = state.get("evidence_chains", [])
    review_records = state.get("review_records", [])
    user_feedback = state.get("user_feedback", "")

    # Reuse global LLM client (fallback to creating one)
    llm_client = get_global_client()
    if llm_client is None:
        from core.llm_client import QwenClient
        llm_client = QwenClient(
            base_url=settings.bailian_base_url,
            api_key=settings.bailian_api_key,
            model=settings.model_name,
        )

    # Run 1-round debate (was 3 rounds; simplified per competition optimization)
    orchestrator = DebateOrchestrator()
    try:
        debate_result = await orchestrator.run_debate(
            llm_client=llm_client,
            hypotheses=active_hyps,
            evidence_chains=evidence_chains,
            review_records=review_records,
            user_feedback=user_feedback,
            rounds=1,  # 精简为 1 轮
        )

        # Update hypothesis with debate results
        updated_hyp_tree = []
        for h in active_hyps:
            h_copy = dict(h)
            if debate_result.strongest_hypothesis_id and h.get("id") == debate_result.strongest_hypothesis_id:
                h_copy["confidence_posterior"] = debate_result.strongest_hypothesis_final_score / 100.0
                if debate_result.debates:
                    latest = debate_result.debates[-1]
                    h_copy["debate_last_score_after"] = latest.judge_score_after
            updated_hyp_tree.append(h_copy)

        # Record debate history
        debate_history = list(state.get("debate_history", []) or [])
        for d in debate_result.debates:
            debate_history.append({
                "round_number": d.round_number,
                "pro_output": d.pro_agent_output[:500],
                "con_output": d.con_agent_output[:500],
                "judge_score_before": d.judge_score_before,
                "judge_score_after": d.judge_score_after,
                "winner_side": d.winner_side,
                "timestamp": d.created_at,
            })

        # Build educational annotation
        edu_text = (
            "【智能体思辨 — 多 Agent 辩论】\n"
            "Pro Agent（辩护方）为正反假设提供证据支撑和逻辑论证；"
            "Con Agent（反辩方）寻找漏洞、替代解释和未控制混杂因子；"
            "Judge Agent（裁判）综合双方论据做出公正裁决。\n"
            "这种对抗性辩论机制模拟了科学共同体中的同行评议过程，"
            "是 AI Scientist 确保假设质量的关键环节。"
        )

        return {
            "hypothesis_tree": updated_hyp_tree,
            "debate_history": debate_history,
            "_debate_completed": True,
            "consensus_reached": debate_result.consensus_reached,
            "current_action": "debate_then_terminate",
            "educational_annotations": [_edu_annotation("debate", edu_text)],
        }

    except Exception as e:
        logger.error(f"[Debate] Failed: {e}")
        return {"current_action": "debate_then_terminate"}


# ──────────────────────────────────────────────
# Compiled graph instance — used by main.py and run_real_data_research.py
#
# Callers import:  from core.graph import cognitive_graph
#                :  result = await cognitive_graph.ainvoke(...)
# So we compile ONCE at module import time and expose the compiled graph.
# ──────────────────────────────────────────────

cognitive_graph = build_cognitive_graph()
