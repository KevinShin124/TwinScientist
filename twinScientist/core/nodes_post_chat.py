"""
Feature 4-Post: Post-report Free Chat Node

Inserted AFTER evolution_manager → allows continuous free-form
dialogue with agent even after research concludes.

If user requests re-analysis: routes back to data_analysis or
hypothesis_generation. If satisfied: returns final signal to END.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


async def _node_post_report_chat(state: AgentState) -> dict:
    """
    Post-report free-form chat node.

    Called after evolution_manager when the entire research pipeline
    has concluded. Instead of silently ending here, we pause and
    let the user engage in extended scientific discussion with the
    AI — question findings, request deeper exploration, or explore
    edge cases.

    Design philosophy: Real science never "ends" — it evolves through
    ongoing discourse. This node embodies that principle.

    Flow:
    1. Assemble comprehensive state summary
    2. Display initial greeting inviting questions
    3. On user message: call LLM → reply → store as chat_turn
    4. On re-analysis request: return routing signal to loop_back
    5. On "end": return terminal signal to flow.end()
    """
    # ---- Gather state components for context assembly ----
    final_report = state.get("final_report", "")
    query = state.get("query", "")
    domain = state.get("domain", "")
    iteration = state.get("iteration", 0)
    max_iter = state.get("_max_iterations_", 200)
    convergence_score = state.get("convergence_score", 0.0)
    stop_reason = state.get("stop_reason", "unknown")

    hypotheses = state.get("hypothesis_tree", [])
    evidence_chains = state.get("evidence_chains", [])
    review_records = state.get("review_records", [])
    experiment_records = state.get("experiment_records", [])
    debate_history = state.get("debate_history", [])
    chat_history = state.get("user_chat_messages", []) or []

    # Detect if there is an incoming user message via interrupt/resume mechanism.
    # LangGraph stores the value passed to graph.resume() inside the state as
    # "_resume_message" when the interrupt_before or interrupt_after hook triggers.
    incoming_user_msg = state.get("_resume_message", None)

    # Build comprehensive state summary for LLM context
    summary_lines = [
        f"**Research Question**: {query}",
        f"**Domain**: {domain}",
        f"**Iterations Completed**: {iteration}/{max_iter}",
        f"**Convergence Score**: {convergence_score:.0%}",
        f"**Stop Reason**: {stop_reason}",
        "",
    ]

    # Hypotheses summary
    active_hyps = [
        h for h in hypotheses if h.get("status") not in ("pruned", "refused")
    ]
    sorted_hyps = sorted(
        active_hyps,
        key=lambda h: h.get("confidence_posterior", h.get("confidence_prior", 0)),
        reverse=True,
    )
    summary_lines.append(f"**Active Hypotheses** ({len(sorted_hyps)}):")
    for i, h in enumerate(sorted_hyps[:10]):
        title = str(h.get("title", "?"))[:70]
        status = h.get("status", "?")
        posterior = h.get("confidence_posterior", h.get("confidence_prior", "?"))
        summary_lines.append(f"  [{i+1}] [{status}] \"{title}\" P(H|D)={posterior}")
    if len(active_hyps) > 10:
        summary_lines.append(f"  ...and {len(active_hyps)-10} more")

    # Evidence chains summary
    if evidence_chains:
        avg_strength = sum(e.get("strength", 0.5) for e in evidence_chains) / len(evidence_chains)
        methods_used = set(str(e.get("method_used", "unknown")) for e in evidence_chains)
        summary_lines.append(
            f"\n**Evidence Chains**: {len(evidence_chains)} | Avg strength: {avg_strength:.2f} | Methods: {', '.join(sorted(methods_used))}"
        )

    # Experiments summary
    analyzed_count = sum(1 for e in experiment_records if e.get("results", {}).get("analysis_complete", False))
    summary_lines.append(f"\n**Experiments Analyzed**: {analyzed_count}/{len(experiment_records)}")

    # Reviews summary
    if review_records:
        recent_scores = [r.get("total_score", "?") for r in review_records[-5:]]
        summary_lines.append(f"\n**Recent Review Scores**: {recent_scores}")

    # Debate history summary
    if debate_history:
        n_rounds = len(debate_history)
        last_r = debate_history[-1]
        summary_lines.append(
            f"\n**Debate Rounds**: {n_rounds} | Last: {last_r.get('score_before','?')}→{last_r.get('score_after','?')} "
            f"| Winner: {last_r.get('winner_side','?')}"
        )

    # Report excerpt
    if final_report:
        report_excerpt = final_report[:1200].replace("\n", " ").replace("\r", "")
        summary_lines.append(f"\n**Final Report (excerpt)**: {report_excerpt}")

    state_summary = "\n".join(summary_lines)

    # ---- Detect user intent from incoming message (if any) ----
    intent_detected = _detect_intent(incoming_user_msg) if incoming_user_msg else "initiated"

    # ---- Determine routing decision based on intent ----
    # Routes back to earlier stages when user explicitly requests
    should_loop_back = False
    loop_target_method = ""

    if incoming_user_msg and incoming_user_msg.strip():
        msg_lower = incoming_user_msg.lower()
        # Re-analysis triggers
        reanalyze_keywords = ["重新分析", "re-analyse", "re analyze", "reanalyze", "换个角度",
                              "从不同角", "another angle", "change perspective", "试一种新方法",
                              "use different method", "换一种思路", "试试其他方法", "重新计算",
                              "re-calculate", "再做一次"]
        if any(kw in msg_lower for kw in reanalyze_keywords):
            should_loop_back = True
            loop_target_method = "data_analysis"
        elif any(kw in msg_lower for kw in ["新的假设", "new hypothesis", "generate new",
                                             "change direction", "调整方向", "换个思路",
                                             "提出新假说", "suggest a new one"]):
            should_loop_back = True
            loop_target_method = "hypothesis_generation"

    # ---- Call LLM to generate response (only if user sent a message) ----
    reply_text = ""
    if incoming_user_msg and incoming_user_msg.strip():
        try:
            llm_client = QwenClient(
                base_url=settings.bailian_base_url,
                api_key=settings.bailian_api_key,
                model=settings.model_name,
            )
            result = await llm_client.chat_complete(
                messages=_build_post_report_messages(state_summary, chat_history, incoming_user_msg),
                temperature=0.7,
                max_tokens=2048,
            )
            choices = result.get("choices", [])
            if choices:
                reply_text = choices[0].get("message", {}).get("content", "")
            else:
                reply_text = "(Could not generate response due to API error)"
        except Exception as e:
            logger.error(f"[PostReportChat] LLM call failed: {e}")
            reply_text = f"Error generating response: {str(e)[:200]}"

    # ---- Build updated state ----
    updates = {
        "current_action": "post_report_chat",
        "post_report_chat_active": True,
        "_max_iterations_": max_iter,
        "iteration": iteration,
    }

    # Append the exchange to chat history
    existing_messages = list(state.get("user_chat_messages", []) or [])
    if incoming_user_msg:
        user_turn = {
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "role": "user",
            "content": incoming_user_msg,
            "intent": intent_detected,
            "created_at": _now_iso(),
        }
        assistant_turn = {
            "id": f"msg_{uuid.uuid4().hex[:8]}_reply",
            "role": "assistant",
            "content": reply_text or state_summary[:500],
            "intent": intent_detected,
            "created_at": _now_iso(),
        }
        existing_messages.extend([user_turn, assistant_turn])
        updates["user_chat_messages"] = existing_messages
    else:
        # Initial entry — no user message yet; just log greeting
        greeting_turn = {
            "id": f"msg_{uuid.uuid4().hex[:8]}_greeting",
            "role": "assistant",
            "content": f"You've completed {iteration} iterations. Your report is ready. What would you like to discuss?",
            "intent": "initiated",
            "created_at": _now_iso(),
        }
        existing_messages.append(greeting_turn)
        updates["user_chat_messages"] = existing_messages

    # Set routing signal for graph router
    if should_loop_back:
        updates["_routing_signal"] = f"loop_back_to_{loop_target_method}"
    elif intent_detected == "accept_and_end":
        updates["_routing_signal"] = "end_session"
    else:
        updates["_routing_signal"] = "continue_chatting"

    return updates


def _detect_intent(message: str | None) -> str:
    """Detect primary user intent from their message."""
    if not message:
        return "unknown"
    msg_lower = message.lower()

    challenge_patterns = ["质疑", "challenge", "wrong", "错误", "质疑你的结论",
                         "有问题", "不同意", "不对", "flawed"]
    if any(p in msg_lower for p in challenge_patterns):
        return "challenge_findings"

    reanalyze_patterns = ["重新分析", "re-analyse", "reanalyze", "换个角度",
                          "从不同角度", "another angle", "换一种方法", "change method",
                          "试一下另一种方式", "re-calculate", "use different approach"]
    if any(p in msg_lower for p in reanalyze_patterns):
        return "request_re_analyze"

    question_patterns = ["请问", "为什么", "explain", "解释一下", "说明一下",
                        "tell me about", "what is", "how does", "是什么意思"]
    if any(p in msg_lower for p in question_patterns):
        return "seek_explanation"

    continue_patterns = ["继续", "done", "ok", "结束", "可以了", "够了",
                         "accept", "accepted", "wrap up", "就这样吧"]
    if any(p in msg_lower for p in continue_patterns):
        return "accept_and_end"

    return "general_question"


# ------------------------------------------------------------------
# Prompt Templates
# ------------------------------------------------------------------


_POST_REPORT_CHAT_SYSTEM_PROMPT = """You are twinScientist's post-research scientific companion.

An autonomous AI-driven investigation has concluded and produced a formal hypothesis-and-evidence report. You now engage in open-ended scientific discussion with a human collaborator who may want to:

1. Understand findings in greater depth
2. Challenge specific hypotheses or methods
3. Request re-analysis from different angles
4. Explore edge cases or alternative interpretations
5. Accept conclusions and wrap up

Respond naturally and scientifically. Reference actual data, methods, and findings from the study. Acknowledge uncertainties honestly. Defend conclusions when warranted but concede gracefully when critique is valid. Never fabricate results or citations.

End each substantive response with a natural follow-up prompt to keep the dialogue productive.
"""


def _build_post_report_messages(state_summary: str, chat_history: list, incoming_message: str) -> list:
    """Build messages for post-report chat LLM call."""
    messages = [{"role": "system", "content": _POST_REPORT_CHAT_SYSTEM_PROMPT}]

    # Inject recent chat turns
    if isinstance(chat_history, list):
        for turn in chat_history[-20:]:
            if isinstance(turn, dict):
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "assistant":
                    messages.append({"role": "assistant", "content": content})
                elif role == "user":
                    messages.append({"role": "user", "content": content})

    # Add user's latest message
    messages.append({"role": "user", "content": incoming_message})
    return messages
