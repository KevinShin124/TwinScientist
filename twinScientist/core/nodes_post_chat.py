"""
Post-report Chat Node — 人在回路 (Human-in-the-Loop)

比赛要求：构建可交互、具备教学意义的人机协作流程。
CLI 模式：输出研究总结后自动退出。
UI 模式：通过 Gradio 界面进行自由对话。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from config.settings import settings
from core.llm_client import QwenClient, get_global_client
from core.state import AgentState

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _node_post_report_chat(state: AgentState) -> dict:
    """
    Post-report HITL node — 比赛要求"人在回路"的最终体现。

    CLI 模式 (auto_confirm=True)：输出研究总结，自动结束。
    UI 模式：等待用户交互。
    """
    auto_confirm = state.get("auto_confirm", False)
    final_report = state.get("final_report", "")
    iteration = state.get("iteration", 0)
    convergence = state.get("convergence_score", 0.0)
    evidence_chains = state.get("evidence_chains", [])
    hypotheses = state.get("hypothesis_tree", [])
    debate_history = state.get("debate_history", [])
    educational_annotations = state.get("educational_annotations", [])

    # Build research summary
    active_hyps = [h for h in hypotheses if h.get("status") not in ("pruned", "refused", "refuted_in_tournament")]
    avg_evidence = round(sum(e.get("strength", 0.5) for e in evidence_chains) / max(len(evidence_chains), 1), 3)

    summary = (
        f"研究完成。共 {iteration} 轮迭代，收敛度 {convergence:.0%}，"
        f"证据强度 {avg_evidence:.3f}，"
        f"活跃假设 {len(active_hyps)} 个，"
        f"辩论轮次 {len(debate_history)} 轮，"
        f"教学注释 {len(educational_annotations)} 条。"
    )

    greeting = {
        "id": f"msg_{uuid.uuid4().hex[:8]}_greeting",
        "role": "assistant",
        "content": summary,
        "created_at": _now_iso(),
    }

    chat_history = list(state.get("user_chat_messages", []) or [])
    chat_history.append(greeting)

    if auto_confirm:
        logger.info(f"[PostReportChat] CLI mode — auto-exit. {summary}")
        return {
            "current_action": "post_report_chat",
            "post_report_chat_active": True,
            "user_chat_messages": chat_history,
            "_routing_signal": "end_session",
            "educational_annotations": [
                {
                    "node": "post_report_chat",
                    "explanation": (
                        "人在回路（Human-in-the-Loop）是 AI Scientist 的核心设计原则。"
                        "研究报告生成后，用户可以质疑结论、要求重新分析、或探索替代解释。"
                        "这确保了 AI 是科学家的助手而非替代品，最终决策权始终在人类手中。"
                    ),
                    "timestamp": _now_iso(),
                }
            ],
        }

    # UI mode — check for incoming user message
    incoming_msg = state.get("_resume_message", None)
    if incoming_msg:
        try:
            llm = get_global_client() or QwenClient(
                base_url=settings.bailian_base_url,
                api_key=settings.bailian_api_key,
                model=settings.model_name,
            )
            result = await llm.chat_complete(
                messages=[
                    {"role": "system", "content": "你是 TwinScientist 的科研助手。用户刚完成了自主研究，请基于报告内容回答用户的问题。保持科学严谨，不确定的地方诚实说明。"},
                    {"role": "user", "content": f"研究报告摘要：{final_report[:2000]}\n\n用户问题：{incoming_msg}"},
                ],
                temperature=0.7,
                max_tokens=2048,
            )
            reply = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"[PostReportChat] LLM error: {e}")
            reply = f"(回复生成失败: {e})"

        chat_history.append({
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "role": "user",
            "content": incoming_msg,
            "created_at": _now_iso(),
        })
        chat_history.append({
            "id": f"msg_{uuid.uuid4().hex[:8]}_reply",
            "role": "assistant",
            "content": reply,
            "created_at": _now_iso(),
        })

    return {
        "current_action": "post_report_chat",
        "post_report_chat_active": True,
        "user_chat_messages": chat_history,
        "_routing_signal": "continue_chatting" if incoming_msg else "end_session",
    }