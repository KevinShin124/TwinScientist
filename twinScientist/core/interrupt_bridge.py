"""
LangGraph ↔ Gradio Bridge — 将用户消息传给中断节点的关键桥梁

Problem: LangGraph interrupts the research flow at certain nodes (human_approval,
post_report_chat), waiting for resume_value. Gradio handles events synchronously,
while LangGraph runs async. We need a bridge that:
1. Listens for interruption signals from the research pipeline
2. Shows user input prompts in the Gradio UI
3. Captures user messages and feeds them back as resume_values

Usage:
    from core.interrupt_bridge import InterruptBridge

    # In the main app flow:
    bridge = InterruptBridge()

    def run_with_interruptions(graph, config):
        result = await bridge.run_graph_with_resume(graph, config)
        return result
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


class InterruptBridge:
    """
    桥接 Gradio 交互与 LangGraph 中断机制。

    This class manages the handshake between:
    - The LangGraph pipeline (which calls interrupt() at configured nodes)
    - The Gradio UI (which displays prompts and captures user responses)

    Flow:
    1. Pipeline starts → iterates through nodes normally
    2. At interrupt node: pipeline pauses, yields state snapshot
    3. Bridge detects pause → shows prompt in UI
    4. User types message → bridge receives it
    5. Bridge calls graph.update_state(state, {"_resume_message": msg})
    6. Pipeline resumes with the new message
    """

    def __init__(self, callback_handler: Optional[Callable] = None):
        """
        Args:
            callback_handler: Optional async callback when interrupt occurs.
                             Called with (node_name, state_snapshot, session_id).
                             Can be used to update UI components.
        """
        self.callback_handler = callback_handler
        self._current_node: str | None = None
        self._session_id: str | None = None

    async def handle_interrupt(
        self,
        node_name: str,
        state_snapshot: dict,
        session_id: str,
        get_user_message: Callable[[str], Coroutine[Any, Any, str]],
    ) -> dict:
        """
        Core interrupt handler. Called when LangGraph pauses at an interrupt node.

        Args:
            node_name: Name of the interrupted node (e.g., "human_approval", "post_report_chat")
            state_snapshot: Full AgentState at time of interruption
            session_id: Unique session identifier
            get_user_message: Async callable that shows prompt in UI and returns user's typed response

        Returns:
            Dict containing the resumed state updates to feed back into LangGraph
        """
        self._current_node = node_name
        self._session_id = session_id

        logger.info(f"[InterruptBridge] Paused at node: {node_name}")

        # Build context summary based on which node was interrupted
        ctx_summary = self._build_context_summary(node_name, state_snapshot)

        # Show prompt to user and wait for their message
        if self.callback_handler:
            await self.callback_handler(node_name, state_snapshot, session_id)

        user_msg = await get_user_message(ctx_summary)

        # Store the message in state so the next node can read it
        updates = {
            "_resume_message": user_msg,
            "user_feedback": user_msg,
            "_interruption_node": node_name,
        }

        logger.info(f"[InterruptBridge] Resumed at {node_name} with user message")

        return {
            "updates": updates,
            "state_snapshot": state_snapshot,
            "user_message": user_msg,
        }

    def _build_context_summary(self, node_name: str, state: dict) -> str:
        """Build human-readable context for the user at each interrupt point."""

        summaries = {
            "human_approval": (
                "**🎛️ 人类审核断点**\n\n"
                f"**当前状态**: 评审已完成，等待你的决定。\n\n"
                "你可以：\n"
                "- 直接批准（输入 'approve'）\n"
                "- 要求修改（输入具体意见）\n"
                "- 深入讨论（在聊天面板提问）\n"
                "- 终止研究（输入 'halt'）"
            ),
            "post_report_chat": (
                "**🔬 研究报告已生成**\n\n"
                "现在你可以自由地与 AI 探讨研究发现。\n\n"
                "示例问题：\n"
                "- '为什么选择这个方法而不是另一种？'\n"
                "- '这个结论可靠吗？证据够充分吗？'\n"
                "- '换个角度重新分析一下'\n"
                "- '帮我找出可能的漏洞'"
            ),
            "literature_review": (
                "**📚 文献调研完成**\n\n"
                "系统已在学术数据库中检索到相关论文。"
                "你想深入了解哪些方面的内容？"
            ),
            "debate_then_terminate": (
                "**⚔️ 辩论完成**\n\n"
                "Pro/Con/Judge 三轮对抗已结束。"
                "你对辩论结果有何看法？是否满意当前结论？"
            ),
        }

        return summaries.get(node_name, f"**{node_name} 断点**\n请提供你的输入：")

    def current_interrupt_node(self) -> str | None:
        """Get the currently interrupted node name."""
        return self._current_node

    @property
    def is_interrupted(self) -> bool:
        """Check if currently waiting for user input."""
        return self._current_node is not None


# ====================================================================
# Convenience Functions for Direct Integration
# ====================================================================


async def simulate_gradio_input(context_prompt: str, test_mode: bool = False, mock_response: str = "") -> str:
    """
    Simulates getting user input — either from real UI (test mode off)
    or from a provided mock string (test mode on).

    In production, this would integrate with Gradio's blocking input mechanism.
    For now, we use asyncio.Event-based signaling.
    """
    if test_mode:
        return mock_response

    # In real Gradio integration, this would block until user submits
    # For testing/CLI usage, just return empty or prompt interactively
    try:
        user_input = input(f"\n[{context_prompt[:50]}...]\n>>> ").strip()
        if user_input:
            return user_input
        return "(empty)"
    except EOFError:
        return ""


def build_interrupt_handler(bridge: InterruptBridge, graph: Any) -> Callable:
    """
    Factory to create interrupt handler bound to a specific LangGraph compiled graph.

    Usage:
        bridge = InterruptBridge()
        graph = build_cognitive_graph()

        handler = build_interrupt_handler(bridge, graph)

        # Use within your streaming loop:
        async for event in graph.astream(...):
            if isinstance(event, dict) and "interrupt" in event:
                # Handle the interrupt
                pass
    """

    async def _handler(node_name: str, node_input: dict, config: dict, **kwargs) -> dict:
        """Generic interrupt handler that works with LangGraph's interrupt system."""
        return {"output": await bridge.handle_interrupt(
            node_name=node_name,
            state_snapshot=node_input,
            session_id=config.get("configurable", {}).get("session_id", "default"),
            get_user_message=lambda ctx: simulate_gradio_input(ctx),
        )}

    return _handler
