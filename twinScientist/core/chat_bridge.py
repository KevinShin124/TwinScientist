"""
LangGraph ↔ Gradio Chat Bridge

This module provides the missing bridge between:
- Gradio's synchronous chat interface (where users type messages)
- LangGraph's async interrupt mechanism (which pauses at configured nodes and waits for resume values)

How it works:
1. Research pipeline runs → reaches interrupt node (e.g., post_report_chat)
2. Pipeline calls interrupt() → returns state snapshot to UI
3. UI shows context summary + prompt for user input
4. User types message in chat panel → send button clicked
5. Bridge captures message → injects as _resume_message into state
6. Pipeline resumes with the new message
7. LLM generates response → updates chat history → routes back or ends
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


class ChatBridge:
    """
    Bridges Gradio's chat input with LangGraph's interrupt/resume mechanism.

    Usage pattern:
        # 1. Initialize
        bridge = ChatBridge()

        # 2. In your research runner:
        async def run_with_chat(graph, initial_state, session_id=None):
            thread = {"configurable": {}}

            # Start graph execution
            async for event in graph.astream(initial_state, stream_mode="updates"):
                if isinstance(event, dict):
                    node_name = list(event.keys())[0]

                    # Check if this node requires interruption
                    if node_name in ("human_approval", "post_report_chat"):
                        # Pause and wait for user message
                        result = await bridge.handle_interrupt(
                            node_name=node_name,
                            state_snapshot=event[node_name],
                            session_id=session_id or "default",
                        )

                        # Inject user message back into state
                        updates = {
                            "_resume_message": result["user_message"],
                            **result.get("state_updates", {}),
                        }

                        # Update checkpoint so next iteration sees the message
                        thread["configurable"]["checkpoint_ns"] = f"node:{node_name}"
                        graph.update_state(thread, updates)

                yield event

    Key points:
    - LangGraph interrupts BEFORE running the node (for interrupt_before)
    - The callback receives (node_name, node_input, config, kwargs)
    - We use a shared state dict to track pending messages per session/node
    - When user submits via Gradio, we set the pending message
    - Next iteration reads _resume_message from state
    """

    def __init__(self, default_prompt: str = "请输入你的消息:",
                 max_wait_seconds: int = 60):
        self.default_prompt = default_prompt
        self.max_wait_seconds = max_wait_seconds

        # Per-session message queue: {session_id: {"pending_message": str, "received": bool}}
        self._pending_messages: dict[str, dict[str, Any]] = {}

    def _get_pending(self, session_id: str) -> dict:
        """Get or create pending message record for a session."""
        if session_id not in self._pending_messages:
            self._pending_messages[session_id] = {
                "pending_message": "",
                "received": False,
                "timestamp": None,
            }
        return self._pending_messages[session_id]

    def clear_pending(self, session_id: str):
        """Clear pending message after processing."""
        if session_id in self._pending_messages:
            del self._pending_messages[session_id]

    async def handle_interrupt(
        self,
        node_name: str,
        state_snapshot: dict,
        session_id: str = "default",
    ) -> dict:
        """
        Handle an interrupt at a specific node.

        This function should be called when LangGraph reaches an interrupt point.
        It builds the context summary, notifies the UI layer, and waits for
        the user's message.

        Returns:
            Dict containing the processed user message and any state updates needed.
        """
        logger.info(f"[ChatBridge] Interrupt at node: {node_name}")

        # Build context summary based on node type
        context_summary = self._build_context_summary(node_name, state_snapshot)

        # Get existing pending message
        pending = self._get_pending(session_id)

        # If no message yet, wait for user to submit (this would integrate with Gradio)
        # For now, we'll use a polling/callback approach
        # In production, you'd block here until Gradio sends the message
        user_message = pending.get("pending_message", "")

        if not user_message:
            # No message available — return initial greeting prompt instead of blocking
            # The UI will show the context summary and keep listening
            logger.info(f"[ChatBridge] No pending message for {session_id}, showing prompt")
            return {
                "prompt": context_summary,
                "wait_for_user": True,
                "state_snapshot": state_snapshot,
            }

        # Process the received message
        pending["received"] = False  # Reset for next time
        logger.info(f"[ChatBridge] Received message: {user_message[:100]}...")

        return {
            "prompt": context_summary,
            "user_message": user_message,
            "wait_for_user": False,
            "state_snapshot": state_snapshot,
        }

    def inject_user_message(self, session_id: str, message: str):
        """
        Called by UI layer when user submits a message.
        This makes the message available to the next interrupt handler call.
        """
        pending = self._get_pending(session_id)
        pending["pending_message"] = message
        pending["received"] = True
        pending["timestamp"] = "now"  # Placeholder for real timestamp
        logger.info(f"[ChatBridge] Injected message for {session_id}: {message[:80]}...")

    def _build_context_summary(self, node_name: str, state: dict) -> str:
        """Build human-readable context for each interrupt point."""
        summaries = {
            "human_approval": (
                "**🎛️ 人类审核断点**\n\n"
                f"**当前状态**: 研究进行中/已完成，等待你的决定。\n\n"
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


# ====================================================================
# Integration Helper Functions
# ====================================================================


async def run_graph_with_chat_bridge(
    graph: Any,
    initial_state: dict,
    chat_bridge: ChatBridge,
    session_id: str = "default",
    recursion_limit: int = 500,
) -> dict:
    """
    Run a LangGraph graph with chat integration.

    This is the main entry point for starting a research session with
    interactive chat capabilities at interrupt points.
    """
    thread = {"configurable": {"recursion_limit": recursion_limit}}
    final_state = None

    async for event in graph.astream(initial_state, stream_mode="updates", config=thread):
        if isinstance(event, dict):
            # Extract node name and state
            node_name = list(event.keys())[0]
            node_input = event[node_name]

            # Check if we need to process an interrupt
            if node_name in ("human_approval", "post_report_chat"):
                result = await chat_bridge.handle_interrupt(
                    node_name=node_name,
                    state_snapshot=node_input,
                    session_id=session_id,
                )

                if result.get("wait_for_user"):
                    # Wait for user message (in production, this would block on Gradio input)
                    # For now, just log and continue
                    logger.info(f"[ChatBridge] Waiting for user at {node_name}")
                else:
                    # Inject user message into next iteration
                    if result.get("user_message"):
                        updates = {
                            "_resume_message": result["user_message"],
                        }
                        graph.update_state(thread, updates)
                        logger.info(f"[ChatBridge] Updated state for {node_name}")

            # Store latest state
            final_state = node_input

    return final_state or initial_state
