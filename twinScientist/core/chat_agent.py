"""
Layer 5 — Chat Agent for Real-Time Human-Agent Interaction

Supports continuous multi-turn dialogue where users can:
- Ask questions about current research state (hypotheses, evidence, experiments)
- Challenge specific hypotheses with counterarguments
- Suggest modifications or new directions
- Participate in ongoing debates as a "human judge"
- Inject domain knowledge during any iteration step

All user messages are stored in state.user_chat_messages and injected into
the next LLM call as context, ensuring continuity across the research loop.

Usage:
    from core.chat_agent import ChatAgent

    agent = ChatAgent(llm_client=client)
    response = await agent.reply(
        user_message="This hypothesis seems weak. What about X?",
        state=current_state,
        action="debate",
    )
    # Returns dict with reply text, updated suggestions, debate annotations
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _create_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:8]}"


@dataclass
class ChatTurn:
    """Single conversation turn (user message + agent response)"""
    id: str
    role: str                  # "user" | "agent"
    content: str
    metadata: dict = None
    created_at: str = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = _now_iso()


class ChatAgent:
    """
    面向科研过程的多轮对话 Agent。

    核心能力：
    - 基于当前研究状态回答用户问题
    - 将用户反馈直接注入后续 LLM prompt 中
    - 在辩论/反思环节作为第三方评审参与
    - 支持对特定假设的深入追问

    对话流程：
    1. 收集用户消息 + 最近历史上下文
    2. 组装系统 prompt（含当前假设树、证据链等摘要）
    3. LLM 生成回复
    4. 存储对话记录到 state.user_chat_messages
    5. 根据用户意图更新 state（如添加 user_guidance）
    """

    SYSTEM_PROMPT = """You are a scientific reasoning assistant engaged in an autonomous research loop.

The user is interacting with you through a research dashboard. They may:
- Ask about the current state of the research
- Challenge or critique specific hypotheses
- Provide domain expertise or point out overlooked factors
- Ask you to reconsider a decision or change direction
- Participate as a judge in the Pro vs Con debate

## Your responsibilities
1. Be honest and scientifically rigorous — admit uncertainty when appropriate
2. Reference actual evidence and data available in the research state
3. When challenged on a hypothesis, explain your reasoning transparently
4. If the user provides new valid insight, incorporate it and show how it changes conclusions
5. Use clear language; avoid jargon unless explaining it

## How this system works
The twinScientist system automatically:
1. Searches literature databases (Crossref, arXiv, Semantic Scholar)
2. Extracts verified facts with cross-checked citations
3. Generates hypotheses via inductive/deductive/abductive reasoning
4. Runs tournament elimination to select candidates
5. Designs and executes experiments with causal inference analysis
6. Subjects results to five-dimension peer review
7. Engages in Pro vs Con debates to stress-test conclusions
8. Reflects on failures and iterates to improve

Your input matters at every stage — you are not just a passive observer but an active collaborator."""


    async def reply(
        self,
        llm_client: Any,
        user_message: str,
        state: dict,
        action: str = "",
    ) -> dict:
        """
        Process one user message and generate a response.

        Args:
            llm_client: QwenClient instance for LLM calls
            user_message: User's text input
            state: Current AgentState (dict copy)
            action: Name of the current cognitive node being executed

        Returns:
            Dict containing:
                - reply: The generated response string
                - updates: State fields to update based on user input
                - chat_turn_id: ID of this conversation turn
                - sentiment: Detected intent ("question", "challenge", "suggestion", "approval")
        """
        # ---- Build system prompt with current state summary ----
        state_summary = self._build_state_summary(state, user_message)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"## Research Context\n{state_summary}\n\n"
                    f"## Current Operation\n**Node**: {action or 'not specified'}\n\n"
                    f"## Your Message\n{user_message}"
                ),
            },
        ]

        # Add recent chat history for context (last 10 turns)
        chat_history = state.get("user_chat_messages", [])
        if isinstance(chat_history, list) and len(chat_history) > 0:
            recent_turns = chat_history[-10:]  # Keep last 10 turns
            for turn in recent_turns:
                if isinstance(turn, dict):
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                    if role == "assistant":
                        messages.append({"role": "assistant", "content": content})
                    else:
                        messages.append({"role": "user", "content": content})

        # ---- Call LLM ----
        try:
            result = await llm_client.chat_complete(
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
            choices = result.get("choices", [])
            if not choices:
                reply_text = "I'm sorry, I couldn't generate a response right now. Please try again."
                detected_intent = "unknown"
            else:
                reply_text = choices[0].get("message", {}).get("content", "")
                if not reply_text:
                    reply_text = "I'm sorry, I couldn't generate a response right now. Please try again."
                detected_intent = self._detect_intent(user_message)

        except Exception as e:
            logger.error(f"[ChatAgent] LLM call failed: {e}")
            reply_text = f"I encountered an error processing your request. Details: {str(e)[:200]}"
            detected_intent = self._detect_intent(user_message)

        # ---- Create chat turn record ----
        turn_id = _create_msg_id()
        chat_turn = {
            "id": turn_id,
            "role": "user",
            "content": user_message,
            "created_at": _now_iso(),
        }

        # Also record agent response
        agent_turn = {
            "id": f"{turn_id}_reply",
            "role": "assistant",
            "content": reply_text,
            "detected_intent": detected_intent,
            "action_context": action,
            "created_at": _now_iso(),
        }

        # ---- Determine state updates based on user intent ----
        updates = self._extract_updates(user_message, detected_intent, reply_text, state)

        return {
            "reply": reply_text,
            "updates": updates,
            "chat_turn_id": turn_id,
            "sentiment": detected_intent,
            "_user_chat_messages_append": [chat_turn, agent_turn],
        }

    def _build_state_summary(self, state: dict, user_message: str) -> str:
        """Build a concise summary of current research state for context."""
        lines = []

        # Query & Domain
        query = state.get("query", "Unknown question")
        domain = state.get("domain", "General science")
        lines.append(f"**Question**: {query}")
        lines.append(f"**Domain**: {domain}")
        lines.append(f"**Iteration**: {state.get('iteration', 0)}/200")

        # Hypothesis pool
        hyp_tree = state.get("hypothesis_tree", [])
        active_hyps = [
            h for h in hyp_tree
            if h.get("status") not in ("pruned", "refused", "refuted_in_tournament")
        ]
        lines.append(f"\n**Active Hypotheses**: {len(active_hyps)}")
        for i, h in enumerate(active_hyps[:5]):
            title = h.get("title", "?")[:50]
            stmt = h.get("statement", "")[:80]
            prior = h.get("confidence_prior", "?")
            posterior = h.get("confidence_posterior", "?")
            status = h.get("status", "?")
            lines.append(
                f"  {i+1}. [{status}] \"{title}\" "
                f"| P(H|D)={posterior} | {stmt}"
            )
        if len(active_hyps) > 5:
            lines.append(f"  ...and {len(active_hyps)-5} more")

        # Evidence chains
        evidence_chains = state.get("evidence_chains", [])
        if evidence_chains:
            avg_strength = sum(
                e.get("strength", 0.5) for e in evidence_chains
            ) / max(len(evidence_chains), 1)
            methods = set(e.get("method_used", "unknown") for e in evidence_chains)
            lines.append(f"\n**Evidence Chains**: {len(evidence_chains)} "
                         f"(avg strength={avg_strength:.2f}, "
                         f"methods={', '.join(sorted(methods))})")

        # Experiments
        experiments = state.get("experiment_records", [])
        analyzed = len([e for e in experiments if e.get("results", {}).get("analysis_complete")])
        lines.append(f"\n**Experiments**: {analyzed}/{len(experiments)} analyzed")

        # Reviews
        reviews = state.get("review_records", [])
        if reviews:
            latest = reviews[-1].get("total_score", "?")
            lines.append(f"\n**Latest Review Score**: {latest}/100")

        # Debate history (if any)
        debate_history = state.get("debate_history", [])
        if debate_history and isinstance(debate_history, list):
            n_rounds = len(debate_history)
            latest_debate = debate_history[-1] if debate_history else {}
            score_before = latest_debate.get("judge_score_before", "?")
            score_after = latest_debate.get("judge_score_after", "?")
            winner = latest_debate.get("winner_side", "?")
            lines.append(f"\n**Debate History**: {n_rounds} rounds completed "
                         f"(latest: {score_before}→{score_after}, winner={winner})")

        # Latest user feedback
        user_feedback = state.get("user_feedback", "")
        if user_feedback:
            lines.append(f"\n**Recent User Feedback**: {user_feedback[:200]}")

        return "\n".join(lines)

    @staticmethod
    def _detect_intent(message: str) -> str:
        """Detect the primary intent of a user message."""
        msg_lower = message.lower()

        # Check for challenge/critique patterns
        challenge_keywords = ["weak", "wrong", "flaw", "bias", "confound", "invalid",
                             "反驳", "质疑", "漏洞", "不足", "偏见"]
        if any(kw in msg_lower or kw in message for kw in challenge_keywords):
            return "challenge"

        # Check for suggestion/new direction
        suggestion_keywords = ["suggest", "try", "consider", "maybe", "how about",
                              "建议", "尝试", "考虑", "方向", "新假设"]
        if any(kw in msg_lower for kw in suggestion_keywords):
            return "suggestion"

        # Check for question patterns
        question_patterns = ["what is", "why does", "explain", "tell me about",
                           "请问", "为什么", "解释", "说明"]
        if any(p in msg_lower for p in question_patterns):
            return "question"

        # Check for approval
        approve_keywords = ["approve", "ok", "yes", "go ahead", "同意", "通过", "好的"]
        if any(kw in msg_lower for kw in approve_keywords):
            return "approval"

        return "general"

    def _extract_updates(self, message: str, intent: str, reply: str,
                         state: dict) -> dict:
        """Extract state updates based on user message intent."""
        updates = {}

        if intent == "approval":
            updates["pending_approval"] = False
            updates["user_feedback"] = message

        elif intent == "challenge":
            # Add to anomaly graph as potential issue
            updates["_add_anomaly_entry"] = {
                "type": "contradiction",
                "description": f"User challenge: {message[:200]}",
                "severity": "medium",
            }
            updates["user_feedback"] = message

        elif intent == "suggestion":
            # Store as guidance for next iteration
            updates["user_guidance"] = message
            updates["user_feedback"] = message

        elif intent == "question":
            # No structural change needed, just record in chat history
            pass

        return updates

    @classmethod
    def process_batch_chat(cls, messages: list[tuple[str, str]], state: dict) -> list[dict]:
        """
        Process a batch of pre-recorded chat messages (for restoring history).

        Args:
            messages: List of (user_message, agent_response) tuples
            state: Current state dict to append to

        Returns:
            Updated state with appended chat records
        """
        cls_instance = cls(None)  # dummy init; won't be called
        updated = dict(state)
        existing_messages = state.get("user_chat_messages", [])

        if not isinstance(existing_messages, list):
            existing_messages = []

        for user_msg, agent_resp in messages:
            uid = _create_msg_id()
            existing_messages.extend([
                {"id": uid, "role": "user", "content": user_msg, "created_at": _now_iso()},
                {"id": f"{uid}_reply", "role": "assistant", "content": agent_resp,
                 "created_at": _now_iso()},
            ])

        updated["user_chat_messages"] = existing_messages
        return updated
