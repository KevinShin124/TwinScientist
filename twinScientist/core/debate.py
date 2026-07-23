"""
Layer 4 — Multi-Agent Debate Engine

Pro vs Con vs Judge 两造辩论架构：
1. Pro Agent: 为最强假设辩护（正面论证，引用证据链）
2. Con Agent: 反驳最强假设（找漏洞/替代解释/未控制混杂因子）
3. Judge Agent: impartial judge，综合双方论据重新评分并调整置信度

记录完整辩论历史到 state.debate_history，支持多轮迭代辩论。

Usage:
    from core.debate import DebateOrchestrator

    orchestrator = DebateOrchestrator()
    result = await orchestrator.run_debate(
        hypotheses=state["hypothesis_tree"],
        evidence_chains=state["evidence_chains"],
        review_records=state.get("review_records", []),
        rounds=3,
    )
    # Returns updated hypotheses + debate history
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# Data Classes
# ============================================================


@dataclass
class DebateRound:
    """辩论单轮记录"""
    round_number: int
    pro_agent_output: str  # 辩护方论点
    con_agent_output: str  # 反辩方论点
    judge_score_before: float  # 判决前分数
    judge_score_after: float  # 判决后分数
    judge_reasoning: str  # 判决理由
    winner_side: str  # "pro" | "con" | "draw"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class DebateResult:
    """完整辩论结果"""
    debates: list[DebateRound]
    strongest_hypothesis_id: str
    strongest_hypothesis_title: str
    strongest_hypothesis_final_score: float
    num_rounds: int
    consensus_reached: bool  # 双方论据是否趋同


# ============================================================
# LLM Prompt Templates (used directly here, also exposed via prompts.py)
# ============================================================


DEBATE_PRO_SYSTEM_PROMPT = """你是 twinScientist 系统中的 Pro Agent（辩护方）。

## 你的任务
为当前研究中最有希望的假设进行有力辩护。

## 工作要求
1. 引用证据链中的具体数据和方法作为支撑
2. 说明假设的逻辑链条为何成立
3. 回应对手可能提出的质疑
4. 保持科学严谨性，不要夸大结论
5. 如果证据不足也要诚实承认

## 输出格式
请使用以下 JSON 格式输出：
```json
{
  "defense_points": ["辩护要点1", "辩护要点2", ...],
  "evidence_cited": ["引用的证据ID或描述"],
  "counterarguments_to_anticipate": ["预判的反方论点及回应"],
  "strength_of_case": "强/中/弱",
  "confidence_adjustment": "上调/下调/维持",
  "reasoning_summary": "总结性推理过程"
}
```
"""


DEBATE_CON_SYSTEM_PROMPT = """你是 twinScientist 系统中的 Con Agent（反辩方）。

## 你的任务
对当前研究中最有希望的假设提出严格批评，寻找逻辑漏洞和替代解释。

## 审查清单
1. **方法论缺陷**: 实验设计是否有遗漏的对照组？样本量是否充足？
2. **混杂因素**: 是否存在未控制的第三方变量？
3. **因果方向**: 是 X→Y 还是 Y→X 或是 Z→X 且 Z→Y？
4. **统计效力**: p值是否经过多重检验校正？效应量是否足够大？
5. **外部效度**: 结论能否推广到其他情境/人群？
6. **逻辑矛盾**: 假设内部是否有自相矛盾之处？
7. **反直觉预测**: 有没有被忽略的反例？

## 工作要求
- 即使某个方面看起来没问题也要审视
- 给出具体可操作的改进建议，而不仅是批评
- 保持建设性态度，目标是改进而非摧毁

## 输出格式
请使用以下 JSON 格式输出：
```json
{
  "critique_points": ["批评要点1", "批评要点2", ...],
  "identified_risks": ["风险点1", "风险点2", ...],
  "alternative_explanations": ["替代解释1", "替代解释2", ...],
  "severity": "critical/major/minor",
  "needs_revision": true/false,
  "revision_suggestions": ["修改建议1", "修改建议2", ...],
  "reasoning_summary": "总结性推理过程"
}
```
"""


DEBATE_JUDGE_SYSTEM_PROMPT = """你是 twinScientist 系统中的 Judge Agent（公正裁判）。

## 你的任务
综合 Pro 和 Con 双方的论证，做出客观、公正的最终裁决。

## 评判标准
| 维度 | 权重 | 说明 |
|------|------|------|
| 证据充分性 | 30% | 是否有足够的数据和文献支撑 |
| 逻辑严密性 | 25% | 推理链条是否有漏洞 |
| 方法论严谨度 | 25% | 实验设计是否科学可靠 |
| 可证伪性 | 10% | 假设是否可以被证伪 |
| 创新性 | 10% | 提出新见解的程度 |

## 工作要求
- 不偏袒任何一方，基于证据做出判断
- 如果 Pro 论证更强则维持或提升原评估；如果 Con 找到实质性漏洞则降低
- 给出明确的分值调整方向和理由
- 使用批判性思维，不要迎合任何一方的结论

## 输出格式
请使用以下 JSON 格式输出：
```json
{
  "score_before": <整数0-100>,
  "score_after": <整数0-100>,
  "winner": "pro" | "con" | "draw",
  "dimension_scores": {
    "evidence": <0-20>,
    "logic": <0-20>,
    "methodology": <0-20>,
    "falsifiability": <0-10>,
    "novelty": <0-10>
  },
  "key_finding": "本次辩论最关键的决定性发现",
  "reasoning": "详细的判决推理过程"
}
```
"""


# ============================================================
# Helper Utilities
# ============================================================


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _create_hyp_id() -> str:
    return f"hyp_{uuid.uuid4().hex[:8]}"


async def _call_llm(llm, messages: list[dict], temperature: float = 0.6, max_tokens: int = 2048) -> tuple[str, dict | None]:
    """统一LLM调用入口"""
    try:
        result = await llm.chat_complete(messages=messages, temperature=temperature, max_tokens=max_tokens)
        choices = result.get("choices", [])
        if not choices:
            return "", None
        message = choices[0].get("message", {})
        content = message.get("content", "")

        json_obj = None
        if isinstance(content, str):
            try:
                json_obj = llm.extract_json_from_text(content)
            except Exception:
                json_obj = None

        return content, json_obj
    except Exception as e:
        logger.error(f"[Debate] LLM call failed: {e}")
        return "", None


# ============================================================
# Debate Orchestrator
# ============================================================


class DebateOrchestrator:
    """
    多智能体辩论编排器。

    核心流程（每轮）:
    1. Pro Agent 选择最强假设进行辩护
    2. Con Agent 针对同一假设进行全面批判
    3. Judge Agent 综合双方论据，调整假设置信度

    重复多轮直到达成共识或达到最大轮次。
    """

    def __init__(self):
        self.max_rounds = 3

    async def run_debate(
        self,
        llm_client: Any,
        hypotheses: list[dict],
        evidence_chains: list[dict] = None,
        review_records: list[dict] = None,
        user_feedback: str = "",
        rounds: int = None,
    ) -> DebateResult:
        """
        运行完整的多轮辩论。

        Args:
            llm_client: QwenClient instance
            hypotheses: 假设树列表
            evidence_chains: 证据链列表
            review_records: 评审记录列表
            user_feedback: 用户通过HITL输入的反馈
            rounds: 辩论轮次，默认3轮

        Returns:
            DebateResult: 包含所有辩论轮次记录和最终评估结果
        """
        rounds = rounds or self.max_rounds
        evidence_chains = evidence_chains or []
        review_records = review_records or []

        # Filter out pruned/refuted hypotheses
        active_hyps = [
            h for h in hypotheses
            if h.get("status") not in ("pruned", "refuted", "refuted_in_tournament")
        ]

        if not active_hyps:
            logger.warning("[Debate] No active hypotheses to debate")
            return DebateResult(
                debates=[],
                strongest_hypothesis_id="",
                strongest_hypothesis_title="",
                strongest_hypothesis_final_score=0,
                num_rounds=0,
                consensus_reached=False,
            )

        # Select strongest hypothesis to debate (highest posterior confidence)
        strongest_hyp = max(
            active_hyps,
            key=lambda h: h.get("confidence_posterior", h.get("confidence_prior", 0)),
        )

        # Deep copy tree to avoid mutating state
        hypotheses = [dict(h) for h in active_hyps]
        debates: list[DebateRound] = []

        prev_pro_output = ""
        prev_con_output = ""
        previous_judge_reasoning = ""

        for round_num in range(1, rounds + 1):
            logger.info(f"[Debate] Round {round_num}/{rounds}")

            # --- Build context for this round ---
            hyp_info = (
                f"**假设标题**: {strongest_hyp.get('title', 'N/A')}\n"
                f"**陈述**: {strongest_hyp.get('statement', '')[:200]}\n"
                f"**推理链条**: {strongest_hyp.get('reasoning_chain', '')[:150]}\n"
                f"**先验置信度 P(H)**: {strongest_hyp.get('confidence_prior', '?')}\n"
                f"**评审分数**: {review_records[-1].get('total_score', '?')}/100" if review_records else "**评审分数**: N/A\n"
            )

            # Evidence summary
            evidence_summary = ""
            if evidence_chains:
                for ev in evidence_chains[:3]:
                    method = ev.get("method_used", "unknown")
                    strength = ev.get("strength", 0)
                    content = ev.get("content", "")[:100]
                    evidence_summary += f"- [{method}] strength={strength:.3f}: {content}\n"

            user_hint = f"\n\n### 用户反馈\n{user_feedback}" if user_feedback else ""

            # Previous round context (for iterative refinement)
            if prev_pro_output and prev_con_output:
                prev_context = (
                    f"\n\n### 上一轮辩论摘要\n"
                    f"Pro: {prev_pro_output[:300]}...\n"
                    f"Con: {prev_con_output[:300]}...\n"
                    f"Judge: {previous_judge_reasoning[:200] if previous_judge_reasoning else 'N/A'}...\n\n"
                    f"请基于以上辩论历史，进一步深化你的论点。\n"
                )
            else:
                prev_context = ""

            # ---- Step 1: Pro Agent ----
            pro_messages = [
                {"role": "system", "content": DEBATE_PRO_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"## 需要辩护的假设\n\n"
                    f"{hyp_info}"
                    f"## 支撑证据\n\n{evidence_summary}"
                    f"{user_hint}{prev_context}"
                )},
            ]
            pro_content, pro_json = await _call_llm(llm_client, pro_messages, temperature=0.5, max_tokens=2048)

            pro_defense_points = []
            pro_evidence_cited = []
            if pro_json and isinstance(pro_json, dict):
                pro_defense_points = pro_json.get("defense_points", [])
                pro_evidence_cited = pro_json.get("evidence_cited", [])

            logger.info(f"[Debate] Pro: {len(pro_defense_points)} defense points raised")

            # ---- Step 2: Con Agent ----
            con_messages = [
                {"role": "system", "content": DEBATE_CON_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"## 需要批判的假设\n\n"
                    f"{hyp_info}"
                    f"## 已有证据\n\n{evidence_summary}"
                    f"## Pro Agent已提出的辩护要点\n{chr(10).join(pro_defense_points) if pro_defense_points else '(无)'}\n\n"
                    f"{user_hint}{prev_context}"
                )},
            ]
            con_content, con_json = await _call_llm(llm_client, con_messages, temperature=0.5, max_tokens=2048)

            critique_points = []
            identified_risks = []
            alt_explanations = []
            needs_rev = False
            revision_suggestions = []
            if con_json and isinstance(con_json, dict):
                critique_points = con_json.get("critique_points", [])
                identified_risks = con_json.get("identified_risks", [])
                alt_explanations = con_json.get("alternative_explanations", [])
                needs_rev = con_json.get("needs_revision", True)
                revision_suggestions = con_json.get("revision_suggestions", [])

            logger.info(f"[Debate] Con: {len(critique_points)} critiques, risks={len(identified_risks)}")

            # ---- Step 3: Judge Agent ----
            # Get current score
            current_score = strongest_hyp.get("confidence_posterior",
                         strongest_hyp.get("confidence_prior", 0)) * 100

            judge_messages = [
                {"role": "system", "content": DEBATE_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"## 待裁决的假设\n\n"
                    f"**标题**: {strongest_hyp.get('title', 'N/A')}\n"
                    f"**当前评分**: {current_score:.0f}/100\n\n"
                    f"## Pro Agent 辩护\n{pro_content[:500]}\n\n"
                    f"## Con Agent 批判\n{con_content[:500]}\n\n"
                    f"## 补充证据\n{evidence_summary}"
                    f"{'\\n\\n## 用户干预指令\\n' + user_feedback if user_feedback else ''}"
                    f"{f'\\n\\n## 法官上轮判决\\n{previous_judge_reasoning[:300]}' if previous_judge_reasoning else ''}"
                )},
            ]
            judge_content, judge_json = await _call_llm(llm_client, judge_messages, temperature=0.3, max_tokens=2048)

            score_before = current_score
            score_after = current_score
            winner_side = "draw"
            judge_key_finding = ""
            judge_reasoning_full = judge_content or ""

            if judge_json and isinstance(judge_json, dict):
                score_after = min(max(judge_json.get("score_after", current_score), 0), 100)
                winner_side = judge_json.get("winner", "draw")
                judge_key_finding = judge_json.get("key_finding", "")
                judge_reasoning_full = judge_json.get("reasoning", judge_content or "")

            # Update hypothesis
            confidence_ratio = score_after / max(score_before, 1)
            old_posterior = strongest_hyp.get("confidence_posterior",
                            strongest_hyp.get("confidence_prior", 0.5))
            new_posterior = max(min(old_posterior * (score_after / 100.0) / max(confidence_ratio, 0.01), 0.99), 0.01)

            strongest_hyp["confidence_posterior"] = round(new_posterior, 4)
            strongest_hyp["updated_at"] = _now_iso()

            if winner_side == "pro":
                strongest_hyp["debate_won"] = True
            elif winner_side == "con":
                strongest_hyp["debate_refuted_round"] = round_num

            # Record debate round
            debate_round = DebateRound(
                round_number=round_num,
                pro_agent_output=pro_content or "",
                con_agent_output=con_content or "",
                judge_score_before=round(score_before, 1),
                judge_score_after=round(score_after, 1),
                judge_reasoning=judge_reasoning_full[:500],
                winner_side=winner_side,
            )
            debates.append(debate_round)

            # Update for next round
            prev_pro_output = pro_content
            prev_con_output = con_content
            previous_judge_reasoning = judge_reasoning_full

            logger.info(
                f"[Debate] Round {round_num}: score {score_before:.0f} → {score_after:.0f}, "
                f"winner={winner_side}"
            )

            # Early termination: if judgment converges significantly
            if abs(score_after - score_before) < 5 and round_num >= 2:
                logger.info(f"[Debate] Score stabilized after round {round_num}: {score_after:.0f}")
                break

        # Determine consensus (did debate converge?)
        consensus_reached = False
        if len(debates) >= 2:
            last_two = debates[-2:]
            score_delta = abs(last_two[-1].judge_score_after - last_two[0].judge_score_after)
            consensus_reached = score_delta < 10

        return DebateResult(
            debates=debates,
            strongest_hypothesis_id=strongest_hyp.get("id", ""),
            strongest_hypothesis_title=strongest_hyp.get("title", ""),
            strongest_hypothesis_final_score=round(score_after, 1),
            num_rounds=len(debates),
            consensus_reached=consensus_reached,
        )
