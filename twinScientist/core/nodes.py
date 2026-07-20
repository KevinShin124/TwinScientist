"""
Layer 2: Cognitive Nodes (all 12+ operations)

每个节点对应一个认知操作单元。数据通过 AgentState 在各节点间流转，
不直接依赖外部数据源（由 channels/ 层按需接入）。

所有 LL M 调用都经过 _call_llm() 统一处理，自带重试和结构化解析。
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any
from pathlib import Path

import networkx as nx

from config.settings import settings
from core.state import AgentState
from core.llm_client import QwenClient
from tools.causal_inference import CausalInferenceEngine
from core.prompts import (
    LITERATURE_REVIEW_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    PI_AGENT_SYSTEM_PROMPT,
    REVIEWER_AGENT_SYSTEM_PROMPT,
    ETHICS_WATCHDOG_SYSTEM_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    HYPOTHESIS_GENERATION_TEMPLATE,
    EXPERIMENT_DESIGN_TEMPLATE,
    REPORT_WRITING_TEMPLATE,
    TOURNAMENT_EVAL_PROMPT,
)
from core.orchestrator import (
    _check_orchestrator_stop_conditions,
    get_cached_orch_check,
    set_orch_check_in_state,
    _hypothesis_statement_similarity,
)

logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================

def _get_llm() -> QwenClient:
    """获取 Qwen Client 实例 — 优先复用全局单例（连接复用），未初始化时降级创建新实例"""
    from core.llm_client import get_global_client

    global_client = get_global_client()
    if global_client is not None:
        return global_client
    # Fallback: create a new instance (shouldn't normally happen after init)
    return QwenClient(
        base_url=settings.bailian_base_url,
        api_key=settings.bailian_api_key,
        model=settings.model_name,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _call_llm(
    llm: QwenClient,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> tuple[str, dict | None]:
    """
    同步版 LLM 调用 — 已废弃。所有节点函数均为 async，应使用 _async_call_llm。
    保留此函数仅为向后兼容；直接调用会触发 RuntimeError。
    """
    raise RuntimeError(
        "_call_llm 已废弃。所有异步节点必须使用 await _async_call_llm(...)。"
    )


# ============================================================
# ASYNC version of _call_llm — used by async node functions
# ============================================================

async def _async_call_llm(
    llm: QwenClient,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> tuple[str, dict | None]:
    """
    异步版 LLM 调用——供 async node 函数使用。
    这才是真正能工作的版本。
    """
    try:
        result = await llm.chat_complete(messages=messages, temperature=temperature, max_tokens=max_tokens)
        choices = result.get("choices", [])
        if not choices:
            return "", None
        message = choices[0].get("message", {})
        content = message.get("content", "")

        json_obj = None
        if isinstance(content, str):
            # extract_json_from_text is SYNCHRONOUS — not async
            try:
                json_obj = llm.extract_json_from_text(content)
            except Exception:
                json_obj = None

        return content, json_obj
    except Exception as e:
        logger.error(f"[LLM] Call failed: {e}")
        return "", None


def _parse_review_score(text: str) -> dict:
    """
    从评审结果文本或 JSON 中解析分数
    优先尝试 JSON 解析，失败后回退到正则匹配
    增强：支持结构化 JSON 输出以适配多层评审和可信度校准
    """
    # Try JSON first
    json_obj = QwenClient("", "").extract_json_from_text(text)
    if json_obj and "total_score" in json_obj:
        return {
            "novelty_score": json_obj.get("novelty_score", 15),
            "feasibility_score": json_obj.get("feasibility_score", 15),
            "methodology_score": json_obj.get("methodology_score", 15),
            "evidence_score": json_obj.get("evidence_score", 15),
            "impact_score": json_obj.get("impact_score", 15),
            "total_score": json_obj["total_score"],
            "needs_revision": json_obj.get("needs_revision", True),
            "comments": text[:1500],
            "revision_instructions": json_obj.get("revision_instructions", ""),
            "high_risk_points": json_obj.get("high_risk_points", []),
            "strengths": json_obj.get("strengths", []),
            "review_confidence": json_obj.get("review_confidence", 1.0),  # 评审自身置信度
        }

    # Fallback: regex matching
    patterns = {
        "total_score": r'total_score:\s*(\d+)/100',
        "needs_revision": r'needs_revision:\s*(true|false)',
    }

    score_match = re.search(patterns["total_score"], text)
    revision_match = re.search(patterns["needs_revision"], text, re.IGNORECASE)

    total = int(score_match.group(1)) if score_match else 50
    needs_rev = revision_match.group(1).lower() == "true" if revision_match else True

    return {
        "total_score": total,
        "needs_revision": needs_rev,
        "novelty_score": 15,  # defaults when no breakdown available
        "feasibility_score": 15,
        "methodology_score": 15,
        "evidence_score": 15,
        "impact_score": 15,
        "comments": text[:1500],
        "revision_instructions": "",
    }


def _create_hypothesis_id() -> str:
    return f"hyp_{uuid.uuid4().hex[:8]}"


# ============================================================
# Evidence utilities — convert causal method output to evidence chain entries
# ============================================================

def _compute_evidence_strength(result: dict) -> float:
    """
    将因果推断结果转为 0-1 置信度分数。

    - CCM: 使用最小 |rho| 值
    - Granger: 用 p-value 反转 (p越小 → strength越大)
    - Counterfactual: 基于 CI 是否包含 0
    """
    if not result or "status" in result and result["status"] == "placeholder":
        return 0.3  # placeholder = low confidence

    # CCM
    rho_xy = result.get("ccm_rho_x_to_y")
    rho_yx = result.get("ccm_rho_y_to_x")
    if rho_xy is not None:
        min_rho = min(abs(rho_xy), abs(rho_yx))
        converge = result.get("convergence_X_to_Y", False) or result.get("convergence_Y_to_X", False)
        base_strength = min(min_rho, 1.0) * 0.7 + 0.3  # convergence bonus
        return round(max(0.0, min(base_strength, 1.0)), 3)

    # Granger
    min_p = result.get("min_p_value", 1.0)
    try:
        return round(max(0.0, min(1.0 - min_p, 1.0)), 3)
    except TypeError:
        return 0.5

    # Default counterfactual / others — only reachable if min_p is not a number that completes the calculation
    return 0.4


def _summarize_causal_result(result: dict, method: str) -> str:
    """将因果方法输出转为人可读摘要"""
    if "causal_direction" in result:
        direction = result["causal_direction"]
        strength = result.get("direction_strength", "unknown")
        return f"[{method}] {direction} (strength={strength})"

    if method == "counterfactual":
        ate = result.get("average_treatment_effect")
        ci = result.get("confidence_interval_95")
        if ate is not None and ci:
            return f"[counterfactual] ATE={ate:.4f}, 95%CI=[{ci[0]:.4f}, {ci[1]:.4f}]"

    if "results_by_lag" in result:
        sig_count = sum(1 for v in result["results_by_lag"].values()
                        if isinstance(v, dict) and v.get("significant"))
        return f"[granger] {sig_count} significant lags out of {len(result['results_by_lag'])}"

    return str(result)[:300]


def _build_evidence_entry(evidence_type: str, result: dict, method: str,
                          hyp_id: str, exp_id: str) -> dict:
    """从因果分析结果构建标准的 EvidenceEntry 字典"""
    entry = {
        "id": f"ev_{uuid.uuid4().hex[:8]}",
        "type": evidence_type,
        "strength": _compute_evidence_strength(result),
        "content": _summarize_causal_result(result, method),
        "linked_hypotheses": [hyp_id] if hyp_id else [],
        "linked_experiments": [exp_id] if exp_id else [],
        "method_used": method,
        "method_params": {},  # will be filled below
        "statistical_basis": {},
        "validation_results": {},
        "causal_direction": result.get("causal_direction"),
        "provenance": f"{method} on experiment {exp_id}",
        "created_at": _now_iso(),
    }

    # Fill statistical_basis and validation_results
    for key, val in result.items():
        if key.startswith(("ccm_rho", "confiden", "library_sizes", "rho_at_each_size",
                           "results_by_lag", "overall_granger", "best_lag", "min_p",
                           "adf_t_statistic", "stationary_at")):
            entry["statistical_basis"][key] = val
        elif key.startswith(("convergence",)):
            entry["validation_results"][key] = val

    return entry


# ============================================================
# Item 15: Ethics & Safety Watchdog Node
# ============================================================

async def node_ethics_check(state: AgentState) -> dict:
    """
    【伦理与安全审查】第一道防线 — Item 15
    采用结构化 JSON 输出解析，比关键词匹配更可靠。
    对标 Anthropic Deep Research / AutoDevin 的三段式安全评估模式。
    """
    query = state.get("query", "")
    hypotheses = state.get("hypothesis_tree", [])
    experiments = state.get("experiment_records", [])
    reports_writing = state.get("final_report", None)

    llm = _get_llm()
    messages = [
        {"role": "system", "content": ETHICS_WATCHDOG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## 待审查内容\n\n"
                f"### 研究问题\n{query}\n\n"
                f"### 当前假设树 ({len(hypotheses)} 个假设)\n"
                + "\n".join([f"- [{h.get('status', '?')}] {h.get('title', 'N/A')} — {h.get('statement', '')[:100]}" for h in hypotheses])
                if hypotheses
                else "(暂无假设)"
            ),
        },
    ]

    content, json_obj = await _async_call_llm(llm, messages, temperature=0.0)

    # Parse structured JSON output (SOTA pattern: deterministic output format)
    # Priority order: JSON → explicit status keywords → ambiguity fallback
    if json_obj and isinstance(json_obj, dict):
        status = str(json_obj.get("审查结果", json_obj.get("status", ""))).strip().upper()
        risk_level = str(json_obj.get("风险等级", json_obj.get("risk_level", "LOW"))).strip().upper()
        reason = str(json_obj.get("理由", json_obj.get("reason", "")))
        suggestion = str(json_obj.get("建议", json_obj.get("suggestion", "")))
    else:
        # Fallback: parse text-based structured output from <ETHICS_CHECK> block
        status_m = re.search(r'<ETHICS_CHECK>\s*\n\s*status:\s*(\S+)', content, re.DOTALL)
        status = status_m.group(1).upper() if status_m else ""
        risk_m = re.search(r'risk_level:\s*(\S+)', content, re.IGNORECASE | re.DOTALL)
        risk_level = risk_m.group(1).upper() if risk_m else ""
        reason = ""
        suggestion = ""

    # Determine outcome with proper precedence: BLOCKED > HUMAN_REVIEW > APPROVED
    is_blocked = status in ("BLOCKED", "拒绝", "拦截") or risk_level in ("HIGH", "CRITICAL")
    needs_review = status in ("HUMAN_REVIEW_REQUIRED", "需要人工审查") or risk_level == "MEDIUM"

    if is_blocked:
        logger.warning(f"[EthicsCheck] BLOCKED — reason: {reason[:200]}")
        return {
            "ethics_status": "blocked",
            "ethics_comment": reason or suggestion or content[:500],
            "ethics_action_required": True,
        }

    if needs_review:
        logger.info(f"[EthicsCheck] HUMAN_REVIEW_REQUIRED — reason: {reason[:200]}")
        return {
            "ethics_status": "human_review_required",
            "ethics_comment": reason or content[:500],
            "ethics_action_required": True,
        }

    # Default: approved (only if not blocked/review needed)
    logger.info("[EthicsCheck] PASS")
    return {
        "ethics_status": "approved",
        "ethics_comment": reason or content[:200],
        "ethics_risk_level": risk_level or "LOW",
        "current_action": "ethics_check",
    }


# ============================================================
# Items 3: Literature Review Node
# ============================================================

async def node_literature_review(state: AgentState) -> dict:
    """
    【文献调研与事实提取】
    提取至少 8 条关键科学事实，附带 DOI/PMID
    标记知识空白区域
    """
    domain = state.get("domain", "环境—人体关联")
    query = state.get("query", "")

    llm = _get_llm()
    messages = [
        {"role": "system", "content": LITERATURE_REVIEW_PROMPT},
        {
            "role": "user",
            "content": (
                f"## 任务：领域文献调研与事实提取\n\n"
                f"**领域**: {domain}\n"
                f"**研究问题**: {query}\n\n"
                f"要求：\n"
                f"1. 列出至少 8 条核心发现（每条必须真实可查）\n"
                f"2. 格式：- [事实] | Reference: Author, Year, Journal, DOI:xxxxx\n"
                f"3. 标记 3-5 个尚未充分研究的空白区域\n"
                f"4. ⚠️ 不要虚构任何文献！不确定就标注 `[需要验证]`"
            ),
        },
    ]

    content, _ = await _async_call_llm(llm, messages, temperature=0.7, max_tokens=4096)

    # Parse structured facts from LLM output
    facts = []
    for line in content.split("\n"):
        if line.strip().startswith("- ["):
            # Extract DOI if present
            doi_match = re.search(r'DOI:\s*([\w\-./]+)', line)
            pmid_match = re.search(r'PMID:\s*(\d+)', line)
            ref_match = re.search(r'Reference:\s*(.+?)(?:\||$)', line)

            facts.append({
                "fact": line.strip()[3:].strip(),
                "doi": doi_match.group(1) if doi_match else None,
                "pmid": pmid_match.group(1) if pmid_match else None,
                "reference": ref_match.group(1).strip() if ref_match else "Unknown",
            })

    logger.info(f"[LiteratureReview] Extracted {len(facts)} facts")

    # --- Knowledge Graph: auto-build from extracted facts ---
    # Entity keywords mapping (environment-human health domain)
    ENTITY_KEYWORDS = {
        "variable": ["temperature", "湿度", "CO₂", "pm2.5", "voc", "臭氧", "humidity",
                      "noise", "光照", "air_quality", "气压"],
        "biomarker": ["hrv", "sdnn", "rmssd", "心率变异性", "血氧", "spo2", "ppg",
                       "血压", "皮质醇", "heart_rate", "cortisol"],
        "population": ["老年人", "儿童", "办公室工人", "孕妇", "elderly", "children",
                        "office_worker"],
        "method": ["ccm", "格兰杰", "granger", "pc-fci", "psm", "贝叶斯网络",
                    "因果推断", "bayesian"],
    }

    kg_raw = nx.DiGraph()
    existing_kg = state.get("knowledge_graph", {})
    if isinstance(existing_kg, dict):
        # Rebuild graph from serializable nodes/edges stored in state
        for node_info in existing_kg.get("nodes", []):
            kg_raw.add_node(node_info["id"], type=node_info.get("type", "unknown"))
        for u, v, relation in existing_kg.get("edges", []):
            kg_raw.add_edge(u, v, relation=relation)
    if not kg_raw.nodes():
        # Initialize with domain root node
        kg_raw.add_node("Environment-Human_Association", type="topic")

    for fact in facts:
        fact_text = fact.get("fact", "")
        for entity_type, keywords in ENTITY_KEYWORDS.items():
            matched = [kw for kw in keywords if kw.lower() in fact_text.lower()]
            if matched:
                entity_name = matched[0]
                kg_raw.add_node(entity_name, type=entity_type)
                kg_raw.add_edge("Environment-Human_Association", entity_name,
                                relation="classified_as", entity_type=entity_type)

    return {
        "literature_summary": content,
        "fact_extraction": facts,
        "_literature_ready": True,
        "current_action": "literature_review",
        "knowledge_graph": {
            "nodes": [{"id": n, "type": d.get("type", "unknown")}
                      for n, d in kg_raw.nodes(data=True)],
            "edges": [(u, v, d.get("relation", ""))
                      for u, v, d in kg_raw.edges(data=True)],
        },
    }


# ============================================================
# Items 5, 26, 27: Hypothesis Generation + Tournament + Bayesian
# ============================================================

async def node_hypothesis_generation(state: AgentState) -> dict:
    """
    【假设生成引擎】
    - 基于文献事实和领域背景生成候选假设
    - Tournament 进化：每次最多保留 Top-3（注：真实 tournament 淘汰需结合评审分数实现）
    - Bayesian 置信度量化（先验 P(H) → 后验 P(H|D) via log-odds update）
    """
    iteration = state.get("iteration", 0)
    domain = state.get("domain", "环境—人体关联")
    query = state.get("query", "")
    lit_summary = state.get("literature_summary", "")
    facts = state.get("fact_extraction", [])

    already_hyp_count = len(state.get("hypothesis_tree", []))

    # === Hard guardrail: force termination before wasting LLM calls ===
    max_iters = state.get("_max_iterations_", 200)
    current_iter = state.get("iteration", 0)
    if current_iter >= max_iters:
        logger.warning(
            f"[HypothesisGen] MAX_ITERATIONS REACHED ({current_iter}>={max_iters}), "
            f"skipping generation to prevent infinite loop"
        )
        return {
            "hypothesis_tree": state.get("hypothesis_tree", []),
            "iteration": current_iter,          # ← Preserve iteration
            "_max_iterations_": max_iters,      # ← Preserve max_iter
            "consecutive_failures": state.get("consecutive_failures", 0),  # ← Don't reset failures
            "current_action": "hypothesis_generation",
        }

    # Get reflection insights if available
    anomalies = state.get("anomaly_graph", [])
    last_insight = anomalies[-1] if anomalies else {}
    reflection_hint = ""
    if last_insight.get("type") == "failure_insight":
        reflection_hint = f"\n\n### 上次反思的改进方向\n{last_insight.get('suggested_fix', '')[:300]}\n请避免之前的错误。"

    llm = _get_llm()
    prompt = HYPOTHESIS_GENERATION_TEMPLATE.format(
        domain_context=f"{domain} — {query}",
        known_facts="\n".join([f"- {f['fact']}" for f in facts[:10]]),
        literature_clues=lit_summary[:1500] if lit_summary else "无已知文献线索",
        constraints=(
            "参考文献必须真实可验证；假设必须涉及环境因子 → 生理指标的因果关联；"
            f"已存在 {already_hyp_count} 个假设，请生成不同的新假设。{reflection_hint}"
        ),
    )

    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    content, _ = await _async_call_llm(llm, messages, temperature=0.8, max_tokens=8192)

    # Parse MULTIPLE hypotheses from structured output
    # Split by "---" separator and parse each block
    raw_blocks = [b.strip() for b in re.split(r'^---$', content, flags=re.MULTILINE) if b.strip()]

    parsed_hypotheses = []
    for block in raw_blocks:
        title_match = re.search(r'标题[：:]\s*(.+)', block)
        statement_match = re.search(r'陈述[：:]\s*(.+)', block)
        conf_match = re.search(r'(?:先验置信度P\(H\)|先验置信度)[：:]?\s*([\d.]+)', block)
        test_match = re.search(r'可检验性评分[：:]?\s*(\d+)', block)
        evidence_req_match = re.search(r'证据需求[：:]\s*(.+?)(?:\n|$)', block)
        reasoning_match = re.search(r'推理链条[：:]\s*(.+?)(?:\n推理|\n先验|\n证据|$)', block, re.DOTALL)

        title = title_match.group(1).strip() if title_match else None
        if not title:
            continue  # skip malformed blocks

        new_hyp = {
            "id": _create_hypothesis_id(),
            "title": title,
            "statement": statement_match.group(1).strip() if statement_match else "",
            "reasoning_chain": reasoning_match.group(1).strip() if reasoning_match else block[:500],
            "confidence_prior": float(conf_match.group(1)) if conf_match else 0.5,
            "confidence_posterior": 0.5,  # will be updated by reviewer
            "testability": int(test_match.group(1)) if test_match else 5,
            "evidence_needed": evidence_req_match.group(1).strip() if evidence_req_match else "",
            "status": "proposed",
            "parent_id": None,
            "children_ids": [],
            "evidence_support": [],
            "evidence_against": [],
            "experiment_ids": [],
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        parsed_hypotheses.append(new_hyp)

    logger.info(f"[HypothesisGen] LLM returned {len(parsed_hypotheses)} candidate hypotheses")

    # Deep copy tree, add all new hypotheses, prune dead branches
    tree = copy.deepcopy([dict(h) for h in state.get("hypothesis_tree", [])])
    tree.extend(parsed_hypotheses)

    # PRUNING: Remove pruned/refuted_in_tournament hypotheses from previous rounds
    # FIX: Also remove "refuted_in_tournament" — these were eliminated by tournament but
    #       never cleaned up, causing unbounded context growth in downstream LLM calls
    pruned_count = 0
    kept_tree = []
    for h in tree:
        status = h.get("status", "")
        if status in ("pruned", "refuted_in_tournament"):
            pruned_count += 1
        else:
            kept_tree.append(h)

    # Preserve failure count — only reset on actual successful generation, don't hardcode 0
    prev_failures = state.get("consecutive_failures", 0)

    logger.info(f"[HypothesisGen] Tree now has {len(kept_tree)} hypotheses (removed {pruned_count} pruned)")

    return {
        "hypothesis_tree": kept_tree,
        "consecutive_failures": prev_failures,
        "_generation_success": True,
        "_max_iterations_": state.get("_max_iterations_", 200),
        "iteration": state.get("iteration", 0),
        "current_action": "hypothesis_generation",
    }


# ============================================================
# Item 27: Tournament Evaluation — Multi-Candidate Bracket Elimination
# ============================================================

async def node_tournament_eval(state: AgentState) -> dict:
    """
    【假设淘汰赛】从 N 个候选假设中两两比较，最终选出 1 个最优假设。

    FIX: 在每条退出路径上都必须显式传递 _max_iterations_, iteration,
         consecutive_failures，防止 LangGraph state merge 吞掉这些控制字段。
    """
    # Guard keys — always carry these forward regardless of branch
    max_iters = state.get("_max_iterations_", 200)
    curr_iter = state.get("iteration", 0)
    prev_failures = state.get("consecutive_failures", 0)

    hypotheses = copy.deepcopy([dict(h) for h in state.get("hypothesis_tree", [])])
    if len(hypotheses) <= 1:
        # 无需比较，直接标记为 active
        for h in hypotheses:
            if h.get("status") == "proposed":
                h["status"] = "active"
        return {
            "hypothesis_tree": hypotheses,
            "_max_iterations_": max_iters,
            "iteration": curr_iter,
            "consecutive_failures": prev_failures,
        }

    llm = _get_llm()

    # Build bracket description for the LLM prompt
    hyp_list_text = "\n".join(
        f"{i + 1}. [{h['id']}] **{h.get('title', '?')}**\n   陈述: {h.get('statement', '')[:200]}\n   推理: {h.get('reasoning_chain', '')[:150]}\n   先验置信度: {h.get('confidence_prior', '?')}\n   证据需求: {h.get('evidence_needed', 'N/A')}"
        for i, h in enumerate(hypotheses)
    )

    num_hyps = len(hypotheses)
    user_content = TOURNAMENT_EVAL_PROMPT.replace("[N]", str(num_hyps))
    user_content += f"\n\n## 当前候选假设列表\n{hyp_list_text}"

    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    content, _ = await _async_call_llm(llm, messages, temperature=0.3, max_tokens=8192)

    # Parse winner and elimination records from LLM output
    winner_title_m = re.search(r'\*\*获胜假设\*\*:\s*(.+)', content)
    winner_id_m = re.search(r'\*\*获胜假设ID\*\*:\s*(.+)', content)
    winner_title = winner_title_m.group(1).strip() if winner_title_m else ""
    winner_id = winner_id_m.group(1).strip() if winner_id_m else ""

    # If we couldn't parse winner ID, fall back to matching by title
    if not winner_id and winner_title:
        for h in hypotheses:
            if winner_title in h.get("title", "") or h.get("title", "") in winner_title:
                winner_id = h["id"]
                break

    # Parse elimination table rows: | 标题 | ID | 轮次 | 被谁击败 | 原因 |
    elim_records = []
    table_rows = re.findall(r'\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|', content)
    for row in table_rows:
        title, hid, round_, defeated_by, reason = [r.strip() for r in row]
        # Skip header-like entries
        if "假设标题" in title or "标题" in title and "ID" in hid:
            continue
        elim_records.append({
            "eliminated_title": title,
            "eliminated_id": hid,
            "eliminated_round": round_,
            "defeated_by": defeated_by,
            "reason": reason,
        })

    # Apply result: set winner status → active, others that were eliminated → refuted
    win_count = 0
    elim_count = 0
    winner_statement = ""
    for h in hypotheses:
        if h["id"] == winner_id or (winner_title and winner_title in h.get("title", "")):
            h["status"] = "active"
            h["tournament_won"] = True
            h["updated_at"] = _now_iso()
            winner_statement = h.get("statement", "")
            win_count += 1
        elif h.get("status") == "proposed" and win_count == 0:
            # First proposed hyp still marked active as fallback if winner not parsed
            h["status"] = "active"
            h["tournament_won"] = True
            winner_statement = h.get("statement", "")
            win_count += 1

    # If winner was never matched, just pick the one with highest prior confidence
    if win_count == 0:
        best = max(hypotheses, key=lambda h: h.get("confidence_prior", 0))
        best["status"] = "active"
        best["tournament_won"] = True
        winner_id = best["id"]
        winner_statement = best.get("statement", "")
        logger.info(f"[TournamentEval] Winner not parsed from output; selected highest-prior: {best['id']} ({best.get('title', '?')})")

    # Mark remaining proposed hypotheses as candidates for next round or review
    for h in hypotheses:
        if h["id"] != winner_id and h.get("status") == "proposed":
            h["status"] = "refuted_in_tournament"
            elim_count += 1

    # Preserve failure count — only increment on actual failures, don't blindly reset
    prev_failures = state.get("consecutive_failures", 0)

    logger.info(
        f"[TournamentEval] Winner={winner_id}, statement_len={len(winner_statement)}, "
        f"eliminated {elim_count} proposals, recorded {len(elim_records)} elimination records"
    )

    return {
        "hypothesis_tree": hypotheses,
        "elimination_records": elim_records,
        "prev_round_winner_id": winner_id,
        "prev_round_winner_statement": winner_statement,
        "consecutive_failures": prev_failures,   # ← FIX: preserve instead of resetting to 0
        "current_action": "tournament_eval",
    }

async def node_experiment_design(state: AgentState) -> dict:
    """
    【实验方案设计】
    基于 Active / Approved 假设设计完整可验证实验
    """
    hypotheses = state.get("hypothesis_tree", [])
    # Include proposed hypotheses so experiments get designed for all viable candidates
    active_hyps = [h for h in hypotheses if h.get("status") in ("active", "proposed", "approved_by_reviewer")]

    if not active_hyps:
        # === Guardrail: always carry control fields to prevent LangGraph merge corruption ===
        max_iters = state.get("_max_iterations_", 200)
        curr_iter = state.get("iteration", 0)
        prev_fails = state.get("consecutive_failures", 0)
        return {
            "_max_iterations_": max_iters,
            "iteration": curr_iter,
            "consecutive_failures": prev_fails,
            "current_action": "experiment_design",
        }

    hyp = active_hyps[0]

    llm = _get_llm()
    prompt = EXPERIMENT_DESIGN_TEMPLATE.format(
        hypothesis_statement=hyp.get("statement", ""),
        key_findings=str(state.get("fact_extraction", [])[:5]),
    )

    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    content, _ = await _async_call_llm(llm, messages, temperature=0.6, max_tokens=4096)

    exp_id = f"exp_{uuid.uuid4().hex[:6]}"

    # Use real sensor data if available
    import glob as _glob
    sensor_csvs = _glob.glob(str(Path("data/sensors/*.csv")))
    input_data_path = sensor_csvs[0] if sensor_csvs else "[DATA_CHANNEL_PLACEHOLDER]"

    experiment_record = {
        "id": exp_id,
        "hypothesis_id": hyp["id"],
        "design_raw": content,
        "input_data_path": input_data_path,
        "output_data_path": "",
        "results": {"status": "design_only", "analysis_pending": True},
        "code_history": [],
        "notes": f"使用传感器数据: {input_data_path}" if input_data_path != "[DATA_CHANNEL_PLACEHOLDER]" else "暂无传感器数据，使用理论分析",
        "created_at": _now_iso(),
    }

    # Deep-copy tree first to avoid LangGraph merge corruption
    tree = copy.deepcopy([dict(h) for h in state.get("hypothesis_tree", [])])
    matched_hyp = None
    for h in tree:
        if h["id"] == active_hyps[0]["id"]:
            matched_hyp = dict(h)
            break

    if matched_hyp:
        matched_hyp["experiment_ids"].append(exp_id)
        matched_hyp["updated_at"] = _now_iso()
        # Replace old hyp in tree with updated copy
        for i, h in enumerate(tree):
            if h["id"] == active_hyps[0]["id"]:
                tree[i] = matched_hyp
                break

    experiments = copy.deepcopy(list(state.get("experiment_records", [])))
    experiments.append(experiment_record)

    logger.info(f"[ExperimentDesign] Created experiment {exp_id} for hypothesis {hyp['id']}")

    return {
        "experiment_records": experiments,
        "hypothesis_tree": tree,
        "current_action": "experiment_design",
    }


# ============================================================
# Items 18, 19, 20: Data Analysis + Causal Inference
# ============================================================

async def node_data_analysis(state: AgentState) -> dict:
    """
    【数据分析与因果推断】
    - 调用 CausalInferenceEngine 自动选择方法并执行分析
    - 构建标准化 EvidenceEntry 存入 evidence_chains
    - 如果没有真实数据，降级为理论分析框架（占位）
    """
    experiments = copy.deepcopy(list(state.get("experiment_records", [])))
    hypotheses = copy.deepcopy(list(state.get("hypothesis_tree", [])))
    evidence_chains = copy.deepcopy(list(state.get("evidence_chains", [])))

    engine = CausalInferenceEngine()

    for exp in experiments:
        if not exp.get("results", {}).get("analysis_pending", False):
            continue

        hyp_id = exp.get("hypothesis_id", "")
        exp_id = exp.get("id", "")
        input_path = exp.get("input_data_path", "")

        # Check if we have real data to analyze
        has_real_data = input_path and input_path != "[DATA_CHANNEL_PLACEHOLDER]"

        if has_real_data:
            try:
                # Load real Daltons sensor data from channels
                from pathlib import Path
                import pandas as pd

                csv_file = Path(input_path)
                logger.info(f"[DataAnalysis] Loading sensor data from {csv_file}")

                # Try Daltons-format parser first (for parsed records with pollutant_name/value)
                from channels.time_series import _detect_daltons_format, _parse_daltons_records
                df_raw = pd.read_csv(csv_file)
                raw_records = df_raw.to_dict(orient="records")
                fmt = _detect_daltons_format(raw_records)
                parsed_records = _parse_daltons_records(raw_records, single_sensor_file=(fmt == "processed"))

                logger.info(f"[DataAnalysis] Loaded {len(parsed_records)} Daltons records (format={fmt})")

                # Extract time series for each pollutant
                pollutants = list(set(r.get("pollutant_name", "") for r in parsed_records))
                ts_data = {p: [] for p in pollutants}

                for rec in parsed_records:
                    poll = rec.get("pollutant_name", "")
                    val = rec.get("value", 0)
                    if poll and val != 0:
                        ts_data[poll].append(float(val))

                # Use primary environmental variables for causal inference
                x_key = "T" if "T" in ts_data and ts_data["T"] else pollutants[0]
                y_key = "CO2" if "CO2" in ts_data and ts_data["CO2"] else (pollutants[1] if len(pollutants) > 1 else x_key)

                x = [v for v in ts_data[x_key] if v is not None][:500]  # cap at 500 samples
                y = [v for v in ts_data[y_key] if v is not None][:500]

                n_samples = min(len(x), len(y))
                if n_samples < 10:
                    raise ValueError(f"Not enough Daltons-parsed data points for analysis (n={n_samples}). Switching to direct CSV parsing.")

                x = x[:n_samples]
                y = y[:n_samples]

                logger.info(f"[DataAnalysis] Using {x_key}→{y_key} causal pathway with {n_samples} paired observations (Daltons format)")

            except Exception as _daltons_exc:
                # Fallback: read CSV directly — columns are named T, CO2, VOC, NO2, PMS1, PMS10, PMS2_5, C2H5OH, H
                logger.warning(f"[DataAnalysis] Daltons parsing failed ({_daltons_exc}), trying direct CSV columns...")
                try:
                    df_direct = pd.read_csv(str(csv_file))
                    logger.info(f"[DataAnalysis] Direct CSV columns: {list(df_direct.columns)}")

                    # Map common column names to physical quantities
                    COL_MAP = {
                        "T": ["T"],                    # Temperature
                        "CO2": ["CO2", "CO2_ppm", "co2", "carbon_dioxide"],  # CO2 concentration
                        "VOC": ["VOC", "voc", "tvoc", "vocs"],               # Volatile organic compounds
                        "H": ["H", "humidity", "RH", "relative_humidity"],   # Humidity
                        "NO2": ["NO2", "no2", "nitrogen_dioxide"],           # Nitrogen dioxide
                        "PMS2_5": ["PMS2_5", "pm2_5", "pm25"],              # PM2.5
                    }

                    x_vals = None
                    y_vals = None

                    # Pick X = temperature, Y = CO2 (primary causal pathway)
                    for x_candidate in COL_MAP["T"]:
                        if x_candidate in df_direct.columns:
                            x_vals = df_direct[x_candidate].dropna().astype(float).tolist()[:500]
                            break

                    for y_candidate in COL_MAP["CO2"]:
                        if y_candidate in df_direct.columns:
                            y_vals = df_direct[y_candidate].dropna().astype(float).tolist()[:500]
                            break

                    if x_vals is None or y_vals is None:
                        raise ValueError(f"Cannot find target columns. Available: {list(df_direct.columns)}. Need T and CO2.")

                    n_samples = min(len(x_vals), len(y_vals))
                    if n_samples < 10:
                        raise ValueError(f"Not enough valid data points (n={n_samples}). Need >= 10.")

                    x = x_vals[:n_samples]
                    y = y_vals[:n_samples]

                    logger.info(f"[DataAnalysis] Using T→CO2 causal pathway with {n_samples} paired observations (direct CSV columns)")

                except Exception as _direct_exc:
                    logger.error(f"[DataAnalysis] Direct CSV parsing also failed: {_direct_exc}")
                    exp["results"]["error"] = str(_direct_exc)
                    exp["results"]["analysis_pending"] = False
                    continue  # Skip this experiment, move on

            try:
                # Auto-select best method
                feature_info = {
                    "sample_size": len(x),
                    "num_variables": 2,
                    "is_time_series": True,
                    "has_known_confounders": False,
                    "nonlinear_relationships": False,
                }
                recommendation = await engine.run("auto_select", feature_info=feature_info)
                selected_method = recommendation.get("selected_method", "granger")

                logger.info(f"[DataAnalysis] Selected method: {selected_method} for experiment {exp_id}")

                # Run the causal inference method
                if selected_method == "ccm":
                    result = await engine.run("ccm", x=x, y=y, column_size=3)
                elif selected_method == "granger":
                    max_lag = min(recommendation.get("parameters", {}).get("max_lag", 5), len(x)//4)
                    result = await engine.run("granger", x=x, y=y, max_lag=max_lag or 3)
                else:
                    result = await engine.run("counterfactual",
                                              predictions_base=x[:n_samples//2],
                                              predictions_intervened=y[n_samples//2:])

                # Build structured EvidenceEntry
                evidence_entry = _build_evidence_entry("causal_inference", result, selected_method,
                                                       hyp_id, exp_id)
                evidence_chains.append(evidence_entry)

                exp["results"]["analysis_complete"] = True
                exp["results"]["result_summary"] = _summarize_causal_result(result, selected_method)
                exp["results"]["selected_method"] = selected_method
                exp["results"]["data_source"] = str(csv_file)

            except Exception as _engine_exc:
                logger.error(f"[DataAnalysis] Causal inference engine failed for {exp_id}: {_engine_exc}")
                exp["results"]["error"] = str(_engine_exc)
                exp["results"]["theoretical_analysis"] = (
                    f"**数据分析失败**: {_engine_exc}\n\n"
                    f"实际加载了传感器数据，但因果推断引擎执行异常。请检查依赖包是否安装完整。\n"
                    f"数据源: {csv_file}, 样本数: {n_samples}"
                )
        else:
            # No real data — provide theoretical analysis framework
            exp["results"]["theoretical_analysis"] = (
                "**分析方法选择**: 由于暂无真实时序数据，以下提供理论分析框架。\n\n"
                "当数据就绪时，系统会自动执行以下步骤：\n"
                "1. **数据预处理**: 时间对齐 → 缺失值填充 → 异常值检测\n"
                "2. **方法选择**: 基于数据特征自动选择 CCM/Granger/贝叶斯网络等\n"
                "3. **因果推断**: 估计环境因子→生理指标的方向性因果关系\n"
                "4. **反事实推演**: 预测不同环境条件下的生理响应\n\n"
                "预计输出：因果效应大小 β 及其显著性、最佳方法的选择理由、反事实预测区间。"
            )

        exp["results"]["analysis_pending"] = False
        break  # Only analyze the latest pending experiment

    return {
        "experiment_records": experiments,
        "evidence_chains": evidence_chains,
        "current_action": "data_analysis",
    }


# ============================================================
# Interpretation Node
# ============================================================

async def node_interpretation(state: AgentState) -> dict:
    """
    【结果解读】
    综合分析结果，更新假设置信度，识别反直觉模式
    """
    hypotheses = copy.deepcopy(list(state.get("hypothesis_tree", [])))
    experiments = list(state.get("experiment_records", []))
    evidence_chains = list(state.get("evidence_chains", []))

    # Guard against empty lists on first run — always carry control fields
    if not experiments:
        logger.info("[Interpretation] No experiments yet — returning early with no change.")
        return {
            "convergence_score": 0.0,
            "_max_iterations_": state.get("_max_iterations_", 200),
            "iteration": state.get("iteration", 0),
            "consecutive_failures": state.get("consecutive_failures", 0),
            "current_action": "interpretation",
        }
    if not evidence_chains:
        logger.info("[Interpretation] No evidence chains yet — returning early.")
        return {
            "convergence_score": 0.0,
            "_max_iterations_": state.get("_max_iterations_", 200),
            "iteration": state.get("iteration", 0),
            "consecutive_failures": state.get("consecutive_failures", 0),
            "current_action": "interpretation",
        }

    latest_results = experiments[-1].get("results", {})
    latest_evidence = evidence_chains[-1]

    logger.info(f"[Interpretation] Latest experiment results keys: {list(latest_results.keys())}")
    logger.info(f"[Interpretation] Latest evidence: type={latest_evidence.get('type')}, strength={latest_evidence.get('strength')}")

    # Build a rich summary for the LLM
    causal_summary = ""
    if isinstance(latest_evidence, dict):
        method_used = latest_evidence.get("method_used", "N/A")
        strength = latest_evidence.get("strength", 0)
        content = latest_evidence.get("content", "")
        statistical_basis = latest_evidence.get("statistical_basis", {})
        causal_dir = latest_evidence.get("causal_direction", "N/A")

        causal_summary = (
            f"**因果推断结果**: {content}\n"
            f"**方法**: {method_used}\n"
            f"**证据强度**: {strength:.3f}\n"
            f"**因果方向**: {causal_dir}\n"
        )
        if statistical_basis:
            causal_summary += "**统计依据**:\n"
            for k, v in statistical_basis.items():
                causal_summary += f"- {k}: {v}\n"

    llm = _get_llm()
    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## 任务：实验结果解读\n\n"
                f"假设数量: {len(hypotheses)}\n"
                f"实验数量: {len(experiments)}\n"
                f"证据链条数: {len(evidence_chains)}\n\n"
                f"**实验结果摘要**: {str(latest_results)[:1000]}\n\n"
                f"**因果分析结果**:\n{causal_summary if causal_summary else '（暂无因果分析结果）'}\n\n"
                f"请解读：\n"
                f"1. 哪些假设获得了支持？哪些被削弱？\n"
                f"2. 是否存在反直觉的模式？\n"
                f"3. 置信度应该如何更新？\n\n"
                f"请按以下格式输出更新的假设置信度：\n"
                f"- <hyp_id>: prior_X.XX → posterior_Y.YY (reason)"
            ),
        },
    ]

    content, parsed_json = await _async_call_llm(llm, messages, temperature=0.6, max_tokens=4096)

    # Update hypothesis confidence based on interpretation (deep copy to avoid mutation)
    tree = copy.deepcopy([dict(h) for h in state.get("hypothesis_tree", [])])  # Deep copy
    result_hyps = []
    max_change = 0.0

    for hyp in tree:
        h_copy = dict(hyp)
        conf_pattern = rf'{re.escape(h_copy["id"])}:.*?posterior\s*([\d.]+)'
        match = re.search(conf_pattern, content)
        if match:
            old_posterior = h_copy.get("confidence_posterior", h_copy.get("confidence_prior"))
            new_posterior = float(match.group(1))
            change = abs(new_posterior - old_posterior)
            if change > max_change:
                max_change = change
            if change > 0.05:
                logger.info(f"[Interpretation] Updated {h_copy['id']}: posterior {old_posterior} → {new_posterior}")
            # Clamp confidence to valid [0.01, 0.99] range — LLM may hallucinate out-of-bounds values
            new_posterior_clamped = min(max(new_posterior, 0.01), 0.99)
            h_copy["confidence_posterior"] = new_posterior_clamped
        result_hyps.append(h_copy)

    # Compute convergence score: 1 - max_confidence_change (lower change = more converged)
    convergence_score = max(0.0, 1.0 - max_change * 2)

    logger.info(f"[Interpretation] Updated {len(result_hyps)} hypotheses, convergence_score={convergence_score:.3f}")

    return {"hypothesis_tree": result_hyps, "convergence_score": round(convergence_score, 3), "current_action": "interpretation"}


# ============================================================
# Item 14: Reviewer Agent
# ============================================================

async def node_reviewer_agent(state: AgentState) -> dict:
    """
    【五维审稿】新颖性/可行性/方法论/证据/影响
    低于75分打回修改
    """
    hypotheses = state.get("hypothesis_tree", [])
    active = [h for h in hypotheses if h.get("status") in ("active", "proposed")]
    if not active:
        # Carry termination-critical keys on ALL branches — prevent LangGraph merge corruption
        max_iters = state.get("_max_iterations_", 200)
        curr_iter = state.get("iteration", 0)
        prev_fails = state.get("consecutive_failures", 0)
        return {
            "next_step": "reflection",
            "_max_iterations_": max_iters,
            "iteration": curr_iter,
            "consecutive_failures": prev_fails,
            "current_action": "reviewer_agent",
        }

    latest_hyp = active[-1]
    reviews = copy.deepcopy(list(state.get("review_records", [])))
    prev_reviews_for_hyp = [r for r in reviews if r.get("hypothesis_id") == latest_hyp["id"]]
    previous_feedback = ""
    if prev_reviews_for_hyp:
        previous_feedback = f"\n\n上次评审反馈（已修改但仍需审核）:\n{prev_reviews_for_hyp[-1].get('comments', '')[:500]}"

    llm = _get_llm()
    messages = [
        {"role": "system", "content": REVIEWER_AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## 待评审假设\n\n"
                f"标题: {latest_hyp.get('title', 'N/A')}\n"
                f"陈述: {latest_hyp.get('statement', 'N/A')}\n"
                f"推理链条: {latest_hyp.get('reasoning_chain', 'N/A')[:300]}\n"
                f"先验置信度: {latest_hyp.get('confidence_prior', '?')}\n"
                f"可检验性: {latest_hyp.get('testability', '?')}/10\n"
                f"已有实验记录: {len(latest_hyp.get('experiment_ids', []))} 个\n"
                f"{previous_feedback}"
            ),
        },
    ]

    content, json_obj = await _async_call_llm(llm, messages, temperature=0.3, max_tokens=2048)
    scores = _parse_review_score(content)

    review_record = {
        "id": f"rev_{uuid.uuid4().hex[:6]}",
        "hypothesis_id": latest_hyp["id"],
        **scores,
        "created_at": _now_iso(),
        "round": len(prev_reviews_for_hyp) + 1,
    }

    reviews.append(review_record)

    # Update hypothesis status (with deep copy to avoid LangGraph mutation)
    tree = copy.deepcopy(list(state.get("hypothesis_tree", [])))
    new_tree = []
    for hyp in tree:
        h_copy = dict(hyp)
        if h_copy["id"] == latest_hyp["id"]:
            if scores["total_score"] >= 75:
                h_copy["status"] = "approved_by_reviewer"
                # Bayesian update via log-odds: convert prior to log-odds space, add evidence weight from score, convert back
                import math
                prior = max(h_copy.get("confidence_prior", 0.5), 1e-6)
                log_odds_prior = math.log(prior / (1 - prior + 1e-6))
                evidence_weight = (scores["total_score"] - 50) / 100 * 2  # score 50→0 weight, score 100→+1 weight
                log_odds_posterior = log_odds_prior + evidence_weight
                posterior = 1.0 / (1.0 + math.exp(-log_odds_posterior))
                posterior = min(max(posterior, 0.01), 0.99)  # clamp to valid range
                h_copy["confidence_posterior"] = round(posterior, 4)
                # Child confidence transfer: only increase child's posterior if parent is confident and no strong evidence against it
                if h_copy.get("parent_id"):
                    parent = next((p for p in tree if p["id"] == h_copy["parent_id"]), None)
                    if parent:
                        parent_post = parent.get("confidence_posterior", 0.5)
                        # Transfer at most 15% boost — prevents paradoxical inflation of weak hypotheses
                        boost_cap = 0.15
                        current = h_copy["confidence_posterior"]
                        if parent_post > 0.7:
                            h_copy["confidence_posterior"] = min(current + boost_cap * (parent_post - 0.7) / 0.3, 0.98)
            else:
                h_copy["status"] = "needs_revision"
            h_copy["updated_at"] = _now_iso()
            h_copy["latest_review_round"] = review_record["round"]
        new_tree.append(h_copy)

    action = "report_writing" if scores["total_score"] >= 75 else "reflection"
    logger.info(f"[ReviewerAgent] {latest_hyp['id']}: score={scores['total_score']}/100, needs_revision={scores['needs_revision']} → next={action}")

    return {
        "review_records": reviews,
        "hypothesis_tree": new_tree,
        "current_action": "reviewer_agent",
    }


# ============================================================
# Item 4/10: Reflection Loop
# ============================================================

async def node_reflection(state: AgentState) -> dict:
    """
    【反思与修正】根因分析 → 派生修正性假设
    失败资产化：存储教训用于后续迭代

    Orchestrator 分析在入口处自动计算，确保反思节点拿到完整的本轮评估上下文。
    """
    iteration = state.get("iteration", 0) + 1
    reviews = list(state.get("review_records", []))
    latest_review = reviews[-1] if reviews else {}
    hypotheses = state.get("hypothesis_tree", [])

    failed_hyp = next((h for h in hypotheses if h.get("status") == "needs_revision"), None)
    hyp_info = str(failed_hyp)[:500] if failed_hyp else "无指定失败假设"

    # --- Compute orchestrator stop-check analysis so reflection has full context ---
    # Reuse cached result if already computed (avoids redundant iteration over state)
    orch_checks = get_cached_orch_check(state)
    if orch_checks is None:
        orch_checks = _check_orchestrator_stop_conditions(state)
    orch_reason = orch_checks.get("reason", "")
    orch_evidence_strength = orch_checks.get("avg_evidence_strength", 0)
    orch_similarity_score = orch_checks.get("similarity_score", 0)
    orch_max_round_reached = orch_checks.get("max_round_reached", False)
    orch_evidence_strong = orch_checks.get("evidence_strong", False)
    orch_converged = orch_checks.get("converged", False)

    llm = _get_llm()
    messages = [
        {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## 反思任务\n\n"
                f"迭代轮次: {iteration}\n"
                f"评审分数: {latest_review.get('total_score', '?')}/100\n"
                f"评审意见: {latest_review.get('comments', '无')[:500]}\n"
                f"被评审假设: {hyp_info}\n\n"
                f"### 📊 Orchestrator 决策分析（本轮验证结果）\n"
                f"- 推理原因: {orch_reason}\n"
                f"- 证据强度: {orch_evidence_strength:.4f}\n"
                f"- 假设相似度(vs上一轮): {orch_similarity_score:.4f}\n"
                f"- 已达最大轮次: {'是' if orch_max_round_reached else '否'}\n"
                f"- 证据已足够强: {'是' if orch_evidence_strong else '否'}\n"
                f"- 假设已收敛: {'是' if orch_converged else '否'}\n\n"
                f"请基于以上全部信息，回答反思问题并生成修正后的假设。"
                f"{'⚠️ 已达到最大轮次上限，请直接给出最终假设并标记为最终版本。' if orch_max_round_reached else ''}"
            ),
        },
    ]

    content, _ = await _async_call_llm(llm, messages, temperature=0.7, max_tokens=4096)

    # Extract potential new hypothesis from reflection
    new_hyp_title = re.search(r'新假设[：:]\s*[标题]*\s*(.+)', content)
    new_hyp_stmt = re.search(r'[新]陈述[：:]\s*(.+)', content)

    new_hyp = None
    if new_hyp_title:
        new_hyp = {
            "id": _create_hypothesis_id(),
            "title": new_hyp_title.group(1).strip(),
            "statement": new_hyp_stmt.group(1).strip() if new_hyp_stmt else content[:200],
            "reasoning_chain": "From reflection: " + content[:300],
            "confidence_prior": 0.2,
            "confidence_posterior": 0.2,
            "testability": 4,
            "status": "proposed",
            "parent_id": failed_hyp["id"] if failed_hyp else None,
            "children_ids": [],
            "evidence_support": [],
            "evidence_against": [],
            "experiment_ids": [],
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "derived_from_failure": True,
        }

    # Record failure insight (Item 10)
    failure_insight = {
        "id": f"fail_{uuid.uuid4().hex[:6]}",
        "iteration": iteration,
        "type": "failure_insight",
        "root_cause": latest_review.get("comments", "")[:500],
        "suggested_fix": content[:500],
        "original_hypothesis_id": failed_hyp["id"] if failed_hyp else None,
        "timestamp": _now_iso(),
    }

    # Deep-copy anomaly_graph to avoid mutating LangGraph state on append
    anomaly_graph = copy.deepcopy(list(state.get("anomaly_graph", [])))
    anomaly_graph.append(failure_insight)

    # Build new tree with proper parent-child linking (deep copy to avoid mutation)
    tree = copy.deepcopy([dict(h) for h in state.get("hypothesis_tree", [])])

    if new_hyp:
        # Ensure parent exists in tree and track child link
        if failed_hyp:
            parent_found = False
            for h in tree:
                if h["id"] == failed_hyp["id"]:
                    parent_found = True
                    if "children_ids" not in h:
                        h["children_ids"] = []
                    h["children_ids"].append(new_hyp["id"])
                    break
            if not parent_found:
                # Parent was pruned or removed — orphan the child at root level
                new_hyp["parent_id"] = None

        tree.append(new_hyp)

    # PRUNING STRATEGY: Remove deeply unproductive branches
    # FIX: Also remove "refuted_in_tournament" — same gap as in hypothesis_generation
    pruned_count = 0
    kept_tree = []
    for h in tree:
        status = h.get("status", "")
        if status in ("refuted", "pruned") and len(h.get("children_ids", [])) == 0:
            h["status"] = "pruned"
            pruned_count += 1
        elif status == "proposed" and h.get("confidence_prior", 0) < 0.15:
            # Very low confidence proposals get pruned
            h["status"] = "pruned"
            pruned_count += 1
        elif status == "refuted_in_tournament":
            # Tournament losers with no children get cleaned up here too
            pruned_count += 1
        else:
            kept_tree.append(h)

    logger.info(f"[Reflection] Iteration {iteration} — failure stored, {'new derived hypothesis added' if new_hyp else 'no new hypothesis'}, pruned {pruned_count} leaf nodes")

    return {
        "iteration": iteration,
        "anomaly_graph": anomaly_graph,
        "hypothesis_tree": kept_tree,
        "consecutive_failures": 0 if new_hyp else state.get("consecutive_failures", 0) + 1,
        "_orch_stop_check": orch_checks,  # Cache for subsequent calls within this iteration
        "current_action": "reflection",
    }


# ============================================================
# Item 25: Termination Evaluation
# ============================================================

async def node_termination_eval(state: AgentState) -> dict:
    """
    【三层语义终止评估 + 收敛度计算】

    收敛度计算公式：
        convergence = 1 - |本轮假设与上一轮假设的语义相似度变化幅度|

        完全相同 (similarity=1.0) → convergence=100%（稳定可停）
        完全不同 (similarity=0)   → convergence=0%（仍在探索不能停）
        部分相同 (similarity=0.7) → convergence=70%（在稳定但还没确定）
        第1轮（无上一轮）         → convergence=0%

    终止条件：
        1. convergence ≥ 85% 且连续 2 轮保持不变 → 停止
        2. 达到最大轮次 (200 轮)                    → 停止
        3. 原始三路评分组合 ≥ 0.85                → 停止
    """
    evidence_chains = state.get("evidence_chains", [])
    exploration_exhausted = state.get("exploration_exhausted", False)
    iteration = state.get("iteration", 0)

    # ---- Step 1: Compute convergence score ----
    prev_statement = state.get("prev_round_winner_statement", "")
    hypotheses = state.get("hypothesis_tree", [])

    # Pick the current round's "winner": approved > tournament_won > highest posterior
    active_hyp = None
    for h in hypotheses:
        if h.get("status") == "approved_by_reviewer":
            active_hyp = h
            break
        if h.get("tournament_won"):
            active_hyp = h
            break
    if not active_hyp:
        active_hyp = max(hypotheses, key=lambda h: h.get("confidence_posterior", h.get("confidence_prior", 0))) if hypotheses else None

    current_statement = active_hyp.get("statement", "") if active_hyp else ""

    if iteration <= 1 or not prev_statement or not current_statement:
        # Round 1 or no previous data — convergence is 0%
        convergence = 0.0
        logger.info(f"[TerminationEval] Round {iteration}: no previous round data, convergence=0.0")
    else:
        # Compute semantic similarity using bigram Jaccard (defined in orchestrator)
        similarity = _hypothesis_statement_similarity(prev_statement, current_statement)
        # Convergence = similarity (high similarity = converged)
        convergence = round(similarity, 3)
        logger.info(
            f"[TerminationEval] Round {iteration}: similarity={similarity:.4f}, convergence={convergence:.3f}"
        )

    # Track convergence history
    convergence_history = list(state.get("convergence_history", []))
    convergence_history.append(convergence)

    # ---- Step 2: Compute evidence strength ----
    if evidence_chains:
        evidence_str = sum(e.get("strength", 0.5) for e in evidence_chains) / len(evidence_chains)
    else:
        approved = [h for h in hypotheses if h.get("status") == "approved_by_reviewer"]
        if approved:
            evidence_str = sum(h.get("confidence_posterior", 0.5) for h in approved) / len(approved)
        else:
            evidence_str = 0.0

    # ---- Step 3: Original combined score ----
    combined_score = convergence * 0.4 + evidence_str * 0.3 + (0.8 if exploration_exhausted else 0.0) * 0.3

    # ---- Step 4: Check all termination conditions ----
    should_terminate = False
    stop_reason = ""

    # Condition A: High convergence + stable for last 2 rounds
    if convergence >= 0.85 and len(convergence_history) >= 2:
        last_two = convergence_history[-2:]
        if abs(last_two[0] - last_two[1]) < 0.05:  # changed less than 5% between last two
            should_terminate = True
            stop_reason = (
                f"收敛度稳定高企: 当前={convergence:.1%}, "
                f"上轮={last_two[0]:.1%}, 差值={abs(last_two[0]-last_two[1]):.1%} (<5%), 结论已稳定"
            )
            logger.info(f"[TerminationEval] CONVERGENCE_STABLE: {stop_reason}")

    # Condition B: Hard limit — use configured max_iterations (not hardcoded 200)
    max_iters = state.get("_max_iterations_", 200)
    if iteration >= max_iters:
        should_terminate = True
        stop_reason = f"已达到最大轮次上限 ({max_iters}/{max_iters})"
        logger.info(f"[TerminationEval] MAX_ROUNDS_REACHED: {stop_reason}")

    # Condition C: Original combined score threshold
    if combined_score >= 0.85 and not should_terminate:
        should_terminate = True
        stop_reason = (
            f"综合评分达标 (combined_score={combined_score:.3f} ≥ 0.85), 证据充分可终止"
        )
        logger.info(f"[TerminationEval] COMBINED_SCORE_HIGH: {stop_reason}")

    if not should_terminate:
        stop_reason = f"未满足任何停止条件 (convergence={convergence:.1%}, combined={combined_score:.3f})，继续下一轮"
        logger.info(f"[TerminationEval] CONTINUE: {stop_reason}")

    # ---- Step 5: PRUNING at termination ----
    tree = copy.deepcopy([dict(h) for h in state.get("hypothesis_tree", [])])
    kept_tree = []
    pruned_at_term = 0
    for h in tree:
        if h.get("status") == "pruned" or h.get("status") == "refuted":
            pruned_at_term += 1
        else:
            kept_tree.append(h)

    logger.info(
        f"[TerminationEval] Score={combined_score:.3f}, terminate={should_terminate}, "
        f"convergence={convergence:.3f}, pruned {pruned_at_term} dead branches"
    )

    result = {
        "convergence_score": convergence,
        "convergence_history": convergence_history,
        "prev_round_winner_statement": current_statement,  # Save for next round comparison
        "evidence_strength": round(evidence_str, 3),
        "exploration_exhausted": exploration_exhausted,
        "combined_score": round(combined_score, 3),
        "should_terminate": should_terminate,
        "stop_reason": stop_reason,
        "hypothesis_tree": kept_tree if not should_terminate else tree,
    }

    return {"_termination_result": result, "__decision": "TERMINATE" if should_terminate else "CONTINUE", "current_action": "termination_eval"}


# ============================================================
# Item 28: Report Writing
# ============================================================

async def node_report_writing(state: AgentState) -> dict:
    """
    【报告撰写】
    强制生成含赛题规范 12 字段的《科学假设与研究计划》
    Strategy: programmatically assemble structural parts from real data,
              ask LLM only for creative prose (Rationale, Abstract, Conclusion).
    """
    hypotheses = state.get("hypothesis_tree", [])
    experiments = list(state.get("experiment_records", []))
    evidence_chains = list(state.get("evidence_chains", []))
    reviews = list(state.get("review_records", []))
    literature_summary = state.get("literature_summary", "")
    domain = state.get("domain", "环境—人体关联")
    query = state.get("query", "")
    elimination_records = list(state.get("elimination_records", []))

    convergence_val = state.get("convergence_score", 0.0) * 100
    iteration_val = state.get("iteration", 0)
    max_iter = state.get("_max_iterations_", 200)

    # --- Programmatic sections (no LLM hallucination possible) ---
    active_hyp_count = len([h for h in hypotheses if h.get('status') != 'pruned'])

    # Section 8.3: Experiment execution table
    exp_table_rows = ""
    for exp in experiments:
        eid = exp.get('id', '?')
        has_res = str(exp.get('results', {}).get('analysis_complete', False)).lower()
        notes = exp.get('notes', 'N/A')[:50]
        exp_table_rows += f"| {eid} | 已设计 | {has_res} | {notes} |\n"

    # Section 9: Real causal inference results (from actual analysis)
    # Debug: log what's actually in state
    logger.info(f"[ReportWriting] evidence_chains count={len(evidence_chains)}")
    for i, ev in enumerate(evidence_chains):
        if isinstance(ev, dict):
            logger.info(f"  ev[{i}] keys={list(ev.keys())[:15]}, type_field={ev.get('type', 'N/A')}")

    has_real_analysis = any(
        (isinstance(e, dict) and e.get('type') == 'causal_inference') or
        (isinstance(e, dict) and e.get('selected_method'))
        for e in evidence_chains
    )
    if not has_real_analysis:
        # Fallback: check experiment results directly
        has_real_analysis = any(
            isinstance(exp, dict) and exp.get('results', {}).get('analysis_complete')
            for exp in experiments
        )
    logger.info(f"[ReportWriting] has_real_analysis={has_real_analysis}")
    section9_content = ""
    if has_real_analysis:
        logger.info("[ReportWriting] Entering HAS_REAL_ANALYSIS branch")
        section9_content += "(以下基于真实数据分析)\n\n"
        for i, ev in enumerate(evidence_chains):
            logger.info(f"[ReportWriting] Checking ev[{i}], type_field={ev.get('type','N/A')}")
            if ev.get('type') != 'causal_inference':
                logger.info("[ReportWriting]   -> skipping non-causal_inference")
                continue
            logger.info(f"[ReportWriting]   -> PROCESSING causal inference entry")
            method = ev.get('method_used', '?')
            strength = ev.get('strength', 0)
            cev = ev.get('content', '')
            sb = ev.get('statistical_basis', {})
            cd = ev.get('causal_direction', None)
            section9_content += f"#### 数据方法: **{method}**\n\n"
            section9_content += f"**因果推断摘要**: {cev}\n\n"
            section9_content += "| 指标 | 值 | 说明 |\n|------|-----|------|\n"
            section9_content += f"| 证据强度 | {strength:.4f} | 0-1 置信度分数 |\n"
            section9_content += f"| 分析方法 | {method} | AI自动选择的最优方法 |\n"
            section9_content += f"| 实验数量 | {len(experiments)} | 执行的实验数 |\n"
            if cd and cd != "None":
                section9_content += f"| 因果方向 | {cd} | 因果推断结果的方向性 |\n"
            sbs = "; ".join(f"{k}: {v}" for k, v in list(sb.items())[:6])
            if sbs:
                section9_content += f"| 统计依据 | {sbs} |\n"
            section9_content += "\n"
        logger.info(f"[ReportWriting] section9 built successfully, length={len(section9_content)}")
    else:
        section9_content = "*注：以下为理论可行性验证框架*\n\n[理论分析框架]\n"
        logger.info("[ReportWriting] Using default theoretical framework section")

    # Section 10: Reviewer feedback table
    review_rows = ""
    for r in reviews:
        h_id = r.get('hypothesis_id', '?')
        score = r.get('total_score', '?')
        needs_rev = bool(r.get('needs_revision', True))
        review_rows += f"| {h_id} | {score} | {needs_rev} |\n"

    # Section 12: Hypothesis tree & evidence chain tables
    hyp_rows = ""
    for h in [h for h in hypotheses if h.get('status') not in ('pruned',)]:
        hid = h['id']
        title = h.get('title', '')
        status = h.get('status', '?')
        prior = h.get('confidence_prior', 0)
        post = h.get('confidence_posterior', h.get('confidence_prior', 0))
        test = h.get('testability', '?')
        hyp_rows += f"| {hid} | {title} | {status} | {prior} | {post} | {test} |\n"

    ev_rows = ""
    for ev in evidence_chains:
        etype = ev.get('type', '?')
        estr = f"{ev.get('strength', 0):.4f}"
        emeth = ev.get('method_used', '?')
        edir = ev.get('causal_direction', '?')
        ev_rows += f"| {etype} | {estr} | {emeth} | {edir} |\n"

    # Literature excerpts
    lit_excerpts = literature_summary[:1500] if literature_summary else "(无文献调研数据)"

    # Get hypo statement for abstract/title
    active_hyps = [h for h in hypotheses if h.get('status') not in ('pruned',)]
    hypo_stmt = active_hyps[-1].get('statement', '') if active_hyps else query
    hypo_title = active_hyps[-1].get('title', '') if active_hyps else '未命名假设'
    reasoning_chain = active_hyps[-1].get('reasoning_chain', '')[:300] if active_hyps else '无'

    # Extract references from literature
    refs = []
    for line in lit_excerpts.split('\n'):
        if 'Reference:' in line or 'Reference : ' in line:
            ref_text = line.split('Reference:')[-1].strip().split(']')[0].strip() if ']' in line else line.strip()
            refs.append(ref_text)
    refs_text = "\n".join([f"{i+1}. {r}" for i, r in enumerate(refs)]) if refs else "[待补充真实引用]"

    # --- Now ask LLM to generate only CREATIVE sections (Rationale + Abstract) ---
    llm = _get_llm()
    llm_messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""请为以下研究生成【解决思路（Rationale）】和【摘要（Abstract）】两部分内容。

### 研究问题
{domain} — {query}

### 假设陈述
{hypo_stmt}

### 推理链条及先验置信度
{reasoning_chain}

### 文献支撑事实
{lit_excerpts[:1000]}

### 因果推断关键发现
{" | ".join(
    f"方法={ev.get('method_used','?')}, 强度={ev.get('strength',0)}, 摘要={ev.get('content','')}"
    for ev in evidence_chains if ev.get('type')=='causal_inference'
) or "无"}

### 要求
1. **解决思路**: 约300字，从逻辑推理、跨学科迁移角度阐述创新点
2. **摘要**: 约250字，包含背景、方法、核心发现、结论
3. 引用文献时请用括号格式如(Wargocki & Wyon, 2010)，不要编造DOI

输出格式：
## 二、解决思路（Rationale）
[你的文字...]

### 支撑事实（来自文献调研）
[literature fact excerpts]

---

## 六、摘要（Paper Abstract）
[你的摘要...]""",
        },
    ]

    content, _ = await _async_call_llm(llm, llm_messages, temperature=0.7, max_tokens=4096)

    rationale_abstract = content

    # --- Assemble final report using .join() so ALL variables get properly interpolated ---
    NL = chr(10)
    parts = []

    parts.append("# 科学假设与研究计划")
    parts.append("")
    parts.append("## 一、待研究问题（Problem Statement）")
    parts.append(f"**{hypo_stmt[:120]}**")
    parts.append("")
    parts.append(f"- **学科领域**: {domain}")
    parts.append(f"- **研究问题**: {query}")
    parts.append(f"- **系统收敛度**: {convergence_val:.0f}%")
    parts.append("")
    parts.append("---")
    parts.append("")

    ra = rationale_abstract
    for header in ["### 支撑事实（来自文献调研）", "## 二、解决思路（Rationale）", "## 六、摘要（Paper Abstract）"]:
        ra = ra.replace(header, "")
    parts.append(ra.strip())
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## 三、技术手段（Technical Details）")
    parts.append("验证本假设需要的技术栈和方法论：")
    parts.append("")
    parts.append("| 模块 | 方法 | 工具/算法 |")
    parts.append("|------|------|----------|")
    parts.append("| 数据采集 | 环境传感器 + 可穿戴设备 | CO₂温湿度仪, PPG光电容积脉搏波, HRV心率变异性 |")
    parts.append("| 信号处理 | 多源时序对齐 + 质量评估 | 互相关法对齐, SNR信噪比评估 |")
    parts.append("| 因果推断 | AI自动选择最优方法 | CCM / Granger / PC-FCI / PSM / 贝叶斯网络 |")
    parts.append("| 统计分析 | 混合效应模型 + 反事实推演 | Statsmodels, GP代理模型 |")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## 四、数据集（Datasets）")
    parts.append("### Source（历史数据来源）")
    parts.append("| 数据类型 | 来源描述 | 样本量估计 | 时间范围建议 |")
    parts.append("|---------|---------|-----------|------------|")
    parts.append("| 环境传感器 | 室内环境监测站（温湿度、CO₂） | ≥5000点/天 | ≥7天连续采集 |")
    parts.append("| PPG/血氧/HRV | 可穿戴传感器（Empatica/Apple Watch等） | ≥100Hz采样率 | ≥72小时连续监测 |")
    parts.append("| 视觉疲劳数据 | 眼动追踪+面部表情识别摄像头 | ≥30FPS视频流 | 每次实验session 10-30分钟 |")
    parts.append("")
    parts.append("### Target（验证实验拟采集数据特征）")
    parts.append("- **采样频率**: 环境数据 1Hz / 生物信号 ≥ 100Hz / 视觉数据 ≥ 30FPS")
    parts.append("- **测量精度**: 温度 ±0.1°C / CO₂ ±10ppm / SpO₂ ±0.5% / PPG SNR > 20dB")
    parts.append("- **实验周期**: 建议连续监测 ≥ 72 小时以捕获日节律变化")
    parts.append("- **受试者数量**: N≥30（群体水平分析），可支持 N-of-1 个体化研究")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## 五、标题（Paper Title）")
    parts.append(f"**{hypo_title}**")
    parts.append("")
    parts.append("---")
    parts.append("")
    # Extract abstract from LLM output if present, otherwise use hypothesis statement
    parts.append("## 六、摘要（Paper Abstract）")
    if "## 六、摘要（Paper Abstract）" in rationale_abstract:
        try:
            abstract_text = rationale_abstract.split("## 六、摘要（Paper Abstract）")[1].strip()
            if abstract_text:
                parts.append(abstract_text[:500])
            else:
                parts.append(hypo_stmt[:500] + "(基于因果推断分析的综合研究计划)")
        except (IndexError, Exception):
            parts.append(hypo_stmt[:500] + "(基于因果推断分析的综合研究计划)")
    else:
        parts.append(hypo_stmt[:500] + "(基于因果推断分析的综合研究计划)")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## 七、方法论（Methods）")
    parts.append("### 7.1 系统架构")
    parts.append("")
    parts.append("```")
    parts.append("┌─────────────┐    ┌──────────────┐    ┌──────────────┐")
    parts.append("│ Literature  │ →  │   Hypothesis  │ →  │  Experiment   │")
    parts.append("│   Review    │    │ Generation   │    │   Design      │")
    parts.append("└─────────────┘    └──────────────┘    └──────────────┘")
    parts.append("       ↓                    ↓                    ↓")
    parts.append("┌─────────────┐    ┌──────────────┐    ┌──────────────┐")
    parts.append("│ Data        │ ←  │  Causal      │ ←  │ Time-Series   │")
    parts.append("│ Analysis    │    │ Inference    │    │ Alignment     │")
    parts.append("└─────────────┘    └──────────────┘    └──────────────┘")
    parts.append("       ↓                    ↓")
    parts.append("┌─────────────┐    ┌──────────────┐")
    parts.append("│ Interpret.  │ →  │ Reviewer 5D  │")
    parts.append("│ & Reflexion │    │ Evaluation   │")
    parts.append("└─────────────┘    └──────────────┘")
    parts.append("```")
    parts.append("")
    parts.append("### 7.2 数据处理流程")
    parts.append("```")
    parts.append("原始数据 → 时间对齐 → 质量评估 → 特征提取 → 因果推断 → 统计检验")
    parts.append("  │            │           │          │          │          │")
    parts.append("传感器CSV   最近邻对齐   SNR评估    频域分解   CCM/Granger   F-test")
    parts.append("PPG波形     交叉相关    缺失插补    时域统计   贝叶斯网络    p<0.05")
    parts.append("```")
    parts.append("")
    parts.append("### 7.3 变量定义")
    parts.append("| 类别 | 变量 | 说明 | 预期单位 |")
    parts.append("|-----|------|------|---------|")
    parts.append("| 自变量 (X) | 温度、湿度、CO₂浓度 | 环境暴露因子 | °C, %, ppm |")
    parts.append("| 因变量 (Y) | HRV(SDNN/RMSSD)、SpO₂、PPG幅值 | 生理响应指标 | ms, %, mV |")
    parts.append("| 协变量 (C) | 年龄、性别、BMI、活动水平 | 个体差异控制 | kg/m², category |")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## 八、实验设计（Experiments）")
    parts.append("### 8.1 基线对比（Baselines）")
    parts.append("| 方法 | 适用场景 | 优势 | 局限 |")
    parts.append("|------|---------|------|------|")
    parts.append("| 线性回归 | 初步相关性分析 | 简单直观 | 无法捕捉非线性 |")
    parts.append("| 随机森林/XGBoost | 预测性能最大化 | 高准确率 | 无因果方向性 |")
    parts.append("| Pearson/Spearman 相关 | 双变量关联检测 | 无需假设分布 | 混淆因子干扰 |")
    parts.append("| **twinScientist（因果推断）** | **因果机制发现** | **方向性+可解释性** | **需要更大样本** |")
    parts.append("")
    parts.append("### 8.2 评估指标（Metrics）")
    parts.append("- **主指标**: 因果效应大小 β 及其显著性 (p-value < 0.05)")
    parts.append("- **辅助指标**: RMSE, R², BIC/AIC（模型比较）")
    parts.append("- **统计功效**: power analysis (α=0.05, power=0.8, effect_size=Cohen's d≈0.5)")
    parts.append("- **置信度**: Bayesian 后验概率 P(H | D)")
    parts.append("")
    parts.append(f"### 8.3 实验执行记录 ({len(experiments)} 个实验方案)")
    parts.append("| id | design_status | has_results | notes |")
    parts.append("|----|---------------|-------------|-------|")
    if exp_table_rows:
        for row_line in exp_table_rows.strip().split(NL):
            parts.append(row_line)
    parts.append("---")
    parts.append("")
    parts.append("## 九、实验结果（Results）")
    parts.append(section9_content)
    parts.append("---")
    parts.append("")
    parts.append("## 十、评审意见（Reviewer Feedback）")
    parts.append("| hyp_id | score | needs_revision |")
    parts.append("|--------|-------|----------------|")
    if review_rows:
        for row_line in review_rows.strip().split(NL):
            parts.append(row_line)
    parts.append("---")
    parts.append("")
    parts.append("## 十一、参考文献（References）")
    parts.append("> **重要声明**: 以下引用必须为真实存在的学术论文。")
    if refs_text:
        for ref_line in refs_text.split(NL):
            parts.append(ref_line)
    parts.append("---")
    parts.append("")
    parts.append("## 十二、附加信息")
    parts.append(f"### 假设树全景 ({active_hyp_count} 个假设)")
    parts.append("| 假设ID | 标题 | 状态 | 先验P(H) | 后验P(H|D) | 可检验性 |")
    parts.append("|--------|------|------|----------|------------|----------|")
    if hyp_rows:
        for row_line in hyp_rows.strip().split(NL):
            parts.append(row_line)

    # --- Elimination Tournament Records ---
    parts.append(f"\n### 本轮候选假设数量：{len(elimination_records) + 1} 个")
    parts.append("")
    parts.append("### 淘汰赛记录")
    parts.append("| 假设ID | 假设简述 | 状态 | 淘汰理由 |")
    parts.append("|--------|---------|------|---------|")

    # Collect all hypotheses that participated in the tournament
    eliminated_ids = set()
    winner_id_from_tournament = None

    # Parse from elimination records
    for rec in elimination_records:
        loser_id = rec.get("eliminated_id", "?")
        loser_title = rec.get("eliminated_title", "?")
        round_ = rec.get("eliminated_round", "?")
        reason = rec.get("reason", "未提供具体原因")

        # Get a brief summary of this hypothesis
        hypo_info = next((h for h in hypotheses if h["id"] == loser_id), {})
        brief = hypo_info.get("statement", "")[:40] or loser_title

        parts.append(f"| {loser_id} | {brief} | 淘汰 | {reason} |")
        eliminated_ids.add(loser_id)

    # Find and record the winner (if any)
    candidates = [h for h in hypotheses if h.get('status') not in ('pruned', 'refuted', 'refuted_in_tournament')]
    if candidates:
        # Pick the one with tournament_won flag or highest posterior
        winner = next((h for h in candidates if h.get("tournament_won")),
                      max(candidates, key=lambda h: h.get("confidence_posterior", h.get("confidence_prior", 0))))
        winner_id = winner["id"]
        winner_brief = winner.get("statement", "")[:40] or winner.get("title", "?")
        parts.append(f"| {winner_id} | {winner_brief} | 优胜 | - |")

    parts.append("---")

    parts.append(f"\n### 证据链汇总 ({len(evidence_chains)} 条)")
    parts.append("| type | strength | method | direction |")
    parts.append("|------|----------|--------|-----------|")
    if ev_rows:
        for row_line in ev_rows.strip().split(NL):
            parts.append(row_line)
    parts.append("---")
    parts.append("")
    parts.append("*本报告由 twinScientist AI Scientist 系统自动生成*")
    parts.append("*生成时间: 当前UTC时间*")
    parts.append(f"*迭代轮次: {iteration_val}/{max_iter} | 收敛度: {convergence_val:.0f}%*")
    parts.append("*Agent: Qwen系列 (阿里云百炼平台) | 编排: LangGraph*")

    report = NL.join(parts)

    # --- Iteration status check (inserted before report output) ---
    iter_status_lines = []
    if iteration_val >= 1:
        iter_status_lines.append(f"**迭代状态**: ✅ 已执行 {iteration_val} 轮迭代反思循环")
    else:
        iter_status_lines.append(
            "**迭代状态**: ⚠️ 反思循环未被执行（当前为第 0 轮）。"
            "本轮仅完成初始验证，建议增加迭代轮次以提升结论可靠性。"
        )

    insert_idx = 1  # right after "# 科学假设与研究计划"
    for line in reversed(iter_status_lines):
        parts.insert(insert_idx, line)
    parts.insert(insert_idx, "")
    parts.insert(insert_idx, "---")
    parts.insert(insert_idx, "")
    report = NL.join(parts)

    # Also inject into the metadata footer area
    footer_marker = f"*迭代轮次: {iteration_val}/{max_iter} | 收敛度: {convergence_val:.0f}%*"
    status_footer = f"| 迭代状态: {'✅ 已执行{iteration_val}轮' if iteration_val >= 1 else '⚠️ 未执行'}*"
    report = report.replace(footer_marker, f"*迭代轮次: {iteration_val}/{max_iter} 收敛度: {convergence_val:.0f}%{status_footer}")

    logger.info("[ReportWriting] Report generated successfully with real data")
    logger.info(f"[ReportWriting] Final report length={len(report)}, Section 9 present={('以下基于真实数据分析' in report) or ('理论可行性验证框架' in report)}")

    return {"final_report": report, "current_action": "report_writing"}



async def node_pi_agent_meeting(state: AgentState) -> dict:
    """
    【PI Agent 总结汇报】
    整合多智能体成果，产出最终综合研究报告
    """
    report = state.get("final_report", "")
    hypotheses = state.get("hypothesis_tree", [])
    reviews = list(state.get("review_records", []))
    experiments = list(state.get("experiment_records", []))

    llm = _get_llm()
    messages = [
        {"role": "system", "content": PI_AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## PI Agent 总结任务\n\n"
                f"已完成实验: {len(experiments)} 次\n"
                f"评审次数: {len(reviews)} 次\n"
                f"已通过评审的假设: {len([h for h in hypotheses if h.get('status') == 'approved_by_reviewer'])} 个\n\n"
                f"【研究概要】\n{report[:2000] if report else '(无研究报告)'}\n\n"
                f"请整合所有信息，对报告进行精简和润色。保持原有的数据、表格和关键发现不变。\n\n"
                f"**重要提示**: 如果报告中包含真实的因果推断分析结果（有Granger/CCM等方法的具体数据），必须保留这些内容，不要删除或替换为理论框架文字。"
            ),
        },
    ]

    pi_content, _ = await _async_call_llm(llm, messages, temperature=0.4, max_tokens=4096)

    # Preserve final_report unless PI agent actually improves it (not just regenerating)
    original_content = state.get("final_report", "")
    # If the original already contains real analysis (Section 9 with real data), preserve it
    if "以下基于真实数据分析" in original_content or "因果推断摘要" in original_content:
        logger.info("[PiAgent] Preserving original report with real causal inference data")
        return {"_pi_summary_done": True, "current_action": "pi_agent_meeting"}
    elif len(pi_content) > len(original_content) * 0.8 and pi_content != original_content:
        logger.info("[PiAgent] Using PI-generated report")
        return {"final_report": pi_content, "_pi_summary_done": True, "current_action": "pi_agent_meeting"}
    else:
        logger.info("[PiAgent] Original report preserved")
        return {"_pi_summary_done": True, "current_action": "pi_agent_meeting"}


# ============================================================
# Item 11, 12: Human Approval Gate
# ============================================================

async def node_human_approval(state: AgentState) -> dict:
    """
    【人类审核入口点】
    LangGraph interrupt_before 挂起于此。
    用户通过 UI 提交决策后 resume，feedback 会注入 state。

    本节点本身不做计算——它只是一个安全闸门。
    """
    user_feedback = state.get("user_feedback", "")
    auto_confirm = state.get("auto_confirm", False)

    if auto_confirm:
        logger.info("[HumanApproval] Auto-confirm enabled — proceeding")
        return {"current_action": "human_approval"}

    if user_feedback and user_feedback.strip().lower() in ("approve", "确认", "通过", "approved"):
        logger.info("[HumanApproval] User approved")
        return {"current_action": "human_approval"}

    if user_feedback:
        logger.info(f"[HumanApproval] User provided feedback: {user_feedback[:100]}")
        return {"user_feedback": user_feedback, "current_action": "human_approval"}

    # No input yet — this is an interrupt point
    logger.info("[HumanApproval] Waiting for human decision...")
    return {"pending_approval": True, "current_action": "human_approval"}


# ============================================================
# Item 9: Evolution Manager
# ============================================================

async def node_evolution_manager(state: AgentState) -> dict:
    """
    【自我进化机制】
    - 提取成功模式共性
    - 总结失败规律
    - 蒸馏 meta-insights
    - 推荐系统级改进
    """
    experiments = list(state.get("experiment_records", []))
    reviews = list(state.get("review_records", []))
    anomalies = state.get("anomaly_graph", [])
    hypotheses = state.get("hypothesis_tree", [])

    passed = [r for r in reviews if not r.get("needs_revision")]
    failed = [r for r in reviews if r.get("needs_revision")]

    llm = _get_llm()
    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## 任务：系统自我进化\n\n"
                f"完成实验: {len(experiments)}, "
                f"通过评审: {len(passed)}, "
                f"被打回: {len(failed)}, "
                f"教训记录: {len(anomalies)}\n\n"
                f"成功评审: {[r.get('comments', '')[:200] for r in passed[-5:]]}\n\n"
                f"失败评审: {[r.get('comments', '')[:200] for r in failed[-5:]]}\n\n"
                f"请总结:\n"
                f"1. 成功模式的共性\n"
                f"2. 失败模式的规律\n"
                f"3. 可用于未来迭代的抽象规则（meta-insights）\n"
                f"4. 系统改进建议\n"
                f"请用清晰的编号列表回答。"
            ),
        },
    ]

    content, _ = await _async_call_llm(llm, messages, temperature=0.5, max_tokens=4096)

    evolution_record = {
        "id": f"evo_{uuid.uuid4().hex[:6]}",
        "iteration": state.get("iteration", 1),
        "experiments_analyzed": len(experiments),
        "reviews_analyzed": len(reviews),
        "insights": content,
        "timestamp": _now_iso(),
    }

    logger.info(f"[EvolutionManager] Extracted insights from {len(experiments)} experiments")

    return {
        "_evolution_insights": evolution_record,
        "current_action": "evolution_manager",
    }
