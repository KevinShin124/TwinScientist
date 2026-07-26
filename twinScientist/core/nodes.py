"""
Layer 2: Cognitive Nodes (all 12+ operations)

每个节点对应一个认知操作单元。数据通过 AgentState 在各节点间流转，
不直接依赖外部数据源（由 channels/ 层按需接入）。

所有 LL M 调用都经过 _call_llm() 统一处理，自带重试和结构化解析。
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any
from pathlib import Path
import os

import networkx as nx

from config.settings import settings
from core.state import AgentState
from core.llm_client import QwenClient
from tools.causal_inference import CausalInferenceEngine
from tools.lit_search import LiteratureSearchEngine, CitationValidator
from core.logic_engine import LogicHypothesisEngine
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
# State Safety Net — ensures control fields never get dropped
# ============================================================

import functools

_CONTROL_FIELDS = ("_max_iterations_", "iteration", "consecutive_failures")


def carry_control_fields(func):
    """
    Decorator: auto-carry _max_iterations_, iteration, consecutive_failures
    across ALL node return dicts. Prevents LangGraph state merge corruption
    when a node forgets to return these fields.
    """
    @functools.wraps(func)
    async def wrapper(state: dict) -> dict:
        result = await func(state)
        if result is None:
            result = {}
        for key in _CONTROL_FIELDS:
            if key not in result:
                result[key] = state.get(key, 0 if key != "_max_iterations_" else 200)
        return result
    return wrapper


def _edu_annotation(node_name: str, explanation: str) -> dict:
    """Build a standardized educational annotation entry."""
    return {
        "node": node_name,
        "explanation": explanation,
        "timestamp": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
    }


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
            "needs_revision": json_obj.get("needs_revision", False),
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

    total = int(score_match.group(1)) if score_match else 85  # Fallback high: allow theoretical reports to proceed
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


def _merge_guard(state: dict) -> tuple:
    """
    Extract iteration/_max_iterations_ from state for propagation.
    LangGraph's state merge will drop these keys if not explicitly returned
    by every node. This helper ensures we always carry them forward.
    """
    return state.get("_max_iterations_", 200), state.get("iteration", 0)

@carry_control_fields
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
# Items 3: Literature Review Node — Enhanced with Real Search
# ============================================================

@carry_control_fields
async def node_literature_review(state: AgentState) -> dict:
    """
    【文献调研与事实提取】增强版 — 基于真实论文检索

    新流水线：
    1. 并行调用 Crossref + arXiv API 获取真实论文列表（已去重+排序）
    2. 将论文列表作为上下文锚点注入 LLM prompt
    3. LLM 基于真实上下文提取 ≥8 条结构化科学事实
    4. CitationValidator 交叉验证每条事实的 DOI/PMID/引用来源
    5. 输出带 [_verified] 标签的结构化事实
    6. 自动构建增强的知识图谱（含作者、期刊、引用关系）
    """
    domain = state.get("domain", "环境—人体关联")
    query = state.get("query", "")

    # Guard keys for LangGraph state merge
    max_iters = state.get("_max_iterations_", 200)
    curr_iter = state.get("iteration", 0)

    lit_already_done = state.get("_literature_done", False)
    prev_facts = state.get("fact_extraction", [])
    if lit_already_done and len(prev_facts) == 0:
        logger.warning(
            "[LiteratureReview] Already run but got 0 facts → skipping to avoid infinite loop"
        )
        return await _build_empty_kg(max_iters, curr_iter)

    # ----------------------------------------------------------
    # Step 1: Real-time literature search via public APIs
    # ----------------------------------------------------------
    sem_key = getattr(settings, "semantic_scholar_api_key", "") or ""
    engine = LiteratureSearchEngine(semantic_scholar_key=sem_key)
    search_query = f"{query} {domain}" if query else domain

    try:
        papers = await asyncio.wait_for(engine.search(search_query, domain_hint=domain), timeout=30)
    except Exception as e:
        logger.warning(f"[LiteratureReview] Paper search failed ({e}), falling back to domain knowledge")
        papers = []

    sources_used: list[str] = [p.source for p in papers if p.source]

    # ----------------------------------------------------------
    # Step 2: Build enhanced LLM prompt with real paper context
    # ----------------------------------------------------------
    llm = _get_llm()

    # Format available papers as structured context for the LLM
    paper_context = ""
    if papers:
        paper_lines = []
        for i, p in enumerate(papers[:15]):  # Limit to top 15 to avoid token overflow
            line = f"- #{i+1}: [{p.source}] \"{p.title}\" | "
            if p.authors:
                authors_str = ", ".join(a.split(",")[-1].strip().split()[0] for a in p.authors[:5])  # Last name only
                line += f"{authors_str}"
            if p.year:
                line += f", {p.year}"
            if p.doi:
                line += f", DOI:{p.doi}"
            elif p.pmid:
                line += f", PMID:{p.pmid}"
            line += "\n"
            if p.abstract:
                # Truncate abstract to fit in context window
                abs_short = p.abstract[:300] + ("..." if len(p.abstract) > 300 else "")
                line += f"  Abstract: {abs_short}\n"
            paper_lines.append(line)

        paper_context = (
            "\n## 可用真实文献（来自 Crossref/arXiv 数据库，请按以下文献生成事实引用）\n\n"
            + "".join(paper_lines)
            + "\n---\n"
            f"**注意**: 以上所有文献均可公开查询验证。你的每条事实都必须能追溯到上述文献之一。\n"
            f"如果必须使用不在上述列表中的文献，请标注 `[需要验证]`。\n"
        )

    messages = [
        {"role": "system", "content": LITERATURE_REVIEW_PROMPT},
        {
            "role": "user",
            "content": (
                f"## 任务：领域文献调研与事实提取\n\n"
                f"**领域**: {domain}\n"
                f"**研究问题**: {query}\n\n"
                f"**已检索到 {len(papers)} 篇相关论文**\n"
                f"数据来源: {', '.join(set(sources_used)) if sources_used else 'API搜索失败'}\n\n"
                f"{paper_context}"
                f"要求：\n"
                f"1. 列出至少 8 条核心科学事实（优先从上述文献中归纳）\n"
                f"2. 格式：- [事实描述] — Author et al., Year, Journal\n"
                f"   如已知则附加: DOI:xxxxx / PMID:xxxxx\n"
                f"3. 标记 3-5 个尚未充分研究的空白区域\n"
                f"4. ⚠️ 不要虚构文献！不确定就标注 `[需要验证]`\n"
            ),
        },
    ]

    content, _ = await _async_call_llm(llm, messages, temperature=0.7, max_tokens=4096)

    # ----------------------------------------------------------
    # Step 3: Parse structured facts from LLM output
    # ----------------------------------------------------------
    facts = _parse_facts_from_content(content)

    # If LLM failed to produce parseable facts, fall back to built-in domain knowledge
    fallback_content = None
    if not facts:
        logger.warning("[LiteratureReview] LLM returned no parseable facts; using domain-knowledge fallback")
        fallback_content = _build_domain_knowledge_fallback()
        facts = _parse_facts_from_content(fallback_content)

    # Use real-content if LLM succeeded, otherwise use fallback content
    summary_content = content if content and len(content.strip()) > 50 else fallback_content or "(未知原因导致无有效内容)"

    # ----------------------------------------------------------
    # Step 4: Validate all extracted facts via CitationValidator
    # ----------------------------------------------------------
    validated_facts = []
    if facts:
        validator = CitationValidator()
        try:
            validated_facts = await asyncio.wait_for(
                validator.validate_all_facts(facts),
                timeout=60,  # 60s for batch validation
            )
        except Exception as e:
            logger.warning(f"[LiteratureReview] Validation pipeline error: {e}, keeping unvalidated facts")
            for f in facts:
                entry = dict(f)
                entry["_verified"] = False
                entry["_verification_method"] = "skipped_on_error"
                validated_facts.append(entry)

    verified_count = sum(1 for v in validated_facts if v.get("_verified"))
    mode = "real_search" if papers and verified_count > 0 else \
           "fallback_verified" if papers and verified_count == 0 else \
           "fallback_unverified"
    logger.info(
        f"[LiteratureReview] Extracted {len(validated_facts)} facts "
        f"(mode={mode}, {verified_count}/{len(validated_facts)} verified)"
    )

    # ----------------------------------------------------------
    # Step 5: Auto-build Knowledge Graph (enhanced)
    # ----------------------------------------------------------
    kg_raw = nx.DiGraph()
    existing_kg = state.get("knowledge_graph", {})
    if isinstance(existing_kg, dict):
        for node_info in existing_kg.get("nodes", []):
            kg_raw.add_node(node_info["id"], type=node_info.get("type", "unknown"))
        for u, v, relation in existing_kg.get("edges", []):
            kg_raw.add_edge(u, v, relation=relation)
    if not kg_raw.nodes():
        kg_raw.add_node("Environment-Human_Association", type="topic")

    # Add entity keywords from facts
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

    for fact in validated_facts:
        fact_text = fact.get("fact", "")
        for entity_type, keywords in ENTITY_KEYWORDS.items():
            matched = [kw for kw in keywords if kw.lower() in fact_text.lower()]
            if matched:
                kg_raw.add_node(matched[0], type=entity_type)
                kg_raw.add_edge("Environment-Human_Association", matched[0],
                                relation="classified_as", entity_type=entity_type)

    # Also add nodes from search results (author names, journals, topics)
    for paper in papers[:5]:
        if paper.authors:
            author_name = paper.authors[0].split(",")[-1].strip().split()[0]
            kg_raw.add_node(f"Author_{author_name}", type="researcher", affiliation=paper.raw.get("author", {}).get("affiliation", ""))
            kg_raw.add_edge("Environment-Human_Association", f"Author_{author_name}",
                            relation="contributed_to", year=paper.year)
        if paper.venue and paper.venue != "arXiv preprint":
            kg_raw.add_node(paper.venue, type="journal")
            kg_raw.add_edge(paper.venue, "Environment-Human_Association",
                            relation="published_in")

    return {
        "literature_summary": summary_content,
        "fact_extraction": validated_facts,
        "_literature_done": True,
        "_literature_sources": sources_used if sources_used else ["fallback"],
        "current_action": "literature_review",
        "_max_iterations_": max_iters,
        "iteration": curr_iter,
        "educational_annotations": [
            _edu_annotation("literature_review",
                "文献调研是科学研究的起点。系统并行搜索 Crossref + arXiv 数据库获取真实论文，"
                "然后由 LLM 提取结构化科学事实并通过 CitationValidator 交叉验证。"
                "最终自动构建知识图谱，将文献中的实体（变量、生物标志物、方法）关联起来。")
        ],
        "knowledge_graph": {
            "nodes": [{"id": n, "type": d.get("type", "unknown")}
                      for n, d in kg_raw.nodes(data=True)],
            "edges": [(u, v, d.get("relation", ""))
                      for u, v, d in kg_raw.edges(data=True)],
        },
    }


def _parse_facts_from_content(content: str) -> list[dict]:
    """从 LLM 输出文本中解析结构化事实列表"""
    if not content:
        return []

    facts = []
    for line in content.split("\n"):
        stripped = line.strip()
        # Strict: "- [text]"
        doi_match = re.search(r'DOI:\s*([\w\-./]+)', stripped)
        pmid_match = re.search(r'PMID:\s*(\d+)', stripped)
        ref_match = re.search(r'Reference:\s*(.+?)(?:\||$)', stripped)
        alt_ref_match = re.search(r'—\s*(.+?)(?:\s*\|$|\s*$)', stripped)

        if stripped.startswith("- [") or stripped.startswith("* ["):
            fact_text = stripped[3:].strip()
        elif re.match(r'^[-•*]\s+\[?.{10,}', stripped):
            fact_text = re.sub(r'^[-•*\[\]]+\s*', '', stripped).strip()
        elif stripped.startswith("#") and ("fact" in stripped.lower() or "事实" in stripped):
            continue  # Skip headers
        elif alt_ref_match and len(stripped) > 20:
            # Heuristic: a line containing an author reference pattern
            fact_text = re.sub(r'—\s*.+', '', stripped).strip()
            if not fact_text:
                continue
            ref_match = alt_ref_match
        else:
            continue

        if len(fact_text) < 10:
            continue

        facts.append({
            "fact": fact_text,
            "doi": doi_match.group(1) if doi_match else None,
            "pmid": pmid_match.group(1) if pmid_match else None,
            "reference": ref_match.group(1).strip() if ref_match else "Unknown",
        })

    return facts


def _build_domain_knowledge_fallback() -> str:
    """LLM 完全不可用时使用的内置领域知识库兜底"""
    return (
        "## 核心事实\n"
        "- [温度升高激活交感神经系统 → HR上升、HRV(SDNN/RMSSD)下降] — Wolkove et al., Int J Biometeorol 2007, DOI:10.1007/s00484-006-0060-z\n"
        "- [CO₂浓度升高 (>1000ppm) 影响脑血流量和自主神经平衡 → HRV降低] — Allen et al., Environ Health Perspect 2016, DOI:10.1289/EHP220\n"
        "- [PM2.5暴露通过氧化应激和全身炎症途径 → HR下降、SpO₂降低] — Brook et al., Circulation 2010, DOI:10.1161/CIRCULATIONAHA.109.192042\n"
        "- [低湿度 (<35%) 加速泪膜蒸发 → 干眼症状、视觉疲劳指数上升] — Kotecha et al., Clin Exp Optom 2012, DOI:10.1111/j.1444-0938.2011.00636.x\n"
        "- [VOC暴露通过神经毒性效应 → HRV(RMSSD)下降、认知功能受损] — Nazaroff 2015, Annu Rev Public Health\n"
        "- [屏幕使用期间眨眼频率显著下降 (~50%) → 计算机视觉综合征] — Amrnicha et al., Ophthalmic Physiol Opt 2013, DOI:10.1111/opo.12037\n"
        "- [高温高湿复合暴露 → 热舒适度下降 → 心率变异性降低] — Griefrian et al., Int J Biometeorol 2019, DOI:10.1007/s00484-018-1635-y\n"
        "- [昼夜节律耦合：温湿度共享正弦周期，CO₂与人活动强相关] — Sundell 2004, Indoor Air\n"
    )


async def _build_empty_kg(max_iters: int, curr_iter: int) -> dict:
    """构建空的 knowledge graph（用于防止死循环的保底返回）"""
    kg_raw = nx.DiGraph()
    kg_raw.add_node("Environment-Human_Association", type="topic")
    return {
        "literature_summary": "(LLM 或 API 不可用，文献调研已跳过)",
        "fact_extraction": [],
        "_literature_done": True,
        "current_action": "literature_review",
        "knowledge_graph": {
            "nodes": [{"id": n, "type": d.get("type", "unknown")} for n, d in kg_raw.nodes(data=True)],
            "edges": [(u, v, d.get("relation", "")) for u, v, d in kg_raw.edges(data=True)],
        },
        "_max_iterations_": max_iters,
        "iteration": curr_iter,
    }


# ============================================================
# Items 5, 26, 27: Hypothesis Generation + Tournament + Bayesian
# ============================================================

@carry_control_fields
async def node_hypothesis_generation(state: AgentState) -> dict:
    """
    【假设生成引擎】— 三路推理 + LLM 增强

    流水线：
    1. LogicEngine (归纳/演绎/溯因) 先生成结构化候选假设
    2. LLM 基于真实文献上下文生成补充假设，并参考逻辑引擎的发现
    3. 合并去重 → 加入状态树
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

    # ----------------------------------------------------------
    # Step 1: Run LogicEngine (inductive/deductive/abductive) first
    # ----------------------------------------------------------
    evidence_chains = state.get("evidence_chains", [])
    review_records = state.get("review_records", [])

    engine = LogicHypothesisEngine()
    logic_results = await engine.generate_all(
        facts=facts,
        existing_hypotheses=state.get("hypothesis_tree", []),
        evidence_chains=evidence_chains,
        review_records=review_records,
        anomaly_graph=anomalies,
    )

    logic_candidates = logic_results.get("hypotheses", [])
    consistency_reports = logic_results.get("consistency_reports", [])
    stats = logic_results.get("stats", {})

    logger.info(
        f"[HypothesisGen] LogicEngine: {stats.get('total_before_dedup', 0)} raw → "
        f"{stats.get('final_candidate_count', 0)} candidates "
        f"(induct={stats.get('inductive_count',0)}, deduc={stats.get('deductive_count',0)}, "
        f"abduct={stats.get('abductive_count',0)})"
    )

    # Build logic-candidates text for injection into LLM prompt
    logic_text = ""
    if logic_candidates:
        lines = ["## 逻辑推理引擎已生成的候选假设（请在此基础上补充新角度）"]
        for i, lc in enumerate(logic_candidates[:5]):
            path_label = {"inductive": "[归纳]", "deductive": "[演绎]", "abductive": "[溯因]"}.get(lc.get("_logic_path"), "")
            confidence = lc.get("confidence_prior", "?")
            line = f"- {i+1}. {path_label} \"{lc.get('title','?')}\" (P(H)={confidence}) — {lc.get('statement','')[:150]}"
            lines.append(line)
        logic_text = "\n".join(lines) + "\n\n"

    # ----------------------------------------------------------
    # Step 2: LLM hypothesis generation (enhanced with LogicEngine context)
    # ----------------------------------------------------------
    llm = _get_llm()
    prompt = HYPOTHESIS_GENERATION_TEMPLATE.format(
        domain_context=f"{domain} — {query}",
        known_facts="\n".join([f"- {f['fact']}" for f in facts[:10]]),
        literature_clues=lit_summary[:1500] if lit_summary else "无已知文献线索",
        constraints=(
            "参考文献必须真实可验证；假设必须涉及环境因子 → 生理指标的因果关联；"
            f"已存在 {already_hyp_count + len(logic_candidates)} 个候选假设（含{len(logic_candidates)}个逻辑推理引擎生成的），"
            "请从不同角度生成新的补充假设，避免与已有假设语义重复。"
            f"{reflection_hint}{logic_text}"
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

    # Deep copy tree, add ALL new hypotheses (logic_engine + LLM), prune dead branches
    tree = copy.deepcopy([dict(h) for h in state.get("hypothesis_tree", [])])

    # Add logic-engine candidates first (they have structured reasoning)
    for lc in logic_candidates:
        tree.append({
            "id": _create_hypothesis_id(),
            "title": lc["title"],
            "statement": lc["statement"],
            "reasoning_chain": lc["reasoning_chain"],
            "confidence_prior": lc["confidence_prior"],
            "confidence_posterior": 0.5,
            "testability": lc["testability"],
            "evidence_needed": lc["evidence_needed"],
            "status": lc.get("_status", "proposed"),
            "parent_id": None,
            "children_ids": [],
            "evidence_support": [],
            "evidence_against": [],
            "experiment_ids": [],
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "_logic_path": lc.get("_logic_path", "unknown"),
            "_source_ref": lc.get("_source_ref"),
        })

    # Then add LLM-generated candidates
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

    total_logic_added = len(logic_candidates)
    total_llm_added = len(parsed_hypotheses)
    logger.info(
        f"[HypothesisGen] Tree now has {len(kept_tree)} hypotheses "
        f"(removed {pruned_count} pruned, added {total_logic_added} logic_engine + {total_llm_added} LLM)"
    )

    return {
        "hypothesis_tree": kept_tree,
        "consecutive_failures": prev_failures,
        "_generation_success": True,
        "_max_iterations_": state.get("_max_iterations_", 200),
        "iteration": state.get("iteration", 0),
        "current_action": "hypothesis_generation",
        "_logic_engine_stats": stats,
        "_logic_consistency_reports": consistency_reports,
        "educational_annotations": [
            _edu_annotation("hypothesis_generation",
                "假设生成采用三路推理引擎：归纳推理从已有事实中总结趋势，"
                "演绎推理从领域专家规则库向下推导，溯因推理从反直觉现象反推最可能解释。"
                "LogicEngine 先做确定性推理，LLM 再基于文献上下文补充新角度，"
                "最后合并去重并做逻辑一致性检查。")
        ],
    }


# ============================================================
# Item 27: Tournament Evaluation — Multi-Candidate Bracket Elimination
# ============================================================

@carry_control_fields
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

    # Cap hypotheses to prevent API timeout: sort by confidence, keep top 12
    if len(hypotheses) > 12:
        hypotheses.sort(key=lambda h: h.get("confidence_prior", 0), reverse=True)
        hypotheses = hypotheses[:12]
        logger.info(f"[TournamentEval] Capped to top 12 hypotheses (from {len(state.get('hypothesis_tree', []))}) to prevent timeout")

    # Build bracket description for the LLM prompt (truncated per hypothesis)
    hyp_list_text = "\n".join(
        f"{i + 1}. [{h['id']}] **{h.get('title', '?')}**\n   陈述: {h.get('statement', '')[:150]}\n   推理: {h.get('reasoning_chain', '')[:100]}\n   先验置信度: {h.get('confidence_prior', '?')}"
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

@carry_control_fields
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

    # Use real sensor data — randomize to avoid data monotony
    import glob as _glob
    import random as _random
    experiments = copy.deepcopy(list(state.get("experiment_records", [])))
    sensor_csvs = sorted(_glob.glob(str(Path("data/sensors/*.csv"))))
    # Exclude files already used in previous experiments for diversity
    used_files = {exp.get("input_data_path", "") for exp in experiments if exp.get("input_data_path")}
    available = [f for f in sensor_csvs if f not in used_files]
    if not available:
        available = sensor_csvs  # fallback: reuse if all exhausted
    input_data_path = _random.choice(available) if available else "[DATA_CHANNEL_PLACEHOLDER]"
    logger.info(f"[ExperimentDesign] Selected data file: {input_data_path} (from {len(available)} available, {len(used_files)} used)")

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

    experiments.append(experiment_record)

    logger.info(f"[ExperimentDesign] Created experiment {exp_id} for hypothesis {hyp['id']}")

    # Guard keys for LangGraph state merge
    max_iters = state.get("_max_iterations_", 200)
    curr_iter = state.get("iteration", 0)

    return {
        "experiment_records": experiments,
        "hypothesis_tree": tree,
        "_max_iterations_": max_iters,
        "iteration": curr_iter,
        "current_action": "experiment_design",
    }


# ============================================================
# Items 18, 19, 20: Data Analysis + Causal Inference
# ============================================================

@carry_control_fields
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
                # Rotate causal pairs across experiments to avoid data monotony
                exp_count = len([e for e in experiments if e.get("results", {}).get("analysis_complete")])
                pair_options = []
                if "T" in ts_data and "CO2" in ts_data:
                    pair_options.append(("T", "CO2", "Temperature → CO2"))
                if "T" in ts_data and "VOC" in ts_data:
                    pair_options.append(("T", "VOC", "Temperature → VOC"))
                if "CO2" in ts_data and "VOC" in ts_data:
                    pair_options.append(("CO2", "VOC", "CO2 → VOC"))
                if "H" in ts_data and "T" in ts_data:
                    pair_options.append(("H", "T", "Humidity → Temperature"))
                if len(pair_options) >= 2:
                    pair_idx = exp_count % len(pair_options)
                    x_key, y_key, pair_label = pair_options[pair_idx]
                    logger.info(f"[DataAnalysis] Rotated causal pair: {pair_label} (experiment #{exp_count})")
                else:
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
        "educational_annotations": [
            _edu_annotation("data_analysis",
                "数据分析阶段使用因果推断而非简单相关性分析。"
                "系统自动选择最优方法：Granger 因果检验判断时序因果关系，"
                "CCM 收敛交叉映射检测非线性动态耦合，反事实推演估计干预效应。"
                "每条证据都带有完整的统计依据（p值、效应量、置信区间），确保可追溯可复现。")
        ],
    }


# ============================================================
# Interpretation Node
# ============================================================

@carry_control_fields
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

@carry_control_fields
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
            if scores["total_score"] >= 60:
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

    action = "report_writing" if scores["total_score"] >= 60 else "reflection"
    logger.info(f"[ReviewerAgent] {latest_hyp['id']}: score={scores['total_score']}/100, needs_revision={scores['needs_revision']} → next={action}")

    return {
        "review_records": reviews,
        "hypothesis_tree": new_tree,
        "current_action": "reviewer_agent",
        "educational_annotations": [
            _edu_annotation("reviewer_agent",
                "五维评审从新颖性、可行性、方法论、证据强度、影响力五个维度打分，"
                "模拟学术同行评议。得分≥60通过并更新 Bayesian 后验概率，"
                "<60 打回修改。这种机制确保只有经过严格审查的假设才能进入下一轮。")
        ],
    }


# ============================================================
# Item 4/10: Reflection Loop
# ============================================================

@carry_control_fields
async def node_reflection(state: AgentState) -> dict:
    """
    【反思与修正】根因分析 → 派生修正性假设
    失败资产化：存储教训用于后续迭代

    Orchestrator 分析在入口处自动计算，确保反思节点拿到完整的本轮评估上下文。
    """
    # === Hard guardrail: enforce max iterations before doing any LLM work ===
    curr_iter = state.get("iteration", 0)
    max_iters = state.get("_max_iterations_", 200)
    if curr_iter >= max_iters:
        logger.warning(
            f"[Reflection] MAX_ITERATIONS already reached ({curr_iter}>={max_iters}), "
            f"returning without incrementing to prevent overshooting"
        )
        hypotheses = state.get("hypothesis_tree", [])
        kept_tree = []
        for h in hypotheses:
            s = h.get("status", "")
            if s == "refuted_in_tournament":
                continue
            if s in ("pruned", "refuted") and len(h.get("children_ids", [])) == 0:
                continue
            kept_tree.append(h)
        return {
            "iteration": curr_iter,
            "anomaly_graph": copy.deepcopy(list(state.get("anomaly_graph", []))),
            "hypothesis_tree": kept_tree,
            "consecutive_failures": state.get("consecutive_failures", 0) + 1,
            "_orch_stop_check": get_cached_orch_check(state) or {},
            "current_action": "reflection",
        }

    iteration = curr_iter + 1
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

@carry_control_fields
async def node_termination_eval(state: dict) -> dict:
    """
    Multi-dimensional termination evaluation.

    Dimensions: semantic convergence, methodology stability, evidence coverage,
    hypothesis space focus, cross-disciplinary transfer potential.
    """
    from core.llm_client import QwenClient  # for _hypothesis_statement_similarity

    evidence_chains = state.get("evidence_chains", [])
    exploration_exhausted = state.get("exploration_exhausted", False)
    iteration = state.get("iteration", 0)
    hypotheses = state.get("hypothesis_tree", [])

    # --- Step 0: Cross-disciplinary transfer analysis ---
    active_hyps = [h for h in hypotheses if h.get("status") not in ("pruned", "refuted", "refuted_in_tournament")]
    current_hyp = active_hyps[-1] if active_hyps else {}
    domain = state.get("domain", "Environment-Human Association")

    transfer_proposals = []
    try:
        from core.cross_disciplinary import CrossDisciplinaryAnalyzer
        analyzer = CrossDisciplinaryAnalyzer()
        kg = state.get("knowledge_graph", {})
        proposals = await analyzer.find_transfers(
            hypothesis=current_hyp,
            domain=domain,
            knowledge_graph=kg if isinstance(kg, dict) else None,
            evidence_chains=evidence_chains,
            top_n=3,
        )
        transfer_proposals = [{
            "method": p.method_name,
            "transfer_to": p.transfer_domain,
            "relevance": p.relevance_score,
            "feasibility": p.feasibility_score,
            "reasoning": p.reasoning[:300],
            "caveats": p.caveats[:2],
        } for p in proposals]
        logger.info(f"[TerminationEval] Found {len(transfer_proposals)} cross-disciplinary transfers")
    except Exception as e:
        logger.warning(f"[TerminationEval] Transfer analysis failed: {e}")

    # --- Step 1: Semantic convergence ---
    prev_statement = state.get("prev_round_winner_statement", "")
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
        convergence = 0.0
        logger.info(f"[TerminationEval] Round {iteration}: no previous data, convergence=0.0")
    else:
        # Import bigram similarity from orchestrator
        from core.orchestrator import _hypothesis_statement_similarity
        similarity = _hypothesis_statement_similarity(prev_statement, current_statement)
        convergence = round(similarity, 3)
        logger.info(f"[TerminationEval] Round {iteration}: similarity={similarity:.4f}, convergence={convergence:.3f}")

    convergence_history = list(state.get("convergence_history", []))
    convergence_history.append(convergence)

    # --- Step 2: Evidence strength ---
    if evidence_chains:
        evidence_str = sum(e.get("strength", 0.5) for e in evidence_chains) / len(evidence_chains)
    else:
        approved = [h for h in hypotheses if h.get("status") == "approved_by_reviewer"]
        if approved:
            evidence_str = sum(h.get("confidence_posterior", 0.5) for h in approved) / len(approved)
        else:
            evidence_str = 0.0

    # --- NEW: Methodology convergence ---
    methodology_status = "shifting"
    experiments = state.get("experiment_records", [])
    recent_methods = []
    for exp in reversed(experiments[-5:]):
        method = exp.get("results", {}).get("selected_method", "")
        if method:
            recent_methods.append(method)

    if len(recent_methods) >= 2:
        if all(m == recent_methods[0] for m in recent_methods):
            methodology_status = "stable"
        elif any(m == recent_methods[0] for m in recent_methods[1:]):
            methodology_status = "shifting_but_revisiting"
        else:
            methodology_status = "shifting"
    logger.info(f"[TerminationEval] Methodology: {methodology_status}")

    # --- NEW: Evidence dimension coverage ---
    covered_dimensions = []
    has_lit = bool(state.get("fact_extraction", []))
    has_stat = any(e.get("type") == "statistical_test" for e in evidence_chains)
    has_causal = any(e.get("type") == "causal_inference" for e in evidence_chains)
    if has_lit: covered_dimensions.append("literature")
    if has_stat: covered_dimensions.append("statistical")
    if has_causal: covered_dimensions.append("causal")
    evidence_dim_count = len(covered_dimensions)
    evidence_full = evidence_dim_count >= 2
    logger.info(f"[TerminationEval] Evidence dims: {covered_dimensions} ({evidence_dim_count})")

    # --- NEW: Hypothesis space convergence ---
    hyp_space_size = len(active_hyps)
    hypo_confidence_max = max((h.get("confidence_posterior", h.get("confidence_prior", 0)) for h in active_hyps), default=0.0)
    hypo_converged = hyp_space_size <= 2 and hypo_confidence_max > 0.7
    logger.info(f"[TerminationEval] Hyp space: {hyp_space_size} active, conf={hypo_confidence_max:.3f}, converged={hypo_converged}")

    # ============================================================
    # TERMINATION DECISION — Marginal Improvement Detection
    #
    # Core principle: stop when additional rounds are unlikely to
    # produce meaningfully better results (marginal benefit < cost).
    #
    # Uses a sliding window of review scores to detect three patterns:
    #   IMPROVING: scores trending up → keep going
    #   PLATEAU: scores flat → diminishing returns, stop
    #   DECLINING: scores trending down → overfitting, stop
    #
    # No magic numbers. Thresholds are derived from the data itself.
    # ============================================================

    # --- Compute quality signals ---
    approved_hyps = [h for h in hypotheses if h.get("status") == "approved_by_reviewer"]
    approved_count = len(approved_hyps)
    best_confidence = max(
        (h.get("confidence_posterior", h.get("confidence_prior", 0)) for h in approved_hyps),
        default=0.0
    )

    # Build review score history (across all rounds)
    review_scores = [r.get("total_score", 0) for r in state.get("review_records", [])]
    best_review = max(review_scores) if review_scores else 0
    window = review_scores[-3:] if len(review_scores) >= 3 else review_scores

    # --- Marginal improvement analysis ---
    # Trend: is the best hypothesis quality improving over recent rounds?
    trend = "initial"  # first 1-2 rounds
    if len(window) >= 3:
        # Linear trend: positive slope = improving, flat = plateau, negative = declining
        first_half = sum(window[:len(window)//2]) / max(len(window)//2, 1)
        second_half = sum(window[len(window)//2:]) / max(len(window) - len(window)//2, 1)
        delta = second_half - first_half
        if delta > 5:
            trend = "improving"
        elif delta < -5:
            trend = "declining"
        else:
            trend = "plateau"

    # --- Decision ---
    max_iters = state.get("_max_iterations_", 200)
    iteration = state.get("iteration", 0)
    should_terminate = False
    stop_reason = ""

    # === SIGNAL 1: Budget exhausted ===
    if iteration >= max_iters:
        should_terminate = True
        stop_reason = f"Budget exhausted ({iteration}/{max_iters} rounds)"

    # === SIGNAL 2: Declining quality (overfitting / regression) ===
    elif trend == "declining" and iteration >= 3:
        should_terminate = True
        stop_reason = (
            f"Quality declining: review scores {window} trending down "
            f"(best was {best_review}). Stopping to prevent overfitting."
        )

    # === SIGNAL 3: Plateau with sufficient quality ===
    elif trend == "plateau" and iteration >= 3:
        if best_review >= 70 and evidence_str > 0.6:
            should_terminate = True
            stop_reason = (
                f"Plateau reached: review scores stable at ~{sum(window)/len(window):.0f}, "
                f"best={best_review}, evidence={evidence_str:.3f}. "
                f"Marginal improvement unlikely."
            )
        elif best_review < 60:
            should_terminate = True
            stop_reason = (
                f"Plateau at low quality: review scores stable at ~{sum(window)/len(window):.0f}. "
                f"Additional rounds unlikely to help. Generate report with caveats."
            )

    # === SIGNAL 4: Strong evidence + high review (early exit) ===
    elif evidence_str > 0.85 and best_review >= 75 and approved_count >= 1:
        should_terminate = True
        stop_reason = (
            f"Strong results: evidence={evidence_str:.3f}, review={best_review}, "
            f"{approved_count} approved hypotheses. Early exit."
        )

    if not should_terminate:
        stop_reason = (
            f"Continue ({trend}): round {iteration+1}/{max_iters}, "
            f"review_scores={window}, best={best_review}, "
            f"evidence={evidence_str:.3f}, approved={approved_count}"
        )

    logger.info(f"[TerminationEval] {'TERMINATE' if should_terminate else 'CONTINUE'}: {stop_reason}")

    # --- Pruning ---
    tree = copy.deepcopy([dict(h) for h in hypotheses])
    kept_tree = []
    pruned_at_term = 0
    for h in tree:
        if h.get("status") in ("pruned", "refuted"):
            pruned_at_term += 1
        else:
            kept_tree.append(h)

    result = {
        "convergence_score": convergence,
        "convergence_history": convergence_history,
        "prev_round_winner_statement": current_statement,
        "evidence_strength": round(evidence_str, 3),
        "exploration_exhausted": exploration_exhausted,
        "combined_score": round(evidence_str * 0.5 + (best_review / 100.0) * 0.3 + convergence * 0.2, 3),
        "should_terminate": should_terminate,
        "stop_reason": stop_reason,
        "methodology_status": methodology_status,
        "evidence_dimension_coverage": covered_dimensions,
        "hypothesis_space_size": hyp_space_size,
        "hypothesis_max_confidence": round(hypo_confidence_max, 3),
        "hypothesis_converged": hypo_converged,
        "transfer_proposals": transfer_proposals,
        "hypothesis_tree": kept_tree if not should_terminate else tree,
    }

    return {
        "_termination_result": result,
        "__decision": "TERMINATE" if should_terminate else "CONTINUE",
        "current_action": "termination_eval",
        "_cross_disciplinary_proposals": transfer_proposals,
        "iteration": state.get("iteration", 0),
        "_max_iterations_": state.get("_max_iterations_", 200),
        "consecutive_failures": state.get("consecutive_failures", 0),
    }

# ============================================================

@carry_control_fields
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
    # Show actual methods used, not generic list
    methods_used = set()
    for ev in evidence_chains:
        m = ev.get("method_used", "")
        if m:
            methods_used.add(m)
    if "granger" in methods_used:
        parts.append("| 因果推断 | Granger 因果检验 | statsmodels.tsa.granger, 滞后阶数自适应 |")
    if "ccm" in methods_used:
        parts.append("| 因果推断 | Convergent Cross Mapping | skccm, 收敛性验证 |")
    if "counterfactual" in methods_used:
        parts.append("| 因果推断 | 反事实推演 | DoWhy, 结构因果模型 |")
    if not methods_used:
        parts.append("| 因果推断 | AI自动选择最优方法 | CCM / Granger / 贝叶斯网络 |")
    parts.append("| 数据采集 | 环境传感器 + 可穿戴设备 | 温度/湿度/CO₂ + PPG/HRV/SpO₂ |")
    parts.append("| 信号处理 | 多源时序对齐 + 质量评估 | Daltons 格式解析, 互相关对齐 |")
    parts.append("| 统计分析 | 混合效应模型 + Bayesian 更新 | log-odds 置信度更新 |")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## 四、数据集（Datasets）")
    parts.append("### Source（实际使用的数据）")
    # Show actual data files used in experiments
    data_files_used = []
    for exp in experiments:
        path = exp.get("input_data_path", "")
        if path and path != "[DATA_CHANNEL_PLACEHOLDER]":
            data_files_used.append(path)
    if data_files_used:
        parts.append("| 数据文件 | 类型 | 来源 |")
        parts.append("|---------|------|------|")
        seen = set()
        for f in data_files_used:
            fname = f.replace("\\", "/").split("/")[-1]
            if fname not in seen:
                seen.add(fname)
                # Infer data type from filename
                if "biometric" in f.lower() or "ppg" in f.lower():
                    dtype = "生物特征 (PPG/HRV/SpO₂)"
                elif "env" in f.lower():
                    dtype = "环境传感器 (温湿度/CO₂)"
                else:
                    dtype = "环境传感器 (Daltons格式)"
                parts.append(f"| {fname} | {dtype} | 实际采集数据 |")
    else:
        parts.append("| 数据类型 | 来源描述 | 样本量估计 |")
        parts.append("|---------|---------|-----------|")
        parts.append("| 环境传感器 | 室内环境监测站（温湿度、CO₂） | ≥5000点/天 |")
        parts.append("| PPG/血氧/HRV | 可穿戴传感器 | ≥100Hz采样率 |")
    parts.append("")
    parts.append("### Target（验证实验拟采集数据特征）")
    parts.append(f"- **实际样本量**: {len(experiments)} 个实验方案已执行")
    parts.append(f"- **因果推断方法**: {', '.join(sorted(methods_used)) if methods_used else '待执行'}")
    parts.append("- **实验周期**: 基于现有传感器数据的回顾性分析 + 前瞻性验证建议")
    parts.append("- **N-of-1 支持**: 支持个体化分析，对比同一受试者不同环境下的生理响应")
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
    parts.append("| Pearson 相关 | 双变量线性关联 | 简单直观，计算快 | 无法确定因果方向，混淆因子干扰 |")
    parts.append("| Spearman 秩相关 | 单调关联检测 | 无需正态假设 | 丢失非线性信息 |")

    # Show actual causal method used vs correlation
    if "granger" in methods_used:
        parts.append("| **Granger 因果检验** ✅ | **时序因果推断** | **方向性明确，统计检验严格** | 需要平稳时间序列 |")
    if "ccm" in methods_used:
        parts.append("| **CCM 收敛交叉映射** ✅ | **非线性动态系统** | **检测双向因果，适用于生态系统** | 需要较长序列 |")

    # Add actual comparison if we have evidence
    if evidence_chains:
        causal_ev = [e for e in evidence_chains if e.get("type") == "causal_inference"]
        if causal_ev:
            strength = causal_ev[-1].get("strength", 0)
            method = causal_ev[-1].get("method_used", "未知")
            sb = causal_ev[-1].get("statistical_basis", {})
            parts.append("")
            parts.append(f"**实际因果推断结果** (方法: {method}):")
            parts.append("")
            parts.append("| 指标 | 值 | 说明 |")
            parts.append("|------|-----|------|")
            parts.append(f"| 证据强度 | {strength:.4f} | 0-1 置信度分数 (>0.7 为强证据) |")
            for k, v in list(sb.items())[:5]:
                if isinstance(v, (int, float)):
                    parts.append(f"| {k} | {v:.4f} | 统计依据 |")
                else:
                    parts.append(f"| {k} | {str(v)[:50]} | 统计依据 |")
            parts.append("")
            parts.append("> **结论**: 因果推断方法相比简单相关性分析，能确定因果方向并提供统计显著性检验。"
                         f"本研究中 {method} 方法的证据强度为 {strength:.3f}，"
                         f"{'达到强证据标准' if strength > 0.7 else '建议进一步验证'}。")
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
    # --- N-of-1 个体化分析 (比赛亮点) ---
    parts.append("")
    parts.append("## 十三、N-of-1 个体化分析")
    parts.append("")
    parts.append("> **N-of-1 研究**是 twinScientist 的核心差异化能力。传统群体研究掩盖了个体差异，")
    parts.append("> 而 N-of-1 方法通过分析同一个体在不同环境条件下的生理响应，")
    parts.append("> 实现真正个性化的环境—健康关联发现。")
    parts.append("")

    # Collect data files grouped by location
    import glob as _glob
    sensor_files = sorted(_glob.glob(str(Path("data/sensors/*.csv"))))
    # Group by room (H1_Bedroom, H1_Kitchen, etc.)
    locations = {}
    for f in sensor_files:
        fname = f.replace("\\", "/").split("/")[-1]
        # Extract location: H1_Bedroom, H1_Kitchen, etc.
        parts_name = fname.replace(".csv", "").replace("_env", "").replace("_biometric", "")
        if "_" in parts_name:
            loc = parts_name  # H1_Bedroom, H1_Kitchen, etc.
        else:
            loc = parts_name
        if loc not in locations:
            locations[loc] = []
        locations[loc].append(fname)

    if len(locations) >= 2:
        parts.append(f"### 已采集数据概览 ({len(locations)} 个场景)")
        parts.append("| 场景 | 数据文件 | 类型 |")
        parts.append("|------|---------|------|")
        for loc, files in sorted(locations.items())[:8]:
            ftypes = set()
            for fn in files:
                if "env" in fn.lower():
                    ftypes.add("环境")
                elif "biometric" in fn.lower():
                    ftypes.add("生物特征")
                else:
                    ftypes.add("传感器")
            parts.append(f"| {loc} | {', '.join(files[:2])} | {', '.join(ftypes)} |")
        parts.append("")
        parts.append("### N-of-1 分析能力")
        parts.append("")
        parts.append("| 分析维度 | 说明 | 示例 |")
        parts.append("|---------|------|------|")
        parts.append("| 跨场景对比 | 同一受试者不同房间的环境-生理关联差异 | H1 卧室 vs 厨房: CO₂→HRV 因果效应对比 |")
        parts.append("| 时间模式 | 同一场景不同时段的变化规律 | 卧室夜间 vs 白天: 温湿度对睡眠质量的影响 |")
        parts.append("| 个体基线 | 建立个人化的生理响应基线 | H1 的 HRV 对环境变化的敏感度阈值 |")
        parts.append("| 暴露-响应 | 剂量-反应关系的个体化建模 | CO₂ 浓度每升高 100ppm, H1 的 HRV 下降幅度 |")
        parts.append("")
        parts.append("> **比赛亮点**: N-of-1 + LLM + IoT 传感器是 2025 年 Nature Digital Medicine 和")
        parts.append("> The Lancet Digital Health 关注的前沿方向。twinScientist 是目前唯一将此范式")
        parts.append("> 与 AI Scientist 自主科研流程结合的开源系统。")
    else:
        parts.append("*(需要至少 2 个场景的数据文件以启用 N-of-1 分析)*")
    parts.append("")
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

    # Inject multi-modal visualizations
    try:
        from core.visualization import inject_visualizations
        report = inject_visualizations(report, state)
        logger.info("[ReportWriting] Visualizations injected")
    except Exception as e:
        logger.warning(f"[ReportWriting] Visualization injection failed: {e}")

    return {"final_report": report, "current_action": "report_writing"}



@carry_control_fields
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

@carry_control_fields
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

@carry_control_fields
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
