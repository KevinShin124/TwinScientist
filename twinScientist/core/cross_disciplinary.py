"""
Layer 3 — Cross-Disciplinary Technology Transfer Analyzer

识别并建议跨学科技术迁移路径，挖掘方法在其他学科中的应用潜力。

工作原理：
1. 加载方法—学科映射表 (data/method_domain_mapping.json)
2. 对当前假设中涉及的技术/分析方法，查找其在其他学科的成功应用案例
3. 生成迁移提案（TransferProposal），标注适用条件和需注意的域差异
4. 结合假设语义与目标领域描述做语义相似度增强排序

Usage:
    from core.cross_disciplinary import CrossDisciplinaryAnalyzer

    analyzer = CrossDisciplinaryAnalyzer()
    proposals = await analyzer.find_transfers(
        hypothesis=hypothesis_dict,
        domain="环境健康",
        knowledge_graph=kg_dict,
    )
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================
# Data Classes
# ============================================================


@dataclass
class MethodEntry:
    """方法—学科映射表中一个条目"""
    method: str
    full_name: str
    source_domains: list[str]
    target_applications: list[str]


@dataclass
class TransferProposal:
    """跨学科技术迁移提案"""
    method_name: str
    method_full_name: str
    current_domain: str
    transfer_domain: str
    transfer_application: str
    relevance_score: float  # 0-1
    feasibility_score: float  # 0-1
    confidence: float  # 0-1
    reasoning: str
    caveats: list[str]
    suggested_data_requirements: list[str]
    similar_methods_in_target: list[str]


# ============================================================
# Mapping Table Loader
# ============================================================


def load_method_domain_mapping(path: str = "") -> list[MethodEntry]:
    """加载方法—学科映射表"""
    if not path:
        candidates = [
            "./twinScientist/data/method_domain_mapping.json",
            "./data/method_domain_mapping.json",
        ]
        for c in candidates:
            if Path(c).exists():
                path = c
                break
        if not path:
            logger.warning("[CrossDisciplinary] method_domain_mapping.json not found")
            return []

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        methods_data = data.get("methods", [])
        entries = [
            MethodEntry(
                method=m["method"],
                full_name=m["full_name"],
                source_domains=m.get("source_domains", []),
                target_applications=m.get("target_applications", []),
            )
            for m in methods_data
        ]
        logger.info(f"[CrossDisciplinary] Loaded {len(entries)} method-domain mappings")
        return entries
    except Exception as e:
        logger.warning(f"[CrossDisciplinary] Failed to load mapping from {path}: {e}")
        return []


# ============================================================
# Similarity Utilities
# ============================================================


class SemanticSimilarity:
    """轻量级语义相似度计算（中文友好）"""

    @staticmethod
    def bigram_jaccard(a: str, b: str) -> float:
        """字符级 bigram Jaccard 相似度"""
        a_norm = " ".join(a.strip().split())
        b_norm = " ".join(b.strip().split())

        def bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1)) if len(s) > 1 else {s}

        a_set, b_set = bigrams(a_norm), bigrams(b_norm)
        if not a_set or not b_set:
            return 0.0
        intersection = len(a_set & b_set)
        union = len(a_set | b_set)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def keyword_overlap(a: str, b: str) -> float:
        """关键词重叠率（基于中英文常见分词启发式）"""
        # Simple tokenization: split on non-alphanumeric + underscore
        tokens_a = set(re.findall(r'[a-zA-Z]+|[一-鿿]', a.lower()))
        tokens_b = set(re.findall(r'[a-zA-Z]+|[一-鿿]', b.lower()))

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def combined(a: str, b: str) -> float:
        """组合相似度：bigram 60% + 关键词 40%"""
        bg = SemanticSimilarity.bigram_jaccard(a, b)
        kw = SemanticSimilarity.keyword_overlap(a, b)
        return 0.6 * bg + 0.4 * kw


# ============================================================
# Transfer Analysis Engine
# ============================================================


class CrossDisciplinaryAnalyzer:
    """
    跨学科技术迁移分析器。

    核心流程：
    1. 从假设和知识图谱中提取涉及的技术方法
    2. 对每个方法，在映射表中找所有目标应用领域
    3. 将目标应用与假设上下文做语义匹配 → 排序
    4. 对高相关度的匹配，生成详细迁移提案
    """

    def __init__(self, mapping_path: str = ""):
        self.entries = load_method_domain_mapping(mapping_path)
        self._similarity = SemanticSimilarity()

        # Build lookup indexes
        self._method_by_keyword: dict[str, list[MethodEntry]] = {}
        self._build_keyword_index()

    def _build_keyword_index(self):
        """构建方法关键词索引，加速匹配"""
        for entry in self.entries:
            keywords = set()
            # Add English keywords
            tokens = re.findall(r'[a-zA-Z_\-]+', entry.method.upper())
            keywords.update(t.lower() for t in tokens)
            # Add common abbreviations/acronyms
            abbrev = re.sub(r'[^A-Z]', '', entry.method).lower()
            if abbrev and len(abbrev) >= 2:
                keywords.add(abbrev)
            # Add full name components
            fn_tokens = re.findall(r'[a-zA-Z]+', entry.full_name)
            keywords.update(t.lower() for t in fn_tokens[:5])  # First 5 words

            for kw in keywords:
                if kw not in self._method_by_keyword:
                    self._method_by_keyword[kw] = []
                self._method_by_keyword[kw].append(entry)

    async def find_transfers(
        self,
        hypothesis: dict,
        domain: str,
        knowledge_graph: dict = None,
        evidence_chains: list[dict] = None,
        top_n: int = 5,
    ) -> list[TransferProposal]:
        """
        为给定假设寻找跨学科技术迁移机会。

        Args:
            hypothesis: 假设字典（含 statement, title, reasoning_chain）
            domain: 当前研究领域
            knowledge_graph: 知识图谱节点/边数据
            evidence_chains: 证据链列表（从中提取已使用方法名）
            top_n: 返回最多 N 个提案

        Returns:
            TransferProposal 对象列表，按 relevance_score 降序排列
        """
        # Collect method mentions from multiple sources
        all_text = ""
        stmt = hypothesis.get("statement", "")
        title = hypothesis.get("title", "")
        reasoning = hypothesis.get("reasoning_chain", "")
        all_text += f"{title} {stmt} {reasoning}"

        # Extract method mentions from text
        matched_entries = self._match_methods(all_text)

        # Also extract from knowledge graph nodes
        kg_text = ""
        kg_nodes = []
        if knowledge_graph and isinstance(knowledge_graph, dict):
            for node in knowledge_graph.get("nodes", []):
                nid = node.get("id", "")
                ntype = node.get("type", "")
                kg_text += f" {nid} {ntype}"
                kg_nodes.append(node)
            for _, _, edge in knowledge_graph.get("edges", []):
                kg_text += f" {edge}"

        kg_matched = self._match_methods(kg_text)
        matched_entries.extend(kg_matched)

        # Extract method names from evidence chains (causal inference methods used)
        evidence_text = ""
        if evidence_chains:
            for ev in evidence_chains:
                method = ev.get("method_used", "")
                content = ev.get("content", "")
                evidence_text += f" {method} {content}"

        ev_matched = self._match_methods(evidence_text)
        matched_entries.extend(ev_matched)

        # Deduplicate entries
        seen_methods: set[str] = set()
        unique_entries = []
        for entry in matched_entries:
            if entry.method not in seen_methods:
                seen_methods.add(entry.method)
                unique_entries.append(entry)

        # For each matched method, generate transfer proposals
        proposals: list[TransferProposal] = []
        for entry in unique_entries:
            proposals.extend(self._generate_proposals_for_entry(
                entry, domain, hypothesis, matched_texts=all_text
            ))

        # Rank by relevance score
        proposals.sort(key=lambda p: p.relevance_score, reverse=True)

        logger.info(
            f"[CrossDisciplinary] Found {len(proposals)} transfer proposals "
            f"(from {len(unique_entries)} methods)"
        )

        return proposals[:top_n]

    def _match_methods(self, text: str) -> list[MethodEntry]:
        """
        在给定的文本中匹配已知的映射方法。

        多级匹配策略：
        1. Exact match (CCM, Granger, HRV...)
        2. Keyword substring match (heart rate → HRV Analysis)
        3. Full name component match (Causal Forest → causal_forest mention)
        """
        if not text:
            return []

        text_lower = text.lower()
        matches: list[tuple[float, MethodEntry]] = []

        # Strategy 1: Exact method name matching
        for entry in self.entries:
            method_upper = entry.method.upper()
            method_variants = [
                entry.method,
                entry.method.upper(),
                entry.method.lower(),
                re.sub(r'[^A-Za-z]', '', entry.method),  # Remove spaces/hyphens
            ]
            for variant in method_variants:
                if variant and variant.lower() in text_lower:
                    # Higher weight for exact/no-space match
                    score = 1.0 if ' ' not in variant and '-' not in variant else 0.7
                    matches.append((score, entry))
                    break

        # Strategy 2: Keyword-level matching
        for entry in self.entries:
            # Check individual words from method name
            words = re.findall(r'[a-zA-Z]+', entry.method)
            word_scores = []
            for word in words:
                if word.lower() in text_lower:
                    word_scores.append(0.5)
                elif word.lower()[:3] in text_lower:  # Trigram tolerance
                    word_scores.append(0.3)

            if word_scores and sum(word_scores) / len(word_scores) > 0.3:
                matches.append((sum(word_scores) / max(len(word_scores), 1), entry))

        # Strategy 3: Semantic similarity against full names
        for entry in self.entries:
            sim = self._similarity.combined(text[:500], entry.full_name)
            if sim > 0.3:
                matches.append((sim * 0.5, entry))

        # Deduplicate by method name, keeping highest score
        best_per_method: dict[str, tuple[float, MethodEntry]] = {}
        for score, entry in matches:
            if entry.method not in best_per_method or score > best_per_method[entry.method][0]:
                best_per_method[entry.method] = (score, entry)

        result = [v[1] for v in best_per_method.values() if v[0] > 0]
        result.sort(key=lambda e: e.method.upper())  # Alphabetical for determinism
        return result

    def _generate_proposals_for_entry(
        self,
        entry: MethodEntry,
        domain: str,
        hypothesis: dict,
        matched_texts: str = "",
    ) -> list[TransferProposal]:
        """为单个方法条目生成迁移提案"""
        proposals = []

        stmt = hypothesis.get("statement", "")
        title = hypothesis.get("title", "")

        for target_app in entry.target_applications:
            # Compute relevance scores
            relevance = self._similarity.combined(matched_texts[:500], target_app)
            relevance = min(relevance * 2.0, 1.0)  # Boost for semantic proximity

            # Feasibility depends on domain overlap with method's source domains
            feasibility_factors = [
                relevance,  # Higher relevance → easier to justify
                0.3,  # Base feasibility (any cross-disciplinary transfer needs effort)
            ]
            feasibility = sum(feasibility_factors) / len(feasibility_factors)
            feasibility = min(max(feasibility, 0.1), 1.0)

            confidence = relevance * 0.6 + feasibility * 0.4
            confidence = round(confidence, 3)

            # Generate reasoning
            reasoning = (
                f"**{entry.full_name}** 原本在 {', '.join(entry.source_domains[:3])} 领域有成熟应用。\n\n"
                f"**迁移至**: {target_app.replace('_', ' ').title()}\n\n"
                f"在当前研究中（{domain} — {hypothesis.get('query', '')[:100]}），"
                f"{entry.method} 可能帮助:"
            )

            # Add specific suggestions based on hypothesis context
            if "HRV" in stmt or "心率变异性" in stmt:
                reasoning += "\n- 分析自主神经系统对环境因子的响应模式"
                reasoning += "\n- 识别非线性反馈回路而非简单线性关联"
            if "因果" in stmt or "causal" in stmt.lower():
                reasoning += "\n- 建立环境暴露→生理指标的因果方向性证据"
                reasoning += "\n- 区分混杂因素与真实因果效应"
            if "预测" in stmt or "predict" in stmt.lower():
                reasoning += "\n- 提供多变量联合风险评分模型"
                reasoning += "\n- 评估个体化阈值以指导干预时机"

            # Caveats
            caveats = [
                f"源领域 '{entry.source_domains[0]}' 的数据特性可能与当前环境健康领域存在系统性差异",
                "需要重新校准方法参数以适应新的数据分布",
                "可能需要额外的验证步骤以确保迁移有效性",
            ]
            if entry.source_domains != entry.target_applications:
                caveats.append(f"注意: {entry.source_domains[0]} → {target_app} 的域差距较大")

            # Suggested data requirements
            data_reqs = [
                f"目标指标的时间序列数据（采样频率 ≥ 1Hz）",
                f"至少 {max(100, len(matched_texts) // 50)} 条有效观测记录",
                "必要的协变量信息以控制混杂因子",
            ]

            # Find similar methods in target domain
            similar_methods = [
                m.method for m in self.entries
                if any(kw in target_app.lower() for kw in ["health", "medical", "clinical", "生物", "临床"])
                and m.method != entry.method
            ][:3]

            proposal = TransferProposal(
                method_name=entry.method,
                method_full_name=entry.full_name,
                current_domain=domain,
                transfer_domain=target_app.replace("_", " ").title(),
                transfer_application=target_app,
                relevance_score=round(relevance, 3),
                feasibility_score=round(feasibility, 3),
                confidence=confidence,
                reasoning=reasoning,
                caveats=caveats,
                suggested_data_requirements=data_reqs,
                similar_methods_in_target=similar_methods,
            )
            proposals.append(proposal)

        return proposals

    def get_all_target_domains(self, source_domain: str) -> list[str]:
        """获取某个源领域的所有可迁移目标领域"""
        targets = set()
        for entry in self.entries:
            if source_domain.lower() in [d.lower() for d in entry.source_domains]:
                targets.update(entry.target_applications)
        return sorted(targets)
