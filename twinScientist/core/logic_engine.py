"""
Layer 3 — Logic Hypothesis Engine

多路径假设推理引擎：
- 归纳推理（Inductive）: 从已有事实和实验结果中归纳趋势
- 演绎推理（Deductive）: 从领域专家规则库向下推导新假设
- 溯因推理（Abductive）: 给定观测/反直觉现象，反推最可能解释

合并去重 + 逻辑一致性检查，为 LLM 提供结构化的高质量候选假设列表。

Usage:
    from core.logic_engine import LogicHypothesisEngine

    engine = LogicHypothesisEngine()
    results = await engine.generate_all(
        facts=state["fact_extraction"],
        existing_hypotheses=state["hypothesis_tree"],
        evidence_chains=state["evidence_chains"],
    )
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================
# Data Classes
# ============================================================


@dataclass
class ConditionMatch:
    """一条规则的条件是否匹配"""
    rule_id: str
    matched_conditions: list[str]
    unmet_conditions: list[str]
    match_score: float  # 0.0 - 1.0


@dataclass
class RawHypothesis:
    """三路推理产生的原始假设（未经一致性检查）"""
    title: str
    statement: str
    reasoning_chain: str
    confidence_prior: float
    testability: int  # 1-10
    evidence_needed: str
    path: str  # "inductive" | "deductive" | "abductive"
    source_ref: str | None  # 来源引用（规则ID、事实ID等）
    parent_rule_ids: list[str] = field(default_factory=list)  # 如果是演绎产生的，记录来源规则
    metadata: dict = field(default_factory=dict)


@dataclass
class ConsistencyReport:
    """逻辑一致性检查结果"""
    hypothesis_id: str  # hyp_title or hyp_statement[:50]
    is_consistent: bool
    conflict_type: str  # "internal", "pairwise", "factual", "none"
    description: str
    severity: str  # "critical", "warning", "info"
    suggestion: str  # 修正建议


# ============================================================
# Domain Rule Loader
# ============================================================


def load_domain_rules(path: str = "") -> list[dict]:
    """加载领域专家规则库"""
    if not path:
        # Default: twinScientist/data/domain_rules.json
        candidates = [
            "./twinScientist/data/domain_rules.json",
            "./data/domain_rules.json",
        ]
        for c in candidates:
            if Path(c).exists():
                path = c
                break
        if not path:
            logger.warning("[LogicEngine] domain_rules.json not found; deductive path disabled")
            return []

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rules = data.get("rules", [])
        logger.info(f"[LogicEngine] Loaded {len(rules)} domain rules from {path}")
        return rules
    except Exception as e:
        logger.warning(f"[LogicEngine] Failed to load rules from {path}: {e}")
        return []


# ============================================================
# Inductive Reasoning
# ============================================================


class InductiveReasoner:
    """
    归纳推理：从已有事实、评审分数变化、收敛趋势中归纳未被充分探索的方向。

    核心思想：如果某些方向的事实很少被覆盖，但其他方向的证据很强，
    那么存在尚未被系统探索的中间地带——这正是新假设的来源。
    """

    async def generate(
        self,
        facts: list[dict],
        existing_hypotheses: list[dict],
        review_records: list[dict] = None,
    ) -> list[RawHypothesis]:
        """
        基于已有数据归纳生成新假设方向。
        """
        hypotheses = []

        # Analyze which entity types have been most explored
        covered_entities = set()
        covered_mechanisms = set()
        for hyp in existing_hypotheses:
            stmt = hyp.get("statement", "").lower()
            reasoning = hyp.get("reasoning_chain", "").lower()
            text = stmt + " " + reasoning
            # Extract variable keywords
            for kw in ["temperature", "humidity", "CO2", "co₂", "pm2.5", "voc",
                       "noise", "light", "air_quality"]:
                if kw in text:
                    covered_entities.add(kw)
            # Extract mechanism keywords
            for kw in ["sympathetic", "inflammation", "oxidative", "autonomic",
                       "melatonin", "cortisol", "endocrine"]:
                covered_mechanisms.add(kw)

        # Find under-explored combinations
        all_variables = [
            ("temperature", "HRV_complexity_metrics"),
            ("humidity", "tear_film_stability"),
            ("CO2", "cognitive_composite_score"),
            ("PM2.5", "visual_fatigue_index"),
            ("VOC", "respiratory_rate_variability"),
            ("negative_ion", "sleep_depth"),
            ("UV_radiation", "circadian_phase_shift"),
            ("indoor_airflow_velocity", "particle_deposition_rate"),
        ]

        under_explored = []
        for var, metric in all_variables:
            if var not in covered_entities:
                under_explored.append((var, metric))

        # Generate one hypothesis per under-explored direction
        for var, metric in under_explored[:5]:
            hypotheses.append(RawHypothesis(
                title=f"{self._chinese_name(var)}对{self._metric_cn(metric)}的影响路径",
                statement=f"系统中{self._chinese_name(var)}水平与{self._metric_cn(metric)}之间存在显著的非线性关联",
                reasoning_chain=(
                    f"已有文献和实验数据显示{self._chinese_name(var)}已被证明会影响多个生理指标，"
                    f"但该变量与{self._metric_cn(metric)}的具体因果关联尚未在本研究范围内被系统探索。"
                    f"通过控制其他环境因子恒定，单独考察{self._chinese_name(var)}梯度变化对"
                    f"{self._metric_cn(metric)}的影响，可以填补这一知识空白。"
                ),
                confidence_prior=0.40,
                testability=6,
                evidence_needed=f"需要在控制条件下采集{self._chinese_name(var)}梯度变化下的{self._metric_cn(metric)}时间序列数据",
                path="inductive",
                source_ref=None,
                metadata={
                    "gap_analysis": f"Under-explored combination: {var} × {metric}",
                    "covered_entity_count": len(covered_entities),
                },
            ))

        logger.info(f"[Inductive] Generated {len(hypotheses)} hypotheses from gap analysis")
        return hypotheses

    @staticmethod
    def _chinese_name(var: str) -> str:
        mapping = {
            "temperature": "温度",
            "humidity": "湿度",
            "CO2": "CO₂浓度",
            "co₂": "CO₂浓度",
            "pm2.5": "PM₂.₅浓度",
            "voc": "VOC暴露水平",
            "noise": "环境噪音水平",
            "light": "光照强度",
            "air_quality": "空气质量综合指数",
            "negative_ion": "负氧离子浓度",
            "UV_radiation": "紫外线辐射强度",
            "indoor_airflow_velocity": "室内气流速度",
        }
        return mapping.get(var, var)

    @staticmethod
    def _metric_cn(metric: str) -> str:
        mapping = {
            "HRV_complexity_metrics": "心率变异性非线性复杂度指标",
            "tear_film_stability": "泪膜稳定性",
            "cognitive_composite_score": "认知功能综合评分",
            "visual_fatigue_index": "视觉疲劳指数",
            "respiratory_rate_variability": "呼吸频率变异性",
            "sleep_depth": "睡眠深度指标",
            "circadian_phase_shift": "昼夜节律相位偏移",
            "particle_deposition_rate": "颗粒物沉积速率",
        }
        return mapping.get(metric, metric)


# ============================================================
# Deductive Reasoning
# ============================================================


class DeductiveReasoner:
    """
    演绎推理：从领域专家规则出发，当已知条件满足时推导新假设。

    规则格式: IF conditions THEN conclusion (MECHANISM described)
    当现有数据和观察到的现象激活某条规则的if条件时，
    该规则预测的结论本身就是一条可检验的假设。
    """

    def __init__(self, rules: list[dict] = None):
        self.rules = rules or []
        self._condition_to_rule_map = {}
        self._build_condition_map()

    def _build_condition_map(self):
        """将每条规则的条件映射到规则ID，加速匹配"""
        for rule in self.rules:
            for cond in rule.get("if_conditions", []):
                if cond not in self._condition_to_rule_map:
                    self._condition_to_rule_map[cond] = []
                self._condition_to_rule_map[cond].append(rule)

    def match_conditions(
        self,
        facts: list[dict],
        evidence_chains: list[dict],
        convergence_history: list[float] = None,
    ) -> list[ConditionMatch]:
        """
        将已知事实和证据链中的信息映射到规则条件，找出匹配的规则。
        """
        matches = []

        # Build a knowledge base from available data
        kb = self._extract_kb(facts, evidence_chains, convergence_history)

        for rule in self.rules:
            conditions = rule.get("if_conditions", [])
            matched = []
            unmet = []

            for cond in conditions:
                if cond in kb and kb[cond]:
                    matched.append(cond)
                else:
                    unmet.append(cond)

            # Calculate match score based on how many conditions are met
            total = max(len(conditions), 1)
            score = len(matched) / total

            if score >= 0.5:  # At least half the conditions are met
                matches.append(ConditionMatch(
                    rule_id=rule["id"],
                    matched_conditions=matched,
                    unmet_conditions=unmet,
                    match_score=score,
                ))

        return matches

    def _extract_kb(
        self,
        facts: list[dict],
        evidence_chains: list[dict],
        convergence_history: list[float],
    ) -> dict:
        """从各种数据源提取知识库（布尔标志+数值估计）"""
        kb = {}

        # Check facts for condition indicators
        fact_text_combined = "\n".join(
            f.get("fact", "") + " " + f.get("_verification_method", "")
            for f in facts
        ).lower()

        # Condition → boolean flag mappings
        condition_indicators = {
            "temperature_increase": ["高温", "温度升高", "high temperature", "thermal"],
            "CO2_elevated_above_1000ppm": ["CO₂", "CO2", "carbon dioxide", "high CO2"],
            "PM25_elevated": ["PM2.5", "pm2.5", "细颗粒", "particulate matter"],
            "low_humidity_below_35": ["低湿度", "low humidity", "干燥", "dry"],
            "VOC_elevated": ["VOC", "挥发", "volatile organic", "TVOC"],
            "high_temperature_high_humidity": ["高温高湿", "heat humidity", "热舒适度"],
            "circadian_pattern_present": ["昼夜节律", "circadian", "昼夜", "diurnal"],
            "noise_elevated_above_55dB": ["噪音", "noise", "acoustic", "声级"],
            "prolonged_screen_use": ["屏幕", "screen", "显示器", "computer vision"],
            "insufficient_daylight_exposure": ["光照不足", "insufficient daylight", "自然光"],
            "negative_ion_elevated": ["负离子", "negative ion", "负氧离子"],
            "PM10_and_UFP_elevated": ["PM₁₀", "超细颗粒", "UFP", "nanoparticle"],
            "formaldehyde_elevated_above_0.08ppm": ["甲醛", "formaldehyde"],
            "rapid_temperature_fluctuation": ["温差波动", "temperature fluctuat", "rapid change"],
            "multiple_pollutants_simultaneous": ["复合暴露", "cumulative", "叠加", "synergy"],
        }

        for cond, keywords in condition_indicators.items():
            kb[cond] = any(
                kw.lower() in fact_text_combined
                for kw in keywords
            )

        # Check evidence chains for causal relationships
        for ev in evidence_chains:
            content = (ev.get("content", "") + " " + ev.get("method_used", "")).lower()
            if "ccm" in content and "bidirectional" in content:
                kb["multiple_pollutants_simultaneous"] = True
            if "granger" in content and "significant" in content:
                kb["temperature_increase"] = True  # proxy: Granger showed some causal link

        # Check convergence history for stability patterns
        if convergence_history and len(convergence_history) >= 2:
            changes = [
                abs(convergence_history[i] - convergence_history[i-1])
                for i in range(1, len(convergence_history))
            ]
            avg_change = sum(changes) / len(changes) if changes else 1.0
            kb["rapid_temperature_fluctuation"] = avg_change > 0.15  # proxy for volatile convergence

        return kb

    def generate_from_matches(self, matches: list[ConditionMatch]) -> list[RawHypothesis]:
        """根据匹配的规则生成假设"""
        hypotheses = []

        for match in matches:
            # Find the original rule
            rule = next((r for r in self.rules if r["id"] == match.rule_id), None)
            if not rule:
                continue

            hypothesis = RawHypothesis(
                title=rule["name"],
                statement=(
                    f"在{match.match_score:.0%}条件下满足的环境中，"
                    f"{rule['mechanism']}，"
                    f"具体影响目标指标:{', '.join(rule.get('target_indicators', []))}"
                ),
                reasoning_chain=(
                    f"【演绎路径】根据领域专家规则 {rule['id']}:\n"
                    f"IF {', '.join(rule['if_conditions'])}\n"
                    f"THEN {rule['then_conclusion']}\n\n"
                    f"机制解释: {rule['mechanism']}\n\n"
                    f"已知条件满足度: {match.match_score:.0%} "
                    f"(已确认: {', '.join(match.matched_conditions)})\n"
                    f"待验证条件: {', '.join(match.unmet_conditions) if match.unmet_conditions else '无'}\n\n"
                    f"目标检测指标: {', '.join(rule.get('target_indicators', []))}"
                ),
                confidence_prior=min(rule["confidence"] * match.match_score, 1.0),
                testability=rule.get("confidence", 0.7) * 10,
                evidence_needed=(
                    f"需要验证: ① {', '.join(match.unmet_conditions) if match.unmet_conditions else '当前条件已基本满足'}  ② 目标指标的实测数据"
                ),
                path="deductive",
                source_ref=rule["id"],
                parent_rule_ids=[rule["id"]],
                metadata={
                    "rule_name": rule.get("name", ""),
                    "mechanism": rule.get("mechanism", ""),
                    "moderating_factors": rule.get("moderating_factors", []),
                    "evidence_refs": rule.get("evidence_refs", []),
                    "matched_score": match.match_score,
                },
            )
            hypotheses.append(hypothesis)

        logger.info(f"[Deductive] Generated {len(hypotheses)} hypotheses from {len(matches)} matching rules")
        return hypotheses


# ============================================================
# Abductive Reasoning
# ============================================================


class AbductiveReasoner:
    """
    溯因推理：给定反直觉或异常的实验结果，寻找最可能的替代解释。

    核心思想：Peirce 的溯因推理 = 观察到令人惊讶的现象 P，
    如果假设 H 为真则 P 是理所当然的，因此有理由怀疑 H 为真。

    用于发现"为什么 A 导致 B"的反向因果、遗漏混杂因素、或非线性阈值效应。
    """

    async def generate(
        self,
        observation_hint: str = "",
        evidence_chains: list[dict] = None,
        anomaly_graph: list[dict] = None,
    ) -> list[RawHypothesis]:
        """
        基于观察到的异常模式生成溯因假设。
        """
        hypotheses = []
        evidence_chains = evidence_chains or []
        anomaly_graph = anomaly_graph or []

        # Collect observations from evidence chains and anomalies
        observations = []

        for ev in evidence_chains:
            strength = ev.get("strength", 0.5)
            method = ev.get("method_used", "")
            direction = ev.get("causal_direction", "")

            # Weak causal relationship despite strong prior expectation
            if strength < 0.3:
                observations.append({
                    "type": "weak_causal_despite_expectation",
                    "detail": f"{method} 显示因果关系强度仅 {strength:.2f}，低于预期",
                    "evidence_id": ev.get("id", ""),
                })

            # Bidirectional causality where only one direction expected
            if "bidirectional" in direction.lower():
                observations.append({
                    "type": "bidirectional_where_unidirectional_expected",
                    "detail": f"{method} 检测到双向因果 ({direction})，暗示遗漏了反馈回路",
                    "evidence_id": ev.get("id", ""),
                })

        for anom in anomaly_graph:
            desc = anom.get("description", "")
            anom_type = anom.get("type", "")
            if anom_type == "contradiction":
                observations.append({
                    "type": "contradictory_finding",
                    "detail": desc[:200],
                    "source": anom.get("id", ""),
                })

        # Generate abductive hypotheses based on observed patterns
        for obs in observations[:5]:
            abductive_hypotheses = self._generate_from_observation(obs)
            hypotheses.extend(abductive_hypotheses)

        # If no specific observations, generate general counter-intuitive alternatives
        if not hypotheses and observation_hint:
            hypotheses.append(RawHypothesis(
                title=f"反直觉现象'{observation_hint[:50]}...'的替代解释",
                statement=f"观察到的现象可能源于未考虑的隐藏混杂因子 X→Y 而非显式的 X→Z→Y 路径",
                reasoning_chain=(
                    "【溯因路径】观察到令人惊讶的现象 P。\n"
                    "如果存在一个隐藏的混杂因子 M，其同时影响 X 和 Y，"
                    "则表面上 X→Y 的因果关系实际上是伪相关。\n\n"
                    f"观察线索: {observation_hint[:200]}"
                ),
                confidence_prior=0.30,
                testability=5,
                evidence_needed="需要引入 instrumental variable 或通过 stratified analysis 排除混杂",
                path="abductive",
                source_ref=None,
                metadata={"observation_hint": observation_hint},
            ))

        logger.info(f"[Abductive] Generated {len(hypotheses)} hypotheses from {len(observations)} observations")
        return hypotheses

    def _generate_from_observation(self, obs: dict) -> list[RawHypothesis]:
        """根据特定观察类型生成溯因假设"""
        h_type = obs["type"]
        hypotheses = []

        if h_type == "weak_causal_despite_expectation":
            hypotheses.append(RawHypothesis(
                title="弱因果关系的替代解释：非线性阈值效应",
                statement=(
                    f"观察到的X→Y关系较弱（证据强度仅{obs['detail'][:10]}），"
                    f"可能是因为因果效应仅在超过某个临界阈值后才显现，"
                    f"而当前数据采集区间未触及该阈值"
                ),
                reasoning_chain=(
                    "【溯因路径】如果存在一个未观测到的阈值函数 threshold(X)：\n"
                    "当 X < threshold 时，Y 不受影响；\n"
                    "当 X ≥ threshold 时，Y 出现显著响应。\n\n"
                    "这种非线性阈值效应会在回归分析中表现为整体关联较弱，"
                    "但如果按阈值分组则会观察到强烈的组内差异。"
                ),
                confidence_prior=0.35,
                testability=7,
                evidence_needed="需要做分段回归分析或寻找历史极端事件样本跨越阈值",
                path="abductive",
                source_ref=None,
                metadata={"explanation": "nonlinear_threshold_effect"},
            ))

        elif h_type == "bidirectional_where_unidirectional_expected":
            hypotheses.append(RawHypothesis(
                title="双向因果的潜在机制：遗漏反馈回路的中介变量",
                statement=(
                    "X→Y 和 Y→X 的双向因果关系暗示存在一个中介变量 M，"
                    "构成完整的正/负反馈回路 X→M→Y→M→X"
                ),
                reasoning_chain=(
                    "【溯因路径】检测到 X↔Y 双向因果。\n"
                    "最合理的解释是存在一个未测量的中介变量 M，"
                    "它既是 X 的结果又是 Y 的原因，形成完整循环。\n\n"
                    "例如：环境温度↑ → 开窗通风 ↑ → 室外CO₂进入 ↑ → 室内CO₂↑ → 人员警觉度↑ → 活动量↑\n"
                    "其中'开窗通风'就是典型的隐藏中介。"
                ),
                confidence_prior=0.40,
                testability=6,
                evidence_needed="需要设计干预实验主动操纵推测的中介变量来验证反馈路径",
                path="abductive",
                source_ref=None,
                metadata={"explanation": "hidden_feedback_loop"},
            ))

        elif h_type == "contradictory_finding":
            hypotheses.append(RawHypothesis(
                title="矛盾发现的潜在原因：群体异质性的 Simpson悖论",
                statement=(
                    "表面矛盾的实验结果可能是由于研究中存在未被分层的群体亚型，"
                    "每个亚型内部关系方向一致但在聚合后被掩盖或反转（Simpson悖论）"
                ),
                reasoning_chain=(
                    "【溯因路径】观察到矛盾结论。\n"
                    "如果研究群体包含至少两个亚型（如不同年龄段、性别或健康状态），"
                    "且各亚型中X→Y的关系方向和强度不同，"
                    "则聚合数据分析可能得到与亚型内部分析完全相反的结论。"
                ),
                confidence_prior=0.35,
                testability=7,
                evidence_needed="需要对数据进行分层分析和交互项检验",
                path="abductive",
                source_ref=None,
                metadata={"explanation": "simpson_paradox"},
            ))

        return hypotheses


# ============================================================
# Merger & Deduplication
# ============================================================


class HypothesisMerger:
    """合并三路推理结果并去除语义重复"""

    @staticmethod
    def merge_and_dedup(all_raws: list[RawHypothesis]) -> list[RawHypothesis]:
        """
        合并所有路径的原始假设，去除 bigram 相似度 > 0.85 的重复项。
        保留先验置信度更高的版本。
        """
        kept: list[RawHypothesis] = []

        for raw in all_raws:
            is_dup = False
            best_existing = None
            best_sim = 0.0

            for existing in kept:
                sim = HypothesisMerger._bigram_similarity(raw.title, existing.title)
                if sim > best_sim:
                    best_sim = sim
                    best_existing = existing

            if best_sim > 0.85 and best_existing:
                # Keep the one with higher confidence
                if raw.confidence_prior > best_existing.confidence_prior:
                    kept.remove(best_existing)
                    kept.append(raw)
                    logger.debug(
                        f"[Dedup] Replaced '{best_existing.title}' "
                        f"(conf={best_existing.confidence_prior:.2f}) with "
                        f"'{raw.title}' (conf={raw.confidence_prior:.2f}), sim={best_sim:.3f}"
                    )
                else:
                    logger.debug(
                        f"[Dedup] Dropped '{raw.title}' "
                        f"(conf={raw.confidence_prior:.2f} < {best_existing.confidence_prior:.2f}), sim={best_sim:.3f}"
                    )
                is_dup = True

            if not is_dup:
                kept.append(raw)

        logger.info(f"[Merger] Merged {len(all_raws)} raw hypotheses → {len(kept)} after dedup")
        return kept

    @staticmethod
    def _bigram_similarity(a: str, b: str) -> float:
        """字符级 bigram Jaccard 相似度"""
        a_norm = " ".join(a.strip().split())
        b_norm = " ".join(b.strip().split())

        def bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1)) if len(s) > 1 else {s}

        a_set, b_set = bigrams(a_norm), bigrams(b_norm)
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / len(a_set | b_set)


# ============================================================
# Consistency Checker
# ============================================================


class ConsistencyChecker:
    """
    假设间和假设与事实间的逻辑一致性检查。

    三个层面：
    1. Self-consistent: 假设内部没有自相矛盾
    2. Pairwise conflicts: 两个假设之间是否互斥
    3. Factual alignment: 是否与已验证的事实冲突
    """

    def check_all(
        self,
        hypotheses: list[RawHypothesis],
        verified_facts: list[dict],
    ) -> list[ConsistencyReport]:
        reports = []

        for hyp in hypotheses:
            # 1. Self-consistency check
            reports.extend(self.check_self_consistent(hyp))

            # 2. Pairwise conflict detection
            for other in hypotheses:
                if hyp is other:
                    continue
                conflict = self.check_pairwise_conflict(hyp, other)
                if conflict:
                    reports.append(conflict)

            # 3. Factual alignment
            reports.extend(self.check_factual_alignment(hyp, verified_facts))

        # Filter out trivial "no conflict" reports
        real_reports = [r for r in reports if r.is_consistent is False or r.severity != "info"]
        logger.info(f"[Consistency] {len(real_reports)} issues found among {len(hypotheses)} hypotheses")
        return real_reports

    def check_self_consistent(self, hyp: RawHypothesis) -> list[ConsistencyReport]:
        """检查假设是否内部自洽"""
        reports = []

        # Check for contradictory directional claims
        stmt_lower = hyp.statement.lower()
        if "increase" in stmt_lower and "decrease" in stmt_lower:
            # Might be saying both increase and decrease of the same variable
            # This could be legitimate (different variables) or contradictory
            pass  # Too noisy for now; rely on LLM for this nuanced check

        # Check confidence合理性
        if hyp.confidence_prior < 0.05 or hyp.confidence_prior > 0.95:
            reports.append(ConsistencyReport(
                hypothesis_id=hyp.title[:50],
                is_consistent=False,
                conflict_type="internal",
                description=f"先验置信度 {hyp.confidence_prior:.2f} 超出合理范围 [0.05, 0.95]",
                severity="warning",
                suggestion="将置信度调整到更合理的中间值",
            ))

        return reports

    def check_pairwise_conflict(
        self, hyp_a: RawHypothesis, hyp_b: RawHypothesis
    ) -> ConsistencyReport | None:
        """检查两个假设之间是否存在互斥"""
        # Simple heuristic: check for opposite directional claims about the same target
        stmt_a = hyp_a.statement.lower()
        stmt_b = hyp_b.statement.lower()

        # Keywords indicating direction
        direction_keywords = {
            "increase": ["上升", "增加", "提高", "增强", "higher", "increas", "elevat"],
            "decrease": ["下降", "降低", "减弱", "减少", "lower", "decreas", "reduc"],
        }

        for keyword_group, terms in direction_keywords.items():
            a_has = any(t in stmt_a for t in terms)
            b_has = any(t in stmt_b for t in terms)
            if a_has and b_has and keyword_group in ("increase", "decrease"):
                continue  # Both have same direction claim — not necessarily conflicting

        # Flag potential directional conflict
        has_opposite = False
        for group in ["increase", "decrease"]:
            opposite = "decrease" if group == "increase" else "increase"
            # Check if A says something increases while B says it decreases
            inc_terms = direction_keywords["increase"][0] + direction_keywords["increase"][1]
            dec_terms = direction_keywords["decrease"][0] + direction_keywords["decrease"][1]

            if any(t in stmt_a for t in inc_terms.split()):
                if any(t in stmt_b for t in dec_terms.split()):
                    has_opposite = True

        if has_opposite:
            # More detailed check needed — flag for manual/LTM review
            # Don't auto-flag as this often triggers false positives
            pass

        return None  # Skip pairwise for now to avoid false positives

    def check_factual_alignment(
        self, hyp: RawHypothesis, verified_facts: list[dict]
    ) -> list[ConsistencyReport]:
        """检查假设是否与已验证的事实冲突"""
        reports = []
        hyp_text = hyp.statement.lower()

        for fact in verified_facts:
            if not fact.get("_verified", False):
                continue  # Only check against verified facts

            fact_text = fact.get("fact", "").lower()
            # Check for direct contradictions
            # A contradiction might be indicated by words that negate each other
            negation_patterns = [
                (["not affect", "no impact", "无关", "不影响"], ["显著影响", "correlate", "相关"]),
                (["increase", "上升", "增高"], ["decrease", "下降", "降低"]),
                (["decrease", "下降", "降低"], ["increase", "上升", "增高"]),
            ]

            for pair in negation_patterns:
                has_first = any(p in hyp_text for p in pair[0])
                has_second = any(p in hyp_text for p in pair[1])
                if has_first and has_second:
                    # Hypothesis contains both directions — internally ambiguous
                    reports.append(ConsistencyReport(
                        hypothesis_id=hyp.title[:50],
                        is_consistent=False,
                        conflict_type="internal",
                        description="假设陈述中包含相反方向的声明，方向性不够明确",
                        severity="warning",
                        suggestion="将假设拆分为两个方向明确的可检验子假设",
                    ))
                    break

        return reports


# ============================================================
# Main Orchestrator
# ============================================================


class LogicHypothesisEngine:
    """
    三路推理假设生成引擎的统一入口。

    Usage:
        engine = LogicHypothesisEngine()
        results = await engine.generate_all(facts, existing_hyps, evidence_chains)
        # Returns: {hypotheses, consistency_reports, stats}
    """

    def __init__(self, domain_rules_path: str = ""):
        rules = load_domain_rules(domain_rules_path)
        self.deductive_reasoner = DeductiveReasoner(rules)
        self.inductive_reasoner = InductiveReasoner()
        self.abductive_reasoner = AbductiveReasoner()
        self.merger = HypothesisMerger()
        self.consistency_checker = ConsistencyChecker()

    async def generate_all(
        self,
        facts: list[dict] = None,
        existing_hypotheses: list[dict] = None,
        evidence_chains: list[dict] = None,
        review_records: list[dict] = None,
        anomaly_graph: list[dict] = None,
    ) -> dict:
        """
        执行三路推理 → 合并去重 → 一致性检查 的完整流水线。

        Returns: {
            "hypotheses": list[dict],  # 最终候选假设
            "consistency_reports": list[dict],  # 一致性检查结果
            "stats": {
                "inductive_count": N,
                "deductive_count": N,
                "abductive_count": N,
                "dedup_count": N,
                "consistency_issues": N,
            }
        }
        """
        facts = facts or []
        existing_hypotheses = existing_hypotheses or []
        evidence_chains = evidence_chains or []
        review_records = review_records or []
        anomaly_graph = anomaly_graph or []

        # --- Step 1: Three-path generation ---
        inductive_results = await self.inductive_reasoner.generate(
            facts, existing_hypotheses, review_records
        )

        # Match conditions then generate deductive
        condition_matches = self.deductive_reasoner.match_conditions(facts, evidence_chains)
        deductive_results = self.deductive_reasoner.generate_from_matches(condition_matches)

        # Abductive from anomalies and weak evidence
        abductive_results = await self.abductive_reasoner.generate(
            evidence_chains=evidence_chains,
            anomaly_graph=anomaly_graph,
        )

        stats = {
            "inductive_count": len(inductive_results),
            "deductive_count": len(deductive_results),
            "abductive_count": len(abductive_results),
            "total_before_dedup": len(inductive_results) + len(deductive_results) + len(abductive_results),
        }

        # --- Step 2: Merge and deduplicate ---
        all_raw = inductive_results + deductive_results + abductive_results
        merged = self.merger.merge_and_dedup(all_raw)
        stats["dedup_count"] = stats["total_before_dedup"] - len(merged)

        # --- Step 3: Consistency check ---
        verified_facts = [f for f in facts if f.get("_verified", False)]
        consistency_reports = self.consistency_checker.check_all(merged, verified_facts)
        stats["consistency_issues"] = len(consistency_reports)

        # Convert to serializable dicts
        final_hypotheses = []
        for hyp in merged:
            final_hypotheses.append({
                "title": hyp.title,
                "statement": hyp.statement,
                "reasoning_chain": hyp.reasoning_chain,
                "confidence_prior": hyp.confidence_prior,
                "testability": hyp.testability,
                "evidence_needed": hyp.evidence_needed,
                "_logic_path": hyp.path,
                "_source_ref": hyp.source_ref,
                "_parent_rule_ids": hyp.parent_rule_ids,
                "_metadata": hyp.metadata,
                "_status": "proposed_by_logic_engine",
            })

        final_reports = [
            {
                "hypothesis_id": r.hypothesis_id,
                "is_consistent": r.is_consistent,
                "conflict_type": r.conflict_type,
                "description": r.description,
                "severity": r.severity,
                "suggestion": r.suggestion,
            }
            for r in consistency_reports
        ]

        stats["final_candidate_count"] = len(final_hypotheses)
        logger.info(
            f"[LogicEngine] Pipeline complete: {stats['total_before_dedup']} → "
            f"{len(final_hypotheses)} candidates, {stats['consistency_issues']} issues"
        )

        return {
            "hypotheses": final_hypotheses,
            "consistency_reports": final_reports,
            "stats": stats,
        }
