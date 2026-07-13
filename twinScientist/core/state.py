"""
Layer 3: State Space & Memory Management — Four-Dimensional Memory Body

从 Base Agent 的 Dual-Track（messages + data_profile）扩展为四维记忆体：

L1 / Kernel          → WorkingMemory: 当前迭代轮次的活跃状态
L2 / SQLite+Vector   → EpisodicMemory: 历史实验记录、交互上下文快照
L3 / KnowledgeGraph+Vector → SemanticMemory: 结构化科学知识、文献图谱
L4 / EvidenceChain    → EvidenceTrack: 实验完整记录和因果证据链
L5 / AnomalyGraph     → AnomalyTrack: 反常规数据片段和异常模式库
"""

from __future__ import annotations

import json
import networkx as nx
from pathlib import Path
from typing import Any, TypedDict


class StateCheckpoint:
    """
    状态检查点 — 实现时间旅行能力 (Item 17)

    在每次关键节点操作后自动保存 checkpoint，
    支持历史状态回滚和对比分析。
    """

    def __init__(self, checkpoint_dir: str = "./data/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_count = 0

    def save(self, state: dict, label: str = "") -> str:
        """保存当前状态为检查点"""
        self._checkpoint_count += 1
        fname = f"checkpoint_{self._checkpoint_count:04d}_{label}"
        path = self.checkpoint_dir / f"{fname}.json"
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def load(self, index: int) -> dict | None:
        """加载指定序号的检查点"""
        patterns = [f"checkpoint_{index:04d}_*.json"]
        files = list(self.checkpoint_dir.glob(patterns[0]))
        if not files:
            return None
        return json.loads(files[0].read_text(encoding="utf-8"))

    def list_all(self) -> list[tuple[int, str]]:
        """列出所有检查点 (序号, 文件名)"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
        return [(i + 1, cp.name) for i, cp in enumerate(checkpoints)]


def serialize_state_for_export(state: dict) -> dict:
    """导出时清理 LangGraph 内部字段，生成干净的 JSON"""
    clean = {}
    skip_keys = {"_max_iterations_", "_termination_result"}
    for k, v in state.items():
        if k.startswith("_") or k.startswith("__"):
            continue
        if isinstance(v, dict):
            clean[k] = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        else:
            clean[k] = v
    return clean


# ---- Type definitions preserved below ----

class HypothesisNode(TypedDict, total=False):
    """假设树中的一个节点"""
    id: str
    statement: str               # 假设陈述
    title: str                   # 简短标题
    reasoning_chain: str         # 推理链条
    confidence_prior: float      # Bayesian 先验概率 P(H)
    confidence_posterior: float  # 后验概率 P(H|D)
    testability: int             # 可检验性评分 1-10
    status: str                  # "proposed" | "active" | "confirmed" | "refuted" | "pruned"
    parent_id: str | None        # 父节点 ID
    children_ids: list[str]
    evidence_support: list[str]  # 支撑该假设的证据 ID
    evidence_against: list[str]  # 反驳该假设的证据 ID
    experiment_ids: list[str]    # 关联的实验 ID
    created_at: str              # ISO 时间戳
    updated_at: str


class ExperimentRecord(TypedDict, total=False):
    """单次实验的完整记录"""
    id: str
    hypothesis_id: str
    design: dict                 # 实验设计方案
    input_data_path: str         # 输入数据路径
    output_data_path: str        # 输出数据路径
    results: dict                # 实验结果（含指标）
    code_history: list[str]      # 执行的代码序列
    notes: str                   # 人工备注
    created_at: str


class ReviewRecord(TypedDict, total=False):
    """Reviewer Agent 的五维评审记录"""
    id: str
    hypothesis_id: str
    novelty_score: int           # 0-20
    feasibility_score: int       # 0-20
    methodology_score: int       # 0-20
    evidence_score: int          # 0-20
    impact_score: int            # 0-20
    total_score: int
    comments: str
    needs_revision: bool
    revision_instructions: str
    created_at: str


class AnomalyEntry(TypedDict, total=False):
    """异常图谱中的一个条目"""
    id: str
    type: str                    # "outlier" | "contradiction" | "pattern_break"
    source_experiment_id: str
    description: str
    severity: str                # "low" | "medium" | "high"
    metadata: dict


class EvidenceEntry(TypedDict, total=False):
    """
    证据链节点 — 国际 SOTA 标准化结构

    包含因果推断的完整统计依据和方法参数，使每条证据都可追溯、可复现。
    """
    id: str                      # UUID
    type: str                    # "data" | "literature" | "causal_inference" | "statistical_test"
    strength: float              # 0-1 置信度
    content: str                 # 原始内容或摘要
    linked_hypotheses: list[str]
    linked_experiments: list[str]

    # Causal-specific fields
    method_used: str | None      # "ccm" | "granger" | "counterfactual"
    method_params: dict | None   # All parameters passed to the causal method
    statistical_basis: dict | None  # p-values, F-stats, rho values, etc.
    validation_results: dict | None  # convergence tests, stationarity checks

    causal_direction: str | None  # "A→B" | "B→A" | "bidirectional" | None
    provenance: str | None        # Human-readable description of how generated
    created_at: str


class AgentState(TypedDict, total=False):
    """
    twinScientist 全局共享状态 —— 四维记忆体驱动

    每条消息都通过 LangGraph annotator（operator.add 或 operator.replace）管理，
    确保状态的正确追加与替换。
    """

    # ============================================================
    # Input
    # ============================================================
    query: str                          # 用户初始问题
    domain: str                         # 学科领域（如 "环境—人体关联"）

    # ============================================================
    # L1: Working Memory (Kernel) — current iteration context
    # ============================================================
    iteration: int                      # 当前迭代轮次
    round_message: list[dict]           # 当前轮的 LLM 对话消息（追加）
    current_action: str                 # 当前认知操作类型
    pending_approval: bool              # 是否等待人类确认
    auto_confirm: bool                  # 是否开启自动确认
    user_feedback: str | None           # 用户在 HITL 断点处输入的反馈/指令

    # ============================================================
    # L2: Episodic Memory (SQLite + Vector DB) — historical snapshots
    # ============================================================
    experiment_records: list[dict]      # 历史实验记录列表（追加）
    review_records: list[dict]          # 历史评审记录列表（追加）

    # ============================================================
    # L3: Semantic Memory (Knowledge Graph + Vector)
    # ============================================================
    knowledge_graph: dict               # NetworkX graph (raw) + serializable nodes/edges
    fact_extraction: list[dict]         # 从文献中提取的关键事实
    literature_summary: str             # 领域文献综述摘要

    # ============================================================
    # L4-L5: Evidence Chain & Anomaly Graph
    # ============================================================
    evidence_chains: list[dict]         # 证据链集合（追加）
    anomaly_graph: list[dict]           # 异常图谱（追加）

    # ============================================================
    # Hypothesis Tree
    # ============================================================
    hypothesis_tree: list[dict]         # 动态假设树根节点列表（直接替换）
    elimination_records: list[dict]     # 淘汰赛记录：{round, pair, winner_id, loser_id, reason}

    # ============================================================
    # Orchestrator Runtime State
    # ============================================================
    orchestrator_state: str             # "researching" | "experimenting" | "reviewing" | "writing" | "converged"
    uncertainty_level: float            # 当前不确定性估计 0-1
    convergence_score: float            # 语义收敛度 0-1
    convergence_history: list[float]    # 历史收敛度序列，用于检测连续稳定 (e.g. [0.0, 0.72, 0.68])
    exploration_exhausted: bool         # 探索是否已穷尽
    consecutive_failures: int           # 连续失败计数（防死循环）
    prev_round_winner_id: str | None    # 上一轮优胜假设 ID（用于跨轮相似度比较）
    prev_round_winner_statement: str | None  # 上一轮优胜假设的陈述（用于语义相似度计算）
    next_step: str | None               # 下一步操作（供路由使用）

    # ============================================================
    # Output Channel
    # ============================================================
    final_report: str | None            # 最终生成的标准化报告
    export_paths: list[str]             # 导出文件路径列表
