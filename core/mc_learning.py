"""
Monte Carlo Reinforcement Learning Engine — 蒙特卡洛经验学习系统

为 twinScientist 的 Orchestrator 路由决策提供基于经验的策略学习。

核心算法:
- First-Visit Monte Carlo: 每个 episode 中，对每个 state-action pair 只更新首次访问的回报
- Discounted Returns: G_t = R_{t+1} + γ·R_{t+2} + γ²·R_{t+3} + ...
- Epsilon-Greedy Exploration: 以 ε 概率随机探索，1-ε 概率利用最优策略
- State Discretization: 将连续特征离散化为有限状态空间
- Incremental Mean: Q(s,a) ← Q(s,a) + α·(G_t - Q(s,a))

设计原则:
- 不改变现有 node/graph/LLM 调用逻辑
- 纯 append-only 持久化，失败不会破坏运行
- 特征向量从现有 state 字段派生，零额外依赖
- 与 experience.py 协作：experience 记录原始数据，mc_learning 学习策略

Usage:
    from core.mc_learning import mc_policy

    # After each node execution, log the step:
    mc_policy.log_step(state, "hypothesis_generation")

    # Get RL recommendation for next routing decision:
    recommendation = mc_policy.recommend(state)

    # Format as LLM prompt context:
    rl_context = mc_policy.format_recommendation_for_prompt(recommendation)

    # At session end, update Q-values from episode:
    mc_policy.update_from_episode()
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# State Discretization — 将连续特征映射到离散状态空间
# ============================================================

# Discretization bins for continuous features
# Each bin boundary creates a categorical bucket
BINS = {
    "iteration_phase": [0, 1, 3, 5, 10, 50, 200],       # 研究阶段
    "avg_evidence": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],     # 证据强度
    "convergence": [0.0, 0.3, 0.6, 0.85, 1.0],           # 收敛度
    "review_score": [0, 50, 65, 75, 85, 100],             # 评审分数
    "num_hyps": [0, 1, 3, 5, 10],                        # 假设数量
    "uncertainty": [0.0, 0.3, 0.6, 1.0],                 # 不确定性
    "consecutive_failures": [0, 1, 2, 3],                # 连续失败
}


def _discretize(value: float, bins: list[float]) -> int:
    """将连续值映射到离散桶索引"""
    for i, boundary in enumerate(bins):
        if value <= boundary:
            return i
    return len(bins)


def extract_state_key(state: dict) -> str:
    """
    从 AgentState 提取离散化的状态键。

    状态空间维度:
    - iteration_phase: 7 bins (研究阶段)
    - avg_evidence: 6 bins (证据强度)
    - convergence: 5 bins (收敛度)
    - review_score: 6 bins (评审分数)
    - num_hyps: 5 bins (假设数量)
    - prev_action: 分类 (上一步操作)

    总状态空间 ≈ 7 × 6 × 5 × 6 × 5 = 6300 个离散状态
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("_max_iterations_", 200)

    # Evidence strength
    evidence_chains = state.get("evidence_chains", [])
    if evidence_chains:
        avg_evidence = sum(e.get("strength", 0.5) for e in evidence_chains) / len(evidence_chains)
    else:
        avg_evidence = 0.0

    # Convergence
    convergence = state.get("convergence_score", 0.0)

    # Review score
    reviews = state.get("review_records", [])
    latest_review = reviews[-1].get("total_score", 0) if reviews else 0

    # Hypothesis count
    hypotheses = state.get("hypothesis_tree", [])
    num_hyps = len([h for h in hypotheses if h.get("status") not in ("pruned", "refuted")])

    # Previous action
    prev_action = state.get("current_action", "none")

    # Consecutive failures
    failures = state.get("consecutive_failures", 0)

    # Build composite state key
    parts = [
        f"iter{_discretize(iteration, BINS['iteration_phase'])}",
        f"ev{_discretize(avg_evidence, BINS['avg_evidence'])}",
        f"conv{_discretize(convergence, BINS['convergence'])}",
        f"rev{_discretize(latest_review, BINS['review_score'])}",
        f"hyps{_discretize(num_hyps, BINS['num_hyps'])}",
        f"prev_{prev_action}",
    ]
    return "|".join(parts)


def extract_feature_vector(state: dict) -> dict[str, Any]:
    """提取原始特征向量（用于存储和调试）"""
    evidence_chains = state.get("evidence_chains", [])
    reviews = state.get("review_records", [])
    hypotheses = state.get("hypothesis_tree", [])
    experiments = state.get("experiment_records", [])
    anomaly_graph = state.get("anomaly_graph", [])

    # Hypothesis status distribution
    approved = sum(1 for h in hypotheses if h.get("status") == "approved_by_reviewer")
    proposed = sum(1 for h in hypotheses if h.get("status") == "proposed")
    refuted  = sum(1 for h in hypotheses
                   if h.get("status") in ("refuted", "refuted_in_tournament"))
    needs_rev = sum(1 for h in hypotheses if h.get("status") == "needs_revision")
    pruned   = sum(1 for h in hypotheses if h.get("status") == "pruned")
    active   = sum(1 for h in hypotheses if h.get("status") == "active")

    rev_scores = [r.get("total_score", 50) for r in reviews]

    return {
        "iteration": state.get("iteration", 0),
        "remaining_budget": max(
            state.get("_max_iterations_", 200) - state.get("iteration", 0), 0
        ),
        "avg_evidence": round(
            sum(e.get("strength", 0.5) for e in evidence_chains) / max(len(evidence_chains), 1), 3
        ),
        "max_evidence": round(
            max((e.get("strength", 0.5) for e in evidence_chains), default=0.0), 3
        ),
        "min_evidence": round(
            min((e.get("strength", 0.5) for e in evidence_chains), default=0.0), 3
        ),
        "convergence": state.get("convergence_score", 0.0),
        "latest_review": rev_scores[-1] if rev_scores else 0,
        "avg_review": round(
            sum(rev_scores) / max(len(rev_scores), 1), 1
        ) if rev_scores else 0,
        "num_hyps": len(hypotheses),
        "approved": approved,
        "proposed": proposed,
        "active": active,
        "refuted": refuted,
        "needs_revision": needs_rev,
        "pruned": pruned,
        "num_evidence": len(evidence_chains),
        "num_experiments": len(experiments),
        "consecutive_failures": state.get("consecutive_failures", 0),
        "prev_action": state.get("current_action", "none"),
        "uncertainty": state.get("uncertainty_level", 0.5),
        "anomaly_count": len(anomaly_graph),
        "has_real_analysis": any(
            isinstance(e, dict) and e.get("type") == "causal_inference"
            for e in evidence_chains
        ),
    }


# ============================================================
# Reward Computation — 多信号融合奖励函数
# ============================================================

# Valid routing actions (matches AVAILABLE_ACTIONS in orchestrator.py)
VALID_ACTIONS = [
    "literature_review",
    "hypothesis_generation",
    "experiment_design",
    "data_analysis",
    "interpretation",
    "reviewer_agent",
    "reflection",
    "termination_eval",
    "report_writing",
    "pi_agent_meeting",
    "evolution_manager",
]


def compute_step_reward(state: dict, action: str, next_state: dict | None = None) -> float:
    """
    计算单步奖励 — 基于当前状态和选择的动作。

    奖励信号来源:
    1. 评审分数变化 (Δreview_score) — 假设质量改善
    2. 证据强度增长 (Δevidence) — 数据支撑增强
    3. 收敛度提升 (Δconvergence) — 研究趋于稳定
    4. 动作合理性 (action_fitness) — 是否选择了逻辑上合适的动作
    5. 效率奖励 (efficiency) — 避免无意义循环

    Returns: float in [-1.0, 1.0]
    """
    # --- Signal 1: Review score quality (bounded to [-0.15, +0.15]) ---
    review_r = 0.0
    reviews = state.get("review_records", [])
    if reviews:
        latest_score = reviews[-1].get("total_score", 50)
        if isinstance(latest_score, (int, float)) and not isinstance(latest_score, bool):
            latest_score = max(0, min(100, latest_score))
            # Normalize to [-0.5, 0.5] range: score 0→-0.5, 50→0, 100→+0.5
            review_r = (latest_score - 50) / 100.0 * 0.3

    # --- Signal 2: Evidence strength (bounded to [0, +0.2]) ---
    evidence_r = 0.0
    evidence_chains = state.get("evidence_chains", [])
    if evidence_chains:
        avg_strength = sum(
            max(0.0, min(1.0, e.get("strength", 0.5))) for e in evidence_chains
        ) / len(evidence_chains)
        evidence_r = avg_strength * 0.2

    # --- Signal 3: Convergence progress (bounded to [0, +0.15]) ---
    convergence = max(0.0, min(1.0, state.get("convergence_score", 0.0)))
    convergence_r = convergence * 0.15

    # --- Signal 4: Action fitness (domain knowledge heuristics) ---
    hypotheses = state.get("hypothesis_tree", [])
    proposed = [h for h in hypotheses if h.get("status") == "proposed"]
    active = [h for h in hypotheses if h.get("status") == "active"]
    approved = [h for h in hypotheses if h.get("status") == "approved_by_reviewer"]
    prev_action = state.get("current_action", "none")
    iteration = state.get("iteration", 0)
    max_iter = state.get("_max_iterations_", 200)

    fitness = 0.0
    # Good: propose experiments for active hypotheses
    if action == "experiment_design" and (active or proposed):
        fitness += 0.15
    # Good: analyze after experiment design
    if action == "data_analysis" and prev_action == "experiment_design":
        fitness += 0.1
    # Good: interpret after analysis
    if action == "interpretation" and prev_action == "data_analysis":
        fitness += 0.1
    # Good: review after interpretation
    if action == "reviewer_agent" and prev_action == "interpretation":
        fitness += 0.1
    # Good: reflect when score is low
    if action == "reflection" and reviews and reviews[-1].get("total_score", 50) < 75:
        fitness += 0.15
    # Good: terminate when evidence is strong
    if action in ("termination_eval", "report_writing") and evidence_chains:
        avg_ev = sum(e.get("strength", 0.5) for e in evidence_chains) / len(evidence_chains)
        if avg_ev > 0.7:
            fitness += 0.15
    # Bad: repeat the same action
    if action == prev_action:
        fitness -= 0.2
    # Bad: generate more hypotheses when there are already many untested ones
    if action == "hypothesis_generation" and len(proposed) > 5:
        fitness -= 0.1
    # Bad: try to write report with no evidence
    if action == "report_writing" and not evidence_chains and not approved:
        fitness -= 0.15
    # Bad: ignore max iterations
    if iteration >= max_iter - 2 and action not in ("termination_eval", "report_writing"):
        fitness -= 0.1

    # Bound fitness to [-0.2, +0.15] before weighting
    fitness = max(-0.2, min(0.15, fitness))
    fitness_r = fitness * 0.35

    # --- Combine with explicit clamping ---
    reward = review_r + evidence_r + convergence_r + fitness_r
    return round(max(-1.0, min(1.0, reward)), 4)


def compute_terminal_reward(state: dict) -> float:
    """
    计算 episode 终止时的终端奖励。

    基于最终研究质量:
    - 评审通过 → 正奖励
    - 证据充分 → 正奖励
    - 高效完成 (少迭代) → 效率奖励
    - 失败终止 (连续失败≥3) → 负奖励
    """
    reward = 0.0

    reviews = state.get("review_records", [])
    evidence_chains = state.get("evidence_chains", [])
    iteration = state.get("iteration", 0)
    max_iter = state.get("_max_iterations_", 200)
    failures = state.get("consecutive_failures", 0)

    # Review quality
    if reviews:
        best_score = max(r.get("total_score", 0) for r in reviews)
        reward += (best_score - 50) / 50.0 * 0.4  # -0.4 to +0.4

    # Evidence quality
    if evidence_chains:
        avg_ev = sum(e.get("strength", 0.5) for e in evidence_chains) / len(evidence_chains)
        reward += avg_ev * 0.3

    # Efficiency bonus (fewer iterations = more efficient)
    efficiency = max(0, 1 - iteration / max(max_iter, 1))
    reward += efficiency * 0.15

    # Failure penalty
    if failures >= 3:
        reward -= 0.3

    return round(max(-1.0, min(1.0, reward)), 4)


# ============================================================
# SQLite Schema — Q-value 存储
# ============================================================

MC_SCHEMA_SQL = """
-- Q-value table: state-action value estimates
CREATE TABLE IF NOT EXISTS mc_qvalues (
    state_key       TEXT NOT NULL,
    action          TEXT NOT NULL,
    q_value         REAL DEFAULT 0.0,     -- Estimated Q(s,a)
    visit_count     INTEGER DEFAULT 0,    -- Number of times this (s,a) was visited
    last_updated    REAL NOT NULL,        -- Timestamp of last update
    PRIMARY KEY (state_key, action)
);

-- Episode history: full state-action-reward trajectories
CREATE TABLE IF NOT EXISTS mc_episodes (
    episode_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    domain          TEXT NOT NULL,
    started_at      REAL NOT NULL,
    ended_at        REAL,
    total_return    REAL DEFAULT 0.0,     -- Sum of discounted rewards
    num_steps       INTEGER DEFAULT 0,
    terminal_reward REAL DEFAULT 0.0,
    status          TEXT DEFAULT 'active' -- 'active' | 'completed' | 'failed'
);

-- Step-level trajectory data
CREATE TABLE IF NOT EXISTS mc_steps (
    episode_id      INTEGER REFERENCES mc_episodes(episode_id),
    step_idx        INTEGER NOT NULL,
    state_key       TEXT NOT NULL,
    action          TEXT NOT NULL,
    reward          REAL DEFAULT 0.0,     -- Immediate reward
    mc_return       REAL DEFAULT 0.0,     -- Discounted cumulative return G_t
    feature_json    TEXT,                 -- Raw feature vector for debugging
    ts              REAL NOT NULL,
    PRIMARY KEY (episode_id, step_idx)
);

CREATE INDEX IF NOT EXISTS idx_mc_steps_state ON mc_steps(state_key);
CREATE INDEX IF NOT EXISTS idx_mc_qvalues_state ON mc_qvalues(state_key);

-- Policy summary cache (aggregated from Q-values)
CREATE TABLE IF NOT EXISTS mc_policy_cache (
    domain          TEXT NOT NULL,
    state_pattern   TEXT NOT NULL,        -- Partial state key pattern
    best_action     TEXT NOT NULL,
    best_q_value    REAL DEFAULT 0.0,
    confidence      REAL DEFAULT 0.0,     -- visit_count / total_visits
    total_visits    INTEGER DEFAULT 0,
    PRIMARY KEY (domain, state_pattern)
);
"""


# ============================================================
# Monte Carlo Policy — 核心学习引擎
# ============================================================

class MonteCarloPolicy:
    """
    蒙特卡洛策略学习器。

    使用 First-Visit MC 方法从完整的 episode 轨迹中学习 Q(s,a) 值。
    学到的策略通过 epsilon-greedy 方式推荐动作给 LLM Orchestrator。

    Attributes:
        gamma: 折扣因子 (0-1)，控制未来奖励的权重
        epsilon: 探索概率 (0-1)，epsilon-greedy 策略的随机性
        alpha: 学习率 (0-1)，增量更新的步长
        min_visits: 最小访问次数，低于此值时不做推荐（数据不足）
    """

    def __init__(
        self,
        db_path: str | None = None,
        gamma: float = 0.9,
        epsilon: float = 0.2,
        alpha: float = 0.1,
        min_visits: int = 3,
    ):
        self._db_path = Path(db_path or "./data/mc_policy.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._gamma = gamma
        self._epsilon = epsilon
        self._alpha = alpha
        self._min_visits = min_visits

        # Per-episode in-memory buffers
        self._current_episode_id: int | None = None
        self._pending_steps: list[dict] = []
        self._domain: str = ""

    # ---- Connection management ----

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            try:
                self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.executescript(MC_SCHEMA_SQL)
                logger.info(f"[MCPolicy] Initialized MC DB at {self._db_path}")
            except Exception as e:
                logger.warning(f"[MCPolicy] Failed to open DB ({e}), running without persistence")
                raise
        return self._conn

    def close(self):
        if self._conn:
            self.conn.commit()
            self._conn.close()
            self._conn = None

    # ---- Episode lifecycle ----

    def begin_episode(self, domain: str, query: str) -> int | None:
        """开始新的 episode（对应一次完整的研究 session）"""
        try:
            cursor = self.conn.execute(
                """INSERT INTO mc_episodes (domain, started_at, status)
                   VALUES (?, ?, 'active')""",
                (domain, time.time()),
            )
            self.conn.commit()
            self._current_episode_id = cursor.lastrowid
            self._pending_steps = []
            self._domain = domain
            logger.info(f"[MCPolicy] Episode #{cursor.lastrowid} started for domain='{domain}'")
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"[MCPolicy] Failed to begin episode: {e}")
            return None

    def log_step(self, state: dict, action: str) -> bool:
        """
        记录一步状态-动作对。

        应在每个认知节点执行完毕后、路由决策做出前调用。
        action 参数是**即将选择的下一个动作**，不是刚执行完的节点。

        每一步都立即持久化到 SQLite，防止进程中断导致数据丢失。
        """
        if self._current_episode_id is None:
            logger.warning("[MCPolicy] log_step skipped: no active episode")
            return False
        try:
            state_key = extract_state_key(state)
            features = extract_feature_vector(state)
            reward = compute_step_reward(state, action)

            step = {
                "step_idx": len(self._pending_steps),
                "state_key": state_key,
                "action": action,
                "reward": reward,
                "feature_json": json.dumps(features, ensure_ascii=False),
                "ts": time.time(),
            }
            self._pending_steps.append(step)

            # Incremental persistence — write step to DB immediately
            try:
                self.conn.execute(
                    """INSERT OR REPLACE INTO mc_steps
                       (episode_id, step_idx, state_key, action, reward, mc_return, feature_json, ts)
                       VALUES (?, ?, ?, ?, ?, 0, ?, ?)""",
                    (self._current_episode_id, step["step_idx"], state_key,
                     action, reward, step["feature_json"], step["ts"]),
                )
                self.conn.commit()
                logger.info(f"[MCPolicy] Step {step['step_idx']} persisted: {action}")
            except Exception as e:
                logger.warning(f"[MCPolicy] Incremental write failed: {e}")

            return True
        except Exception as e:
            logger.debug(f"[MCPolicy] Step logging failed: {e}")
            return False

    def update_from_episode(self, final_state: dict | None = None) -> int | None:
        """
        Episode 结束后调用：计算折扣回报并更新 Q-values。

        使用 First-Visit Monte Carlo 方法：
        1. 从 episode 末尾向前计算每步的折扣累积回报 G_t
        2. 对每个首次出现的 (state, action) 对，增量更新 Q(s,a)

        G_T = R_T + terminal_reward
        G_t = R_t + γ · G_{t+1}
        Q(s,a) ← Q(s,a) + α · (G_t - Q(s,a))
        """
        eid = self._current_episode_id
        if eid is None or not self._pending_steps:
            return None

        try:
            # Compute terminal reward
            terminal_reward = 0.0
            if final_state:
                terminal_reward = compute_terminal_reward(final_state)

            # --- Step 1: Compute discounted returns (backward pass) ---
            n = len(self._pending_steps)
            returns = [0.0] * n

            # Last step: return = immediate_reward + terminal_reward
            returns[n - 1] = self._pending_steps[n - 1]["reward"] + terminal_reward

            # Backward: G_t = R_t + γ · G_{t+1}
            for t in range(n - 2, -1, -1):
                returns[t] = self._pending_steps[t]["reward"] + self._gamma * returns[t + 1]

            # Assign returns to steps
            for i, step in enumerate(self._pending_steps):
                step["mc_return"] = round(returns[i], 4)

            # --- Step 2: First-Visit MC update ---
            seen_pairs = set()
            for step in self._pending_steps:
                pair = (step["state_key"], step["action"])
                if pair in seen_pairs:
                    continue  # First-Visit: skip subsequent visits
                seen_pairs.add(pair)

                self._update_q_value(
                    step["state_key"],
                    step["action"],
                    step["mc_return"],
                )

            # --- Step 3: Persist to database ---
            total_return = sum(s["mc_return"] for s in self._pending_steps) / n if n > 0 else 0.0

            self.conn.execute(
                """UPDATE mc_episodes SET
                    ended_at = ?, total_return = ?, num_steps = ?,
                    terminal_reward = ?, status = 'completed'
                   WHERE episode_id = ?""",
                (time.time(), round(total_return, 4), n, terminal_reward, eid),
            )

            # Batch-insert steps
            rows = [
                (eid, s["step_idx"], s["state_key"], s["action"],
                 s["reward"], s["mc_return"], s["feature_json"], s["ts"])
                for s in self._pending_steps
            ]
            self.conn.executemany(
                """INSERT OR REPLACE INTO mc_steps
                   (episode_id, step_idx, state_key, action, reward, mc_return, feature_json, ts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

            # Refresh policy cache
            self._refresh_policy_cache()

            self.conn.commit()

            logger.info(
                f"[MCPolicy] Episode #{eid} updated: {n} steps, "
                f"terminal_reward={terminal_reward:.3f}, avg_return={total_return:.3f}"
            )
            return eid

        except Exception as e:
            logger.error(f"[MCPolicy] Episode update failed for #{eid}: {e}", exc_info=True)
            return None
        finally:
            self._current_episode_id = None
            self._pending_steps = []

    # ---- Q-value updates ----

    def _update_q_value(self, state_key: str, action: str, mc_return: float):
        """增量更新 Q(s,a) ← Q(s,a) + α·(G_t - Q(s,a))"""
        try:
            conn = self.conn
            row = conn.execute(
                "SELECT q_value, visit_count FROM mc_qvalues WHERE state_key=? AND action=?",
                (state_key, action),
            ).fetchone()

            if row is None:
                # First time seeing this (s,a) pair
                conn.execute(
                    """INSERT INTO mc_qvalues (state_key, action, q_value, visit_count, last_updated)
                       VALUES (?, ?, ?, 1, ?)""",
                    (state_key, action, round(mc_return, 4), time.time()),
                )
            else:
                old_q, visits = row
                new_q = old_q + self._alpha * (mc_return - old_q)
                conn.execute(
                    """UPDATE mc_qvalues SET q_value=?, visit_count=?, last_updated=?
                       WHERE state_key=? AND action=?""",
                    (round(new_q, 4), visits + 1, time.time(), state_key, action),
                )
        except Exception as e:
            logger.debug(f"[MCPolicy] Q-value update failed: {e}")

    # ---- Policy recommendation ----

    def recommend(self, state: dict) -> dict:
        """
        基于学到的 Q-values 推荐下一步动作。

        使用 epsilon-greedy 策略:
        - 以 ε 概率随机选择（探索）
        - 以 1-ε 概率选择 Q(s,a) 最大的动作（利用）

        Returns: {
            "recommended_action": str,
            "q_values": dict[str, float],     # All actions and their Q-values
            "best_action": str,               # Action with highest Q-value
            "best_q_value": float,
            "confidence": float,              # visit_count / min_visits (capped at 1.0)
            "is_exploring": bool,             # Whether this is an exploratory choice
            "state_key": str,                 # The discretized state key
            "method": str,                    # "exploit" | "explore" | "no_data"
        }
        """
        state_key = extract_state_key(state)
        result = {
            "recommended_action": None,
            "q_values": {},
            "best_action": None,
            "best_q_value": 0.0,
            "confidence": 0.0,
            "is_exploring": False,
            "state_key": state_key,
            "method": "no_data",
        }

        try:
            # Query Q-values for this state
            rows = self.conn.execute(
                "SELECT action, q_value, visit_count FROM mc_qvalues WHERE state_key=?",
                (state_key,),
            ).fetchall()

            if not rows:
                # No data for this state — try partial match on prev_action
                rows = self._fuzzy_match_qvalues(state_key)

            if not rows:
                result["method"] = "no_data"
                return result

            # Build Q-value dict and find best action
            q_values = {}
            best_action = None
            best_q = float("-inf")
            total_visits = 0

            for action, q_val, visits in rows:
                if action in VALID_ACTIONS:
                    q_values[action] = q_val
                    total_visits += visits
                    if q_val > best_q:
                        best_q = q_val
                        best_action = action

            result["q_values"] = q_values
            result["best_action"] = best_action
            result["best_q_value"] = round(best_q, 4) if best_action else 0.0

            # Confidence = how much data we have
            max_visits = max(v for _, _, v in rows if _ in q_values) if rows else 0
            result["confidence"] = min(1.0, max_visits / self._min_visits)

            # Epsilon-greedy selection
            if result["confidence"] < 0.3:
                # Not enough data — don't recommend, let LLM decide
                result["method"] = "no_data"
                return result

            import random
            if random.random() < self._epsilon:
                # Explore: random action from valid set
                explore_action = random.choice(list(q_values.keys()))
                result["recommended_action"] = explore_action
                result["is_exploring"] = True
                result["method"] = "explore"
            else:
                # Exploit: best known action
                result["recommended_action"] = best_action
                result["method"] = "exploit"

        except Exception as e:
            logger.debug(f"[MCPolicy] Recommendation failed: {e}")
            result["method"] = "error"

        return result

    def _fuzzy_match_qvalues(self, state_key: str) -> list[tuple]:
        """
        当精确状态没有数据时，通过部分匹配找到相似状态的 Q-values。

        匹配策略:
        1. 忽略 prev_action 匹配（只看状态特征部分）
        2. 放宽 iteration_phase 的精度
        3. 如果还没有数据，返回空
        """
        try:
            # Split state key into parts: "iter3|ev2|conv1|rev3|hyps2|prev_reflection"
            parts = state_key.split("|")
            if len(parts) < 6:
                return []

            # Try matching without prev_action (first 5 parts)
            state_prefix = "|".join(parts[:5]) + "|"
            rows = self.conn.execute(
                "SELECT action, q_value, visit_count FROM mc_qvalues WHERE state_key LIKE ?",
                (f"{state_prefix}%",),
            ).fetchall()

            if rows:
                return rows

            # Try even broader: match on evidence + convergence + review only
            # Skip iteration and hyps for maximum generalization
            ev_part = parts[1]  # e.g., "ev2"
            conv_part = parts[2]  # e.g., "conv1"
            rev_part = parts[3]  # e.g., "rev3"
            pattern = f"%|{ev_part}|{conv_part}|{rev_part}|%"
            rows = self.conn.execute(
                "SELECT action, q_value, visit_count FROM mc_qvalues WHERE state_key LIKE ?",
                (pattern,),
            ).fetchall()

            return rows
        except Exception:
            return []

    # ---- Prompt formatting ----

    def format_recommendation_for_prompt(self, recommendation: dict) -> str:
        """
        将 MC 策略推荐格式化为 LLM prompt 注入文本。

        输出示例:
        ## 📊 蒙特卡洛策略推荐（基于历史学习）
        当前状态: iter3|ev2|conv1|rev3|hyps2|prev_reflection
        推荐动作: hypothesis_generation (Q=0.723, 置信度=0.85)
        策略: exploit（利用最优策略）

        候选动作 Q-values:
        | 动作 | Q值 | 访问次数 |
        |------|-----|---------|
        | hypothesis_generation | 0.723 | 12 |
        | reflection | 0.456 | 8 |
        | experiment_design | 0.321 | 5 |
        """
        if not recommendation or recommendation.get("method") == "no_data":
            return ""

        state_key = recommendation.get("state_key", "?")
        rec_action = recommendation.get("recommended_action", "?")
        best_action = recommendation.get("best_action", "?")
        best_q = recommendation.get("best_q_value", 0.0)
        confidence = recommendation.get("confidence", 0.0)
        method = recommendation.get("method", "?")
        q_values = recommendation.get("q_values", {})

        lines = [
            "## 📊 蒙特卡洛策略推荐（基于历史学习）",
            f"当前状态: `{state_key}`",
            f"推荐动作: **{rec_action}** (Q={best_q:.3f}, 置信度={confidence:.0%})",
            f"策略: {method}（{'利用最优策略' if method == 'exploit' else '随机探索'}）",
            "",
            "### 候选动作 Q-values",
            "| 动作 | Q值 |",
            "|------|-----|",
        ]

        # Sort by Q-value descending, show top 5
        sorted_actions = sorted(q_values.items(), key=lambda x: x[1], reverse=True)[:5]
        for action, q_val in sorted_actions:
            marker = " ← 推荐" if action == rec_action else ""
            lines.append(f"| {action} | {q_val:.3f}{marker} |")

        lines.append("")
        lines.append("⚠️ 以上为历史学习建议，请结合当前研究状态做出最终决策。")

        return "\n".join(lines)

    # ---- Policy cache refresh ----

    def _refresh_policy_cache(self):
        """从 Q-values 刷新策略缓存表"""
        try:
            conn = self.conn

            # Get all unique state keys
            state_keys = conn.execute(
                "SELECT DISTINCT state_key FROM mc_qvalues"
            ).fetchall()

            for (state_key,) in state_keys:
                # Find best action for this state
                best = conn.execute(
                    """SELECT action, q_value, visit_count FROM mc_qvalues
                       WHERE state_key=? ORDER BY q_value DESC LIMIT 1""",
                    (state_key,),
                ).fetchone()

                if not best:
                    continue

                action, q_val, visits = best
                total = conn.execute(
                    "SELECT SUM(visit_count) FROM mc_qvalues WHERE state_key=?",
                    (state_key,),
                ).fetchone()[0] or 1

                # Use state_key as pattern (could be generalized later)
                conn.execute(
                    """INSERT OR REPLACE INTO mc_policy_cache
                       (domain, state_pattern, best_action, best_q_value, confidence, total_visits)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (self._domain, state_key, action, q_val,
                     round(visits / total, 3), total),
                )

            conn.commit()
        except Exception as e:
            logger.debug(f"[MCPolicy] Policy cache refresh failed: {e}")

    # ---- Query helpers ----

    def get_top_actions(self, n: int = 5) -> list[dict]:
        """获取全局 Q-value 最高的 top-N 动作"""
        try:
            rows = self.conn.execute(
                """SELECT state_key, action, q_value, visit_count
                   FROM mc_qvalues
                   WHERE visit_count >= ?
                   ORDER BY q_value DESC LIMIT ?""",
                (self._min_visits, n),
            ).fetchall()

            return [
                {
                    "state_key": sk,
                    "action": action,
                    "q_value": round(q, 4),
                    "visits": visits,
                }
                for sk, action, q, visits in rows
            ]
        except Exception:
            return []

    def get_learning_stats(self) -> dict:
        """获取学习统计摘要"""
        try:
            conn = self.conn
            episodes = conn.execute(
                "SELECT COUNT(*), AVG(total_return), AVG(num_steps) FROM mc_episodes WHERE status='completed'"
            ).fetchone()
            q_entries = conn.execute("SELECT COUNT(*) FROM mc_qvalues").fetchone()[0]
            total_visits = conn.execute("SELECT SUM(visit_count) FROM mc_qvalues").fetchone()[0] or 0

            return {
                "total_episodes": episodes[0] or 0,
                "avg_return": round(episodes[1] or 0, 3),
                "avg_steps": round(episodes[2] or 0, 1),
                "q_table_size": q_entries,
                "total_visits": total_visits,
                "gamma": self._gamma,
                "epsilon": self._epsilon,
                "alpha": self._alpha,
            }
        except Exception:
            return {}

    def get_episode_history(self, limit: int = 20) -> list[dict]:
        """获取最近的 episode 历史"""
        try:
            rows = self.conn.execute(
                """SELECT episode_id, domain, started_at, ended_at,
                          total_return, num_steps, terminal_reward, status
                   FROM mc_episodes ORDER BY episode_id DESC LIMIT ?""",
                (limit,),
            ).fetchall()

            return [
                {
                    "episode_id": eid,
                    "domain": domain,
                    "duration_sec": round((ended or 0) - started, 1),
                    "total_return": round(ret, 3),
                    "num_steps": steps,
                    "terminal_reward": round(tr, 3),
                    "status": status,
                }
                for eid, domain, started, ended, ret, steps, tr, status in rows
            ]
        except Exception:
            return []

    def reset(self):
        """清除所有学习数据（调试用）"""
        try:
            self.conn.execute("DELETE FROM mc_steps")
            self.conn.execute("DELETE FROM mc_episodes")
            self.conn.execute("DELETE FROM mc_qvalues")
            self.conn.execute("DELETE FROM mc_policy_cache")
            self.conn.commit()
            logger.info("[MCPolicy] All learning data reset")
        except Exception as e:
            logger.warning(f"[MCPolicy] Reset failed: {e}")


# Singleton instance
mc_policy = MonteCarloPolicy()
