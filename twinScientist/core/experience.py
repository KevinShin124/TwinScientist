"""
Experience Store — 蒙特卡洛经验学习（Monte-Carlo Offline RL）

每次研究 session 结束后自动评估 episode return，将 (state_features, action, reward)
记录到本地 SQLite store。下次 Orchestrator 决策时注入历史成功经验作为 prompt context。

设计理念：
- 不改变任何现有 node / graph / LLM 调用逻辑
- 纯 append-only 持久化，失败不会破坏运行
- feature vector 从现有 state 字段派生，零额外依赖
- LLM 路由的决策质量随使用次数自然累积提升

Usage:
    from core.experience import exp_store
    # Start a new research session:
    exp_store.begin_session(domain="环境—人体关联", query="问题")
    # Record each routing decision:
    exp_store.log_step(state, "hypothesis_generation")
    # Session ends:
    exp_store.flush_session()
    # Next session — get learned policy:
    tips = exp_store.get_policy_tips(domain="环境—人体关联", n=3)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================
# Database Schema (self-contained SQLite, no server needed)
# ============================================================

SCHEMA_SQL = """
-- One row per research session (outer loop metadata)
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    domain          TEXT NOT NULL,
    query_hash      TEXT NOT NULL,       -- SHA-256 prefix of research question
    started_at      REAL NOT NULL,
    ended_at        REAL NOT NULL,
    iteration_end   INTEGER NOT NULL,    -- total iterations consumed
    convergence_end REAL NOT NULL,       -- final convergence_score
    evidence_str    REAL NOT NULL,       -- avg evidence strength at end
    review_scores   TEXT,                -- JSON array of [score_per_round]
    terminated_by   TEXT,                -- reason key
    reward_overall  REAL DEFAULT 0.0,    -- computed at flush
    reward_conv     REAL DEFAULT 0.0,
    reward_evidence REAL DEFAULT 0.0,
    reward_review   REAL DEFAULT 0.0,
);

-- One row per *routing step* inside a session
CREATE TABLE IF NOT EXISTS steps (
    session_id      INTEGER REFERENCES sessions(id),
    step_idx        INTEGER NOT NULL,
    ts              REAL NOT NULL,
    feature_json    TEXT NOT NULL,       -- JSON-encoded feature dict
    action          TEXT NOT NULL,       -- chosen next action
    cum_reward      REAL DEFAULT 0.0,    -- assigned on flush (MC return)
    PRIMARY KEY (session_id, step_idx)
);

CREATE INDEX IF NOT EXISTS idx_steps_action ON steps(action);

-- Pre-aggregated policy cache (fast read during routing)
CREATE TABLE IF NOT EXISTS policy_stats (
    domain          TEXT NOT NULL,
    action          TEXT NOT NULL,
    total_taken     INTEGER DEFAULT 0,
    high_reward_cnt INTEGER DEFAULT 0,
    avg_cum_reward  REAL DEFAULT 0.0,
    PRIMARY KEY (domain, action)
);
"""


class ExperienceStore:
    """Append-only experience replay buffer backed by a local SQLite file."""

    def __init__(self, db_path: str | None = None):
        self._db_path = Path(db_path or "./data/experience.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        # Per-session in-memory buffers
        self._current_session_id: int | None = None
        self._pending_steps: list[dict] = []
        self._domain: str = ""

    # ---- Connection management ----

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            try:
                self._conn = sqlite3.connect(str(self._db_path))
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.executescript(SCHEMA_SQL)
                logger.info(f"[ExpStore] Initialized experience DB at {self._db_path}")
            except Exception as e:
                logger.warning(
                    f"[ExpStore] Failed to open DB ({e}), running without persistence"
                )
                raise
        return self._conn

    def close(self):
        if self._conn:
            self.conn.commit()
            self._conn.close()
            self._conn = None

    # ---- Feature extraction ----

    @staticmethod
    def _feature_vector(state: dict) -> dict[str, Any]:
        """Compact, deterministic feature dict derived from AgentState."""
        hypotheses = state.get("hypothesis_tree", [])
        reviews = state.get("review_records", [])
        evidence_chains = state.get("evidence_chains", [])
        experiments = state.get("experiment_records", [])
        anomaly_graph = state.get("anomaly_graph", [])

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
            "num_hyps": len(hypotheses),
            "approved": approved,
            "proposed": proposed,
            "active": active,
            "refuted": refuted,
            "needs_revision": needs_rev,
            "pruned": pruned,
            "num_evidence": len(evidence_chains),
            "avg_evidence": round(
                sum(e.get("strength", 0.5) for e in evidence_chains)
                / max(len(evidence_chains), 1), 3
            ),
            "max_evidence": round(
                max((e.get("strength", 0.5) for e in evidence_chains), default=0.0), 3
            ),
            "min_evidence": round(
                min((e.get("strength", 0.5) for e in evidence_chains), default=0.0), 3
            ),
            "num_experiments": len(experiments),
            "latest_review": rev_scores[-1] if rev_scores else 0,
            "avg_review": round(
                sum(rev_scores) / max(len(rev_scores), 1), 1
            ) if rev_scores else 0,
            "anomaly_count": len(anomaly_graph),
            "prev_action": state.get("current_action", "none"),
            "has_real_analysis": any(
                isinstance(e, dict) and e.get("type") == "causal_inference"
                for e in evidence_chains
            ),
        }

    # ---- Session lifecycle ----

    def begin_session(self, domain: str, query: str) -> int | None:
        """Start tracking a new research session. Returns session_id or None."""
        try:
            qh = hashlib.sha256(query.encode()).hexdigest()[:12]
            cursor = self.conn.execute(
                """INSERT INTO sessions (domain, query_hash, started_at, iteration_end,
                                         convergence_end, evidence_str, review_scores,
                                         terminated_by)
                   VALUES (?, ?, ?, 0, 0, 0, '', '')""",
                (domain, qh, time.time()),
            )
            self._current_session_id = cursor.lastrowid
            self._pending_steps = []
            self._domain = domain
            return self._current_session_id
        except Exception as e:
            logger.error(f"[ExpStore] Failed to begin session: {e}")
            return None

    def log_step(self, state: dict, action: str) -> bool:
        """Record one routing decision. Called after every decision node."""
        if self._current_session_id is None:
            return False
        try:
            feat = self._feature_vector(state)
            self._pending_steps.append({
                "step_idx": len(self._pending_steps),
                "ts": time.time(),
                "feature_json": json.dumps(feat, ensure_ascii=False),
                "action": action,
            })
            return True
        except Exception as e:
            logger.debug(f"[ExpStore] Step logging failed: {e}")
            return False

    def flush_session(self) -> int | None:
        """Finalize current session: persist, compute rewards, update policy stats."""
        sid = self._current_session_id
        if sid is None or not self._pending_steps:
            return None

        try:
            last_feat = json.loads(self._pending_steps[-1]["feature_json"])
            now = time.time()

            # Compute multi-signal reward
            reward = self._compute_reward(last_feat)
            term_reason = self._infer_termination(last_feat)

            # Update session row with outcomes
            self.conn.execute(
                """UPDATE sessions SET
                    ended_at = ?, iteration_end = ?, convergence_end = ?,
                    evidence_str = ?, review_scores = ?, terminated_by = ?,
                    reward_overall = ?, reward_conv = ?, reward_evidence = ?, reward_review = ?
                 WHERE id = ?""",
                (
                    now,
                    last_feat["iteration"],
                    self._estimate_convergence(last_feat),
                    last_feat["avg_evidence"],
                    json.dumps([last_feat["latest_review"]]),
                    term_reason,
                    reward["overall"],
                    reward["conv"],
                    reward["evidence"],
                    reward["review"],
                    sid,
                ),
            )

            # Batch-insert steps
            rows = [(sid, s["step_idx"], s["ts"], s["feature_json"], s["action"], 0)
                    for s in self._pending_steps]
            self.conn.executemany(
                """INSERT OR REPLACE INTO steps (session_id, step_idx, ts, feature_json, action, cum_reward)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                rows,
            )

            # Assign Monte Carlo cumulative reward (single-step episodes get full return)
            self.conn.execute(
                "UPDATE steps SET cum_reward = ? WHERE session_id = ?",
                (reward["overall"], sid),
            )

            # Flush policy stats
            self._refresh_policy_stats(self._domain)

            logger.info(
                f"[ExpStore] Session #{sid} flushed: {len(rows)} steps, "
                f"reward={reward['overall']:.2f}, domain='{self._domain}'"
            )
            return sid

        except Exception as e:
            logger.error(f"[ExpStore] Flush failed for session #{sid}: {e}", exc_info=True)
            return None
        finally:
            self._current_session_id = None
            self._pending_steps = []

    # ---- Reward computation ----

    @staticmethod
    def _compute_reward(feat: dict) -> dict:
        """Multi-signal reward from pre-computed features."""
        iteration = feat["iteration"]
        remaining = feat["remaining_budget"]
        max_iter = iteration + max(remaining, 1)

        # Convergence quality: stability + speed efficiency
        conv_quality = min(feat["avg_evidence"] * 0.6 + 0.4, 1.0)
        efficiency = max(0, 1 - (iteration / max(max_iter, 1)))
        conv_q = conv_quality * 0.6 + efficiency * 0.4

        # Evidence & review quality
        evidence_s = feat["avg_evidence"]
        review_s = feat["latest_review"] / 100.0 if feat["latest_review"] > 0 else 0.5

        overall = conv_q * 0.35 + evidence_s * 0.35 + review_s * 0.30

        return {
            "overall": round(overall, 3),
            "conv": round(conv_q, 3),
            "evidence": round(evidence_s, 3),
            "review": round(review_s, 3),
        }

    @staticmethod
    def _estimate_convergence(feat: dict) -> float:
        """Estimate final convergence from available features."""
        if feat["latest_review"] >= 75 and feat["avg_evidence"] > 0.5:
            return round(min(feat["avg_evidence"] * 0.8 + 0.2, 1.0), 3)
        elif feat["avg_evidence"] > 0.3:
            return round(feat["avg_evidence"] * 0.5, 3)
        return 0.0

    @staticmethod
    def _infer_termination(feat: dict) -> str:
        remaining = feat["remaining_budget"]
        if remaining <= 0:
            return "max_rounds"
        if feat["latest_review"] >= 75 and feat["avg_evidence"] > 0.7:
            return "evidence"
        if feat["avg_evidence"] > 0.5 and feat["iteration"] > 3:
            return "convergence"
        return "unclear"

    # ---- Policy statistics refresh ----

    def _refresh_policy_stats(self, domain: str):
        """Rebuild the policy_stats cache from accumulated session data."""
        try:
            conn = self.conn
            # Determine top-decile threshold for this domain
            decile_row = conn.execute(
                """SELECT reward_overall FROM sessions
                   WHERE domain=? ORDER BY reward_overall DESC
                   LIMIT 1 OFFSET MAX((SELECT COUNT(*)-1 FROM sessions WHERE domain=?)/10, 0)""",
                (domain, domain),
            ).fetchone()
            high_threshold = decile_row[0] if decile_row else 0.4

            # Aggregate per-domain stats using last-action-per-session heuristic
            rows = conn.execute(
                """SELECT action, COUNT(*),
                          SUM(CASE WHEN cum_reward >= ? THEN 1 ELSE 0 END),
                          AVG(cum_reward)
                   FROM steps
                   JOIN sessions ses ON ses.id = steps.session_id
                   WHERE ses.domain = ? AND step_idx = (
                       SELECT MAX(step_idx) FROM steps WHERE session_id = steps.session_id
                   )
                   GROUP BY action""",
                (high_threshold, domain),
            ).fetchall()

            for action, cnt, high_cnt, avg_r in rows:
                conn.execute(
                    """INSERT OR REPLACE INTO policy_stats (domain, action, total_taken,
                             high_reward_cnt, avg_cum_reward)
                       VALUES (?, ?, ?, ?, ?)""",
                    (domain, action, cnt, high_cnt or 0, avg_r or 0.0),
                )

            # Cross-domain general stats (domain='*' matches any via OR clause)
            gen_rows = conn.execute(
                """SELECT action, COUNT(*),
                          SUM(CASE WHEN cum_reward >= 0.5 THEN 1 ELSE 0 END),
                          AVG(cum_reward)
                   FROM steps WHERE cum_reward > 0
                   GROUP BY action""",
            ).fetchall()
            for action, cnt, high_cnt, avg_r in gen_rows:
                conn.execute(
                    """INSERT OR REPLACE INTO policy_stats (domain, action, total_taken,
                             high_reward_cnt, avg_cum_reward)
                       VALUES ('*', ?, ?, ?, ?)""",
                    (action, cnt, high_cnt or 0, avg_r or 0.0),
                )

            conn.commit()
        except Exception as e:
            logger.warning(f"[ExpStore] Policy refresh failed: {e}")

    # ---- Query helpers ----

    def get_policy_tips(self, domain: str = "", n: int = 3) -> list[dict]:
        """Return top-N most rewarded actions from history."""
        try:
            if domain:
                conditions = "(ps.domain = ? OR ps.domain = '*')"
                params = [domain, n]
            else:
                conditions = "ps.domain = '*'"
                params = [n]

            rows = self.conn.execute(
                f"""SELECT ps.domain, ps.action, ps.total_taken,
                          ps.high_reward_cnt, ps.avg_cum_reward
                     FROM policy_stats ps
                     WHERE {conditions}
                     ORDER BY ps.avg_cum_reward DESC
                     LIMIT ?""",
                params,
            ).fetchall()

            tips = []
            for d, action, total, high, avg_r in rows:
                pct = f"{high/total*100:.0f}% of {total}" if total > 0 else "insufficient data"
                tips.append({
                    "domain": d,
                    "action": action,
                    "total_taken": total,
                    "high_reward_pct": pct,
                    "avg_cum_reward": round(avg_r, 3),
                })
            return tips
        except Exception as e:
            logger.warning(f"[ExpStore] Query tips failed: {e}")
            return []

    def format_tips_for_prompt(self, tips: list[dict]) -> str:
        """Format policy tips as markdown block for LLM injection."""
        if not tips:
            return ""

        lines = ["## 📊 历史决策经验（基于蒙特卡洛分析的过去研究会话）"]
        lines.append("系统已从过去的研究中学习到以下路由策略的有效性:")
        for i, tip in enumerate(tips, 1):
            lines.append(
                f"- `{tip['action']}`: 被执行{tip['total_taken']}次, "
                f"高回报占比{tip['high_reward_pct']}, "
                f"平均累计奖励{tip['avg_cum_reward']:.2f} → 建议优先考虑"
            )
        return "\n".join(lines)

    def reset(self, domain: str = ""):
        """Clear stored experience (for debugging)."""
        if domain:
            sids = [r[0] for r in self.conn.execute(
                "SELECT id FROM sessions WHERE domain=?", (domain,)
            ).fetchall()]
            ph = ",".join("?" * len(sids)) if sids else "0"
            self.conn.execute(f"DELETE FROM steps WHERE session_id IN ({ph})", sids)
            self.conn.execute("DELETE FROM sessions WHERE domain=?", (domain,))
            self.conn.execute("DELETE FROM policy_stats WHERE domain=?", (domain,))
        else:
            self.conn.execute("DELETE FROM steps")
            self.conn.execute("DELETE FROM sessions")
            self.conn.execute("DELETE FROM policy_stats")
        self.conn.commit()
        logger.info(f"[ExpStore] Experience {'scoped' if domain else ''}reset complete")


# Singleton instance
exp_store = ExperienceStore()
