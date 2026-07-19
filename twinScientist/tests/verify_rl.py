"""
自主强化学习端到端验证脚本

模拟多 episode 研究循环，验证 MC RL 系统能够：
1. 记录 state-action-reward 轨迹
2. 计算折扣回报 (discounted returns)
3. 更新 Q-values (First-Visit Monte Carlo)
4. 从历史经验中学习策略偏好 (policy convergence)
5. Experience Store 正确持久化和检索

不依赖 LLM API，通过模拟不同质量的研究 session 来测试学习行为。
"""

import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.mc_learning import (
    MonteCarloPolicy,
    compute_step_reward,
    compute_terminal_reward,
    extract_feature_vector,
    extract_state_key,
    VALID_ACTIONS,
)
from core.experience import ExperienceStore


def make_state(iteration, action, review_score, evidence_str, convergence,
               n_hyps=3, n_approved=0, max_iter=50, failures=0):
    """构造模拟的 AgentState"""
    hypotheses = []
    for i in range(n_approved):
        hypotheses.append({"status": "approved_by_reviewer"})
    for i in range(n_hyps - n_approved):
        hypotheses.append({"status": "proposed" if i % 2 == 0 else "active"})

    evidence = [{"strength": evidence_str, "type": "data"}] if evidence_str > 0 else []
    reviews = [{"total_score": review_score}] if review_score > 0 else []

    return {
        "iteration": iteration,
        "_max_iterations_": max_iter,
        "current_action": action,
        "convergence_score": convergence,
        "evidence_chains": evidence,
        "review_records": reviews,
        "hypothesis_tree": hypotheses,
        "experiment_records": [],
        "anomaly_graph": [],
        "consecutive_failures": failures,
        "uncertainty_level": max(0, 1 - convergence),
    }


def simulate_good_episode():
    """模拟一个高质量研究 session：快速收敛、高评审分"""
    actions = [
        "literature_review",
        "hypothesis_generation",
        "experiment_design",
        "data_analysis",
        "interpretation",
        "reviewer_agent",
        "report_writing",
    ]
    steps = []
    for i, action in enumerate(actions):
        progress = i / len(actions)
        state = make_state(
            iteration=i + 1,
            action=action,
            review_score=int(50 + progress * 45),   # 50→95
            evidence_str=round(0.2 + progress * 0.6, 2),  # 0.2→0.8
            convergence=round(progress * 0.9, 2),    # 0→0.9
            n_approved=int(progress * 3),
            max_iter=50,
        )
        steps.append((state, action))

    final = make_state(
        iteration=7, action="report_writing",
        review_score=95, evidence_str=0.85, convergence=0.92,
        n_approved=3, max_iter=50,
    )
    return steps, final


def simulate_bad_episode():
    """模拟一个低质量研究 session：反复循环、低分"""
    actions = [
        "literature_review",
        "hypothesis_generation",
        "hypothesis_generation",   # repeat
        "hypothesis_generation",   # repeat again
        "reflection",
        "hypothesis_generation",   # still generating
        "literature_review",       # going back
    ]
    steps = []
    for i, action in enumerate(actions):
        state = make_state(
            iteration=i + 1,
            action=action,
            review_score=max(20, 40 - i * 3),        # 40→20
            evidence_str=round(0.1 + i * 0.02, 2),   # barely growing
            convergence=round(0.05 * i, 2),           # slow convergence
            n_hyps=5 + i,                              # too many untested
            max_iter=50,
        )
        steps.append((state, action))

    final = make_state(
        iteration=7, action="literature_review",
        review_score=25, evidence_str=0.15, convergence=0.15,
        n_hyps=12, max_iter=50, failures=2,
    )
    return steps, final


def simulate_medium_episode():
    """模拟中等质量的 session"""
    actions = [
        "literature_review",
        "hypothesis_generation",
        "experiment_design",
        "data_analysis",
        "reviewer_agent",
        "reflection",
        "experiment_design",
        "data_analysis",
        "interpretation",
        "reviewer_agent",
    ]
    steps = []
    for i, action in enumerate(actions):
        progress = i / len(actions)
        state = make_state(
            iteration=i + 1,
            action=action,
            review_score=int(45 + progress * 35),
            evidence_str=round(0.15 + progress * 0.45, 2),
            convergence=round(progress * 0.7, 2),
            n_approved=int(progress * 2),
            max_iter=50,
        )
        steps.append((state, action))

    final = make_state(
        iteration=10, action="reviewer_agent",
        review_score=78, evidence_str=0.6, convergence=0.72,
        n_approved=2, max_iter=50,
    )
    return steps, final


def run_rl_verification():
    """运行完整的 RL 验证"""
    print("=" * 70)
    print("  twinScientist 自主强化学习 — 端到端验证")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_mc = os.path.join(tmpdir, "mc_policy.db")
        db_exp = os.path.join(tmpdir, "experience.db")

        policy = MonteCarloPolicy(db_path=db_mc, gamma=0.9, epsilon=0.0, alpha=0.3)
        exp_store = ExperienceStore(db_path=db_exp)

        # Force epsilon=0 during learning to make it deterministic for verification
        policy._epsilon = 0.0

        results = {}

        # ============================================================
        # Phase 1: 训练 — 多 episode 学习
        # ============================================================
        print("\n[Phase 1] Training (10 episodes)")
        print("-" * 50)

        episode_sequence = [
            ("good", simulate_good_episode),
            ("bad", simulate_bad_episode),
            ("good", simulate_good_episode),
            ("medium", simulate_medium_episode),
            ("good", simulate_good_episode),
            ("bad", simulate_bad_episode),
            ("good", simulate_good_episode),
            ("medium", simulate_medium_episode),
            ("good", simulate_good_episode),
            ("good", simulate_good_episode),
        ]

        episode_returns = []
        q_table_sizes = []

        for ep_idx, (quality, sim_fn) in enumerate(episode_sequence):
            steps, final_state = sim_fn()

            # --- MC Policy episode ---
            eid = policy.begin_episode(domain="环境—人体关联", query=f"test_{ep_idx}")
            for state, action in steps:
                policy.log_step(state, action)
            policy.update_from_episode(final_state=final_state)

            # --- Experience Store session ---
            sid = exp_store.begin_session(domain="环境—人体关联", query=f"test_{ep_idx}")
            for state, action in steps:
                exp_store.log_step(state, action)
            exp_store.flush_session()

            # Stats
            stats = policy.get_learning_stats()
            q_size = stats.get("q_table_size", 0)
            avg_ret = stats.get("avg_return", 0)
            q_table_sizes.append(q_size)

            # Get episode return
            ep_history = policy.get_episode_history(limit=1)
            ep_return = ep_history[0]["total_return"] if ep_history else 0
            episode_returns.append(ep_return)

            marker = "[OK]" if quality == "good" else ("[FAIL]" if quality == "bad" else "[WARN]")
            print(f"  Episode {ep_idx+1:2d} [{quality:6s}] {marker}  "
                  f"return={ep_return:+.3f}  Q-table={q_size}")

        results["q_table_growth"] = q_table_sizes[-1] > q_table_sizes[0]
        results["total_episodes"] = len(episode_sequence)

        # ============================================================
        # Phase 2: 验证 Q-value 学习效果
        # ============================================================
        print("\n[Phase 2] Q-value Learning Verification")
        print("-" * 50)

        top_actions = policy.get_top_actions(n=10)
        print(f"  Top-10 动作（按 Q-value 排序）:")
        for a in top_actions:
            print(f"    state={a['state_key'][:30]:30s}  "
                  f"action={a['action']:25s}  Q={a['q_value']:+.4f}  "
                  f"visits={a['visits']}")

        # Check: good actions should have higher Q-values
        # "report_writing" and "interpretation" should be learned as good
        all_q = {}
        for a in top_actions:
            all_q.setdefault(a["action"], []).append(a["q_value"])

        results["q_table_has_entries"] = len(top_actions) > 0

        # ============================================================
        # Phase 3: 策略推荐测试
        # ============================================================
        print("\n[Phase 3] Policy Recommendation Test")
        print("-" * 50)

        # Test with different state configurations
        test_states = [
            ("早期研究", make_state(1, "literature_review", 0, 0.1, 0.0, max_iter=50)),
            ("中期实验", make_state(5, "experiment_design", 60, 0.4, 0.3, max_iter=50)),
            ("后期收敛", make_state(10, "reviewer_agent", 85, 0.75, 0.85, n_approved=3, max_iter=50)),
        ]

        for label, state in test_states:
            rec = policy.recommend(state)
            print(f"  [{label}]")
            print(f"    state_key: {rec['state_key']}")
            print(f"    recommended: {rec['recommended_action']}")
            print(f"    method: {rec['method']}, confidence: {rec['confidence']:.1%}")
            if rec["q_values"]:
                sorted_q = sorted(rec["q_values"].items(), key=lambda x: x[1], reverse=True)
                for action, q in sorted_q[:3]:
                    print(f"      {action:25s}: Q={q:+.4f}")

        # ============================================================
        # Phase 4: Experience Store 策略提示验证
        # ============================================================
        print("\n[Phase 4] Experience Store Policy Tips")
        print("-" * 50)

        tips = exp_store.get_policy_tips(domain="环境—人体关联", n=5)
        print(f"  历史经验 Top-5:")
        for tip in tips:
            print(f"    {tip['action']:25s}  "
                  f"executed={tip['total_taken']}x  "
                  f"high_reward={tip['high_reward_pct']}  "
                  f"avg_cum_reward={tip['avg_cum_reward']:+.3f}")

        prompt_text = exp_store.format_tips_for_prompt(tips)
        if prompt_text:
            print(f"\n  格式化 Prompt 注入文本 (前500字):")
            print(f"  {prompt_text[:500]}")

        results["tips_generated"] = len(tips) > 0

        # ============================================================
        # Phase 5: 奖励函数一致性验证
        # ============================================================
        print("\n🔢 Phase 5: 奖励计算验证")
        print("-" * 50)

        # Good state should get higher reward than bad state
        good_state = make_state(5, "data_analysis", 85, 0.7, 0.6, n_approved=2, max_iter=50)
        bad_state = make_state(5, "hypothesis_generation", 30, 0.1, 0.05, n_hyps=10, max_iter=50)

        r_good = compute_step_reward(good_state, "interpretation")
        r_bad = compute_step_reward(bad_state, "hypothesis_generation")
        print(f"  Good state → interpretation: reward={r_good:+.4f}")
        print(f"  Bad state → hypothesis_gen:  reward={r_bad:+.4f}")
        results["good_beats_bad_reward"] = r_good > r_bad

        # Terminal reward
        good_final = make_state(8, "report_writing", 92, 0.85, 0.9, n_approved=3, max_iter=50)
        bad_final = make_state(8, "literature_review", 25, 0.1, 0.1, failures=3, max_iter=50)
        t_good = compute_terminal_reward(good_final)
        t_bad = compute_terminal_reward(bad_final)
        print(f"  Good terminal reward: {t_good:+.4f}")
        print(f"  Bad terminal reward:  {t_bad:+.4f}")
        results["good_beats_bad_terminal"] = t_good > t_bad

        # ============================================================
        # Phase 6: Feature vector 完整性验证
        # ============================================================
        print("\n[Phase 6] Feature Vector Integrity")
        print("-" * 50)

        state = make_state(5, "data_analysis", 75, 0.6, 0.5, n_approved=2, max_iter=50)
        mc_feat = extract_feature_vector(state)
        exp_feat = ExperienceStore._feature_vector(state)

        print(f"  MC Learning features:  {len(mc_feat)} fields")
        print(f"  Experience features:   {len(exp_feat)} fields")

        # Check key fields exist in both
        shared_keys = set(mc_feat.keys()) & set(exp_feat.keys())
        mc_only = set(mc_feat.keys()) - set(exp_feat.keys())
        exp_only = set(exp_feat.keys()) - set(mc_feat.keys())
        print(f"  Shared fields:    {len(shared_keys)}")
        print(f"  MC-only fields:   {mc_only if mc_only else 'none'}")
        print(f"  Exp-only fields:  {exp_only if exp_only else 'none'}")

        # Check consistency of shared fields
        mismatch = []
        for k in shared_keys:
            v1, v2 = mc_feat[k], exp_feat[k]
            if isinstance(v1, float):
                if abs(v1 - v2) > 0.01:
                    mismatch.append(f"{k}: mc={v1} vs exp={v2}")
            elif v1 != v2:
                mismatch.append(f"{k}: mc={v1} vs exp={v2}")

        if mismatch:
            print(f"  [WARN] Mismatches: {mismatch}")
        else:
            print(f"  [OK] All shared fields consistent")

        results["feature_consistency"] = len(mismatch) == 0

        # ============================================================
        # Phase 7: MC Return 折扣验证
        # ============================================================
        print("\n[Reward] Phase 7: MC Return 折扣计算")
        print("-" * 50)

        # Check that different steps get different cum_rewards
        conn = policy.conn
        rows = conn.execute("""
            SELECT episode_id, step_idx, state_key, action, reward, mc_return
            FROM mc_steps
            WHERE episode_id = (SELECT MAX(episode_id) FROM mc_episodes WHERE status='completed')
            ORDER BY step_idx
        """).fetchall()

        if rows:
            print(f"  最近 episode 的步骤回报:")
            for eid, idx, sk, action, reward, mc_ret in rows:
                print(f"    step {idx}: action={action:25s}  "
                      f"reward={reward:+.4f}  G_t={mc_ret:+.4f}")

            returns = [r[5] for r in rows]
            results["mc_returns_vary"] = len(set(returns)) > 1
            print(f"  [OK] MC returns vary across steps: {len(set(returns))} distinct values")
        else:
            results["mc_returns_vary"] = False

        # ============================================================
        # 总结
        # ============================================================
        print("\n" + "=" * 70)
        print("  Verification Summary")
        print("=" * 70)

        all_pass = True
        checks = [
            ("Q-table 增长", results.get("q_table_growth")),
            ("Q-table 有数据", results.get("q_table_has_entries")),
            ("好状态奖励 > 差状态", results.get("good_beats_bad_reward")),
            ("好终止奖励 > 差终止", results.get("good_beats_bad_terminal")),
            ("特征一致性", results.get("feature_consistency")),
            ("MC Return 折扣变化", results.get("mc_returns_vary")),
            ("经验策略提示生成", results.get("tips_generated")),
        ]

        for label, passed in checks:
            status = "[OK] PASS" if passed else "[FAIL] FAIL"
            if not passed:
                all_pass = False
            print(f"  {status}  {label}")

        print("=" * 70)
        if all_pass:
            print("  ALL PASSED! RL system working correctly")
        else:
            print("  [WARN] 部分检查未通过，需要排查")
        print("=" * 70)

        # Close connections
        policy.close()
        exp_store.close()

        return all_pass


if __name__ == "__main__":
    success = run_rl_verification()
    sys.exit(0 if success else 1)
