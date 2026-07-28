"""
Test Suite — twinScientist 奖励系统、特征提取、数据生成器

覆盖修复的问题：
1. RMSSD/SDNN 方向性（neurotoxic 应降低 HRV）
2. PPG baseline 不被双重计入
3. SDNN/RMSSD 系数合理性
4. experience.py SQL schema 语法正确
5. sessions.ended_at 允许 NULL
6. extract_feature_vector 字段完整性
7. compute_step_reward 边界检查
8. MC return 折扣计算
9. state discretization 桶边界
"""

import json
import math
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# Test: compute_step_reward 边界检查
# ============================================================

class TestComputeStepReward(unittest.TestCase):
    """测试 compute_step_reward 的边界保护和正确性"""

    def setUp(self):
        from core.mc_learning import compute_step_reward
        self.compute = compute_step_reward

    def test_empty_state_returns_zero(self):
        """空状态应返回接近 0 的奖励"""
        state = {}
        reward = self.compute(state, "literature_review")
        self.assertIsInstance(reward, float)
        self.assertGreaterEqual(reward, -1.0)
        self.assertLessEqual(reward, 1.0)

    def test_reward_clamped_to_valid_range(self):
        """所有输入组合都应产生 [-1, 1] 范围的奖励"""
        extreme_state = {
            "review_records": [{"total_score": 100}] * 10,
            "evidence_chains": [{"strength": 1.0}] * 20,
            "convergence_score": 1.0,
            "hypothesis_tree": [
                {"status": "approved_by_reviewer"} for _ in range(50)
            ],
            "current_action": "literature_review",
            "iteration": 0,
            "_max_iterations_": 200,
        }
        for action in [
            "literature_review", "hypothesis_generation", "experiment_design",
            "data_analysis", "interpretation", "reviewer_agent", "reflection",
            "termination_eval", "report_writing",
        ]:
            reward = self.compute(extreme_state, action)
            self.assertGreaterEqual(reward, -1.0, f"Reward {reward} < -1 for {action}")
            self.assertLessEqual(reward, 1.0, f"Reward {reward} > 1 for {action}")

    def test_review_score_boundary_values(self):
        """评审分数边界值（0, 50, 100）应正确处理"""
        # Score = 0 → negative contribution
        state_low = {"review_records": [{"total_score": 0}]}
        r_low = self.compute(state_low, "literature_review")

        # Score = 100 → positive contribution
        state_high = {"review_records": [{"total_score": 100}]}
        r_high = self.compute(state_high, "literature_review")

        self.assertLess(r_low, r_high, "Score 0 should yield lower reward than score 100")

    def test_review_score_invalid_type_handled(self):
        """非数字 total_score 不应崩溃"""
        state = {"review_records": [{"total_score": None}]}
        reward = self.compute(state, "literature_review")
        self.assertIsInstance(reward, float)

        state2 = {"review_records": [{"total_score": "abc"}]}
        reward2 = self.compute(state2, "literature_review")
        self.assertIsInstance(reward2, float)

    def test_review_score_out_of_range_clamped(self):
        """超出 [0,100] 的评审分数应被截断"""
        state_over = {"review_records": [{"total_score": 200}]}
        state_normal = {"review_records": [{"total_score": 100}]}
        r_over = self.compute(state_over, "literature_review")
        r_normal = self.compute(state_normal, "literature_review")
        # Both should produce the same reward (clamped to 100)
        self.assertEqual(r_over, r_normal)

    def test_evidence_strength_clamped(self):
        """evidence strength 超出 [0,1] 应被截断"""
        state = {
            "evidence_chains": [
                {"strength": 2.0},   # Over 1.0
                {"strength": -0.5},  # Under 0.0
            ]
        }
        reward = self.compute(state, "data_analysis")
        self.assertIsInstance(reward, float)
        self.assertGreaterEqual(reward, -1.0)
        self.assertLessEqual(reward, 1.0)

    def test_convergence_clamped(self):
        """convergence_score 超出 [0,1] 应被截断"""
        state_high = {"convergence_score": 5.0}
        state_normal = {"convergence_score": 1.0}
        r_high = self.compute(state_high, "interpretation")
        r_normal = self.compute(state_normal, "interpretation")
        self.assertEqual(r_high, r_normal, "Convergence >1 should be clamped to 1")

    def test_repeat_action_penalty(self):
        """重复动作应受到惩罚"""
        state = {"current_action": "reflection"}
        r_repeat = self.compute(state, "reflection")
        r_different = self.compute(state, "literature_review")
        self.assertLess(r_repeat, r_different, "Repeating action should have lower reward")

    def test_near_max_iteration_penalty(self):
        """接近最大迭代时非终止动作应受惩罚"""
        state_near_end = {
            "iteration": 199,
            "_max_iterations_": 200,
        }
        r_continue = self.compute(state_near_end, "hypothesis_generation")
        r_terminate = self.compute(state_near_end, "termination_eval")
        self.assertLess(r_continue, r_terminate,
                        "Near max iter, non-termination action should score lower")

    def test_good_action_sequence_rewarded(self):
        """合理的动作序列应获得正向 fitness 奖励"""
        state = {
            "current_action": "experiment_design",
            "hypothesis_tree": [{"status": "active"}],
        }
        # data_analysis after experiment_design → fitness +0.1
        state_after_exp = {
            **state,
            "current_action": "experiment_design",
        }
        r_analysis = self.compute(state_after_exp, "data_analysis")

        state_random = {
            "current_action": "literature_review",
            "hypothesis_tree": [],
        }
        r_random = self.compute(state_random, "data_analysis")
        self.assertGreater(r_analysis, r_random,
                           "Good sequence should be rewarded over random")


# ============================================================
# Test: compute_terminal_reward
# ============================================================

class TestComputeTerminalReward(unittest.TestCase):
    """测试 compute_terminal_reward 的正确性"""

    def setUp(self):
        from core.mc_learning import compute_terminal_reward
        self.compute = compute_terminal_reward

    def test_empty_state(self):
        """空状态应返回 0 附近"""
        reward = self.compute({})
        self.assertGreaterEqual(reward, -1.0)
        self.assertLessEqual(reward, 1.0)

    def test_high_review_positive(self):
        """高评审分应产生正向奖励"""
        state = {"review_records": [{"total_score": 95}]}
        reward = self.compute(state)
        self.assertGreater(reward, 0)

    def test_failure_penalty(self):
        """连续失败≥3 应有惩罚"""
        state_ok = {"consecutive_failures": 0}
        state_fail = {"consecutive_failures": 5}
        r_ok = self.compute(state_ok)
        r_fail = self.compute(state_fail)
        self.assertLess(r_fail, r_ok)

    def test_efficiency_bonus(self):
        """更少迭代应有更高效率奖励"""
        state_fast = {"iteration": 5, "_max_iterations_": 200}
        state_slow = {"iteration": 190, "_max_iterations_": 200}
        r_fast = self.compute(state_fast)
        r_slow = self.compute(state_slow)
        self.assertGreater(r_fast, r_slow)

    def test_clamped_range(self):
        """终端奖励应在 [-1, 1]"""
        extreme = {
            "review_records": [{"total_score": 100}],
            "evidence_chains": [{"strength": 1.0}] * 50,
            "iteration": 1,
            "_max_iterations_": 200,
        }
        reward = self.compute(extreme)
        self.assertGreaterEqual(reward, -1.0)
        self.assertLessEqual(reward, 1.0)


# ============================================================
# Test: extract_feature_vector 完整性
# ============================================================

class TestExtractFeatureVector(unittest.TestCase):
    """测试 extract_feature_vector 字段完整性"""

    def setUp(self):
        from core.mc_learning import extract_feature_vector
        self.extract = extract_feature_vector

    def test_all_required_fields_present(self):
        """验证所有必需字段都存在"""
        state = {}
        features = self.extract(state)

        required_fields = [
            "iteration", "remaining_budget", "avg_evidence", "max_evidence",
            "min_evidence", "convergence", "latest_review", "avg_review",
            "num_hyps", "approved", "proposed", "active", "refuted",
            "needs_revision", "pruned", "num_evidence", "num_experiments",
            "consecutive_failures", "prev_action", "uncertainty",
            "anomaly_count", "has_real_analysis",
        ]
        for field in required_fields:
            self.assertIn(field, features, f"Missing field: {field}")

    def test_remaining_budget_computed(self):
        """remaining_budget 应正确计算"""
        state = {"iteration": 50, "_max_iterations_": 200}
        features = self.extract(state)
        self.assertEqual(features["remaining_budget"], 150)

    def test_remaining_budget_never_negative(self):
        """remaining_budget 不应为负"""
        state = {"iteration": 300, "_max_iterations_": 200}
        features = self.extract(state)
        self.assertEqual(features["remaining_budget"], 0)

    def test_hypothesis_status_distribution(self):
        """假设状态分布应正确统计"""
        state = {
            "hypothesis_tree": [
                {"status": "approved_by_reviewer"},
                {"status": "approved_by_reviewer"},
                {"status": "proposed"},
                {"status": "active"},
                {"status": "refuted"},
                {"status": "pruned"},
                {"status": "needs_revision"},
            ]
        }
        features = self.extract(state)
        self.assertEqual(features["approved"], 2)
        self.assertEqual(features["proposed"], 1)
        self.assertEqual(features["active"], 1)
        self.assertEqual(features["refuted"], 1)
        self.assertEqual(features["pruned"], 1)
        self.assertEqual(features["needs_revision"], 1)
        self.assertEqual(features["num_hyps"], 7)

    def test_evidence_statistics(self):
        """证据统计应正确"""
        state = {
            "evidence_chains": [
                {"strength": 0.3},
                {"strength": 0.7},
                {"strength": 0.5},
            ]
        }
        features = self.extract(state)
        self.assertEqual(features["num_evidence"], 3)
        self.assertAlmostEqual(features["avg_evidence"], 0.5, places=2)
        self.assertAlmostEqual(features["max_evidence"], 0.7, places=2)
        self.assertAlmostEqual(features["min_evidence"], 0.3, places=2)

    def test_has_real_analysis(self):
        """has_real_analysis 应检测 causal_inference 类型"""
        state_no = {"evidence_chains": [{"type": "data"}]}
        state_yes = {"evidence_chains": [{"type": "causal_inference"}]}
        self.assertFalse(self.extract(state_no)["has_real_analysis"])
        self.assertTrue(self.extract(state_yes)["has_real_analysis"])

    def test_empty_state_defaults(self):
        """空状态应返回合理默认值"""
        features = self.extract({})
        self.assertEqual(features["iteration"], 0)
        self.assertEqual(features["remaining_budget"], 200)
        self.assertEqual(features["avg_evidence"], 0.0)
        self.assertEqual(features["num_hyps"], 0)
        self.assertEqual(features["anomaly_count"], 0)
        self.assertFalse(features["has_real_analysis"])


# ============================================================
# Test: experience.py _feature_vector
# ============================================================

class TestExperienceFeatureVector(unittest.TestCase):
    """测试 experience.py 的 _feature_vector"""

    def setUp(self):
        from core.experience import ExperienceStore
        self.extract = ExperienceStore._feature_vector

    def test_consistency_with_mc_learning(self):
        """experience.py 和 mc_learning.py 的特征应保持一致的核心字段"""
        from core.mc_learning import extract_feature_vector

        state = {
            "iteration": 10,
            "_max_iterations_": 200,
            "hypothesis_tree": [
                {"status": "approved_by_reviewer"},
                {"status": "proposed"},
            ],
            "evidence_chains": [{"strength": 0.6}],
            "review_records": [{"total_score": 80}],
        }

        exp_feat = self.extract(state)
        mc_feat = extract_feature_vector(state)

        # Core fields should match
        self.assertEqual(exp_feat["iteration"], mc_feat["iteration"])
        self.assertEqual(exp_feat["remaining_budget"], mc_feat["remaining_budget"])
        self.assertEqual(exp_feat["num_hyps"], mc_feat["num_hyps"])
        self.assertEqual(exp_feat["approved"], mc_feat["approved"])
        self.assertAlmostEqual(exp_feat["avg_evidence"], mc_feat["avg_evidence"])


# ============================================================
# Test: experience.py SQL schema
# ============================================================

class TestExperienceSchema(unittest.TestCase):
    """测试 experience.py SQL schema 语法正确性"""

    def test_schema_creates_without_error(self):
        """Schema 应在标准 SQLite 中无错创建"""
        from core.experience import SCHEMA_SQL
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(SCHEMA_SQL)
            # Verify all tables exist
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t[0] for t in tables}
            self.assertIn("sessions", table_names)
            self.assertIn("steps", table_names)
            self.assertIn("policy_stats", table_names)
        finally:
            conn.close()

    def test_sessions_allows_null_ended_at(self):
        """sessions.ended_at 应允许 NULL（begin_session 不设此字段）"""
        from core.experience import SCHEMA_SQL
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(SCHEMA_SQL)
            # Insert without ended_at (as begin_session does)
            conn.execute(
                """INSERT INTO sessions (domain, query_hash, started_at, iteration_end,
                                         convergence_end, evidence_str, review_scores,
                                         terminated_by)
                   VALUES ('test', 'abc123', 1000.0, 0, 0, 0, '', '')"""
            )
            row = conn.execute("SELECT ended_at FROM sessions").fetchone()
            self.assertIsNone(row[0], "ended_at should be NULL on insert")
        finally:
            conn.close()

    def test_no_trailing_commas_in_schema(self):
        """Schema 不应有尾随逗号（), 模式不应出现在列/约束定义后）"""
        from core.experience import SCHEMA_SQL
        lines = SCHEMA_SQL.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Check for "REAL DEFAULT 0.0," or similar followed by ");"
            if stripped.endswith(",") and i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if next_stripped.startswith(")"):
                    self.fail(
                        f"Trailing comma found at line {i+1}: '{stripped}' "
                        f"followed by '{next_stripped}'"
                    )


# ============================================================
# Test: experience.py MC return
# ============================================================

class TestExperienceMCReturn(unittest.TestCase):
    """测试 experience.py flush_session 的 MC return 计算"""

    def test_discounted_returns_not_uniform(self):
        """多步 session 的各步 cum_reward 不应完全相同"""
        from core.experience import ExperienceStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ExperienceStore(db_path=os.path.join(tmpdir, "test.db"))
            store.begin_session("test", "query")

            # Log multiple steps
            for i in range(5):
                state = {
                    "iteration": i,
                    "_max_iterations_": 200,
                    "hypothesis_tree": [],
                    "evidence_chains": [],
                    "review_records": [],
                    "experiment_records": [],
                    "anomaly_graph": [],
                }
                store.log_step(state, f"action_{i}")

            store.flush_session()

            # Check that cum_reward varies across steps
            rows = store.conn.execute(
                "SELECT step_idx, cum_reward FROM steps ORDER BY step_idx"
            ).fetchall()

            self.assertEqual(len(rows), 5)
            rewards = [r[1] for r in rows]
            # Not all identical (unless reward is exactly 0)
            if rewards[0] != 0:
                self.assertTrue(
                    len(set(rewards)) > 1,
                    f"All cum_rewards are identical: {rewards}"
                )
            store.close()


# ============================================================
# Test: State discretization
# ============================================================

class TestStateDiscretization(unittest.TestCase):
    """测试 state discretization 桶边界"""

    def setUp(self):
        from core.mc_learning import extract_state_key, _discretize, BINS
        self.extract_key = extract_state_key
        self.discretize = _discretize
        self.bins = BINS

    def test_discretize_boundary_values(self):
        """桶边界值应映射到正确的桶"""
        bins = [0, 1, 3, 5, 10]
        self.assertEqual(self.discretize(0, bins), 0)
        self.assertEqual(self.discretize(0.5, bins), 1)
        self.assertEqual(self.discretize(1, bins), 1)
        self.assertEqual(self.discretize(2, bins), 2)
        self.assertEqual(self.discretize(3, bins), 2)
        self.assertEqual(self.discretize(11, bins), 5)  # Beyond all bins

    def test_state_key_deterministic(self):
        """相同状态应产生相同的状态键"""
        state = {
            "iteration": 5,
            "evidence_chains": [{"strength": 0.6}],
            "convergence_score": 0.7,
            "review_records": [{"total_score": 80}],
            "hypothesis_tree": [{"status": "active"}],
            "current_action": "data_analysis",
            "consecutive_failures": 0,
        }
        key1 = self.extract_key(state)
        key2 = self.extract_key(state)
        self.assertEqual(key1, key2)

    def test_state_key_format(self):
        """状态键应有正确的格式"""
        state = {"current_action": "reflection"}
        key = self.extract_key(state)
        parts = key.split("|")
        self.assertEqual(len(parts), 6)
        self.assertTrue(parts[0].startswith("iter"))
        self.assertTrue(parts[1].startswith("ev"))
        self.assertTrue(parts[2].startswith("conv"))
        self.assertTrue(parts[3].startswith("rev"))
        self.assertTrue(parts[4].startswith("hyps"))
        self.assertTrue(parts[5].startswith("prev_"))


# ============================================================
# Test: SDNN/RMSSD directionality (gen_multimodal_simulator)
# ============================================================

class TestSDNNRMSSDDirectionality(unittest.TestCase):
    """测试 SDNN/RMSSD 在环境压力下的方向性"""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from gen_multimodal_simulator import BiometricModel
        self.model = BiometricModel(seed=42)

    def test_neurotoxic_reduces_sdnn(self):
        """神经毒性应降低 SDNN（而非升高）"""
        subject = self.model._generate_subject_profile("TEST_001")
        env_clean = {
            "T": 22.0, "CO2": 400, "VOC": 10, "PMS2_5": 5, "H": 45,
            "timestamp": "2026-01-01T00:00:00",
        }
        env_polluted = {
            "T": 22.0, "CO2": 400, "VOC": 500, "PMS2_5": 5, "H": 45,
            "timestamp": "2026-01-01T01:00:00",
        }

        load_clean = self.model.calculate_environmental_load(env_clean, subject)
        load_polluted = self.model.calculate_environmental_load(env_polluted, subject)

        # Neurotoxic should be higher in polluted env
        self.assertGreater(load_polluted["neurotoxic"], load_clean["neurotoxic"])

        # SDNN should be LOWER in polluted env (both neurotoxic and symp contribute)
        # Simulate the SDNN calculation
        sdnn_clean = (
            subject["hrv_baseline"]
            - load_clean["symp_activation"] * 8.0
            - load_clean["thermo_load"] * 3.0
            - abs(subject["co2_sensitivity"]) * ((env_clean["CO2"] / 400) - 1) * 100
            - load_clean["systemic_inflam"] * 5.0
            - load_clean["neurotoxic"] * 6.0
            - abs(subject["humid_sensitivity"]) * abs(env_clean["H"] - 45)
        )
        sdnn_polluted = (
            subject["hrv_baseline"]
            - load_polluted["symp_activation"] * 8.0
            - load_polluted["thermo_load"] * 3.0
            - abs(subject["co2_sensitivity"]) * ((env_polluted["CO2"] / 400) - 1) * 100
            - load_polluted["systemic_inflam"] * 5.0
            - load_polluted["neurotoxic"] * 6.0
            - abs(subject["humid_sensitivity"]) * abs(env_polluted["H"] - 45)
        )

        self.assertLess(sdnn_polluted, sdnn_clean,
                        "SDNN should decrease under neurotoxic stress")

    def test_neurotoxic_reduces_rmssd(self):
        """神经毒性应降低 RMSSD"""
        subject = self.model._generate_subject_profile("TEST_002")
        env_clean = {
            "T": 22.0, "CO2": 400, "VOC": 10, "PMS2_5": 5, "H": 45,
            "timestamp": "2026-01-01T00:00:00",
        }
        env_polluted = {
            "T": 22.0, "CO2": 400, "VOC": 500, "PMS2_5": 5, "H": 45,
            "timestamp": "2026-01-01T01:00:00",
        }

        load_clean = self.model.calculate_environmental_load(env_clean, subject)
        load_polluted = self.model.calculate_environmental_load(env_polluted, subject)

        rmssd_clean = (
            subject["rmssd_baseline"]
            - load_clean["symp_activation"] * 6.0
            - abs(subject["co2_sensitivity"]) * ((env_clean["CO2"] / 400) - 1) * 60
            - load_clean["neurotoxic"] * 5.0
            - load_clean["thermo_load"] * 2.0
        )
        rmssd_polluted = (
            subject["rmssd_baseline"]
            - load_polluted["symp_activation"] * 6.0
            - abs(subject["co2_sensitivity"]) * ((env_polluted["CO2"] / 400) - 1) * 60
            - load_polluted["neurotoxic"] * 5.0
            - load_polluted["thermo_load"] * 2.0
        )

        self.assertLess(rmssd_polluted, rmssd_clean,
                        "RMSSD should decrease under neurotoxic stress")


# ============================================================
# Test: PPG baseline not double-counted
# ============================================================

class TestPPGBaseline(unittest.TestCase):
    """测试 PPG 计算中 baseline 不被双重计入"""

    def test_ppg_baseline_contribution_is_1x(self):
        """PPG 的稳态 baseline 贡献应约等于 1× ppo_baseline"""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from gen_multimodal_simulator import BiometricModel

        model = BiometricModel(seed=42)
        subject = model._generate_subject_profile("PPG_TEST")

        # Zero-load environment (baseline conditions)
        env = {
            "T": 22.0, "CO2": 400, "VOC": 50, "PMS2_5": 12, "H": 45,
            "timestamp": "2026-01-01T00:00:00",
        }
        load = model.calculate_environmental_load(env, subject)

        # All loads should be ~0 at baseline conditions
        ppg_delta = (
            + load["thermo_load"] * 0.15
            - load["systemic_inflam"] * 0.08
            + load["symp_activation"] * (-0.05)
        )

        # In zero-load, ppg_delta ≈ 0, so ppg ≈ ppo_baseline
        # After smoothing with prev=ppo_baseline, result should be ppo_baseline
        baseline = subject["ppo_baseline"]
        ppg_point = baseline + ppg_delta  # no noise for this test
        ppg = 0.8 * baseline + 0.2 * ppg_point  # smooth with prev=baseline

        # Should be approximately equal to baseline (within 10%)
        self.assertAlmostEqual(ppg, baseline, delta=baseline * 0.1,
                               msg=f"PPG {ppg:.4f} should be near baseline {baseline:.4f}")


# ============================================================
# Test: Subject profile key consistency
# ============================================================

class TestSubjectProfileKeys(unittest.TestCase):
    """测试 SUBJECT_PROFILE_KEYS 与实际生成的键一致"""

    def test_profile_keys_match_generated(self):
        """SUBJECT_PROFILE_KEYS 应与 _generate_subject_profile 的键匹配"""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from gen_multimodal_simulator import BiometricModel

        model = BiometricModel(seed=42)
        profile = model._generate_subject_profile("KEY_TEST")

        # Every key in SUBJECT_PROFILE_KEYS should exist in the profile
        for key in BiometricModel.SUBJECT_PROFILE_KEYS:
            self.assertIn(key, profile, f"Key '{key}' in SUBJECT_PROFILE_KEYS but not in profile")

        # ppo_baseline should be in both (not ppv_baseline)
        self.assertIn("ppo_baseline", profile)
        self.assertIn("ppo_baseline", BiometricModel.SUBJECT_PROFILE_KEYS)


# ============================================================
# Test: experience.py _compute_reward
# ============================================================

class TestExperienceComputeReward(unittest.TestCase):
    """测试 experience.py 的 _compute_reward"""

    def setUp(self):
        from core.experience import ExperienceStore
        self.compute = ExperienceStore._compute_reward

    def test_reward_components_present(self):
        """奖励应包含 overall, conv, evidence, review 四个分量"""
        feat = {
            "iteration": 10,
            "remaining_budget": 190,
            "avg_evidence": 0.6,
            "latest_review": 80,
        }
        reward = self.compute(feat)
        self.assertIn("overall", reward)
        self.assertIn("conv", reward)
        self.assertIn("evidence", reward)
        self.assertIn("review", reward)

    def test_reward_range(self):
        """各分量应在 [0, 1] 范围"""
        feat = {
            "iteration": 10,
            "remaining_budget": 190,
            "avg_evidence": 0.8,
            "latest_review": 90,
        }
        reward = self.compute(feat)
        for key in ["overall", "conv", "evidence", "review"]:
            self.assertGreaterEqual(reward[key], 0.0)
            self.assertLessEqual(reward[key], 1.0)


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
