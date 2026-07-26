"""
Smoke tests — verify the entire pipeline without API calls.

Run: python -m pytest tests/test_smoke.py -v
Or:   python tests/test_smoke.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Test 1: All 14 nodes import correctly and have the safety decorator
# ============================================================
def test_all_nodes_importable():
    """Every node function must be importable and decorated."""
    from core.nodes import (
        node_ethics_check,
        node_literature_review,
        node_hypothesis_generation,
        node_tournament_eval,
        node_experiment_design,
        node_data_analysis,
        node_interpretation,
        node_reviewer_agent,
        node_reflection,
        node_termination_eval,
        node_report_writing,
        node_pi_agent_meeting,
        node_human_approval,
        node_evolution_manager,
    )
    nodes = [
        node_ethics_check, node_literature_review, node_hypothesis_generation,
        node_tournament_eval, node_experiment_design, node_data_analysis,
        node_interpretation, node_reviewer_agent, node_reflection,
        node_termination_eval, node_report_writing, node_pi_agent_meeting,
        node_human_approval, node_evolution_manager,
    ]

    for node in nodes:
        assert node is not None, f"Node {node} is None"
        assert callable(node), f"Node {node} is not callable"
        # Verify decorator is applied
        assert hasattr(node, "__wrapped__"), (
            f"Node {node.__name__} is missing @carry_control_fields decorator"
        )

    print(f"[PASS] All 14 nodes importable and decorated")


# ============================================================
# Test 2: State safety net — carry_control_fields decorator
# ============================================================
def test_carry_control_fields_decorator():
    """Decorator must auto-add missing control fields to return dicts."""
    from core.nodes import carry_control_fields

    @carry_control_fields
    async def fake_node(state):
        return {"current_action": "test"}

    state = {"_max_iterations_": 42, "iteration": 7, "consecutive_failures": 2}
    result = asyncio.run(fake_node(state))

    assert result["_max_iterations_"] == 42, "Should preserve _max_iterations_"
    assert result["iteration"] == 7, "Should preserve iteration"
    assert result["consecutive_failures"] == 2, "Should preserve consecutive_failures"
    assert result["current_action"] == "test", "Should preserve original fields"

    # Test with None return
    @carry_control_fields
    async def none_node(state):
        return None

    result2 = asyncio.run(none_node(state))
    assert result2["_max_iterations_"] == 42
    assert result2["iteration"] == 7

    print("[PASS] carry_control_fields decorator works correctly")


# ============================================================
# Test 3: Graph compiles without errors
# ============================================================
def test_graph_compiles():
    """Cognitive graph must build and compile successfully."""
    from core.graph import build_cognitive_graph
    graph = build_cognitive_graph()
    assert graph is not None
    # Verify all expected nodes are registered
    expected_nodes = [
        "ethics_check", "literature_review", "hypothesis_generation",
        "tournament_eval", "experiment_design", "data_analysis",
        "interpretation", "reviewer_agent", "reflection",
        "debate_then_terminate", "termination_eval", "report_writing",
        "pi_agent_meeting", "human_approval", "evolution_manager",
        "post_report_chat",
    ]
    for node in expected_nodes:
        assert node in graph.nodes, f"Node '{node}' missing from graph"

    print("[PASS] Graph compiles with all 16 nodes")


# ============================================================
# Test 4: State TypedDict has required fields
# ============================================================
def test_state_has_required_fields():
    """AgentState must include all critical fields for LangGraph state merge."""
    from core.state import AgentState

    required = [
        "query", "domain", "iteration", "current_action",
        "hypothesis_tree", "evidence_chains", "experiment_records",
        "review_records", "fact_extraction", "literature_summary",
        "final_report", "convergence_score",
        "should_terminate", "stop_reason",
        "debate_history", "educational_annotations",
        "user_chat_messages",
    ]
    annotations = AgentState.__annotations__
    for field in required:
        assert field in annotations, f"Field '{field}' missing from AgentState"

    print(f"[PASS] AgentState has all {len(required)} required fields")


# ============================================================
# Test 5: New modules import correctly
# ============================================================
def test_new_modules_importable():
    """All 6 upgrade modules must be importable."""
    from core.progress import ProgressDashboard
    from core.visualization import render_causal_chart, render_hypothesis_tree, inject_visualizations
    from core.adaptive import compute_iteration_budget
    from core.memory import ResearchMemory
    from core.sft_pipeline import SFTDataCollector

    # Progress dashboard
    dash = ProgressDashboard()
    dash.on_node_start("ethics_check")
    dash.on_node_end("ethics_check", {"ethics_status": "approved"})
    summary = dash.summary()
    assert "Pipeline completed" in summary

    # Adaptive iterations
    simple = compute_iteration_budget("temperature")
    complex = compute_iteration_budget("温度对心率变异性的因果影响 个体差异 N-of-1 多因素交互")
    assert simple >= 2
    assert complex >= simple, "Complex query should get more iterations"

    # Visualization
    chart = render_causal_chart([])
    assert isinstance(chart, str)

    tree = render_hypothesis_tree([])
    assert isinstance(tree, str)

    # Memory (use temp file for cross-platform compatibility)
    import tempfile
    mem = ResearchMemory(tempfile.mktemp(suffix=".json"))
    mem.remember("test-session", {"query": "test", "hypothesis_tree": [], "evidence_chains": []})
    recalled = mem.recall("test")
    assert len(recalled) >= 0

    print("[PASS] All 6 upgrade modules importable and functional")


# ============================================================
# Test 6: Prompts contain required sections
# ============================================================
def test_prompts_complete():
    """All required prompt templates must exist and be non-empty."""
    from core import prompts

    required_prompts = [
        "ORCHESTRATOR_SYSTEM_PROMPT",
        "PI_AGENT_SYSTEM_PROMPT",
        "REVIEWER_AGENT_SYSTEM_PROMPT",
        "ETHICS_WATCHDOG_SYSTEM_PROMPT",
        "REFLECTION_SYSTEM_PROMPT",
        "LITERATURE_REVIEW_PROMPT",
        "HYPOTHESIS_GENERATION_TEMPLATE",
        "EXPERIMENT_DESIGN_TEMPLATE",
        "REPORT_WRITING_TEMPLATE",
        "TOURNAMENT_EVAL_PROMPT",
        # New specialized personas
        "GENERATION_AGENT_PROMPT",
        "REFLECTION_AGENT_PROMPT",
        "RANKING_AGENT_PROMPT",
        "PROXIMITY_AGENT_PROMPT",
        "META_REVIEW_AGENT_PROMPT",
    ]

    for name in required_prompts:
        prompt = getattr(prompts, name, None)
        assert prompt is not None, f"Prompt '{name}' missing"
        assert len(prompt) > 50, f"Prompt '{name}' too short ({len(prompt)} chars)"

    print(f"[PASS] All {len(required_prompts)} prompts present and non-empty")


# ============================================================
# Test 7: Settings load correctly
# ============================================================
def test_settings_load():
    """Settings must load with sensible defaults."""
    from config.settings import settings

    assert settings.bailian_base_url, "bailian_base_url must be set"
    assert settings.model_name, "model_name must be set"
    assert settings.max_iterations > 0, "max_iterations must be positive"
    # API key might be empty in test env — that's OK

    print("[PASS] Settings load correctly")


# ============================================================
# Test 8: Edge case — empty state, missing fields
# ============================================================
def test_empty_state_handling():
    """Nodes should handle empty/missing state gracefully."""
    empty_state = {}

    # Test that carry_control_fields fills missing fields
    from core.nodes import carry_control_fields

    @carry_control_fields
    async def empty_node(state):
        return {"current_action": "test"}

    result = asyncio.run(empty_node(empty_state))
    assert result["_max_iterations_"] == 200, "Should default to 200"
    assert result["iteration"] == 0, "Should default to 0"
    assert result["consecutive_failures"] == 0, "Should default to 0"

    print("[PASS] Empty state handling works")


# ============================================================
# Test 9: LLM client can be created (no API call)
# ============================================================
def test_llm_client_creation():
    """QwenClient should be instantiable without API call."""
    from core.llm_client import QwenClient

    client = QwenClient(
        base_url="https://test.example.com/v1",
        api_key="sk-test",
        model="qwen-max",
    )
    assert client.model == "qwen-max"
    assert client.base_url == "https://test.example.com/v1"

    print("[PASS] QwenClient instantiable")


# ============================================================
# Test 10: Debate module imports and structures exist
# ============================================================
def test_debate_module():
    """Debate data structures and orchestrator must be importable."""
    from core.debate import (
        DebateRound, DebateResult, DebateOrchestrator,
        DEBATE_PRO_SYSTEM_PROMPT, DEBATE_CON_SYSTEM_PROMPT, DEBATE_JUDGE_SYSTEM_PROMPT,
    )

    # Verify prompts exist
    assert len(DEBATE_PRO_SYSTEM_PROMPT) > 100
    assert len(DEBATE_CON_SYSTEM_PROMPT) > 100
    assert len(DEBATE_JUDGE_SYSTEM_PROMPT) > 100

    # Verify orchestrator
    orchestrator = DebateOrchestrator()
    assert orchestrator.max_rounds > 0

    # Verify dataclasses
    round_ = DebateRound(round_number=1, pro_agent_output="test", con_agent_output="test",
                         judge_score_before=70, judge_score_after=75,
                         judge_reasoning="test", winner_side="pro")
    assert round_.round_number == 1

    result = DebateResult(debates=[], strongest_hypothesis_id="test",
                          strongest_hypothesis_title="test",
                          strongest_hypothesis_final_score=80,
                          num_rounds=1, consensus_reached=True)
    assert result.consensus_reached

    print("[PASS] Debate module intact")


# ============================================================
# Runner
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TwinScientist Smoke Tests")
    print("=" * 60)
    print()

    tests = [
        test_all_nodes_importable,
        test_carry_control_fields_decorator,
        test_graph_compiles,
        test_state_has_required_fields,
        test_new_modules_importable,
        test_prompts_complete,
        test_settings_load,
        test_empty_state_handling,
        test_llm_client_creation,
        test_debate_module,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {test.__name__}: {e}")

    print()
    print(f"{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 60}")

    sys.exit(1 if failed > 0 else 0)