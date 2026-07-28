# Bug: TerminationEval ignores Orchestrator MAX_ROUNDS_REACHED, causing infinite loop

## Description

When the agent reaches the configured max iteration limit, the Orchestrator correctly detects `MAX_ROUNDS_REACHED` and routes to `termination_eval`. However, `TerminationEval` independently re-evaluates termination criteria using its own convergence/evidence scoring, finds the combined score below threshold, and routes back to `hypothesis_generation` — creating an **infinite loop**.

## Steps to Reproduce

1. Run with `python -m main --question "<any question>"` (default max_iter=5)
2. Agent goes through: EthicsCheck → LiteratureReview → HypothesisGen → TournamentEval → ExperimentDesign → DataAnalysis → Interpretation → ReviewerAgent
3. Reviewer gives moderate scores (55-85/100), orchestrator route: `CONTINUE → reflection` or eventually `MAX_ROUNDS_REACHED → termination_eval`
4. After ~5 iterations, `OrchestratorStop` logs: `MAX_ROUNDS_REACHED: 已达到最大轮次上限 (5/5)`
5. Graph routes to `termination_eval`, which computes:
   ```
   Round 5: similarity=0.2826, convergence=0.283
   CONTINUE: 未满足任何停止条件 (convergence=28.3%, combined=0.113)，继续下一轮
   Score=0.113, terminate=False, convergence=0.283, pruned 0 dead branches
   ```
6. `route_after_termination()` returns `"hypothesis_generation"` because `combined=0.113 < 0.85`
7. Loop restarts: `hypothesis_generation → tournament_eval → experiment_design → ... → reviewer_agent → termination_eval → ...`
8. Each cycle adds 10 more hypotheses (tree grows: 10→20→30→...→120+) with `pruned 0` every time
9. Eventually hits `GraphRecursionError` at LangGraph's recursion limit

## Root Cause Analysis

### Bug 1: TerminationEval doesn't respect Orchestrator's hard stop decision

In `core/graph.py`, `_after_reviewer_route()` correctly detects `stop=true` + `max_round_reached` and routes to `"termination_eval"` with log:
```
[AfterReviewer] ORCHESTRATOR STOPPED (terminate): 已达到最大轮次上限 (5/5)
```

But `node_termination_eval()` in `core/nodes.py` has NO awareness of this orchestrator-level stop signal. It independently computes termination:

```python
def node_termination_eval(state: AgentState) -> dict:
    # ...
    # Condition A: Convergence stable (two consecutive rounds similar)
    # Condition B: Hard limit — max 200 rounds
    # Condition C: Original combined score threshold
    # NONE of these check: "Orchestrator already decided to stop"
```

The `iteration >= 200` check (Condition B) would stop, but the actual `max_iter` may be much lower (default 5). And even if iteration >= 200 were true, if the orchestrator already said stop AND the iteration is 5, the condition isn't triggered either way because the Orchestrator uses `_max_iterations_` dynamically.

### Bug 2: `route_after_termination()` always allows continuation

```python
def route_after_termination(state: AgentState) -> str:
    convergence = state.get("convergence_score", 0.0)
    evidence_str = sum(e.get("strength", 0.5) for e in state.get("evidence_chains", [])) / max(len(state.get("evidence_chains", [])), 1)
    exploration_done = state.get("exploration_exhausted", False)
    combined = convergence * 0.4 + evidence_str * 0.3 + (0.8 if exploration_done else 0.0) * 0.3

    if combined >= 0.85:
        return "report_writing"
    return "hypothesis_generation"  # ← Always loops back when combined < 0.85
```

Since the actual data analysis returns `strength=0.0` (causal inference on temperature-only sensor data produces no cross-variable correlation), `evidence_str` is effectively 0. Convergence stays at 0.2-0.4 because hypotheses change slightly each round. `combined` never exceeds 0.85 → infinite loop back to `hypothesis_generation`.

### Bug 3: HypothesisTree grows without bounds

```
Line 83: Tree now has 20 hypotheses (removed 0 pruned)
Line 98: Tree now has 30 hypotheses (removed 0 pruned)
...
Line 685: Tree now has 100 hypotheses (removed 0 pruned)
Line 735: Tree now has 120 hypotheses (removed 0 pruned)
```

Every call to `HypothesisGen` appends new candidates without removing old ones (`pruned 0`). There's no pruning mechanism engaged, so the tree grows unboundedly across cycles.

## Observed Log Evidence

From the last successful session (cli-session-569fba0c, started 2026-07-14 12:39:26):

```
Line 475: [ReviewerAgent] hyp_6f5f8424: score=75/100, needs_revision=True → next=report_writing
Line 476: [OrchestratorStop] CONTINUE: No stop condition met       ← round 1

Line 498: [ReviewerAgent] hyp_6f5f8424: score=84/100, needs_revision=False → next=report_writing
Line 499: [OrchestratorStop] CONTINUE: No stop condition met       ← round 2

Line 524: [ReviewerAgent] hyp_9915a177: score=79/100, needs_revision=False → next=report_writing
Line 525: [OrchestratorStop] CONTINUE: No stop condition met       ← round 3

Line 550: [ReviewerAgent] hyp_49111dfd: score=80/100, needs_revision=True → next=report_writing
Line 551: [OrchestratorStop] CONTINUE: No stop condition met       ← round 4

Line 600: [ReviewerAgent] hyp_7419a05c: score=70/100, needs_revision=True → next=reflection
Line 601: [OrchestratorStop] MAX_ROUNDS_REACHED: 已达到最大轮次上限 (5/5)  ← round 5, stop detected!
Line 603: [TerminationEval] Round 5: similarity=0.2308, convergence=0.231
Line 604: [TerminationEval] CONTINUE: 未满足任何停止条件 (convergence=23.1%, combined=0.092)，继续下一轮  ← IGNORES stop!
Line 608: [HypothesisGen] Tree now has 70 hypotheses (removed 0 pruned) ← NEW CYCLE STARTED

... repetition continues ...

Line 651: [ReviewerAgent] hyp_bf56f32a: score=35/100, needs_revision=True
Line 652: [OrchestratorStop] MAX_ROUNDS_REACHED: 已达最大轮次上限 (5/5)
Line 654: [TerminationEval] Round 5: similarity=0.0877, convergence=0.088
Line 655: [TerminationEval] CONTINUE: 未满足任何停止条件 (convergence=8.8%, combined=0.035)，继续下一轮

Line 677: [ReviewerAgent] hyp_39c98382: score=80/100
Line 678: [OrchestratorStop] MAX_ROUNDS_REACHED: 已达到最大轮次上限 (5/5)
Line 680: [TerminationEval] Round 5: convergence=23.9% → CONTINUE

... and so on past 120 hypotheses
```

The same pattern repeated ~8 times after `MAX_ROUNDS_REACHED` was first detected. The graph cycled indefinitely despite the orchestrator signaling stop.

## Impact

- **Wasted API costs**: Each useless cycle burns Qwen API calls (~10 LLM calls per cycle × 8+ cycles = ~80 wasted calls)
- **Resource waste**: Hypothesis tree grows to 120+ entries with zero pruning
- **No useful output**: Research completes with a theoretical framework report containing no real analysis (all strength=0.0)
- **First two sessions** hit `GraphRecursionError: Recursion limit of 25 reached` before the third one ran long enough to show 80+ hypotheses

## Proposed Fixes

### Fix 1: Pass Orchestrator stop signal into TerminationEval (Critical)

Store the orchestrator's stop decision in state and check it in `node_termination_eval`:

```python
# In core/orchestrator.py _check_orchestrator_stop_conditions():
result["_orchestrator_stop_reason"] = result.get("reason")

# In core/nodes.py node_termination_eval():
orch_stop = state.get("_orchestrator_stop_reason")
if orch_stop:
    # Orchestrator already decided to stop — honor it
    should_terminate = True
    stop_reason = f"Orchestrator已决定终止: {orch_stop}"
```

### Fix 2: Lower combined_score threshold for termination (Important)

The threshold of 0.85 is unrealistically high given:
- Convergence naturally oscillates around 0.2-0.4 in early research
- Evidence strength is often 0.0 when data doesn't support causal inference
- Combined formula weights these heavily: `convergence*0.4 + evidence*0.3 + exploration*0.3`

Suggested: Lower threshold to 0.5, or add a fallback when `max_iterations` are nearing exhaustion:

```python
remaining_budget = state.get("_max_iterations_", 200) - state.get("iteration", 0)
if remaining_budget <= 2:
    # Force termination when budget is nearly exhausted
    combined = max(combined, 0.9)
```

### Fix 3: Cap hypothesis tree size and enforce pruning (Medium priority)

Add a cap in `node_hypothesis_generation` to prevent unbounded growth:

```python
MAX_HYPOTHESIS_TREE_SIZE = 30
if len(kept_tree) > MAX_HYPOTHESIS_TREE_SIZE:
    # Prune lowest-confidence hypotheses
    kept_tree = prune_lowest_confidence(kept_tree, MAX_HYPOTHESIS_TREE_SIZE)
```

### Fix 4: Increase default max_iter or document the limitation (Low priority)

The CLI default of `--iterations 5` is very low and causes rapid hitting of the limit with weak signals. Consider increasing the default to 10-20 or documenting this as a minimum threshold.

## Environment

- **Platform**: Windows 11 Home China
- **Python**: 3.14 (pythoncore-3.14-64)
- **LLM**: Qwen via DashScope (阿里云百炼平台)
- **LangGraph**: Pregel implementation
- **recursion_limit**: 2000 (set in main.py line 70)
- **max_iterations**: 5 (CLI default, `--iterations` flag)
- **Log file**: `twinScientist/logs/twinscientist.log`
