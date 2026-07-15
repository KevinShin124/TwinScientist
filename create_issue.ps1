$token = $env:GITHUB_TOKEN
if (-not $token) { Write-Error "Please set GITHUB_TOKEN environment variable (e.g., `$env:GITHUB_TOKEN='ghp_xxx')" ; exit 1 }
$body = @"
## Description

When the agent reaches the configured max iteration limit, the Orchestrator correctly detects \`MAX_ROUNDS_REACHED\` and routes to \`termination_eval\`. However, \`TerminationEval\` independently re-evaluates termination criteria using its own convergence/evidence scoring, finds the combined score below threshold, and routes back to \`hypothesis_generation\` — creating an **infinite loop**.

## Steps to Reproduce

1. Run with \`python -m main --question "<any question>"\` (default max_iter=5)
2. Agent goes through: EthicsCheck → LiteratureReview → HypothesisGen → TournamentEval → ExperimentDesign → DataAnalysis → Interpretation → ReviewerAgent
3. After ~5 iterations, \`OrchestratorStop\` logs: \`MAX_ROUNDS_REACHED: 已达到最大轮次上限 (5/5)\`
4. Graph routes to \`termination_eval\`, which computes:
   \`\`\`
   Round 5: similarity=0.2826, convergence=0.283
   CONTINUE: combined=0.113 < 0.85, continues to hypothesis_generation
   \`\`\`
5. Loop restarts each cycle adds 10 hypotheses (tree: 10→20→30→...→120+) with \`pruned 0\`
6. Eventually hits \`GraphRecursionError\` at LangGraph recursion limit

## Root Cause Analysis

### Bug 1: TerminationEval ignores Orchestrator hard stop

In \`core/graph.py\`, \`_after_reviewer_route()\` correctly detects \`stop=true + max_round_reached\` and routes to \`termination_eval\`. But \`node_termination_eval()\` in \`core/nodes.py\` has NO awareness of this orchestrator-level stop signal. It independently computes termination via convergence/evidence scoring.

### Bug 2: route_after_termination() always allows continuation when combined_score < 0.85

\`combined = convergence * 0.4 + evidence * 0.3 + exploration * 0.3\`

Since data analysis returns \`strength=0.0\` (temperature-only sensor data), evidence≈0. Convergence oscillates at 0.2~0.4. Combined never exceeds 0.85 → infinite loop.

### Bug 3: HypothesisTree grows without bounds

Every HypothesisGen call appends new candidates without pruning (\`pruned 0\`). Tree grows unboundedly: 10→20→...→120+ across cycles.

## Observed Log Evidence

From session cli-session-569fba0c (2026-07-14):

\`\`\`
Line 475: [ReviewerAgent] hyp_6f5f8424: score=75/100, needs_revision=True
Line 476: [OrchestratorStop] CONTINUE: No stop condition met       ← round 1
Line 498: [ReviewerAgent] score=84/100
Line 499: [OrchestratorStop] CONTINUE: No stop condition met       ← round 2
Line 550: [ReviewerAgent] score=80/100
Line 551: [OrchestratorStop] CONTINUE: No stop condition met       ← round 4
Line 600: [ReviewerAgent] score=70/100
Line 601: [OrchestratorStop] MAX_ROUNDS_REACHED: 已达最大轮次上限 (5/5)  ← STOP DETECTED!
Line 603: [TerminationEval] Round 5: convergence=23.1%, combined=0.092 → CONTINUE  ← IGNORES!
Line 608: [HypothesisGen] Tree now has 70 hypotheses (removed 0 pruned) ← NEW CYCLE
...repeats 8+ times until 120+ hypotheses accumulated\`\`\`

## Impact

- **Wasted API costs**: ~10 LLM calls × 8+ cycles = ~80 wasted Qwen API calls per run
- **Resource waste**: Hypothesis tree grows to 120+ entries with zero pruning
- **No useful output**: Report contains theoretical framework, no real analysis (all strength=0.0)
- **First two sessions**: hit \`GraphRecursionError: Recursion limit of 25 reached\`

## Proposed Fixes

### Fix 1 (Critical): Pass Orchestrator stop signal into TerminationEval
Store orchestrator stop reason in state and check in \`node_termination_eval()\`:
\`\`\`python
orch_stop = state.get("_orchestrator_stop_reason")
if orch_stop:
    should_terminate = True
\`\`\`

### Fix 2 (Important): Add exhaustion fallback to route_after_termination()
\`\`\`python
remaining_budget = state.get("_max_iterations_", 200) - state.get("iteration", 0)
if remaining_budget <= 2:
    combined = max(combined, 0.9)  # Force termination near budget exhaustion
\`\`\`

### Fix 3 (Medium): Cap hypothesis tree size
Add \`MAX_HYPOTHESIS_TREE_SIZE = 30\` in \`node_hypothesis_generation()\`, prune lowest-confidence.

## Environment

- **Platform**: Windows 11 Home China / Python 3.14
- **LLM**: Qwen via DashScope (阿里云百炼)
- **recursion_limit**: 2000 | **max_iterations**: 5 (CLI default)
- **Log**: \`twinScientist/logs/twinscientist.log\`
"@

$headers = @{
    "Authorization" = "Bearer $token"
    "Accept"        = "application/vnd.github+json"
}

Invoke-RestMethod -Uri "https://api.github.com/repos/18357034693-stack/TwinScientist3/issues" `
    -Method Post -Headers $headers -Body ($body | ConvertTo-Json -Depth 3) -ContentType "application/json"
