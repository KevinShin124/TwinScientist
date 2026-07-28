"""
Streaming Progress Dashboard — Apple/Google-grade real-time pipeline visibility.

Shows each cognitive node as it executes, with timing, status, and key metrics.
Integrates with LangGraph's astream_events() for zero-overhead event capture.

Usage:
    from core.progress import ProgressDashboard
    dashboard = ProgressDashboard()
    async for event in cognitive_graph.astream_events(state, config):
        dashboard.on_event(event)
"""

from __future__ import annotations

import sys
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Node display names and icons for the dashboard (ASCII-safe for Windows)
NODE_DISPLAY = {
    "ethics_check":          ("伦理审查",       "[ETHICS]"),
    "literature_review":      ("文献调研",       "[LIT]"),
    "hypothesis_generation":  ("假设生成",       "[HYPO]"),
    "tournament_eval":        ("淘汰赛评估",     "[TOURN]"),
    "experiment_design":      ("实验设计",       "[EXPT]"),
    "data_analysis":          ("数据分析",       "[DATA]"),
    "interpretation":         ("结果解读",       "[INTER]"),
    "reviewer_agent":         ("五维评审",       "[REVW]"),
    "reflection":             ("反思修正",       "[REFL]"),
    "debate_then_terminate":  ("智能体辩论",     "[DEBATE]"),
    "termination_eval":       ("终止评估",       "[TERM]"),
    "report_writing":         ("报告撰写",       "[RPRT]"),
    "pi_agent_meeting":       ("PI 总结",        "[PI]"),
    "human_approval":         ("人机审核",       "[HITL]"),
    "evolution_manager":      ("自我进化",       "[EVOL]"),
    "post_report_chat":       ("研究总结",       "[CHAT]"),
}


class ProgressDashboard:
    """
    Real-time streaming progress display for the CLI.

    Shows a tree of pipeline steps with live status updates.
    Designed to look like a professional CI/CD pipeline or build system.
    """

    def __init__(self, total_expected: int = 14):
        self._start_time = time.time()
        self._node_times: dict[str, float] = {}
        self._completed: list[str] = []
        self._current: str = ""
        self._total = total_expected
        self._step = 0
        self._quiet = False

    def quiet(self):
        """Suppress output (for non-interactive modes)."""
        self._quiet = True

    def on_node_start(self, node_name: str):
        """Called when a node begins execution."""
        if self._quiet:
            return
        self._current = node_name
        self._step += 1
        display, icon = NODE_DISPLAY.get(node_name, (node_name, "⚙️"))
        elapsed = time.time() - self._start_time
        bar = self._progress_bar()
        sys.stdout.write(
            f"\r  {icon} [{bar}] {self._step}/{self._total}  {display:<16}  ... {elapsed:.0f}s"
        )
        sys.stdout.flush()

    def on_node_end(self, node_name: str, state: dict | None = None):
        """Called when a node completes."""
        if self._quiet:
            return
        self._completed.append(node_name)
        self._node_times[node_name] = time.time() - self._start_time
        display, icon = NODE_DISPLAY.get(node_name, (node_name, "⚙️"))
        elapsed = time.time() - self._start_time
        bar = self._progress_bar()

        # Extract key metrics from state for display
        metrics = self._extract_metrics(node_name, state)

        # Clear line and write completion
        sys.stdout.write(
            f"\r  {icon} [{bar}] {self._step}/{self._total}  {display:<16}  OK {elapsed:.0f}s  {metrics}"
        )
        sys.stdout.write("\n")
        sys.stdout.flush()

    def on_node_error(self, node_name: str, error: str):
        """Called when a node fails."""
        if self._quiet:
            return
        display, icon = NODE_DISPLAY.get(node_name, (node_name, "⚙️"))
        sys.stdout.write(
            f"\r  {icon} {'-' * 10} {self._step}/{self._total}  {display:<16}  ERR {error[:60]}"
        )
        sys.stdout.write("\n")
        sys.stdout.flush()

    def summary(self) -> str:
        """Return a summary of the pipeline execution."""
        total_time = time.time() - self._start_time
        lines = [
            "",
            "+" + "-" * 58 + "+",
            f"  Pipeline completed in {total_time:.0f}s ({len(self._completed)} nodes executed)",
            "+" + "-" * 58 + "+",
        ]
        for node in self._completed:
            display, icon = NODE_DISPLAY.get(node, (node, "⚙️"))
            t = self._node_times.get(node, 0)
            lines.append(f"  {icon} {display:<20} {t:.0f}s")
        return "\n".join(lines)

    def _progress_bar(self, width: int = 10) -> str:
        """Simple ASCII progress bar."""
        filled = int(self._step / max(self._total, 1) * width)
        return "#" * filled + "-" * (width - filled)

    def _extract_metrics(self, node_name: str, state: dict | None) -> str:
        """Extract relevant metrics from state for display."""
        if not state:
            return ""
        try:
            if node_name == "literature_review":
                facts = state.get("fact_extraction", [])
                verified = sum(1 for f in facts if f.get("_verified"))
                return f"({len(facts)} facts, {verified} verified)"
            if node_name == "hypothesis_generation":
                hyps = state.get("hypothesis_tree", [])
                return f"({len(hyps)} hypotheses)"
            if node_name == "tournament_eval":
                return f"(winner selected)"
            if node_name == "data_analysis":
                evidence = state.get("evidence_chains", [])
                if evidence:
                    strength = evidence[-1].get("strength", 0)
                    method = evidence[-1].get("method_used", "?")
                    return f"({method}, strength={strength:.3f})"
            if node_name == "interpretation":
                conv = state.get("convergence_score", 0)
                return f"(convergence={conv:.0%})"
            if node_name == "reviewer_agent":
                reviews = state.get("review_records", [])
                if reviews:
                    score = reviews[-1].get("total_score", "?")
                    return f"(score={score}/100)"
            if node_name == "debate_then_terminate":
                debates = state.get("debate_history", [])
                return f"({len(debates)} rounds)"
            if node_name == "termination_eval":
                term = state.get("_termination_result", {})
                if term.get("should_terminate"):
                    return "(terminating)"
                return "(continuing)"
            if node_name == "report_writing":
                report = state.get("final_report", "")
                return f"({len(report)} chars)"
        except Exception:
            pass
        return ""


# Singleton instance for use across modules
dashboard = ProgressDashboard()