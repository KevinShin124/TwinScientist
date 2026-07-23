"""
Layer 4 — Education Annotation System

Generates human-readable educational explanations for each Agent operation.
Helps users understand:
- Why did the agent choose this node?
- What is the scientific reasoning behind this decision?
- What should be the focus of the next step?

Usage:
    from core.education import EducationAnnotation

    annotation = EducationAnnotation.explain_action(
        action="literature_review",
        reason="Extracting facts...",
        state_context={...},
    )
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Action explanation templates per node type
TEMPLATES = {
    "literature_review": {
        "title": "[LITERATURE REVIEW]",
        "body_template": "Retrieving前沿 research on `{query}` from academic databases (PubMed, CrossRef, arXiv).\n"
                         "\n"
                         "Found {n_papers} papers. Extracted and validated {n_facts} key facts.\n\n"
                         "**Why**: Research must be anchored in verified literature, not LLM hallucination.",
    },
    "hypothesis_generation": {
        "title": "[HYPOTHESIS GENERATION]",
        "body_template": "Generating candidate hypotheses using three reasoning paths:\n"
                         "- **Inductive**: trends from existing data\n"
                         "- **Deductive**: rules → predictions (e.g., IF high CO2 THEN HRV decreases)\n"
                         "- **Abductive**: finding alternative explanations for anomalies\n\n"
                         "Current pool: {n_hyps} candidates, {consistency_issues} consistency issues flagged.",
    },
    "tournament_eval": {
        "title": "[TOURNAMENT EVALUATION]",
        "body_template": "Running elimination bracket: comparing all candidates on four dimensions:\n"
                         "Logic strength | Evidence support | Testability | Validation value\n"
                         "Only ONE hypothesis advances to experiment design.",
    },
    "experiment_design": {
        "title": "[EXPERIMENT DESIGN]",
        "body_template": "Designing verifiable experiment for the winning hypothesis.\n"
                         "Defines IV (environmental factor), DV (biomarker), control group, sampling strategy.\n"
                         "Ready to execute with real sensor data.",
    },
    "data_analysis": {
        "title": "[DATA ANALYSIS]",
        "body_template": "Running causal inference on experimental data.\n"
                         "AI auto-selected method: CCM / Granger / Counterfactual analysis.\n"
                         "Building evidence chains (not simple correlation!).\n"
                         "Evidence strength so far: {avg_evidence:.2f}.",
    },
    "reviewer_agent": {
        "title": "[PEER REVIEW]",
        "body_template": "Five-dimension peer review:\n"
                         "Novelty (20%) | Feasibility (20%) | Methodology (20%) | Evidence (20%) | Impact (20%)\n"
                         "Score: {score}/100 {'PASS' if True else 'NEEDS REVISION'}.",
    },
    "reflection": {
        "title": "[REFLECTION & CORRECTION]",
        "body_template": "Analyzing failures: root cause, confounding factors, alternative interpretations.\n"
                         "Deriving improved hypotheses for next round.\n"
                         "Failure assets are stored as lessons for future iterations.",
    },
    "debate": {
        "title": "[MULTI-AGENT DEBATE]",
        "body_template": "Pro vs Con vs Judge adversarial debate:\n"
                         "- Pro defends the strongest hypothesis with evidence\n"
                         "- Con finds logical gaps and uncontrolled confounders\n"
                         "- Judge impartially re-scores based on argument quality\n"
                         "Round {round_num}/{max_rounds}: Score moved {score_before:.0f} -> {score_after:.0f}",
    },
    "human_approval": {
        "title": "[HUMAN IN THE LOOP]",
        "body_template": "Your input is needed at this checkpoint.\n"
                         "**Current status**: {status_summary}\n\n"
                         "Options: APPROVE | REQUEST REVISION | CHAT WITH AGENT | HALT",
    },
    "termination_eval": {
        "title": "[TERMINATION ASSESSMENT]",
        "body_template": "Evaluating whether research has reached reliable conclusions.\n"
                         "Checks: semantic convergence ({convergence:.0%}), evidence strength ({evidence_strength:.2f}), "
                         "methodology stability, evidence dimension coverage.\n"
                         "Result: {'STOP' if True else 'CONTINUE'}.",
    },
    "default": {
        "title": "[AGENT ACTION]",
        "body_template": "Operation: {action_name}. Status: {iteration}/200 rounds, "
                        "{n_hyps} hypotheses, {n_evidence} evidence chains.",
    },
}


class EducationAnnotation:
    """Generate educational explanations for Agent decisions."""

    @staticmethod
    def explain_action(action: str, reason: str, context: dict | None = None) -> str:
        ctx = context or {}
        t = TEMPLATES.get(action, TEMPLATES["default"])

        try:
            body = t["body_template"].format(
                query=ctx.get("query", ""),
                n_papers=len(ctx.get("_papers_found", []) or []),
                n_facts=len(ctx.get("fact_extraction", []) or []),
                n_hyps=len([h for h in ctx.get("hypothesis_tree", []) if h.get("status") not in ("pruned", "refused")]),
                consistency_issues=len(ctx.get("_logic_consistency_reports", []) or []),
                avg_evidence=sum(e.get("strength", 0.5) for e in ctx.get("evidence_chains", []) or []) / max(len(ctx.get("evidence_chains", []) or []), 1),
                score=ctx.get("review_score", "?"),
                status_summary=ctx.get("current_status", ""),
                round_num=ctx.get("debate_round", 1),
                max_rounds=ctx.get("debate_max_rounds", 3),
                score_before=ctx.get("debate_score_before", 0),
                score_after=ctx.get("debate_score_after", 0),
                convergence=ctx.get("convergence_score", 0.0),
                evidence_strength=ctx.get("evidence_strength", 0.0),
                iteration=ctx.get("iteration", 0),
                n_evidence=len(ctx.get("evidence_chains", []) or []),
                action_name=t["title"],
            )
        except Exception:
            body = reason

        return f"{t['title']}\n\n{body}"

    @staticmethod
    def quick_actions(action: str) -> list[dict]:
        ACTIONS = {
            "human_approval": [
                {"label": "Approve & Continue", "value": "approve"},
                {"label": "Request Revision", "value": "revise", "input_required": True},
                {"label": "Chat With Agent", "value": "chat"},
                {"label": "Halt Research", "value": "halt"},
            ],
            "debate": [
                {"label": "Read Debate Record", "value": "read_debate"},
                {"label": "Challenge Hypothesis", "value": "challenge"},
                {"label": "Provide Supplement Info", "value": "provide_info"},
                {"label": "Accept Verdict", "value": "accept"},
            ],
            "reflection": [
                {"label": "View Root Cause", "value": "view_root_cause"},
                {"label": "Suggest Direction", "value": "suggest_direction"},
                {"label": "View Data Quality", "value": "view_data_quality"},
                {"label": "Continue", "value": "continue"},
            ],
        }
        return ACTIONS.get(action, [
            {"label": "Ask for Details", "value": "chat"},
            {"label": "Proceed", "value": "proceed"},
        ])
