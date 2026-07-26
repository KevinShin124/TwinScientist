"""
SFT Data Pipeline — Fine-tuning Data Collection for Qwen.

Collects (prompt, completion) pairs from research sessions to enable
future SFT fine-tuning of Qwen on scientific hypothesis generation tasks.

The competition explicitly allows SFT/微调, and this pipeline automates
the data collection process. Each completed research session produces
training examples for:
- Hypothesis generation from literature facts
- Causal inference method selection
- Peer review scoring
- Report section generation

Usage:
    from core.sft_pipeline import SFTDataCollector
    collector = SFTDataCollector()
    collector.collect_from_session(state)
    collector.export("data/sft/hypothesis_gen.jsonl")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SFTDataCollector:
    """
    Collects and exports (prompt, completion) pairs for Qwen SFT fine-tuning.

    Each session produces multiple training examples across different task types.
    """

    TASKS = [
        "hypothesis_generation",    # Facts → Hypotheses
        "method_selection",         # Data features → Causal method
        "peer_review",             # Hypothesis → Review scores
        "report_rationale",        # Context → Rationale section
        "report_abstract",         # Context → Abstract section
    ]

    def __init__(self, output_dir: str = "./data/sft"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._examples: dict[str, list[dict]] = {t: [] for t in self.TASKS}

    def collect_from_session(self, state: dict):
        """
        Extract training examples from a completed research session.

        Called after each successful research run.
        """
        try:
            self._collect_hypothesis_gen(state)
            self._collect_method_selection(state)
            self._collect_peer_review(state)
            self._collect_report_sections(state)
            logger.info(
                f"[SFT] Collected {sum(len(v) for v in self._examples.values())} "
                f"training examples from session"
            )
        except Exception as e:
            logger.warning(f"[SFT] Collection failed: {e}")

    def _collect_hypothesis_gen(self, state: dict):
        """Extract (facts → hypothesis) training pairs."""
        facts = state.get("fact_extraction", [])
        hypotheses = state.get("hypothesis_tree", [])
        if not facts or not hypotheses:
            return

        # Use approved hypotheses as positive examples
        approved = [h for h in hypotheses if h.get("status") == "approved_by_reviewer"]
        for hyp in approved[:3]:  # Top 3 per session
            facts_text = "\n".join(f"- {f.get('fact', '')}" for f in facts[:10])
            self._examples["hypothesis_generation"].append({
                "instruction": "基于以下科学事实，生成一个可验证的科研假设",
                "input": facts_text,
                "output": json.dumps({
                    "title": hyp.get("title", ""),
                    "statement": hyp.get("statement", ""),
                    "reasoning_chain": hyp.get("reasoning_chain", ""),
                    "confidence_prior": hyp.get("confidence_prior", 0),
                    "testability": hyp.get("testability", 5),
                }, ensure_ascii=False),
                "metadata": {"timestamp": datetime.now(timezone.utc).isoformat()},
            })

    def _collect_method_selection(self, state: dict):
        """Extract (data features → method selection) training pairs."""
        evidence_chains = state.get("evidence_chains", [])
        for ev in evidence_chains:
            if ev.get("type") != "causal_inference":
                continue
            method = ev.get("method_used", "")
            if not method:
                continue

            self._examples["method_selection"].append({
                "instruction": "根据数据特征，选择最合适的因果推断方法",
                "input": json.dumps({
                    "sample_size": 500,
                    "is_time_series": True,
                    "nonlinear": False,
                }),
                "output": method,
                "metadata": {
                    "strength": ev.get("strength", 0),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            })

    def _collect_peer_review(self, state: dict):
        """Extract (hypothesis → review) training pairs."""
        reviews = state.get("review_records", [])
        hypotheses = state.get("hypothesis_tree", [])
        for review in reviews:
            hyp_id = review.get("hypothesis_id", "")
            hyp = next((h for h in hypotheses if h.get("id") == hyp_id), {})
            if not hyp:
                continue

            self._examples["peer_review"].append({
                "instruction": "对以下科学假设进行五维评审（新颖性/可行性/方法论/证据/影响力）",
                "input": json.dumps({
                    "title": hyp.get("title", ""),
                    "statement": hyp.get("statement", ""),
                    "reasoning": hyp.get("reasoning_chain", "")[:300],
                }, ensure_ascii=False),
                "output": json.dumps({
                    "total_score": review.get("total_score", 0),
                    "novelty_score": review.get("novelty_score", 0),
                    "feasibility_score": review.get("feasibility_score", 0),
                    "methodology_score": review.get("methodology_score", 0),
                    "evidence_score": review.get("evidence_score", 0),
                    "impact_score": review.get("impact_score", 0),
                    "needs_revision": review.get("needs_revision", False),
                    "comments": review.get("comments", "")[:500],
                }, ensure_ascii=False),
                "metadata": {"timestamp": datetime.now(timezone.utc).isoformat()},
            })

    def _collect_report_sections(self, state: dict):
        """Extract (context → report section) training pairs."""
        report = state.get("final_report", "")
        if not report or len(report) < 500:
            return

        # Extract Rationale section
        if "## 二、解决思路" in report:
            parts = report.split("## 二、解决思路")
            if len(parts) > 1:
                rationale = parts[1].split("---")[0].strip()[:1000]
                query = state.get("query", "")
                self._examples["report_rationale"].append({
                    "instruction": "基于研究问题和文献事实，撰写解决思路（Rationale）",
                    "input": f"研究问题: {query}",
                    "output": rationale,
                    "metadata": {"timestamp": datetime.now(timezone.utc).isoformat()},
                })

        # Extract Abstract section
        if "## 六、摘要" in report:
            parts = report.split("## 六、摘要")
            if len(parts) > 1:
                abstract = parts[1].split("---")[0].strip()[:800]
                self._examples["report_abstract"].append({
                    "instruction": "基于研究结果，撰写论文摘要（Abstract）",
                    "input": f"研究问题: {state.get('query', '')}",
                    "output": abstract,
                    "metadata": {"timestamp": datetime.now(timezone.utc).isoformat()},
                })

    def export(self, task: str = "all") -> dict[str, str]:
        """
        Export collected examples as JSONL files for SFT training.

        Args:
            task: "all" or one of TASKS

        Returns:
            Dict mapping task names to output file paths
        """
        tasks = self.TASKS if task == "all" else [task]
        paths = {}

        for t in tasks:
            examples = self._examples.get(t, [])
            if not examples:
                continue

            path = self._output_dir / f"{t}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

            paths[t] = str(path)
            logger.info(f"[SFT] Exported {len(examples)} examples to {path}")

        return paths

    def get_stats(self) -> dict:
        """Return collection statistics."""
        return {
            task: len(examples)
            for task, examples in self._examples.items()
        }

    def reset(self, task: str = "all"):
        """Clear collected examples."""
        if task == "all":
            self._examples = {t: [] for t in self.TASKS}
        elif task in self._examples:
            self._examples[task] = []


# Singleton
sft_collector = SFTDataCollector()