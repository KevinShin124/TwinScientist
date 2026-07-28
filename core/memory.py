"""
Research Memory — Cross-Session Knowledge Persistence (Google-style).

Remembers findings from previous research sessions and retrieves
relevant knowledge for new queries. Enables the system to build
on past discoveries rather than starting from scratch each time.

Design: Simple JSON file-based persistence. No external dependencies.
Production would use a vector database; this is the competition-grade
lightweight implementation.

Usage:
    from core.memory import ResearchMemory
    memory = ResearchMemory()
    memory.remember(session_id, state)
    relevant = memory.recall(query, top_k=5)
"""

from __future__ import annotations

import json
import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ResearchMemory:
    """
    Persistent knowledge store across research sessions.

    Stores: hypotheses, evidence chains, key findings, and conclusions.
    Retrieves: relevant past findings for new research questions.
    """

    def __init__(self, db_path: str = "./data/research_memory.json"):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict] = self._load()

    def _load(self) -> list[dict]:
        """Load existing memory entries."""
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save(self):
        """Persist memory to disk."""
        self._path.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def remember(self, session_id: str, state: dict) -> dict:
        """
        Extract and store key findings from a completed research session.

        Returns the memory entry that was stored.
        """
        hypotheses = state.get("hypothesis_tree", [])
        evidence_chains = state.get("evidence_chains", [])
        final_report = state.get("final_report", "")
        query = state.get("query", "")
        domain = state.get("domain", "")
        iteration = state.get("iteration", 0)
        convergence = state.get("convergence_score", 0.0)

        # Extract key findings
        approved_hyps = [
            h for h in hypotheses
            if h.get("status") == "approved_by_reviewer"
        ]
        best_hyp = max(
            approved_hyps,
            key=lambda h: h.get("confidence_posterior", h.get("confidence_prior", 0)),
        ) if approved_hyps else (hypotheses[0] if hypotheses else {})

        evidence_summary = []
        for ev in evidence_chains:
            if ev.get("type") == "causal_inference":
                evidence_summary.append({
                    "method": ev.get("method_used", "?"),
                    "strength": ev.get("strength", 0),
                    "direction": ev.get("causal_direction", "?"),
                    "summary": ev.get("content", "")[:200],
                })

        entry = {
            "id": hashlib.sha256(session_id.encode()).hexdigest()[:12],
            "session_id": session_id,
            "query": query,
            "domain": domain,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iterations": iteration,
            "convergence": convergence,
            "best_hypothesis": {
                "title": best_hyp.get("title", ""),
                "statement": best_hyp.get("statement", "")[:300],
                "confidence": best_hyp.get("confidence_posterior",
                           best_hyp.get("confidence_prior", 0)),
            },
            "evidence_summary": evidence_summary,
            "report_excerpt": final_report[:1000] if final_report else "",
            "keywords": self._extract_keywords(query, domain, best_hyp),
        }

        # Avoid duplicates
        existing_ids = {e["id"] for e in self._entries}
        if entry["id"] not in existing_ids:
            self._entries.append(entry)
            self._save()
            logger.info(f"[Memory] Stored session {entry['id']}: '{query[:50]}...'")

        return entry

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve relevant past research findings for a new query.

        Uses simple keyword matching (production would use embeddings).
        """
        if not self._entries:
            return []

        query_lower = query.lower()
        scored = []

        for entry in self._entries:
            score = 0.0
            # Keyword match
            entry_keywords = entry.get("keywords", [])
            for kw in entry_keywords:
                if kw.lower() in query_lower:
                    score += 1.0
            # Domain match
            if entry.get("domain", "").lower() in query_lower:
                score += 0.5
            # Evidence quality bonus
            evidence = entry.get("evidence_summary", [])
            if evidence:
                avg_strength = sum(e.get("strength", 0) for e in evidence) / len(evidence)
                score += avg_strength * 0.5
            # Recency bonus
            score += 0.1  # slight preference for recent entries

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def format_context(self, relevant: list[dict]) -> str:
        """
        Format recalled memories as context for LLM prompts.
        """
        if not relevant:
            return ""

        lines = [
            "",
            "## 历史研究发现（来自过往研究会话）",
            "以下发现来自之前的科研会话，可供当前研究参考：",
            "",
        ]
        for i, entry in enumerate(relevant, 1):
            hyp = entry.get("best_hypothesis", {})
            lines.append(f"### 历史发现 {i}: {entry.get('query', '?')[:80]}")
            lines.append(f"- 最优假设: {hyp.get('title', '?')}")
            lines.append(f"- 置信度: {hyp.get('confidence', 0):.0%}")
            lines.append(f"- 迭代轮次: {entry.get('iterations', '?')}")
            lines.append(f"- 收敛度: {entry.get('convergence', 0):.0%}")

            evidence = entry.get("evidence_summary", [])
            if evidence:
                lines.append("- 证据链:")
                for ev in evidence[:3]:
                    lines.append(f"  - [{ev.get('method', '?')}] strength={ev.get('strength', 0):.3f}: {ev.get('summary', '')[:100]}")
            lines.append("")

        return "\n".join(lines)

    def _extract_keywords(self, query: str, domain: str, best_hyp: dict) -> list[str]:
        """Extract keywords from query, domain, and hypothesis."""
        keywords = set()
        # Split on common delimiters
        for text in [query, domain, best_hyp.get("title", ""), best_hyp.get("statement", "")]:
            # Extract 2-4 character Chinese terms and English words
            chinese_terms = re.findall(r'[一-鿿]{2,4}', text)
            english_terms = re.findall(r'[a-zA-Z]{3,}', text)
            keywords.update(chinese_terms[:5])
            keywords.update(english_terms[:3])
        return list(keywords)[:10]

    def get_stats(self) -> dict:
        """Return memory statistics."""
        domains = {}
        for e in self._entries:
            d = e.get("domain", "unknown")
            domains[d] = domains.get(d, 0) + 1
        return {
            "total_sessions": len(self._entries),
            "domains": domains,
            "avg_iterations": round(
                sum(e.get("iterations", 0) for e in self._entries) / max(len(self._entries), 1), 1
            ),
            "avg_convergence": round(
                sum(e.get("convergence", 0) for e in self._entries) / max(len(self._entries), 1), 2
            ),
        }


# Singleton
memory = ResearchMemory()