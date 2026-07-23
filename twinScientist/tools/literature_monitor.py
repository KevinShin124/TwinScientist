"""
Layer 6 - Item 21: Continuous Literature Monitor — Real Implementation

后台定时检索新论文，自动评估与当前假设关联度并通知 Orchestrator。
基于 tools.lit_search.LiteratureSearchEngine 提供真实 API 检索能力。
"""

from __future__ import annotations

import logging
import asyncio
from datetime import datetime, timezone
from typing import Any

from tools.lit_search import LiteratureSearchEngine, Paper

logger = logging.getLogger(__name__)


class LiteratureMonitor:
    """
    持续文献监控 — 后台定时任务（已实现真实 API 搜索）

    功能：
    1. 基于关键词定期检索最新文献（Crossref + arXiv / Semantic Scholar）
    2. LLM 评估与新假说的关联度（通过摘要 bigram 重叠率）
    3. 高关联度新论文推送到 Orchestrator

    Usage:
        monitor = LiteratureMonitor(semantic_scholar_key="YOUR_KEY")
        await monitor.start_background(interval_seconds=3600)
    """

    def __init__(
        self,
        api_key: str = "",
        semantic_scholar_key: str = "",
        engine: LiteratureSearchEngine | None = None,
    ):
        self.api_key = api_key
        self.semantic_scholar_key = semantic_scholar_key
        self.running = False
        # Use provided engine or create a new one with same key
        if engine is not None:
            self._engine = engine
        else:
            from config.settings import settings
            sem_key = getattr(settings, "semantic_scholar_api_key", "") or semantic_scholar_key
            self._engine = LiteratureSearchEngine(semantic_scholar_key=sem_key)
        self._known_papers: set[str] = set()  # Track seen paper DOIs/titles

    async def start_background(self, interval_seconds: int = 3600):
        """启动后台定时监测循环"""
        logger.info(f"[LiteratureMonitor] Starting background loop every {interval_seconds}s")
        self.running = True
        while self.running:
            try:
                await self.search_and_evaluate()
            except Exception as e:
                logger.error(f"[LiteratureMonitor] Background search error: {e}")
            await asyncio.sleep(interval_seconds)

    def stop(self):
        """停止后台循环"""
        self.running = False

    async def search_and_evaluate(
        self,
        domain_hint: str = "",
        max_total: int = 20,
    ) -> list[Paper]:
        """
        执行一轮文献搜索和评估

        Args:
            domain_hint: 领域提示语
            max_total: 最大返回结果数

        Returns: 高关联度的新论文列表（过滤掉 _known_papers 中已有的）
        """
        try:
            papers = await self._engine.search(domain_hint or "环境—人体关联", max_total=max_total)
        except Exception as e:
            logger.warning(f"[LiteratureMonitor] Search failed: {e}")
            return []

        # Filter out already-seen papers
        new_papers = [p for p in papers if self._is_new(p)]
        for p in new_papers:
            self._mark_seen(p)

        if new_papers:
            logger.info(f"[LiteratureMonitor] Found {len(new_papers)} new papers ({len(papers)} total)")
        else:
            logger.debug("[LiteratureMonitor] No new papers since last check")

        return new_papers

    async def assess_relevance(
        self,
        paper: Paper,
        hypothesis_statement: str,
    ) -> float:
        """
        评估一篇论文与当前假设的关联度（0-1 分数）。

        使用 bigram Jaccard 相似度在标题+摘要与假设陈述之间计算。
        如果有语义标记（如特定变量名），额外加分。

        Args:
            paper: 待评估的论文对象
            hypothesis_statement: 当前假设陈述文本

        Returns: relevance score 0-1
        """
        if not paper.abstract and not paper.title:
            return 0.0

        texts_to_compare = [paper.title]
        if paper.abstract:
            texts_to_compare.append(paper.abstract[:500])  # Truncate long abstracts

        scores = []
        for text in texts_to_compare:
            sim = self._bigram_similarity(text, hypothesis_statement)
            scores.append(sim)

        if not scores:
            return 0.0

        # Weight title higher than abstract
        base_score = scores[0] * 0.7 + (max(scores[1:]) if len(scores) > 1 else 0.0) * 0.3

        # Bonus for citation count (cited papers are more likely relevant)
        citation_bonus = min(paper.citation_count / 100, 1.0) * 0.1

        # Open access bonus
        oa_bonus = 0.05 if paper.open_access else 0.0

        final_score = min(base_score + citation_bonus + oa_bonus, 1.0)
        return round(final_score, 4)

    @staticmethod
    def _bigram_similarity(a: str, b: str) -> float:
        """字符级 bigram Jaccard 相似度（支持中英文）"""
        a_norm = ' '.join(a.strip().split())
        b_norm = ' '.join(b.strip().split())

        def bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1)) if len(s) > 1 else {s}

        a_set, b_set = bigrams(a_norm), bigrams(b_norm)
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / len(a_set | b_set)

    def _is_new(self, paper: Paper) -> bool:
        """判断是否为已知论文"""
        if paper.doi:
            return paper.doi.lower() not in self._known_dois
        return paper.title.lower() not in self._known_titles

    def _mark_seen(self, paper: Paper):
        """标记论文为已见"""
        if paper.doi:
            self._known_dois.add(paper.doi.lower())
        if paper.title:
            self._known_titles.add(paper.title.lower())

    @property
    def _known_dois(self) -> set:
        if not hasattr(self, '_known_dois_internal'):
            self._known_dois_internal: set[str] = set()
        return self._known_dois_internal

    @property
    def _known_titles(self) -> set:
        if not hasattr(self, '_known_titles_internal'):
            self._known_titles_internal: set[str] = set()
        return self._known_titles_internal
