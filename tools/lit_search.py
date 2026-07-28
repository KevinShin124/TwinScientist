"""
Layer 5 — Literature Search & Citation Verification Engine

整合多个学术数据源的统一文献检索与引用验证系统：
- Crossref REST API（免费无需密钥）—— DOI、作者、期刊、引用次数
- arXiv Atom feed（免费无需密钥）—— 预印本论文搜索
- Semantic Scholar API（可选，需 API Key）—— 语义相似度搜索 + 引用图谱

工作流程：
1. search() → 并行调用所有可用数据源，去重后返回 Top-N 论文列表
2. extract_facts() → LLM 基于真实论文上下文提取事实
3. validate_all_facts() → 交叉验证每条事实引用的真实性

Usage:
    from tools.lit_search import LiteratureSearchEngine, CitationValidator

    engine = LiteratureSearchEngine()
    papers = await engine.search("indoor temperature heart rate variability")

    validator = CitationValidator()
    verified = await validator.validate_all_facts(facts)
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# ============================================================
# Data Classes
# ============================================================


@dataclass
class Paper:
    """统一的论文数据结构，兼容三个数据源的异构输出"""

    title: str = ""
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    pmid: str = ""
    url: str = ""
    source: str = "crossref"  # "crossref" | "arxiv" | "semantic_scholar"
    citation_count: int = 0
    keywords: list[str] = field(default_factory=list)
    open_access: bool = False
    raw: dict = field(default_factory=dict)

    @property
    def citation_str(self) -> str:
        """格式化引用字符串，供 LLM prompt 使用"""
        parts = []
        if self.authors:
            parts.append(", ".join(self.authors[:5]))
            if len(self.authors) > 5:
                parts.append(" et al.")
        if self.year:
            parts.append(f" ({self.year}).")
        else:
            parts.append(".")
        parts.append(f" {self.title}.")
        if self.venue:
            parts.append(f" {self.venue}.")
        if self.doi:
            parts.append(f" DOI:{self.doi}")
        return "".join(parts)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "pmid": self.pmid,
            "url": self.url,
            "source": self.source,
            "citation_count": self.citation_count,
            "keywords": self.keywords,
        }


# ============================================================
# Base Configuration
# ============================================================


def _get_optional_env(key: str, default: str = "") -> str:
    """从环境变量读取可选配置项"""
    import os
    return os.environ.get(key, default)


# ============================================================
# CrossRef Search Engine
# ============================================================


class CrossRefSearcher:
    """Crossref REST API — 免费无需密钥"""

    BASE_URL = "https://api.crossref.org/works"

    def __init__(self):
        self._client_base = "https://api.crossref.org"

    async def search(self, query: str, max_results: int = 20) -> list[Paper]:
        """
        通过 Crossref API 搜索论文。

        Args:
            query: 搜索关键词（自动 URL encode，非 ASCII 字符会被过滤）
            max_results: 最大结果数（API 限制为 20）

        Returns:
            Paper 对象列表，按相关度排序
        """
        # Crossref API rejects queries with non-ASCII characters (400 Bad Request).
        # Map Chinese scientific terms to English keywords for the API call.
        CN_TO_EN_KEYWORDS = {
            "温度": "temperature", "湿度": "humidity", "心率": "heart rate",
            "心率变异性": "heart rate variability", "HRV": "HRV",
            "血氧": "blood oxygen", "SpO2": "SpO2", "CO2": "CO2",
            "二氧化碳": "carbon dioxide", "环境": "environment",
            "人体": "human", "健康": "health", "影响": "effect",
            "因果": "causal", "传感器": "sensor", "PM2.5": "PM2.5",
            "VOC": "VOC", "视觉": "visual", "疲劳": "fatigue",
            "个体化": "personalized", "N-of-1": "N-of-1",
            "睡眠": "sleep", "血压": "blood pressure",
            "室内": "indoor", "空气": "air", "质量": "quality",
            "暴露": "exposure", "响应": "response",
            "生理": "physiological", "指标": "indicator",
            "生物": "biometric", "特征": "feature",
            "天气": "weather", "气候": "climate",
            "污染物": "pollutant", "臭氧": "ozone",
            "噪声": "noise", "光照": "light",
            "通风": "ventilation", "建筑": "building",
            "办公室": "office", "学校": "school",
            "老年人": "elderly", "儿童": "children",
            "长期": "long-term", "短期": "short-term",
            "监测": "monitoring", "可穿戴": "wearable",
            "研究": "study", "分析": "analysis",
        }
        ascii_query = query
        for cn, en in CN_TO_EN_KEYWORDS.items():
            ascii_query = ascii_query.replace(cn, en)
        # Also strip remaining non-ASCII
        ascii_query = re.sub(r'[^\x00-\x7F]+', ' ', ascii_query).strip()
        if not ascii_query or len(ascii_query) < 3:
            ascii_query = "indoor environment human health physiological response"

        params = {
            "query": ascii_query,
            "rows": min(max_results, 20),
            "mailto": "twinScientist@research",
        }

        try:
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)) as client:
                resp = await client.get(
                    f"{self._client_base}/works",
                    params=params,
                    headers={"Accept": "application/json", "User-Agent": "TwinScientist/1.0 (mailto:twinScientist@research)"},
                )
                resp.raise_for_status()
                data = resp.json()

                items = data.get("message", {}).get("items", [])
                return [self._parse_item(item) for item in items if item.get("DOI")]

        except Exception as e:
            logger.warning(f"[CrossRef] Search failed for '{query}': {e}")
            return []

    def _parse_item(self, item: dict) -> Paper:
        """将 Crossref JSON 响应解析为标准 Paper 对象"""
        authors_raw = item.get("author", [])
        authors = []
        for a in authors_raw:
            given = a.get("given", "")
            family = a.get("family", "")
            if given and family:
                authors.append(f"{given} {family}".strip())
            elif family:
                authors.append(family)
            elif given:
                authors.append(given)

        # Extract year from published-print or published-online
        year = None
        for date_field in ("published-print", "published-online", "created"):
            date_info = item.get(date_field)
            if date_info and isinstance(date_info, dict):
                date_parts = date_info.get("date-parts", [[], ])
                if date_parts[0]:
                    year = date_parts[0][0]
                    break
            if not date_info and date_field == "created":
                # Fallback: use timestamp field
                ts = item.get("created", {}).get("timestamp")
                if ts:
                    year = int(ts) // 10000000000  # epoch ms to year approx

        # Venue / journal name
        container_titles = item.get("container-title", [])
        venue = container_titles[0] if container_titles else item.get("publisher", "")

        # Categories for keywords
        categories = []
        for cat_list in item.get("category", []):
            if isinstance(cat_list, dict):
                categories.append(cat_list.get("scheme", "").rstrip(": "))
            elif isinstance(cat_list, str):
                categories.append(cat_list)

        doi = item.get("DOI", "")

        return Paper(
            title=item.get("title", [""])[0],
            abstract="",  # Crossref doesn't provide abstracts without premium subscription
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            url=item.get("URL", ""),
            source="crossref",
            citation_count=item.get("is-referenced-by-count", 0),
            keywords=categories,
            raw=item,
        )


# ============================================================
# arXiv Search Engine
# ============================================================


class ArXivSearcher:
    """arXiv API — 免费无需密钥，支持 RSS/Atom feed 解析"""

    BASE_URL = "https://export.arxiv.org/api/query"

    async def search(self, query: str, max_results: int = 20) -> list[Paper]:
        """
        通过 arXiv API 搜索论文。

        优势: 提供免费摘要和全文链接
        劣势: 仅限物理/计算机/数学等领域（无直接生物医学）
        """
        # arXiv query syntax: ti=title, au=author, abs=abstract, all=any
        safe_query = quote_plus(query)
        url = (
            f"{self.BASE_URL}?search_query=all:{safe_query}"
            f"&start=0&max_results={min(max_results, 30)}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )

        try:
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)) as client:
                resp = await client.get(url, headers={"Accept": "application/atom+xml", "User-Agent": "TwinScientist/1.0"})
                resp.raise_for_status()

                return self._parse_atom(resp.text)

        except httpx.TimeoutException:
            logger.warning(f"[arXiv] Timeout for '{query[:80]}' — skipping")
            return []
        except Exception as e:
            logger.warning(f"[arXiv] Search failed for '{query[:80]}': {type(e).__name__}: {e}")
            return []

    def _parse_atom(self, xml_content: str) -> list[Paper]:
        """解析 arXiv Atom XML 响应为标准 Paper 列表"""
        namespace = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"[arXiv] XML parse error: {e}")
            return []

        papers = []
        for entry in root.findall("atom:entry", namespace):
            title_el = entry.find("atom:title", namespace)
            summary_el = entry.find("atom:summary", namespace)

            if title_el is None or summary_el is None:
                continue

            # Strip inline formatting tags from titles/summaries
            title = self._strip_tags(title_el.text).replace("\n", " ").strip()
            abstract = self._strip_tags(summary_el.text).replace("\n", " ").strip()

            # Authors
            author_elems = entry.findall("atom:author", namespace)
            authors = []
            for a in author_elems:
                name = a.findtext("atom:name", "", namespace)
                if name:
                    authors.append(name.strip())

            # Published date
            published = entry.findtext("atom:published", "", namespace)
            year = None
            if published:
                try:
                    year = int(published[:4])
                except ValueError:
                    pass

            # arXiv ID → extract id
            id_el = entry.findtext("atom:id", "", namespace)
            arxiv_id = ""
            if id_el:
                # id format: http://arxiv.org/abs/2301.xxxxx
                m = re.search(r"/(\d{4}\.\w+)", id_el)
                if m:
                    arxiv_id = m.group(1)

            # Categories
            categories = []
            for cat in entry.findall("atom:category", namespace):
                term = cat.get("term", "")
                if term:
                    categories.append(term.replace("_", " "))

            papers.append(Paper(
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                venue="arXiv preprint",
                doi="",  # arXiv papers may have DOI but we need separate lookup
                url=f"https://arxiv.org/abs/{arxiv_id}",
                source="arxiv",
                keywords=categories,
                raw={"arxiv_id": arxiv_id},
            ))

        return papers

    @staticmethod
    def _strip_tags(text: str) -> str:
        """去除 XML 中的 HTML 标签（arXiv 摘要可能含 LaTeX formatting）"""
        text = re.sub(r"<[^>]+>", "", text)
        # Replace common LaTeX macros with readable text
        latex_replacements = {
            r"\emph{": "(", r"}": ")",
            r"\textbf{": "[", r"}": "]",
            r"\tt ": "",
        }
        for old, new in latex_replacements.items():
            text = text.replace(old, new)
        return text


# ============================================================
# Semantic Scholar Search Engine (Optional, requires API Key)
# ============================================================


class SemanticScholarSearcher:
    """Semantic Scholar API — 免费注册后可用，最全面的学术搜索引擎"""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    async def search(self, query: str, max_results: int = 20) -> list[Paper]:
        """
        搜索 Semantic Scholar。免费 tier 无需 API key（rate-limited to 1 req/s）。
        """
        params = {
            "query": query,
            "limit": min(max_results, 20),
            "fields": "title,abstract,authors,year,externalIds,citationCount,venue,isOpenAccess",
        }

        headers = {"Accept": "application/json", "User-Agent": "TwinScientist/1.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)) as client:
                resp = await client.get(
                    self.BASE_URL,
                    params=params,
                    headers=headers,
                )
                if resp.status_code == 429:
                    logger.warning(f"[SemScholar] Rate limited — skipping")
                    return []
                resp.raise_for_status()
                data = resp.json()

                papers_data = data.get("data", [])
                return [self._parse_paper(p) for p in papers_data if p.get("title")]

        except httpx.TimeoutException:
            logger.warning(f"[SemScholar] Timeout for '{query[:80]}' — skipping")
            return []
        except Exception as e:
            logger.warning(f"[SemScholar] Search failed for '{query[:80]}': {type(e).__name__}: {e}")
            return []

    def _parse_paper(self, data: dict) -> Paper:
        """解析 Semantic Scholar JSON 响应"""
        # Authors: SS returns [{"name": "...", ...}]
        authors = [a.get("name", "") for a in data.get("authors", [])]

        # External IDs: extract DOI and PMID
        ext_ids = data.get("externalIds", {})
        doi = ext_ids.get("DOI", "")
        pmid = ext_ids.get("MEDLINE", ext_ids.get("PMID", ""))

        tldr = ""
        tldr_data = data.get("tldr")
        if isinstance(tldr_data, dict):
            tldr = tldr_data.get("text", "")
        elif isinstance(tldr_data, str):
            tldr = tldr_data

        return Paper(
            title=data.get("title", ""),
            abstract=data.get("abstract", tldr) or tldr,
            authors=authors,
            year=data.get("year"),
            venue=data.get("venue", ""),
            doi=doi,
            pmid=pmid,
            url=f"https://www.semanticscholar.org/paper/{data.get('paperId', '')}",
            source="semantic_scholar",
            citation_count=data.get("citationCount", 0),
            open_access=data.get("isOpenAccess", False),
            raw=data,
        )


# ============================================================
# Unified Literature Search Engine
# ============================================================


class LiteratureSearchEngine:
    """
    统一的文献搜索引擎。

    集成 Crossref、arXiv、Semantic Scholar 三个数据源，
    自动去重、排序，返回最佳匹配论文列表。
    """

    def __init__(
        self,
        semantic_scholar_key: str = "",
        max_crossref: int = 15,
        max_arxiv: int = 10,
        max_semantic_scholar: int = 15,
    ):
        self.crossref = CrossRefSearcher()
        self.arxiv = ArXivSearcher()
        self.semantic_scholar = SemanticScholarSearcher(semantic_scholar_key)
        self.max_crossref = max_crossref
        self.max_arxiv = max_arxiv
        self.max_semantic_scholar = max_semantic_scholar

    async def search(
        self,
        query: str,
        domain_hint: str = "",
        max_total: int = 20,
    ) -> list[Paper]:
        """
        并行搜索所有可用数据源，去重后返回综合排名结果。

        Args:
            query: 搜索查询
            domain_hint: 领域提示（用于调整各源权重）
            max_total: 返回总数上限

        Returns:
            Paper 对象列表
        """
        logger.info(f"[LitSearch] Searching: '{query}' (domain={domain_hint})")

        # Run searches in parallel
        results = await asyncio.gather(
            self._run_source("crossref", self.crossref.search, query, self.max_crossref),
            self._run_source("arxiv", self.arxiv.search, query, self.max_arxiv),
            self._run_source("sem-scholar", self.semantic_scholar.search, query, self.max_semantic_scholar),
        )

        all_papers: list[Paper] = []
        sources_used: list[str] = []
        for idx, source_name in enumerate(["crossref", "arxiv", "semantic_scholar"]):
            src_papers = results[idx]
            if src_papers:
                all_papers.extend(src_papers)
                sources_used.append(source_name)
                logger.info(f"[LitSearch] {source_name}: {len(src_papers)} papers found")

        if not all_papers:
            logger.warning("[LitSearch] No papers found from any source")
            return []

        # Deduplicate by DOI first, then by title similarity
        deduped = self._deduplicate(all_papers)

        # Re-rank by combined score (citation count + recency + source quality)
        ranked = self._rank_papers(deduped, query, domain_hint)

        return ranked[:max_total]

    async def _run_source(
        self,
        name: str,
        search_fn,
        query: str,
        max_results: int,
    ) -> list[Paper]:
        """Run a single search source with error handling"""
        try:
            papers = await asyncio.wait_for(
                search_fn(query, max_results),
                timeout=25.0,  # 25s timeout per source
            )
            return papers
        except asyncio.TimeoutError:
            logger.warning(f"[LitSearch] Source '{name}' timed out after 25s")
            return []
        except Exception as e:
            logger.warning(f"[LitSearch] Source '{name}' error: {e}")
            return []

    def _deduplicate(self, papers: list[Paper]) -> list[Paper]:
        """
        基于 DOI 和标题相似度去重。

        优先级: Semantic Scholar (有DOI/PMID最完整) > Crossref > arXiv
        """
        seen_dois: set[str] = set()
        seen_titles_lower: set[str] = set()
        kept: list[Paper] = []

        # Sort by source priority (SS has most complete metadata)
        source_priority = {
            "semantic_scholar": 0,
            "crossref": 1,
            "arxiv": 2,
        }
        sorted_papers = sorted(papers, key=lambda p: source_priority.get(p.source, 99))

        for paper in sorted_papers:
            # Deduplicate by DOI
            if paper.doi:
                doi_lower = paper.doi.lower().strip()
                if doi_lower in seen_dois:
                    logger.debug(f"[LitSearch] Duplicate DOI: {paper.doi}")
                    continue
                seen_dois.add(doi_lower)

            # Deduplicate by title similarity (fuzzy match)
            normalized_title = re.sub(r"\s+", "", paper.title.lower())
            if normalized_title in seen_titles_lower:
                logger.debug(f"[LitSearch] Duplicate title: {paper.title}")
                continue
            # Check against all existing titles (approximate fuzzy match)
            is_duplicate = False
            for existing_title in seen_titles_lower:
                if self._title_similarity(normalized_title, existing_title) > 0.85:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            seen_titles_lower.add(normalized_title)
            kept.append(paper)

        return kept

    @staticmethod
    def _title_similarity(a: str, b: str) -> float:
        """计算两个标题的 bigram 重叠率（中文字符友好）"""
        if not a or not b:
            return 0.0
        # Use character-level bigram Jaccard
        def bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1)) if len(s) > 1 else {s}
        a_set, b_set = bigrams(a), bigrams(b)
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / len(a_set | b_set)

    def _rank_papers(
        self,
        papers: list[Paper],
        query: str,
        domain_hint: str,
    ) -> list[Paper]:
        """
        重新排序论文列表，按综合质量评分。

        评分因子:
        - 年份 (越新越好): weight=0.2
        - 引用次数 (越多越好): weight=0.3
        - 来源完整性 (SS > Crossref > arXiv): weight=0.2
        - 有摘要者优先: weight=0.1
        - 开放获取优先: weight=0.1
        - 相关性衰减 (按原始位置): weight=0.1
        """
        max_year = max((p.year or 2000) for p in papers)
        min_year = min((p.year or 2000) for p in papers)
        year_range = max_year - min_year or 1

        max_citations = max((p.citation_count for p in papers), default=1) or 1

        scored: list[tuple[float, Paper]] = []
        for i, paper in enumerate(papers):
            s = 0.0
            # Recency
            if paper.year:
                s += 0.2 * ((paper.year - min_year) / year_range)
            # Citations
            s += 0.3 * min(paper.citation_count / max_citations, 1.0)
            # Source quality
            source_scores = {"semantic_scholar": 1.0, "crossref": 0.7, "arxiv": 0.5}
            s += 0.2 * source_scores.get(paper.source, 0.3)
            # Has abstract
            s += 0.1 * (1.0 if paper.abstract else 0.0)
            # Open access
            s += 0.1 * (1.0 if paper.open_access else 0.0)
            # Relevance decay (original order position)
            s += 0.1 * (1.0 - i / max(len(papers), 1))

            scored.append((s, paper))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored]


# ============================================================
# Citation Validator
# ============================================================


class CitationValidator:
    """
    引用校验器：交叉验证每条事实引用的真实性。

    支持的校验方式：
    - DOI 验证（Crossref REST API）
    - PMID 验证（NCBI E-utilities）
    - Title-based best-effort verification（Crossref）
    """

    def __init__(self):
        self.crossref = CrossRefSearcher()

    async def verify_doi(self, doi: str) -> bool:
        """
        通过 Crossref API 验证 DOI 是否存在。

        Crossref /works/{doi} endpoint 对存在的 DOI 返回 200，不存在返回 404。
        """
        doi_clean = doi.strip().replace("https://doi.org/", "").replace("http://dx.doi.org/", "")

        try:
            import httpx

            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"https://api.crossref.org/works/{quote_plus(doi_clean)}",
                    headers={"Accept": "application/json"},
                )
                exists = resp.status_code == 200
                if exists:
                    logger.debug(f"[CiteValidator] DOI verified: {doi_clean}")
                return exists

        except Exception as e:
            logger.debug(f"[CiteValidator] DOI verification failed for {doi_clean}: {e}")
            return False

    async def verify_pmid(self, pmid: str) -> bool:
        """
        通过 NCBI E-utilities 验证 PMID 是否存在。
        """
        pmid_clean = pmid.strip()

        if not pmid_clean.isdigit():
            return False

        try:
            import httpx

            url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=pubmed&id={pmid_clean}&retmode=json"
            )

            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                # Resultset has non-empty results means PMID exists
                results = data.get("result", {})
                exists = bool(results and pmid_clean in results)
                return exists

        except Exception as e:
            logger.debug(f"[CiteValidator] PMID verification failed for {pmid_clean}: {e}")
            return False

    async def verify_by_title(
        self, title: str, author: str = "", year: int | None = None
    ) -> bool:
        """
        通过标题+作者组合搜索 Crossref，判断是否存在匹配论文。
        不严格一一匹配，只要排名第一的结果标题相似度 > 0.8 即视为存在。
        """
        try:
            papers = await self.crossref.search(title, max_results=1)
            if not papers:
                return False

            sim = self._title_similarity(
                re.sub(r"\s+", "", title.lower()),
                re.sub(r"\s+", "", papers[0].title.lower()),
            )
            matches = sim > 0.8
            if matches:
                logger.debug(
                    f"[CiteValidator] Title verified: '{title}' "
                    f"(match='{papers[0].title}', sim={sim:.3f})"
                )
            return matches

        except Exception as e:
            logger.debug(f"[CiteValidator] Title search failed: {e}")
            return False

    async def validate_fact(self, fact_entry: dict) -> dict:
        """
        校验单条事实引用。

        Args:
            fact_entry: {"fact": "...", "doi": ..., "pmid": ..., "reference": "..."}

        Returns:
            增加 "_verified" 字段的标准结构化结果
        """
        doi = fact_entry.get("doi")
        pmid = fact_entry.get("pmid")
        reference = fact_entry.get("reference", "")

        verified = False
        verification_method = "none"

        if doi:
            verified = await self.verify_doi(doi)
            verification_method = "doi" if verified else "doi_invalid"

        if not verified and pmid:
            verified = await self.verify_pmid(pmid)
            verification_method = "pmid" if verified else "pmid_invalid"

        # If no DOI/PMID, try best-effort title search from reference string
        if not verified and reference and reference != "Unknown":
            # Try to extract potential title from reference string like
            # "Author, Year, Journal, DOI:xxx"
            title_candidates = [reference]
            # Sometimes reference contains the actual paper title before the journal name
            if ", " in reference and "." in reference:
                # Heuristic: title often appears between first comma and last period
                parts = reference.split(", ")
                if len(parts) >= 3:
                    candidate = ", ".join(parts[:-1]).strip()
                    title_candidates.append(candidate)

            for candidate_title in title_candidates[:2]:
                if len(candidate_title) < 10:
                    continue
                verified = await self.verify_by_title(candidate_title)
                if verified:
                    verification_method = "title_match"
                    break
                # Also verify that candidate looks like a real paper title
                # (has some capitalization, length > 20 chars)
                if len(candidate_title) > 20:
                    verified = await self.verify_by_title(candidate_title[:80])
                    if verified:
                        verification_method = "title_match"
                        break

        if not verified:
            verification_method = "unverified"

        result = dict(fact_entry)
        result["_verified"] = verified
        result["_verification_method"] = verification_method
        result["_verified_at"] = datetime.now(timezone.utc).isoformat()

        return result

    async def validate_all_facts(
        self, facts: list[dict], max_concurrent: int = 10
    ) -> list[dict]:
        """
        批量校验所有事实的引用。

        Args:
            facts: 事实条目列表
            max_concurrent: 最大并发校验数（避免被 API 限流）

        Returns:
            每个事实都增加了 _verified / _verification_method / _verified_at 字段
        """
        validated: list[dict] = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def validated_one(fact):
            async with semaphore:
                return await self.validate_fact(fact)

        tasks = [asyncio.create_task(validated_one(f)) for f in facts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"[CiteValidator] Fact {i} validation error: {result}")
                # Mark unverified on error
                entry = dict(facts[i])
                entry["_verified"] = False
                entry["_verification_method"] = "error"
                validated.append(entry)
            else:
                validated.append(result)

        verified_count = sum(1 for v in validated if v.get("_verified"))
        logger.info(
            f"[CiteValidator] Validated {verified_count}/{len(facts)} facts "
            f"({verified_count/max(len(facts),1)*100:.0f}% verified)"
        )
        return validated
