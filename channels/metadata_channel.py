"""
Layer 5 - SQLite Channel for metadata and knowledge graph storage
"""

import sqlite3
import json
from pathlib import Path
from typing import Any, Literal

ALLOWED_TABLES = frozenset({"hypotheses", "experiments", "evidence_chains", "anomalies", "literature_facts", "reviews"})
"""SQL injection guard: only pre-approved table names can be queried."""


class MetadataChannel:
    """轻量级 SQLite 元数据通道（L2 情景记忆的持久化存储）"""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    async def connect(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    async def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _create_tables(self) -> None:
        """创建必要的元数据表"""
        assert self._conn is not None
        cursor = self._conn.cursor()

        # Hypothesis tree
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                statement TEXT,
                confidence_prior REAL DEFAULT 0.0,
                confidence_posterior REAL DEFAULT 0.0,
                testability INTEGER DEFAULT 5,
                status TEXT DEFAULT 'proposed',
                parent_id TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Experiment records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                hypothesis_id TEXT,
                design_summary TEXT,
                input_data_path TEXT,
                output_data_path TEXT,
                results TEXT,
                notes TEXT,
                created_at TEXT
            )
        """)

        # Evidence chains
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evidence_chains (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                strength REAL DEFAULT 0.5,
                content TEXT,
                method_used TEXT,
                causal_direction TEXT,
                linked_hypotheses TEXT,  -- JSON array of IDs
                created_at TEXT
            )
        """)

        # Anomaly graph
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                source_experiment_id TEXT,
                description TEXT,
                severity TEXT DEFAULT 'low',
                metadata TEXT,
                created_at TEXT
            )
        """)

        # Literature facts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS literature_facts (
                id TEXT PRIMARY KEY,
                fact TEXT NOT NULL,
                reference_doi TEXT,
                reference_pmid TEXT,
                domain TEXT,
                extracted_at TEXT
            )
        """)

        # Review records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id TEXT PRIMARY KEY,
                hypothesis_id TEXT,
                novelty_score INTEGER,
                feasibility_score INTEGER,
                methodology_score INTEGER,
                evidence_score INTEGER,
                impact_score INTEGER,
                total_score INTEGER,
                comments TEXT,
                needs_revision INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)

        self._conn.commit()

    async def insert(self, table: str, record: dict[str, Any]) -> str:
        """插入一条记录"""
        if table not in ALLOWED_TABLES:
            raise ValueError(f"Table '{table}' not allowed")
        assert self._conn is not None
        columns = ", ".join(record.keys())
        placeholders = ", ".join(["?"] * len(record))
        values = list(record.values())

        self._conn.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()
        return record.get("id", "unknown")

    async def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """执行查询"""
        assert self._conn is not None
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
