from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings


# Singleton pattern — env loaded once, settings created lazily on first access
_settings_instance: "Settings | None" = None


def _get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        load_dotenv()
        _settings_instance = Settings()
    return _settings_instance


class Settings(BaseSettings):
    # ============================================================
    # Layer 1: Base Model — Qwen via Alibaba Cloud Bailian
    # ============================================================
    bailian_api_key: str = ""
    bailian_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: str = "qwen-max"

    # ============================================================
    # Layer 5: Data Pipeline Channels
    # ============================================================
    sqlite_db_path: str = ""
    sensor_data_dir: str = ""
    biometric_data_dir: str = ""
    visual_fatigue_data_dir: str = ""

    # ============================================================
    # Layer 3: Memory System
    # ============================================================
    vector_db_uri: str = ""

    # ============================================================
    # Layer 2: Orchestrator
    # ============================================================
    max_iterations: int = 15

    # ============================================================
    # Layer 4: Human-in-the-Loop
    # ============================================================
    human_approval_enabled: bool = True

    # ============================================================
    # Output & Logging
    # ============================================================
    output_target: str = "console"
    output_dir: str = ""
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _init_defaults(self):
        """Initialize with .env / os defaults if not explicitly set"""
        import os as _os

        self.bailian_base_url = self.bailian_base_url or _os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model_name = self.model_name or _os.getenv("MODEL_NAME", "qwen-max")
        self.max_iterations = self.max_iterations or int(_os.getenv("MAX_ITERATIONS", "15"))
        self.human_approval_enabled = self.human_approval_enabled if isinstance(self.human_approval_enabled, bool) else _os.getenv("HUMAN_APPROVAL_ENABLED", "true").lower() == "true"
        self.output_target = self.output_target or _os.getenv("OUTPUT_TARGET", "console")
        self.log_level = self.log_level or _os.getenv("LOG_LEVEL", "INFO")
        data_parent = Path(__file__).parent.parent
        self.sqlite_db_path = self.sqlite_db_path or _os.getenv("SQLITE_DB_PATH", str(data_parent / "data" / "cc_switch.db"))
        self.sensor_data_dir = self.sensor_data_dir or _os.getenv("SENSOR_DATA_DIR", str(data_parent / "data" / "sensors"))
        self.biometric_data_dir = self.biometric_data_dir or _os.getenv("BIOMETRIC_DATA_DIR", str(data_parent / "data" / "biometric"))
        self.visual_fatigue_data_dir = self.visual_fatigue_data_dir or _os.getenv("VISUAL_FATIGUE_DATA_DIR", str(data_parent / "data" / "visual_fatigue"))
        self.output_dir = self.output_dir or str(data_parent / "output")
        return self

    @model_validator(mode="after")
    def validate_bailian_api_key(self):
        if not self.bailian_api_key.strip():
            raise ValueError("BAILIAN_API_KEY environment variable is required to run twinScientist")
        return self

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Lazy load Settings — avoids crash at import time if .env is missing."""
    global _settings_instance
    if _settings_instance is None:
        load_dotenv()
        _settings_instance = Settings()
    return _settings_instance


# Backwards-compat alias for existing imports
settings = get_settings()
