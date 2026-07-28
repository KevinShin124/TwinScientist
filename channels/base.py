"""
Data Channel — Base class for all data pipelines
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseChannel(ABC):
    """所有数据通道的抽象基类"""

    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        ...

    @abstractmethod
    async def query(self, sql_or_params: str | dict[str, Any]) -> list[dict[str, Any]]:
        """读取/查询数据"""
        ...

    @abstractmethod
    async def ingest(self, record: dict[str, Any] | list[dict[str, Any]]) -> dict:
        """写入/入库数据"""
        ...

    @abstractmethod
    async def export(self, format: str = "json") -> str:
        """导出数据"""
        ...

