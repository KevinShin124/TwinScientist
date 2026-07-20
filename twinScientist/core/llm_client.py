"""
Layer 1 - LLM Client: Qwen via Alibaba Cloud Bailian API (OpenAI-compatible)

功能：
- 完整对话补全（非流式 / 流式）
- Function Calling（tools）
- 自动重试（指数退避，处理 rate limit 和网络抖动）
- Token 用量追踪（支持百炼 response headers）
- 结构化输出解析（从 JSON blob 中提取结果）
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)


class QwenClient:
    """阿里云百炼平台 Qwen LLM 客户端"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "qwen-max",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Token 统计
        self._input_tokens: int = 0
        self._output_tokens: int = 0

        # 持久化 HTTP 客户端 — 跨调用复用 TCP/TLS 连接，避免每次重建握手开销
        self._client = httpx.AsyncClient(timeout=180.0)

    @property
    def total_input_tokens(self) -> int:
        return self._input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._output_tokens

    def reset_token_stats(self):
        self._input_tokens = 0
        self._output_tokens = 0

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # 百炼特有请求头（部分接口需要）
        headers["X-DashScope-SSE"] = "enable"
        return headers

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.8,
    ) -> dict[str, Any]:
        """
        非流式调用 — 返回完整响应 JSON
        带指数退避重试机制
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._build_headers(),
                    json=payload,
                )

                # 记录 Token
                usage = resp.json().get("usage", {})
                self._input_tokens += usage.get("input_tokens", 0)
                self._output_tokens += usage.get("output_tokens", 0)

                if resp.status_code == 429:
                    # Rate limit — 获取 Retry-After 或使用默认延迟
                    retry_after = float(resp.headers.get("Retry-After", self.retry_delay * (2 ** attempt)))
                    logger.warning(f"[QwenClient] Rate limited on attempt {attempt}. Waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue

                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Server error {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )

                resp.raise_for_status()
                result = resp.json()

                logger.debug(
                    f"[QwenClient] Completion OK — model={self.model}, "
                    f"in_tokens={usage.get('input_tokens', '?')}, "
                    f"out_tokens={usage.get('output_tokens', '?')}"
                )
                return result

            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                last_error = e
                wait_time = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(f"[QwenClient] Connection error on attempt {attempt}/{self.max_retries}: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(wait_time)
            except httpx.HTTPStatusError as e:
                last_error = e
                # 4xx 非 429 通常是请求错误，不重试
                if 400 <= e.response.status_code < 500:
                    break
                logger.warning(f"[QwenClient] HTTP {e.response.status_code} on attempt {attempt}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))

        raise RuntimeError(
            f"[QwenClient] Failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        ) from last_error

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """异步流式生成 token，逐 token yield"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        for attempt in range(1, self.max_retries + 1):
            try:
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._build_headers(),
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
                return  # 成功完成

            except Exception as e:
                logger.warning(f"[QwenClient] Stream error on attempt {attempt}: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))

        raise RuntimeError(f"[QwenClient] Stream failed after {self.max_retries} attempts")

    async def call_tool(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
    ) -> dict[str, Any]:
        """
        函数调用（function calling）
        返回 {tool_calls: [...], content: "..."}
        """
        result = await self.chat_complete(messages=messages, tools=tools, temperature=0.0)

        choices = result.get("choices", [])
        if not choices:
            return {"tool_calls": [], "content": "", "error": "No choices returned"}

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])
        content = message.get("content", "")

        return {
            "tool_calls": tool_calls,
            "content": content,
        }

    def extract_json_from_text(self, text: str) -> dict | None:
        """
        Synchronous JSON extraction from LLM output text.
        Handles Markdown code block wrapping (```json ... ``` or ``` ... ```)
        Extracts first valid JSON object if no code block found.
        """
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 提取 ```json ... ``` 或 ``` ... ``` 块
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 提取第一个合法的 JSON 对象
        matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        for m in matches:
            try:
                return json.loads(m)
            except json.JSONDecodeError:
                continue

        return None


async def create_client(
    base_url: str,
    api_key: str,
    model: str = "qwen-max",
) -> QwenClient:
    """工厂函数：创建 QwenClient 实例（每次新建，用于测试/隔离场景）"""
    return QwenClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
    )


# ============================================================
# 全局单例客户端 — 所有节点共享同一个 QwenClient，复用 TCP/TLS 连接
# ============================================================

_global_client: QwenClient | None = None


async def init_global_client(base_url: str, api_key: str, model: str = "qwen-max") -> QwenClient:
    """首次调用时创建全局单例客户端，后续调用直接返回同一实例"""
    global _global_client
    if _global_client is None:
        _global_client = QwenClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
    return _global_client


def get_global_client() -> QwenClient | None:
    """获取全局单例客户端（可能尚未初始化，返回 None）"""
    return _global_client


async def close_global_client() -> None:
    """关闭全局客户端并释放 HTTP 连接池资源。应在程序退出前调用"""
    global _global_client
    if _global_client is not None:
        await _global_client._client.aclose()
        _global_client = None
