"""OpenAI 兼容 chat 客户端（M4 §4、决策记录 §4.5）。**只做协议适配，不含业务逻辑。**

本地主路径是 7B-AWQ 单卡（vLLM/SGLang 的 OpenAI 兼容端点），所以这一层不引任何
厂商 SDK，直接用 httpx 打 `/chat/completions`。

三个必须留在这一层的细节：

1. **deadline 传递**（决策记录 §4.9）：入口算绝对截止时间戳，每层发起下游调用前
   先看剩余预算。否则「每层 2s」会在四层链路上累积成 8s。
2. **模型适配点**：部分推理型模型不接受 `temperature != 1`，需要按 model id 强制
   改写（PaperQA2 的 `_update_temperature` 就是干这个的）。本项目主路径是本地 7B
   不触发这条，但**适配点必须留出来**——否则将来接一个云端 fallback 模型会在
   运行时报参数错误。
3. **`top_p` 显式写 1.0**：`temperature=0` 时采样已退化为贪心，`top_p` 不再起作用。
   显式设 1.0 是为了避免两个旋钮同时拧——同时写 `temperature=0.1, top_p=0.9`
   这种组合，出问题时无法归因是哪个在起作用（M4 §4）。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

import httpx

from citequest.config.rag import GenerationSettings
from citequest.core.errors import UpstreamTimeoutError
from citequest.observability.logging import get_logger

__all__ = ["ChatClient", "ChatMessage", "Deadline"]

_logger = get_logger(__name__)

#: 不接受 `temperature != 1` 的模型前缀。命中即把温度强制改写为 1.0 并打日志——
#: **静默改写比报错更危险，所以必须留痕**：温度在本系统里是可调试性参数
#: （首句闸门要求可复现，M4 §4 第 3 条论证），被悄悄改掉会让闸门通过率不可复现。
_FIXED_TEMPERATURE_MODEL_PREFIXES: tuple[str, ...] = ("o1", "o3", "gpt-5")


@dataclass(slots=True, frozen=True)
class ChatMessage:
    """一条对话消息。三区分离（System/Evidence/User）由调用方组装成消息列表。"""

    role: str
    content: str

    def as_payload(self) -> dict[str, str]:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(f"非法 role：{self.role}")
        return {"role": self.role, "content": self.content}


class Deadline:
    """绝对截止时间戳。

    用绝对时间而不是「剩余毫秒」，是因为剩余毫秒必须在每一层重新计算，
    而漏算一层的表现是**超时预算被悄悄放大一倍**，不报错。
    """

    __slots__ = ("_at",)

    def __init__(self, at: float) -> None:
        self._at = at

    @classmethod
    def after_ms(cls, budget_ms: int) -> Deadline:
        return cls(time.monotonic() + budget_ms / 1000.0)

    def remaining_s(self) -> float:
        return max(0.0, self._at - time.monotonic())

    def expired(self) -> bool:
        return self.remaining_s() <= 0.0


class ChatClient:
    """生成与路由共用的 chat 客户端。

    生成模型与路由小模型共用一个类、不同 `model` 参数：它们的协议完全相同，
    为「小模型」单开一个类只会多一层没有行为差异的间接。
    """

    def __init__(self, settings: GenerationSettings, *, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {settings.api_key}"},
        )

    def _payload(
        self,
        messages: Iterable[ChatMessage],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict[str, object]:
        effective_temperature = temperature
        if model.lower().startswith(_FIXED_TEMPERATURE_MODEL_PREFIXES) and temperature != 1.0:
            _logger.warning("temperature_forced", model=model, requested=temperature, applied=1.0)
            effective_temperature = 1.0
        return {
            "model": model,
            "messages": [m.as_payload() for m in messages],
            "temperature": effective_temperature,
            # 见模块 docstring 第 3 条：显式写 1.0，不让两个采样旋钮同时生效。
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    async def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        deadline: Deadline | None = None,
    ) -> str:
        """一次非流式生成，返回完整文本。

        HyDE、decompose、路由分类都走这条——它们的产物要整体解析，
        流式对它们没有价值，只会多一层拼接。
        """
        payload = self._payload(
            messages,
            model=model or self._settings.model_id,
            temperature=self._settings.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self._settings.max_tokens,
            stream=False,
        )
        timeout = deadline.remaining_s() if deadline else None
        if deadline is not None and deadline.expired():
            raise UpstreamTimeoutError("发起 chat 调用前预算已耗尽")
        try:
            response = await self._client.post("/chat/completions", json=payload, timeout=timeout)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(f"chat 调用超时：{exc}") from exc
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise UpstreamTimeoutError("chat 响应没有 choices —— 端点协议不兼容或模型未就绪")
        content: str = choices[0].get("message", {}).get("content", "")
        return content

    async def stream(
        self,
        messages: Iterable[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        deadline: Deadline | None = None,
    ) -> AsyncIterator[str]:
        """流式生成，逐 token 产出增量文本。

        **这里产出的是 token 增量，不是句子**。按句边界推送是上层
        `rag.streaming` 的职责——放行粒度与校验粒度必须一致，而校验是句子级的
        （MiniCheck API 即句子级；Anthropic Citations API 不引用小于一句的片段）。
        把切句放在客户端层会让「换一个推理端点」顺带换掉切句实现。
        """
        payload = self._payload(
            messages,
            model=model or self._settings.model_id,
            temperature=self._settings.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self._settings.max_tokens,
            stream=True,
        )
        timeout = deadline.remaining_s() if deadline else None
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload, timeout=timeout) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    delta = _parse_sse_delta(line)
                    if delta:
                        yield delta
        except httpx.TimeoutException as exc:
            # 超时语义全系统只有一种（决策记录 §4.9）：**已产生内容一律保留**。
            # 这里只把异常抛上去，已 yield 的增量不回收——上层据此做「只降级不撤回」。
            raise UpstreamTimeoutError(f"chat 流式调用超时：{exc}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


def _parse_sse_delta(line: str) -> str:
    """解析一行 OpenAI 兼容流式响应，返回增量文本（无增量则返回空串）。

    单独成函数是为了让它可以在没有推理服务的情况下被单测断言——流式解析的
    坑（`data: [DONE]`、心跳空行、一行多字段）全在这里，而它们只有在真实端点
    抖动时才暴露，靠集成测试碰运气不现实。
    """
    if not line or not line.startswith("data:"):
        return ""
    body = line[len("data:") :].strip()
    if not body or body == "[DONE]":
        return ""
    try:
        chunk = json.loads(body)
    except json.JSONDecodeError:
        _logger.warning("chat_stream_bad_json", raw_len=len(body))
        return ""
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta: str = choices[0].get("delta", {}).get("content") or ""
    return delta
