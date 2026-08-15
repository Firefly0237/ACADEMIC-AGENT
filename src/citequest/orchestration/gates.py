"""GPU 资源闸门：每类任务各持一个信号量 + 有界队列 + 排队超时（M6 §5.2）。

**每类任务的额度不共享。** 共享额度会让秒级的生成任务饿死几十毫秒的编码任务——
一次生成占用秒级、一次 HHEM 前向只占几十毫秒，混在一个池子里排队，搜索路的
首字节延迟会被 RAG 路污染。

**队列必须有界。** 无界队列在过载时不拒绝任何人，它只是把所有人的延迟无限拉长：
用户体验从「一部分人被降级」变成「所有人都超时」，而监控上表现为延迟缓慢爬升
而不是错误率跳变，更难发现。

**排队超时必须短于该阶段的价值窗口。** 生成排队超时 2s，而 RAG 首 token 的 p95
预算是 2.5s——排队已经花掉 2s，即使拿到额度也必破 SLO，那份等待是纯浪费，
不如早点降级成只给搜索结果。这条约束由 `Settings._cross_group_invariants` 在
启动时校验，不靠人记。

放在 orchestration 而不是 api，是因为闸门属于 service 层：MCP adapter 那条路
同样要受同一套 GPU 额度约束，闸门定义在 HTTP 层就意味着 MCP 绕过了它。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum

from citequest.config.api import ConcurrencySettings
from citequest.observability.logging import get_logger

__all__ = ["GateBusy", "GateName", "GpuGates"]

_logger = get_logger(__name__)


class GateName(StrEnum):
    """四类 GPU 任务。取值同时是 Prometheus 的标签，所以固定成枚举。"""

    RERANK = "rerank"
    GENERATE = "generate"
    HHEM = "hhem"
    ROUTER = "router"


class GateBusy(Exception):
    """排队超时或队列已满。

    **它不是错误而是降级信号**：调用方接住它之后走各自的降级分支
    （rerank → RRF 名次直出；生成 → 只给搜索结果；HHEM → 只保确定性校验；
    路由 → 默认路由）。所以它不继承 `CiteQuestError`——继承了就会被 api 层的
    异常处理器翻译成一个 HTTP 错误，而这里要的恰恰是**不报错**。
    """

    def __init__(self, gate: GateName, waited_ms: int) -> None:
        super().__init__(f"{gate.value} 闸门繁忙，已等待 {waited_ms}ms")
        self.gate = gate
        self.waited_ms = waited_ms


@dataclass(slots=True)
class _Gate:
    """一个闸门的三件套：额度、队列上限、排队超时。"""

    name: GateName
    semaphore: asyncio.Semaphore
    limit: int
    queue_limit: int
    timeout_ms: int
    waiting: int = 0


class GpuGates:
    """进程内的 GPU 闸门集合。

    **刻意是进程内状态而不是分布式配额**（M6 §4.2）：限流器可以 fail-open，
    因为最后一道闸门是本地的——即使 Redis 整个瞎掉，GPU 也不会被打穿。
    一旦把这层也做成分布式，这个「最后一道本地闸门」的性质就没了。

    连带约束写在 `cli.serve` 的 docstring 里：**多 worker 会让额度乘以 worker 数**，
    所以默认单 worker，扩容靠多容器。
    """

    def __init__(self, settings: ConcurrencySettings) -> None:
        self._gates: dict[GateName, _Gate] = {
            GateName.RERANK: _Gate(
                GateName.RERANK,
                asyncio.Semaphore(settings.sem_rerank),
                settings.sem_rerank,
                settings.queue_rerank,
                settings.queue_timeout_rerank_ms,
            ),
            GateName.GENERATE: _Gate(
                GateName.GENERATE,
                asyncio.Semaphore(settings.sem_generate),
                settings.sem_generate,
                settings.queue_generate,
                settings.queue_timeout_generate_ms,
            ),
            GateName.HHEM: _Gate(
                GateName.HHEM,
                asyncio.Semaphore(settings.sem_hhem),
                settings.sem_hhem,
                settings.queue_hhem,
                settings.queue_timeout_hhem_ms,
            ),
            GateName.ROUTER: _Gate(
                GateName.ROUTER,
                asyncio.Semaphore(settings.sem_router),
                settings.sem_router,
                settings.queue_router,
                settings.queue_timeout_router_ms,
            ),
        }

    def snapshot(self) -> dict[str, dict[str, int]]:
        """当前占用情况，供 `/healthz` 详情与 Prometheus 采集。

        `waiting` 是三个「静默失败」指标之外最值得盯的一个：它涨起来的时候
        错误率还是 0，只有延迟在爬。
        """
        return {
            gate.name.value: {"limit": gate.limit, "waiting": gate.waiting, "queue_limit": gate.queue_limit}
            for gate in self._gates.values()
        }

    @asynccontextmanager
    async def acquire(self, name: GateName, *, timeout_ms: int | None = None) -> AsyncIterator[None]:
        """获取一个额度，超时或队列满时抛 `GateBusy`。

        两段拒绝的顺序是有讲究的：**先判队列长度、再等超时**。队列已经满了还去
        排队等 2 秒，只是把过载的代价从「立刻降级」变成「慢一点再降级」，
        而用户已经多等了 2 秒。

        `finally` 里必须释放：客户端断连时 `asyncio.CancelledError` 会从 `yield`
        处抛出（M6 §2.4 第 3 条），不释放就会把额度永久泄漏给一个已经没人要的请求。
        """
        gate = self._gates[name]
        budget_ms = timeout_ms if timeout_ms is not None else gate.timeout_ms

        if gate.waiting >= gate.queue_limit:
            _logger.warning("gate_queue_full", gate=name.value, waiting=gate.waiting, queue_limit=gate.queue_limit)
            raise GateBusy(name, waited_ms=0)

        gate.waiting += 1
        started = asyncio.get_running_loop().time()
        try:
            await asyncio.wait_for(gate.semaphore.acquire(), timeout=budget_ms / 1000)
        except TimeoutError as exc:
            waited_ms = int((asyncio.get_running_loop().time() - started) * 1000)
            _logger.warning("gate_queue_timeout", gate=name.value, waited_ms=waited_ms, budget_ms=budget_ms)
            raise GateBusy(name, waited_ms=waited_ms) from exc
        finally:
            gate.waiting -= 1

        try:
            yield
        finally:
            gate.semaphore.release()
