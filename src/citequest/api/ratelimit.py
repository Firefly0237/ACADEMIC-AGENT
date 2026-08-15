"""限流：令牌桶（M6 §4）。

**两类限流保护的是两种东西**：

| 规则 | 阈值 | 保护对象 | 超限行为 |
|---|---|---|---|
| 全局请求限流 | `api.rate_limit_search_per_minute`，桶容量 20 | 应用层与 Milvus | 429 + `Retry-After` |
| **AI 回答限流** | `api.rate_limit_answer_per_minute`，桶容量 5 | **GPU** | **不返回 429，降级为只给搜索结果** |

AI 回答单独限流，是因为它和检索差着一个数量级的成本：一次检索是 Milvus 的一次
查询；一次 AI 回答是 HyDE 生成 + 48 pair 的 cross-encoder 重排 + 带 3k token
上下文的生成 + 一次 HHEM 前向，全部落在 GPU 上。用同一个阈值管这两件事，
等于用最便宜那件事的成本给最贵那件事定价。

**为什么是令牌桶而不是滑动窗口**：用户的真实行为是「搜一次 → 快速翻 2 页 →
切一次排序」，几秒内 4 个请求完全正常；滑动窗口会把这个正常行为误判为攻击。
令牌桶的桶容量正是为这种行为准备的。代价是 `X-RateLimit-Remaining` 的语义是
「还能立刻发几次」而不是「本分钟还剩几次」——**这个区别必须写进 API 文档**，
否则调用方会按次数理解它。

**故障策略：fail-open。** 限流是保护措施不是业务逻辑，让保护措施的故障直接变成
服务不可用，是把可用性押在一个旁路组件上。前提是**最后一道闸门必须是本地的**
——AI 回答的并发信号量是进程内的（`orchestration.gates`），所以即使限流器整个
瞎掉，GPU 也不会被打穿。这就是为什么是「限速 + 限并发」两道而不是一道。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from fastapi import Request

from citequest.config.api import ApiSettings
from citequest.core.errors import RateLimitedError
from citequest.observability.logging import get_logger

__all__ = [
    "TOKEN_BUCKET_LUA",
    "RateLimitDecision",
    "TokenBucketLimiter",
    "client_key",
    "limiter_from_settings",
    "quota_headers",
]

_logger = get_logger(__name__)

#: 分布式版本的 Lua 脚本（M6 §4.2）。**接真实实现**：多副本部署时改用
#: `EVALSHA` 调用它，并处理 `NOSCRIPT`（Redis 重启或 `SCRIPT FLUSH` 后脚本缓存会空）。
#:
#: 四个实现细节各自对应一个真实会踩的坑：
#: 1. 时钟取自 `redis.call('TIME')` 而不是应用传进来的 now——多副本时钟漂移会让
#:    「时间倒流」的副本把桶算多。Redis 5 之后复制传播写效果而不是脚本本身，这才安全；
#: 2. **冷启动时桶是满的**，空桶开局会让每个新用户的第一次请求都被拒；
#: 3. `EXPIRE` 让空闲桶自动回收——限流 key 的基数等于活跃 IP 数，不设 TTL 就是
#:    一个只增不减的内存泄漏；
#: 4. `retry_after` 由脚本算好返回，而不是让客户端猜。
#:
#: 本项目当前是单进程部署（`cli.serve` 默认单 worker），所以主实现是下面的进程内
#: 令牌桶——**不为一个还没有的部署形态先引一个 Redis 依赖**。
TOKEN_BUCKET_LUA = """
local rate, cap, need, ttl = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3]), tonumber(ARGV[4])
local t   = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local st     = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(st[1])
local ts     = tonumber(st[2])
if tokens == nil then tokens, ts = cap, now end
tokens = math.min(cap, tokens + math.max(0, now - ts) * rate)
local allowed, retry_after = 0, 0
if tokens >= need then
  tokens  = tokens - need
  allowed = 1
else
  retry_after = (need - tokens) / rate
end
redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', KEYS[1], ttl)
return {allowed, tostring(tokens), tostring(retry_after)}
"""


@dataclass(slots=True)
class RateLimitDecision:
    """一次限流判定。

    `remaining` 是一个**连续量**（桶里还剩多少令牌），不是「本分钟还剩几次」。
    这两者在数值上接近、语义上不同，配额头的文档必须写清楚。
    """

    allowed: bool
    remaining: float
    retry_after_ms: int
    limit_per_minute: int
    reset_after_ms: int


@dataclass(slots=True)
class _Bucket:
    tokens: float
    ts: float


class TokenBucketLimiter:
    """进程内令牌桶。

    每 key 内存 O(1)（两个字段），单次命令全 O(1)，且**允许桶容量内的突发**——
    这三条正是选它而不是 ZSET 滑动窗口的理由（后者每 key 是 O(limit)，
    且严格不允许突发）。
    """

    def __init__(self, *, rate_per_minute: int, capacity: int, name: str) -> None:
        self._rate = rate_per_minute / 60.0
        self._capacity = float(capacity)
        self._name = name
        self._buckets: dict[str, _Bucket] = {}

    def check(self, key: str, *, need: float = 1.0) -> RateLimitDecision:
        """判定并扣减。**冷启动时桶是满的**，不是空的。"""
        now = time.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=self._capacity, ts=now)
            self._buckets[key] = bucket

        bucket.tokens = min(self._capacity, bucket.tokens + max(0.0, now - bucket.ts) * self._rate)
        bucket.ts = now

        if bucket.tokens >= need:
            bucket.tokens -= need
            return RateLimitDecision(
                allowed=True,
                remaining=bucket.tokens,
                retry_after_ms=0,
                limit_per_minute=int(self._rate * 60),
                reset_after_ms=int((self._capacity - bucket.tokens) / self._rate * 1000),
            )

        retry_after_ms = int((need - bucket.tokens) / self._rate * 1000)
        _logger.info("rate_limited", limiter=self._name, key=key, retry_after_ms=retry_after_ms)
        return RateLimitDecision(
            allowed=False,
            remaining=bucket.tokens,
            retry_after_ms=retry_after_ms,
            limit_per_minute=int(self._rate * 60),
            reset_after_ms=retry_after_ms,
        )

    def enforce(self, key: str) -> RateLimitDecision:
        """超限直接抛 `RateLimitedError`（→ 429 + 配额头）。

        **只有搜索路用它。** AI 路走 `check` 然后静默降级——429 会让用户什么都
        拿不到，降级让用户仍然拿到搜索结果，只是没有 AI 回答。同一个系统里
        一类超限返回 429、另一类静默降级，这个不整齐必须在文档里解释清楚。
        """
        decision = self.check(key)
        if not decision.allowed:
            raise RateLimitedError(
                "Request quota exceeded, please retry after backoff.",
                retry_after_ms=decision.retry_after_ms,
            )
        return decision


def client_key(request: Request) -> str:
    """限流维度：IP。

    系统无登录、公开可搜，所以没有更好的标识可用。**已知的误伤要主动说**：
    IP 维度会误伤 NAT / 校园网共享出口。缓解是把限额设在正常单用户远达不到的
    水平，并且让 AI 限流只降级不拒绝——共享出口的用户最差是拿不到 AI 回答，
    不会拿不到搜索结果。
    """
    if request.client is not None and request.client.host:
        return request.client.host
    # 拿不到对端地址时退回一个共享 key：**宁可把这批请求算在一起限流，
    # 也不要放行**——放行意味着任何隐藏了对端地址的调用方都能绕过限流。
    return "unknown"


def quota_headers(decision: RateLimitDecision) -> dict[str, str]:
    """配额头。**429 必须同时给出它们**（M6 §1.4 第 2 条）。"""
    return {
        "X-RateLimit-Limit": str(decision.limit_per_minute),
        "X-RateLimit-Remaining": str(int(decision.remaining)),
        "X-RateLimit-Reset": str(max(1, decision.reset_after_ms // 1000)),
    }


def limiter_from_settings(settings: ApiSettings, *, ai_path: bool) -> TokenBucketLimiter:
    """按用途建限流器。桶容量取值来自 M6 §4.1：全局 20、AI 路 5。

    桶容量为什么不是配置项：它与速率是一对（「容量 20、速率 1/秒」表达的是
    「允许瞬间 20 次的正常操作簇，长期速率仍限死」），拆成两个独立旋钮会让人
    单独改一个，而单改任何一个都会破坏这句话的语义。
    """
    if ai_path:
        return TokenBucketLimiter(rate_per_minute=settings.rate_limit_answer_per_minute, capacity=5, name="answer")
    return TokenBucketLimiter(rate_per_minute=settings.rate_limit_search_per_minute, capacity=20, name="search")
