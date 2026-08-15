"""服务层参数：HTTP、缓存、限流、并发闸门与超时（决策记录 §4.6 / §4.9）。

超时表有一条贯穿全表的取法：**单次超时 = 该阶段 p95 预算 × 2，向上取到易记刻度**。
卡在 p95 附近的超时会在正常波动时就大面积降级。

排队超时另有一条：**必须短于该阶段的价值窗口**——生成排 2s 之后即使拿到额度，
也必破 2.5s 的 p95，那份等待是纯浪费。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["ApiSettings", "CacheSettings", "ConcurrencySettings", "LatencyBudget", "TimeoutSettings"]


class ApiSettings(BaseModel):
    """FastAPI 服务本身。"""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, ge=1, le=65535)
    root_path: str = Field(default="")
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    query_max_length: int = Field(
        default=500,
        ge=1,
        description=(
            "超出返回 400 而不是静默截断。截断会让**用户看到的输入**与**实际执行的 query** 不一致，"
            "而且截断后的 query 会以截断形态进缓存 key，同一段长文本的不同前缀可能命中同一条缓存"
        ),
    )
    accept_client_request_id: bool = Field(
        default=True,
        description="沿用客户端 X-Request-Id 便于跨服务串联，但**必须校验字符集与长度再落日志**——这个字段会进日志和 trace，不校验就是一个日志注入面",
    )
    healthz_expose_details: bool = Field(
        default=False,
        description="匿名调用只返回存活信号，不泄露模型名、版本、endpoint 等配置",
    )
    rate_limit_search_per_minute: int = Field(default=60, ge=1)
    rate_limit_answer_per_minute: int = Field(default=10, ge=1, description="AI 路更贵，配额单列")


class CacheSettings(BaseModel):
    """一级半缓存（决策记录 D19）。

    L3 embedding 缓存**物理上不存在**——Data-in/Data-out 之后编码在服务端发生，
    客户端根本没有 embedding 可缓存。这不是砍掉的，是架构后果（O6）。
    """

    l1_enabled: bool = Field(default=True, description="全量响应缓存")
    l1_ttl_seconds: int = Field(default=900, ge=1)
    l2_enabled: bool = Field(default=True, description="热门 query 的 paper 级结果")
    l2_ttl_seconds: int = Field(default=3_600, ge=1)
    key_includes_index_version: bool = Field(
        default=True,
        description="key 必须含 index_version + **全部影响结果的参数**。不含 top_k 之类参数是正确性 bug 而不是命中率问题",
    )
    retracted_paper_exact_invalidation: bool = Field(
        default=True,
        description="撤稿论文必须按 paper_id 精确失效——这是 TTL 之外唯一的例外，因为它是合规要求，不能等 TTL 自然过期",
    )


class ConcurrencySettings(BaseModel):
    """GPU 信号量与有界队列（决策记录 §4.9）。

    各类**不共享额度**：共享额度会让秒级的生成任务饿死几十毫秒的编码任务。
    队列一律有界：无界队列在过载时把所有人的延迟无限拉长，且监控上表现为
    延迟缓升而非错误率跳变——那是最难发现的一类过载。
    """

    sem_rerank: int = Field(default=4, ge=1)
    sem_generate: int = Field(
        default=12,
        ge=1,
        description="= min(显存上限 60, 延迟 SLO 上限 12)。只算显存会得出「能开 60」，然后 20 并发时首 token 全线崩",
    )
    sem_hhem: int = Field(default=8, ge=1)
    sem_router: int = Field(default=12, ge=1)

    queue_rerank: int = Field(default=8, ge=1)
    queue_generate: int = Field(default=12, ge=1)
    queue_hhem: int = Field(default=8, ge=1)
    queue_router: int = Field(default=16, ge=1)

    queue_timeout_rerank_ms: int = Field(default=500, ge=1, description="超时 → 退回 RRF 名次直出")
    queue_timeout_generate_ms: int = Field(default=2_000, ge=1, description="超时 → 只给搜索结果")
    queue_timeout_hhem_ms: int = Field(default=200, ge=1, description="超时 → 只保确定性引用校验")
    queue_timeout_router_ms: int = Field(default=300, ge=1, description="超时 → 走默认路由")


class TimeoutSettings(BaseModel):
    """单次调用超时（决策记录 §4.9）。逐项对齐 §4.7 的 p95 预算 × 2。"""

    router_ms: int = Field(default=800, ge=1)
    milvus_ms: int = Field(default=400, ge=1)
    milvus_total_deadline_ms: int = Field(default=1_000, ge=1, description="含 1 次重试")
    hyde_ms: int = Field(default=2_000, ge=1)
    rerank_ms: int = Field(default=800, ge=1)
    first_sentence_gate_ms: int = Field(default=200, ge=1)
    first_token_ms: int = Field(default=6_000, ge=1)
    correction_loop_ms: int = Field(default=5_000, ge=1)
    sse_total_ms: int = Field(default=60_000, ge=1)
    propagate_deadline: bool = Field(
        default=True,
        description="逐层传递绝对截止时间戳。否则「每层 2s」在四层链路上累积成 8s",
    )


class LatencyBudget(BaseModel):
    """延迟**预算**，不是性能（决策记录 §4.7）。

    **全表无一行实测（U4）。** M7 实测替换前，对外只讲机制与量级，
    任何具体毫秒数都不能引用。这些值只用于告警阈值与回归基线，不用于对外宣称。
    """

    search_p50_ms: int = Field(default=100, ge=1)
    search_p95_ms: int = Field(default=250, ge=1)
    rag_first_token_p50_ms: int = Field(default=1_000, ge=1)
    rag_first_token_p95_ms: int = Field(default=2_500, ge=1)
    rag_with_correction_p95_ms: int = Field(default=4_000, ge=1)
    measured: bool = Field(
        default=False,
        description="False 表示以上全部为推算值。**为 True 之前禁止在任何对外材料里引用具体毫秒数**",
    )
