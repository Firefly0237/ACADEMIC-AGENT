"""LangGraph 检索纠正循环与会话记忆参数（决策记录 §4.5 / §4.10）。

**定性先说清楚**：本项目主体是 workflow（routing + parallelization），不是 agent。
只有一条窄的 LangGraph 检索纠正循环。不能宣称「做了 agent 系统」。

§4.10 那一组（记忆层）的口径是全表最严的一档：**本层无一项来自已定参数，
全部为新设计值，只有自己的推导，没有外部出处可引用。** 收进配置是为了闭合
「查不到的数值不许写」这条规矩，不是为它们背书。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["AgentSettings", "MemorySettings"]


class AgentSettings(BaseModel):
    """LangGraph 纠正循环。"""

    correction_loop_enabled: bool = Field(default=True)
    recursion_limit: int = Field(
        default=8,
        ge=1,
        description=(
            "**显式设置**。官方默认是 1000（1.0.6 起），离安全边界差两个数量级，"
            "所以显式设更必要——无论官方默认到底是多少，「显式设 8」这个动作都成立（V4）"
        ),
    )
    max_correction_rounds: int = Field(default=1, ge=0, le=1, description="循环上限 1 次。超时退回首轮结果**不报错**")
    correction_loop_timeout_ms: int = Field(default=5_000, ge=1)
    checkpointer_backend: str = Field(
        default="memory",
        description=(
            "**尚未选型（U8）**。官方明确 InMemorySaver 仅供开发测试、进程重启即丢。"
            "后端选定前**不得声称已具备持久化能力**"
        ),
    )
    # 止损判据事先写死，与「HyDE 跑不赢 direct 就下线」同一套纪律（K10）
    stop_loss_min_improvement: float = Field(
        default=0.05,
        gt=0.0,
        description="纠正循环相对静态 pipeline 提升 < 5% → 只保留工具化与 MCP，砍掉循环",
    )
    parallel_tool_calls_max: int = Field(default=3, ge=1, description="同轮并行工具调用上限，与 decompose 上限同量级")
    observation_max_tokens: int = Field(
        default=800,
        ge=1,
        description="观察层裁剪。21,600 → 3,600 → 800 两次收敛；不裁剪则三轮 64,800 token",
    )
    trace_jsonl_enabled: bool = Field(default=True, description="轨迹 JSONL 落盘，供离线复盘与轨迹级指标计算")
    no_progress_rounds_to_stop: int = Field(
        default=2,
        ge=1,
        description="连续 N 轮新增有效 chunk 数为 0 即止损。与调用指纹去重一起构成仅有的两层确定性判据（H12）",
    )


class MemorySettings(BaseModel):
    """研究会话记忆（决策记录 §4.10）。

    **范围收窄**：只做研究会话（主键 research_id，跟课题走），**不做用户画像记忆**
    （跟人走）。这是对 K14「跨会话 memory 风险为负」的收窄而不是推翻——K14 否的
    是用户画像那一半。研究会话的收益可确定性测量：重复推荐率是可数的。
    """

    enabled: bool = Field(default=False, description="v1 不默认开启——它是 §4.10 那一组「无外部出处」参数的载体")
    compress_after_turns: int = Field(
        default=6,
        ge=1,
        description="单轮增量约 900 token → 6 轮 ≈ 5,400，覆盖一个完整探索段落。降到 3 压缩次数翻倍且输入信息量不足",
    )
    compress_after_tokens: int = Field(
        default=8_000, ge=1, description="≈8–9 轮，比轮数闸门略松一档，专治「单轮特别长」"
    )
    summary_max_tokens: int = Field(
        default=400,
        ge=1,
        description="仅压缩 narrative 字段。结构化字段做集合运算不做自然语言压缩，因此不漂移",
    )
    keep_raw_turns: int = Field(
        default=2,
        ge=0,
        description="指代先行词绝大多数落在最近 1–2 轮；N=3 起收益衰减而常驻成本线性涨",
    )
    incremental_compression: bool = Field(
        default=True,
        description="增量叠加 O(n) vs 全量重算 O(n²)：n=10 时 58,000 vs 297,000 token",
    )
