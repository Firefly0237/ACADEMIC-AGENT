"""四条必测曲线（M7 §9.1）。**这份清单是写死的，不是「建议」。**

每一条都对应一个具体设计选择的**唯一数字来源**——不做，那个选择就只能靠推测辩护。

| # | 实验 | 定的是什么 |
|---|---|---|
| ① | grouping 下的组召回曲线：用 FLAT 暴力算 ground truth，扫 C ∈ {2,4,8,16} | 定死 `ef = max(C × group_limit × group_size, 64)` 里的 **C** |
| ② | 固定 recall 目标扫 ef，得 recall–QPS 曲线 | 扩展性论证的数据来源 |
| ③ | mmap 开关 + HNSW_SQ（refine on/off）的内存–召回对照 | 降本三档的代价量化 |
| ④ | 不同过滤选择率下的 P99 | 过滤（按年份、按 granularity）是本语料的刚需 |

**为什么 ① 必须做**：grouping 场景下 ef 该怎么取，官方文档没有任何指引（该页既未
枚举 `group_by_field` 的字段类型，也未说明内部如何扩大候选来凑组）。所以那条派生式
是一条**自定的工程规则**，**C=4 是起点不是依据**。在曲线跑出来之前，只能说
「我知道 ef 必须按 entity 预算而不是组数派生，因为 grouping 下两者不等价」，
**不能给具体倍率**。

本模块与 `evaluation/` 下其余模块的分工很清楚：那些是纯函数，没有外部依赖也能被
单测；**这里的每一条都必须连一个真实的 Milvus 才有意义**，所以它单独放在
`runners/` 下，而不是混进指标模块。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from citequest.config.settings import Settings
from citequest.indexing.client import MilvusIndexClient
from citequest.observability.logging import get_logger

__all__ = ["CurvePoint", "CurveResult", "group_recall_curve", "recall_qps_curve"]

_logger = get_logger(__name__)

#: 曲线 ① 要扫的候选倍率。当前配置取 C=4（`MilvusSettings.ef_candidate_multiplier`），
#: 这条曲线的全部意义就是把这个 4 从「起点」变成「依据」。
DEFAULT_C_SWEEP: tuple[int, ...] = (2, 4, 8, 16)


@dataclass(slots=True)
class CurvePoint:
    """曲线上的一个点。`extras` 放该曲线特有的量（内存占用、过滤选择率等）。"""

    x: float
    recall: float
    qps: float | None = None
    extras: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class CurveResult:
    """一条曲线的完整产物。`ground_truth_source` 必须记。

    组召回的 ground truth 只能来自 FLAT——**唯一保证精确结果的索引**。用 HNSW
    自己的结果当 ground truth 会得到一条恒等于 1 的曲线，而它看起来完全正常。
    """

    name: str
    points: list[CurvePoint]
    ground_truth_source: str
    notes: list[str] = field(default_factory=list)


def group_recall_curve(
    client: MilvusIndexClient,
    settings: Settings,
    queries: Sequence[str],
    *,
    c_sweep: Sequence[int] = DEFAULT_C_SWEEP,
) -> CurveResult:
    """曲线 ①：grouping 下的组召回随 C 的变化。

    做法：同一批 query 各跑两次——一次用 FLAT 暴力检索并在客户端按 paper_id 分组
    得到**精确的 top-N 组**，一次用 HNSW + `group_by_field` 且 ef 按 C 派生。
    组召回 = 两者组集合的交集大小 / 精确组数。

    **注意组召回不等于 entity 召回**：一个组只要有一条 chunk 被召回就算命中，
    所以组召回天然高于 entity 召回。**这正是那条派生式必须按 entity 预算
    （`C × group_limit × group_size`）而不是按组数派生的原因**——两者在 grouping
    下不等价，而这条曲线就是用来把这句话变成数字的。

    **接真实实现**：
    1. `client.raw.search(..., search_params={"metric_type": "COSINE"}, ...)` 走一份
       临时建的 FLAT 索引取 ground truth；
    2. `client.hybrid_search(requests, GroupingSpec(group_limit=..., group_size=..., ef=C×gl×gs))`
       取实测；
    3. 逐 query 算交集，按 query 平均（与指标口径一致，macro）。
    """
    if not queries:
        raise ValueError("组召回曲线至少需要一条 query——空样本会产出一条恒为 1 的假曲线")
    gl = settings.retrieval.rag_group_limit
    gs = settings.retrieval.rag_group_size
    _logger.info("group_recall_curve_start", queries=len(queries), c_sweep=list(c_sweep), group_limit=gl, group_size=gs)
    raise NotImplementedError(
        "接真实实现：需要一份 FLAT 索引做 ground truth（唯一保证精确结果的索引）。"
        f"扫 ef = C × {gl} × {gs}，C ∈ {list(c_sweep)}，逐 query 算组交集后按 query 平均"
    )


def recall_qps_curve(
    client: MilvusIndexClient,
    settings: Settings,
    queries: Sequence[str],
    *,
    ef_sweep: Sequence[int],
) -> CurveResult:
    """曲线 ②：固定 recall 目标扫 ef，得 recall–QPS 权衡。

    它是扩展性论证的数据来源：「加 ef 能把召回提到多少、代价是多少 QPS」这句话
    在曲线跑出来之前没有数字支撑。

    **QPS 必须在与线上相同的并发下测**：单线程串行测出来的 QPS 与并发下的 QPS
    不是同一个量，而报告里通常不会写清楚是哪一个——这是延迟数字最容易被拆穿的
    地方之一（全部延迟数字当前一律定性为「预算」，见 `LatencyBudget.measured`）。
    """
    if not ef_sweep:
        raise ValueError("ef_sweep 不能为空")
    _logger.info("recall_qps_curve_start", queries=len(queries), ef_sweep=list(ef_sweep))
    raise NotImplementedError(
        "接真实实现：对每个 ef 跑同一批 query，记 recall（对 FLAT ground truth）与并发下的 QPS。"
        "串行 QPS 与并发 QPS 不可混报"
    )
