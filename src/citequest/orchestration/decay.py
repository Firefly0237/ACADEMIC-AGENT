"""时间衰减：**只作用在记忆层，不作用在检索排序**（M5 §8）。

先划硬边界：**检索排序里没有时效权重。** 相关性、引用量、时间是三个不可比的量，
加权成一个分数会产生两个后果——权重零出处；同一篇论文在不同 query 下拿到不同的
分，**排序不可复现也不可缓存**。所以排序模式是用户可切换的三选一
（relevance / citations / date），见 `core.enums.SortMode`。

那衰减发生在哪？两处，都在记忆层：

| 位置 | 衰减什么 | 形态 |
|---|---|---|
| 已确认结论 | 第 3 轮确认的结论，第 30 轮还能不能当前提用 | **加时间标注 + 到期重验** |
| 跨会话研究会话 | 三周前「已确认」的东西今天还成不成立 | 超阈值的条目从「已确认」降级为「待重验」 |

**为什么是「重验」而不是「打折」**：打折需要一个可解释的衰减函数和一个可比的
分数，两样都没有。而「超过 N 天的结论重新检索一次」是确定性的、可实现的、
代价可算的——它把一个需要标定的连续量换成了一个只需要定阈值的二值判定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from citequest.observability.logging import get_logger
from citequest.orchestration.memory import ConfirmedClaim

__all__ = [
    "CitationPercentileTable",
    "DecayPolicy",
    "RevalidationVerdict",
    "assert_ranking_is_decay_free",
    "needs_revalidation",
]

_logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class DecayPolicy:
    """衰减策略。**只有一个可调量，而且它没有被拍数**。"""

    revalidate_after_days: int | None = None
    """结论重验阈值。**【待实测】未标定，默认 None = 不重验。**

    标定方式（M5 §8.3）：用语料的年份分层构造样本——把 2022 年的结论对 2023–2025
    的摘要层重验，测不同阈值下的**漏检率**与**无谓重验率**，两个方向的错误分开报。
    在标定完成之前默认关闭，是因为**拍一个数比不做更糟**：它会让这层看起来
    已经工作了，而实际行为无人验证过。
    """

    high_impact_percentile: float = 0.10
    """高引例外的切点：**同发表年、同一级领域** `cited_by_count` 的 top 10%【设计】。

    十分位是引用分布上常用的「显著高引」切点；引用分布是重尾的，用均值会被极端值
    拖走，所以必须用分位数而不是均值。
    """

    min_group_size: int = 30
    """(year, field) 分组的最小样本量。不足时**必须显式回退到更粗的分组**——
    不能在 3 篇论文里算 top 10%，那个分位点没有统计意义。"""


@dataclass(slots=True)
class CitationPercentileTable:
    """(year, field) → 分位切点的查表结构。

    **建库时离线算好，查询期只是查表，不引入任何在线计算**（M5 §8.2）。
    `cited_by_count` 是论文元数据里现成的字段，分位数按 (year, field) 分组算。

    为什么不能用绝对引用阈值（比如「引用 > 1000 的不衰减」）：**引用数与发表年份
    强相关**——2010 年的论文有 15 年积累时间，2024 年的没有。用绝对阈值等于把
    衰减规则又变回了年份规则，绕一圈回到原点。分位数把年份积累效应在分母里消掉了。
    """

    #: (year, primary_category) → 该组 top-p 的 cited_by_count 切点
    thresholds: dict[tuple[int, str], int] = field(default_factory=dict)
    #: (year, primary_category) → 该组样本量，用于判定是否需要回退到更粗分组
    group_sizes: dict[tuple[int, str], int] = field(default_factory=dict)
    #: year → 该年全领域的切点，样本量不足时的回退档
    year_fallback: dict[int, int] = field(default_factory=dict)

    def threshold_for(self, year: int, category: str | None, policy: DecayPolicy) -> int | None:
        """取该 (年份, 领域) 的高引切点，样本量不足时回退到按年分组。

        回退是**显式的**而不是静默的：回退这件事会打进日志，因为「切点是在多大的
        样本上算的」直接决定了这个切点值不值得信。
        """
        key = (year, category or "")
        size = self.group_sizes.get(key, 0)
        if category is not None and size >= policy.min_group_size:
            return self.thresholds.get(key)
        fallback = self.year_fallback.get(year)
        _logger.info("citation_percentile_fallback", year=year, category=category, group_size=size, threshold=fallback)
        return fallback


@dataclass(slots=True)
class RevalidationVerdict:
    """一条结论的重验判定。

    `high_impact_exempt` 单独带出来是为了让例外规则的触发率可观测：
    切点过松会把大量老论文豁免掉，**等于把衰减整个关掉**，而这件事在指标上
    只表现为「重验次数变少」，不打点就看不出来（决策记录 U8 清单第 5 项）。
    """

    claim: ConfirmedClaim
    needs_revalidation: bool
    age_days: int
    high_impact_exempt: bool = False
    reason: str = ""


def needs_revalidation(
    claim: ConfirmedClaim,
    policy: DecayPolicy,
    *,
    now: datetime | None = None,
    cited_year_category: dict[str, tuple[int, str | None]] | None = None,
    cited_counts: dict[str, int] | None = None,
    percentiles: CitationPercentileTable | None = None,
) -> RevalidationVerdict:
    """判断一条已确认结论是否要降级为「待重验」。

    **例外的效果是「不降级」，不是「加权提升」**——加权提升会把它推回 §8.1 已经
    排除掉的那条路（时效信号进排序）。所以这里返回的是一个二值判定，
    没有任何地方产出分数。

    :param cited_year_category: paper_id → (发表年, 一级领域)。论文级元数据，
        与全库精确分面同一个载体（DuckDB 读 parquet）。
    :param percentiles: 建库时离线算好的分位表。缺表时**不做例外判定**——
        没有分位表就无法判断「高引」，而用绝对阈值凑一个是错的。
    """
    moment = now or datetime.now(UTC)
    age_days = max((moment - claim.confirmed_at).days, 0)

    if policy.revalidate_after_days is None:
        return RevalidationVerdict(
            claim=claim,
            needs_revalidation=False,
            age_days=age_days,
            reason="重验阈值未标定（【待实测】），本层默认关闭",
        )

    if age_days < policy.revalidate_after_days:
        return RevalidationVerdict(claim=claim, needs_revalidation=False, age_days=age_days, reason="未到阈值")

    if _is_high_impact(claim, cited_year_category, cited_counts, percentiles, policy):
        # 例外命中：不降级，**但仍然标注年份**——例外是「不降级」，不是「装作它是新的」。
        return RevalidationVerdict(
            claim=claim,
            needs_revalidation=False,
            age_days=age_days,
            high_impact_exempt=True,
            reason="同年同领域 cited_by_count 高引例外",
        )

    return RevalidationVerdict(
        claim=claim,
        needs_revalidation=True,
        age_days=age_days,
        reason=f"超过重验阈值 {policy.revalidate_after_days} 天",
    )


def _is_high_impact(
    claim: ConfirmedClaim,
    cited_year_category: dict[str, tuple[int, str | None]] | None,
    cited_counts: dict[str, int] | None,
    percentiles: CitationPercentileTable | None,
    policy: DecayPolicy,
) -> bool:
    """结论的任一支撑论文命中高引例外即豁免。

    「任一」而不是「全部」：一条结论只要还站在一篇公认的奠基工作上，
    把它降级为待重验的收益就很低，而重验的代价是一次完整检索。

    > **能力边界，主动讲**：这条规则用的是**引用计数**，不是引用图遍历。
    > 本系统的引用相关能力是「引用上下文检索」（别人在什么语境下引用 X），
    > 不是图遍历（A 引了谁）——因为摘要层的引用关系字段在实测中大面积为空。
    > 所以能用「被引多少次」，不能用「被谁引用、引用链多深」。
    """
    if percentiles is None or cited_counts is None or cited_year_category is None:
        return False
    for paper_id in claim.cited:
        meta = cited_year_category.get(paper_id)
        count = cited_counts.get(paper_id)
        if meta is None or count is None:
            continue
        year, category = meta
        threshold = percentiles.threshold_for(year, category, policy)
        if threshold is not None and count >= threshold:
            return True
    return False


def assert_ranking_is_decay_free(sort_mode: str, memory_touched: bool) -> None:
    """护栏：**确认三种排序模式的输出与记忆层状态无关**（M5 §8.3 第 4 条测项）。

    这不是一个业务函数，是一条防止设计腐化的断言：日后有人「顺手」把研究会话的
    已读集合接进排序，这里会立刻炸。之所以要一条运行时断言而不是只写在文档里，
    是因为这类腐化的表现是**指标悄悄变好又悄悄变差**，没有任何报错。
    """
    if memory_touched:
        raise AssertionError(f"排序模式 {sort_mode} 读取了记忆层状态——衰减泄漏到排序（M5 §8.1 硬边界）")
