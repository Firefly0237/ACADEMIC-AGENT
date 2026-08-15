"""granularity 路由：本模块唯一「不做就会静默产生幻觉」的分支（M3 §3）。

**为什么它单独成一个文件**：路由层其余所有分支不做只是效果差一点；这一条不做
是产出错误答案。用户问「XX 方法用的什么损失函数」，不过滤时检索会照常返回一批
2024 年论文的**摘要**——它们与 query 语义高度相关，rerank 也会给高分，因为
**它们确实高度相关**。然后模型顺着摘要里那句「我们提出了一种新的对比学习目标」
把损失函数形式、温度系数、数据集名字全补齐了。

这个失败模式有三个性质（M3 §3.2）：
1. **它不报错**——检索有结果、rerank 分数正常、[N] 引用合法可点、引用的确实是
   那篇论文，**所有下游校验全部通过**。幻觉防线里没有任何一层拦得住它。
2. **它是系统性的**——只要问方法细节且相关论文在 2023 年之后就必然发生。
3. **它伪装成能力**——回答质量越高，伤害越大。

判错的代价不对称（M3 §3.6）：假阴性（该过滤没过滤）= 静默幻觉；
假阳性（不该过滤却过滤了）= 召回面缩小、时间截到 2022，**但不会编**。
所以词表宁可宽不做窄，且深度信号放在 Level 1（确定性）而不是交给 Level 2 的模型。
"""

from __future__ import annotations

from dataclasses import dataclass

from citequest.core.enums import DegradeReason, Granularity, GranularityRoute
from citequest.core.query_plan import QueryFilters, QuerySignals

__all__ = [
    "ABSTRACT_ONLY_YEAR_MIN",
    "COVERAGE_NOTE_EN",
    "COVERAGE_NOTE_ZH",
    "FULLTEXT_YEAR_MAX",
    "FULLTEXT_YEAR_MIN",
    "GranularityDecision",
    "decide_granularity",
    "fallback_on_insufficient_groups",
]

#: 全文层的年份上界（决策记录 §2.1）：unarXive open subset 取 cs.* / 2019–2022。
#: **2023 年以后的论文一篇全文都没有**——两层语料在时间上完全互补，
#: 所以任何时间条件都直接决定了打到哪一层。
FULLTEXT_YEAR_MAX = 2022
FULLTEXT_YEAR_MIN = 2019

#: 「只有摘要」的区间下界。时效信号或 `year_from >= 2023` 命中即落进这里。
ABSTRACT_ONLY_YEAR_MIN = 2023


@dataclass(slots=True)
class GranularityDecision:
    """三种处置之一（M3 §3.4）及其对外表现。"""

    route: GranularityRoute
    granularity: Granularity
    coverage_note: str | None = None
    """写进回答的能力边界声明。**只在第三种处置下非空**——
    它是唯一一个无法靠过滤解决的组合，所以唯一诚实的做法是把边界讲出来。"""

    depth_hit: bool = False
    recency_hit: bool = False


#: 第三种处置的边界声明。**这段话是回答的一部分，不是免责声明**（M3 §3.4）——
#: 静默过滤到 fulltext 等于悄悄把「最新」这个约束丢掉（返回一批 2022 年的论文却
#: 不说明），而不过滤就回到 §3.2 的幻觉路径。
COVERAGE_NOTE_ZH = "2023 年以后的论文在本库中只有摘要粒度，方法与实验细节无法由这些证据支撑。"
COVERAGE_NOTE_EN = (
    "Papers after 2022 are indexed at abstract granularity only in this corpus; "
    "method-level and experiment-level details cannot be supported by that evidence."
)


def _recency_from_filters(filters: QueryFilters) -> bool:
    """年份条件本身也是时效信号（M3 §3.3 第 ② 类）。

    `year_from >= 2023` 与「latest」在语义上是同一件事：都把候选面推进了
    只有摘要的区间。分开判会让「2024 年之后关于损失函数的论文」漏掉第三种处置。
    """
    return filters.year_from is not None and filters.year_from >= ABSTRACT_ONLY_YEAR_MIN


def decide_granularity(
    signals: QuerySignals,
    filters: QueryFilters,
    *,
    needs_fulltext_hint: bool | None = None,
    response_language: str = "EN",
) -> GranularityDecision:
    """三种处置的判定（M3 §3.4）。

    :param needs_fulltext_hint: Level 2 小 LLM 输出的意图信号（第 ③ 类）。
        它**只在规则未命中时补位**——它复用的是一次本来就要发生的 LLM 调用，
        多输出一个字段的边际成本是几个 token（这与 §2.4 被否决的 embedding
        锚点层性质完全不同：后者复用的是一个架构上不存在的东西）。
    :param response_language: 决定边界声明用哪种语言。回答语言与 query 语言一致，
        边界声明是回答的一部分，跟着走。
    """
    depth_hit = bool(signals.method_words) or (needs_fulltext_hint is True and not signals.method_words)
    recency_hit = bool(signals.recency_words) or _recency_from_filters(filters)

    if depth_hit and recency_hit:
        # ① 与 ② 同时命中：**唯一无法靠过滤解决的组合**。用户既要最新又要细节，
        # 而这两件事在本库里不同时存在。两层并召 + 在回答里显式声明覆盖边界。
        note = COVERAGE_NOTE_ZH if response_language == "ZH" else COVERAGE_NOTE_EN
        return GranularityDecision(
            route=GranularityRoute.BOTH_WITH_COVERAGE_NOTE,
            granularity=Granularity.ANY,
            coverage_note=note,
            depth_hit=True,
            recency_hit=True,
        )

    if depth_hit:
        # ① 命中、② 不命中：granularity == "fulltext" 硬过滤。
        # 候选面从 27 万篇缩到 5 万篇且时间截到 2022——这个代价是明知的，
        # 因为反方向的代价是错误答案（§3.6）。
        return GranularityDecision(
            route=GranularityRoute.FULLTEXT_ONLY,
            granularity=Granularity.FULLTEXT,
            depth_hit=True,
        )

    # ① 不命中：两层并召，不加 granularity 过滤。这是默认形态。
    return GranularityDecision(route=GranularityRoute.BOTH, granularity=Granularity.ANY, recency_hit=recency_hit)


def fallback_on_insufficient_groups(
    decision: GranularityDecision,
    *,
    returned_groups: int,
    requested_groups: int,
    response_language: str = "EN",
) -> tuple[GranularityDecision, DegradeReason | None]:
    """档 1 降级：fulltext 硬过滤后凑不满 `group_limit`（M3 §3.5）。

    **这条降级链里没有引入任何新的阈值常数**——触发条件完全由既有参数
    `group_limit` 派生。「凑不满预算」本身就是一个可观测事实，不需要再拍一个
    「少于 8 组」之类的数。本模块被删掉的东西里占比最大的一类就是这种没有出处的判据。

    档 2（放开后仍凑不满，或 rerank 后无一条过阈）由生成层的「信息不足」处理，
    判定发生在首句闸门**之前**——此时尚未推送任何 token，「不展示」才是可执行的。

    :returns: `(放开过滤后的决策, 降级原因)`。第二项为 None 表示无需降级。
    """
    if decision.route is not GranularityRoute.FULLTEXT_ONLY:
        return decision, None
    if returned_groups >= requested_groups:
        return decision, None

    # 放开 granularity 过滤两层并召，但**保留第三行的边界声明**：
    # 用户问的仍然是方法细节，只是现在证据里混进了摘要块，这个事实必须说出来。
    note = COVERAGE_NOTE_ZH if response_language == "ZH" else COVERAGE_NOTE_EN
    relaxed = GranularityDecision(
        route=GranularityRoute.BOTH_WITH_COVERAGE_NOTE,
        granularity=Granularity.ANY,
        coverage_note=note,
        depth_hit=decision.depth_hit,
        recency_hit=decision.recency_hit,
    )
    return relaxed, DegradeReason.FULLTEXT_INSUFFICIENT
