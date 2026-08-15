"""全局枚举与口径标签。

这里只放**跨模块共享**的取值集合。之所以集中在一处，是因为这些取值同时出现在
Milvus schema、API 响应体、SSE 事件、日志字段和评测报告里——分散定义必然漂移，
而其中好几个（granularity / retrieval_mode / citation_support）漂移的后果是
静默的语义错误，不是报错。

取值全部来自设计文档：
- Granularity / SecType          → 决策记录 §4.1、M1 §5.5/§6.1
- RouteType / GranularityRoute   → M3 §2.1、§8.1
- RetrievalMode / RewriteStrategy→ 决策记录 §4.4、M3 §8.1
- SortMode                       → 决策记录 §4.4「排序模式」
- CitationSupport                → M4 §1.3（`citation_support[]`）
- SseEventType                   → M6 §2.1 事件全表
- DegradeReason                  → M6 §5.4 降级矩阵
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "CitationSupport",
    "DegradeReason",
    "Granularity",
    "GranularityRoute",
    "LicenseId",
    "PipelineStage",
    "RetrievalMode",
    "RetrievalQualityGrade",
    "RewriteStrategy",
    "RouteType",
    "SecType",
    "SortMode",
    "SourceDataset",
    "SseEventType",
]


class Granularity(StrEnum):
    """语料粒度。

    这不是一个内部实现细节，而是**刻意暴露给调用方的能力边界**（M6 §1.2）：
    2019–2022 有 section 级全文，2023–2025 只有摘要。方法级问题只有 fulltext
    答得了，路由层不强制过滤就会让模型拿摘要编方法细节（决策记录 §2.7 第 2 条）。
    """

    FULLTEXT = "fulltext"
    ABSTRACT = "abstract"
    #: 仅用于**请求参数**的「不限」，不会出现在 chunk 记录上
    ANY = "any"


class SecType(StrEnum):
    """章节的 IMRaD 类型，由 unarXive 直接给出，本项目不做标题正则识别（M1 §6.1）。

    用途有二：检索侧可选过滤维度；切块质量校验（与 unarXive_imrad_clf 比对分布）。
    """

    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    METHOD = "method"
    EXPERIMENT = "experiment"
    RESULT = "result"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    RELATED_WORK = "related_work"
    ABSTRACT = "abstract"
    OTHER = "other"


class SourceDataset(StrEnum):
    """来源数据集。逐条写入 chunk，不做全局声明（决策记录 D2 第 4 条）。"""

    UNARXIVE_OPEN_SUBSET = "unarXive_open_subset"
    OPENALEX = "openalex"


class LicenseId(StrEnum):
    """许可标识。两层许可不同，必须逐条标注——汇编（Collection）不是改编（Adaptation），
    所以不能用一句全局混合声明糊过去（决策记录 D2 第 4 条）。
    """

    CC_BY_SA_4_0 = "CC-BY-SA-4.0"
    CC0 = "CC0"


class RouteType(StrEnum):
    """查询路由类型（M3 §2.1）。

    STRUCTURED 不是独立链路，它是「SEARCH_ONLY + 一组非空 filter」；
    单列出来只是为了让监控能看见结构化条件的占比。
    """

    SEARCH_ONLY = "SEARCH_ONLY"
    SEARCH_PLUS_QA = "SEARCH_PLUS_QA"
    SEARCH_PLUS_MULTIHOP = "SEARCH_PLUS_MULTIHOP"
    STRUCTURED = "STRUCTURED"


class GranularityRoute(StrEnum):
    """粒度路由的三种处置（M3 §8.1）。

    `both_with_coverage_note` 存在的理由：既不能把 2023 年后的论文一律排除，
    也不能让摘要冒充方法细节，所以第三种处置是「都检索，但在响应里标覆盖说明」。
    """

    FULLTEXT_ONLY = "fulltext_only"
    BOTH = "both"
    BOTH_WITH_COVERAGE_NOTE = "both_with_coverage_note"


class RetrievalMode(StrEnum):
    """本次检索实际走了哪几路（决策记录 D6 / §4.4）。

    这个字段的作用是**把降级度量出来而不是掩盖掉**：中文 query 打英文文档时
    BM25 命中率≈0，此时只发 dense 一路并打点 dense_only；带引号的确切短语
    只发 sparse 一路（单路下不存在量纲问题，BM25 分数直接就是排序）。
    """

    HYBRID = "hybrid"
    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only"


class RewriteStrategy(StrEnum):
    """查询改写策略（决策记录 D12）。

    只有三个取值：multi-query 与 rewrite 已删除（ARAGOG 实测 multi-query 降低
    retrieval precision），step-back 只进离线消融。`direct` 是所有改写必须打败的基线。
    """

    DIRECT = "direct"
    HYDE = "hyde"
    DECOMPOSE = "decompose"


class SortMode(StrEnum):
    """排序模式（决策记录 §4.4）。

    引用量与时效**不进单一相关性分数**——六信号加权公式已被剔除（H1），
    因为 `max_citations` 取候选集内最大值会让同一篇论文在不同 query 下拿到
    不同引用分，排序不可复现也不可缓存。改为用户可切换的三档。
    """

    RELEVANCE = "relevance"
    CITATIONS = "citations"
    DATE = "date"


class CitationSupport(StrEnum):
    """逐引用边的支持强度（M4 §1.3，随 `ai_answer_done` 下发）。

    `unverified` 不是失败，它是 HHEM 不可用或该边尚未判定时的诚实取值——
    在线只在首句跑一次 HHEM（决策记录 §4.6），其余边默认就是 unverified。
    """

    SUPPORTED = "supported"
    WEAK = "weak"
    UNVERIFIED = "unverified"


class RetrievalQualityGrade(StrEnum):
    """检索质量评估的三档（决策记录 §4.5）。

    仅概念层引 CRAG（arXiv:2401.15884），机制层照 LangGraph 官方 agentic-rag 教程。
    **必须声明的偏离**：CRAG 的 Incorrect 分支是 web search，本项目不做——
    Incorrect 在这里只触发一次改写重检索，循环上限 1 次。
    """

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class SseEventType(StrEnum):
    """`/v1/answer` 的 SSE 事件类型（M6 §2.1）。

    三条不变量写在这里而不是散在实现里：
    1. `META` 永远是第一个事件，且在任何下游调用之前就发（它同时是「代理没有缓冲」的探针）；
    2. `AI_ANSWER_START` 出现即意味着首句已过 grounding 闸门；
    3. `AI_ANSWER_FLAGGED` 只能降级不能撤回，它永远指向一个已推送的 seq。
    """

    META = "meta"
    SEARCH_RESULTS = "search_results"
    AI_ANSWER_START = "ai_answer_start"
    AI_ANSWER_DELTA = "ai_answer_delta"
    AI_ANSWER_FLAGGED = "ai_answer_flagged"
    AI_ANSWER_DONE = "ai_answer_done"
    AI_ANSWER_ERROR = "ai_answer_error"
    ERROR = "error"
    DONE = "done"


class DegradeReason(StrEnum):
    """降级原因（M6 §5.4）。底线永远是「只返回搜索结果，不返回 AI 失败」（K9）。

    这些取值同时是 Prometheus 的分桶标签——`degraded` 非空的请求占比是三个
    「静默失败」监控项之一（M6 §6.5）：长期跑在降级状态时所有健康检查都是绿的。
    """

    RERANK_TIMEOUT = "rerank_timeout"
    #: fulltext 硬过滤后 grouping 凑不满 group_limit → 放开过滤两层并召（M3 §3.5 档 1）。
    #: 它的触发率高说明**全文层覆盖不足，正确的补救是扩语料而不是调路由**。
    FULLTEXT_INSUFFICIENT = "fulltext_insufficient"
    RERANK_UNAVAILABLE = "rerank_unavailable"
    HYDE_TIMEOUT = "hyde_timeout"
    ROUTER_FALLBACK = "router_fallback"
    HHEM_UNAVAILABLE = "hhem_unavailable"
    GENERATION_TIMEOUT = "generation_timeout"
    GENERATION_UNAVAILABLE = "generation_unavailable"
    CORRECTION_LOOP_TIMEOUT = "correction_loop_timeout"
    SPARSE_CHANNEL_SKIPPED = "sparse_channel_skipped"
    #: 关系库（分面）挂 → `facets_exact` 缺省为 null，筛选器不显示精确计数（M6 §5.4）。
    #: 它不影响检索结果本身，所以是降级不是错误。
    EXACT_FACETS_UNAVAILABLE = "exact_facets_unavailable"
    #: 缓存旁路（Redis 熔断）。同时要把 GPU 信号量收紧到一半——用主动降级换住不崩。
    CACHE_UNAVAILABLE = "cache_unavailable"


class PipelineStage(StrEnum):
    """span 树的节点名（M6 §6.5：route → search → rerank → hyde → generate → ground → postprocess）。

    固定成枚举是为了让 Langfuse span 名、Prometheus 直方图标签、结构化日志的
    stage 字段三处用同一套字面值——三处各写各的字符串，聚合时就对不上了。
    """

    ROUTE = "route"
    HYDE = "hyde"
    DECOMPOSE = "decompose"
    SEARCH = "search"
    RERANK = "rerank"
    GENERATE = "generate"
    GROUND = "ground"
    POSTPROCESS = "postprocess"
    CORRECTION_LOOP = "correction_loop"
