"""两路 `AnnSearchRequest` 与融合器的装配（决策记录 D6/D9，M2 §12）。

一次请求在索引层走过的路：

```
query
 ├─ 中文？        → dense 单路（retrieval_mode=dense_only）
 ├─ 带引号短语？  → sparse 单路 + text_exact 的 PHRASE_MATCH 过滤，BM25 分数直接排序
 └─ 英文常规
      ├─ AnnSearchRequest #1 : dense,  HNSW/COSINE, ef = max(4×gl×gs, 64)
      └─ AnnSearchRequest #2 : sparse, SPARSE_INVERTED_INDEX/BM25
            ↓ 两路都由 Milvus 服务端从 query 原文生成（TEI Function / BM25 Function）
      hybrid_search(ranker=RRFRanker(k=60), group_by_field="paper_id", ...)
```

两条不可协商的约束：

1. **融合器唯一 `RRFRanker(k=60)`**。WeightedRanker 与全部权重字段已删除——官方
   自己承认量纲问题，它的解法 arctan 只压值域不对齐分布，issue #40836 有实例证明
   归一化会翻转两个文档的相对顺序。需要通道偏置的场景改为**单路不融合**：单路下
   量纲问题根本不存在，所以不需要一个加权器来解决一个可以避免制造的问题。
2. **`data` 传的是 query 原文字符串**，不是向量。这是 Data-in/Data-out 的直接体现：
   客户端拿不到也不需要 query embedding。

`AnnSearchRequest` **不支持** grouping 的四个参数（`group_by_field` / `group_size` /
`strict_group_size` / `rank_group_scorer`），它们只能在 `hybrid_search` 顶层传——
这一点官方 API 文档没写，由 pymilvus 源码核实（`prepare.py` 把它们塞进 `rank_params`）。
"""

from __future__ import annotations

from pymilvus import AnnSearchRequest, RRFRanker

from citequest.config.milvus import MilvusSettings
from citequest.config.retrieval import RetrievalSettings
from citequest.core.enums import RetrievalMode
from citequest.indexing.index_params import build_dense_search_params, build_sparse_search_params

__all__ = [
    "GroupingSpec",
    "build_ann_requests",
    "build_ranker",
    "grouping_kwargs",
]


class GroupingSpec:
    """一条检索路径的 grouping 参数（决策记录 §4.4）。

    两条必须写进接口文档的语义：

    1. **`limit` 是组数不是条数**——所以对外参数叫 `group_limit`，不叫 `top_k`。
    2. **`group_size` 是上界不是保证**——摘要层每篇只有 1 个 chunk，对这些论文
       要 3 段会**静默返回 1 条**。加上 `strict_group_size=False`，实际返回条数
       ≤ `group_limit × group_size`。下游的上下文预算必须按「可能少于 48」写，
       不能按 48 写死，否则就是一个偶发的下标越界。
    """

    __slots__ = ("group_limit", "group_size", "strict")

    def __init__(self, group_limit: int, group_size: int, *, strict: bool = False) -> None:
        if group_limit < 1 or group_size < 1:
            raise ValueError(f"group_limit({group_limit}) 与 group_size({group_size}) 必须为正")
        self.group_limit = group_limit
        self.group_size = group_size
        # strict_group_size=True 官方警告在数据分布不均时降性能，而混合粒度语料
        # 就是典型的分布不均——摘要层每篇 1 块、全文层每篇约 25 块。
        self.strict = strict

    @property
    def entity_budget(self) -> int:
        """底层至少要扫出的 entity 数，也是 ef 派生式的输入。"""
        return self.group_limit * self.group_size

    @classmethod
    def for_search(cls, retrieval: RetrievalSettings) -> GroupingSpec:
        """搜索路：60 组 × 1，对应浅分页 3 页 × 20 篇。"""
        return cls(retrieval.search_group_limit, retrieval.search_group_size, strict=retrieval.strict_group_size)

    @classmethod
    def for_rag(cls, retrieval: RetrievalSettings) -> GroupingSpec:
        """RAG 路：16 组 × 3 = ≤48 chunk（16×3 ≤ rerank 输入上限 50）。

        为什么是 16×3 而不是 24×2：学术回答的可信度**先取决于引了几篇不同的论文**，
        其次才是每篇挖多深；而 2 段/篇正是被判定为不够的那个值——方法级问题的证据
        物理上分布在 Method 段与 Experiment 段，`group_size=2` 等于用参数把换源
        买来的能力关掉。**M7 必做 group_size ∈ {2,3,4} 的消融来定这个取舍。**
        """
        return cls(retrieval.rag_group_limit, retrieval.rag_group_size, strict=retrieval.strict_group_size)

    @classmethod
    def for_eval(cls, retrieval: RetrievalSettings) -> GroupingSpec:
        """评测路：100 组 × 1，覆盖 Recall@100。

        报告里必须**显式声明**评测用的是加大 limit 的同一条管线，否则口径与线上
        的 60 不一致，指标就不能拿来给线上背书。
        """
        return cls(retrieval.eval_group_limit, 1, strict=retrieval.strict_group_size)


def build_ranker(retrieval: RetrievalSettings) -> RRFRanker:
    """唯一的融合器。

    k=60 有两个独立出处：Milvus 官方默认就是 60（推荐区间 [10,100]）；Cormack
    SIGIR'09 里 60 是 pilot investigation 定下的值，同文 Table 1 显示整个区间几乎
    平坦（k=10→MAP .2123、k=60→.2145、k=100→.2142），而 k=0 掉到 .2072、k=500 掉到
    .2098。**取 60 既是官方默认也是论文原值，是零辩护成本的选择。**

    **接口硬约束**：RRF 分数没有物理含义（k=60 时上界约 2/61≈0.033），禁止任何上游
    拿它做相关性阈值。全系统唯一可做阈值的是 rerank 的 sigmoid 归一化分。
    """
    return RRFRanker(k=retrieval.rrf_k)


def build_ann_requests(
    query_text: str,
    milvus: MilvusSettings,
    grouping: GroupingSpec,
    *,
    mode: RetrievalMode = RetrievalMode.HYBRID,
    filter_expr: str = "",
) -> list[AnnSearchRequest]:
    """按检索模式组装 1–2 路请求。

    `data=[query_text]` 传的是**原文字符串**：dense 侧由 Milvus 调 TEI Function 编码，
    sparse 侧由 BM25 Function 分词打权重。写入期与检索期共用同一份 Function 声明，
    「灌库用 A 模型、查询用 B 模型」这类不一致**在结构上不可能发生**。

    `limit` 这里传的是 `entity_budget`（组数 × 每组条数）而不是组数：单路 ANN 阶段
    还没有分组概念，分组发生在融合之后。传组数会让两路各自只召回 60 条 entity，
    而这 60 条可能只来自 10 篇论文。

    :param mode: 三类 query 的通道组合。中文只组 dense（BM25 打英文文档命中率≈0，
        显式降级并打点，而不是假装 hybrid 还在工作）；引号短语只组 sparse
        （单路下 BM25 分数直接就是排序，量纲问题不存在）。
    """
    if not query_text.strip():
        raise ValueError("query 文本不能为空——空串会让 BM25 Function 产出空 sparse 向量并静默返回随机结果")

    requests: list[AnnSearchRequest] = []
    limit = grouping.entity_budget

    if mode in (RetrievalMode.HYBRID, RetrievalMode.DENSE_ONLY):
        requests.append(
            AnnSearchRequest(
                data=[query_text],
                anns_field=milvus.dense_field,
                param=build_dense_search_params(
                    milvus, group_limit=grouping.group_limit, group_size=grouping.group_size
                ),
                limit=limit,
                expr=filter_expr or None,
            )
        )

    if mode in (RetrievalMode.HYBRID, RetrievalMode.SPARSE_ONLY):
        requests.append(
            AnnSearchRequest(
                data=[query_text],
                anns_field=milvus.sparse_field,
                param=build_sparse_search_params(milvus),
                limit=limit,
                expr=filter_expr or None,
            )
        )

    if not requests:  # pragma: no cover - RetrievalMode 是闭集，这里只防未来新增枚举
        raise ValueError(f"检索模式 {mode} 没有产生任何通道")
    return requests


def grouping_kwargs(grouping: GroupingSpec, retrieval: RetrievalSettings) -> dict[str, object]:
    """grouping 的四个 kwargs，只能在 `hybrid_search` 顶层传。

    `rank_group_scorer` **不设**——源码确认参数存在并透传，但默认值与合法取值均
    **未核实**，而本项目对未核实的参数一律不动。
    """
    return {
        "group_by_field": retrieval.group_by_field,
        "group_size": grouping.group_size,
        "strict_group_size": grouping.strict,
    }
