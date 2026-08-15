"""检索层（M2/M4）：一次查询如何变成候选 chunk 列表，以及结果怎么组装成响应。

两条不可协商的约束：

1. **融合器唯一 `RRFRanker(k=60)`**。没有 WeightedRanker、没有权重字段、
   没有 `dense_weight`——它们已被整体删除。需要通道偏置的场景改为单路不融合
   （中文 query 走 dense_only，引号短语走 sparse_only）。
2. **RRF 分数无物理含义**（k=60 时上界约 2/61≈0.033），禁止任何上游拿它做
   相关性阈值。全系统唯一可做阈值的分数是 rerank 的 sigmoid 归一化分。

模块划分：

- `hybrid.py`      —— dense + sparse 双路 + RRF + grouping 的执行（含 HyDE 多路合并）
- `exact_match.py` —— TEXT_MATCH / PHRASE_MATCH(slop) 的检索层入口
- `filters.py`     —— `QueryPlan` → 过滤表达式；撤稿黑名单在应用侧剔除
- `grouping.py`    —— grouping 结果解析：rows → 组 → `PaperHit`
- `rerank.py`      —— 应用侧 bge-reranker-v2-m3，`normalize=True`，48→8，含降级链
- `snippets.py`    —— 片段与高亮（展示层，不复刻服务端 analyzer）
- `facets.py`      —— 两类分面口径：结果内近似 + 全库精确

rerank 放应用侧而不是用 Milvus 内置 Model Ranker，是因为官方硬限制
「Model rankers cannot be used with grouping searches」。两个官方能力互斥时
选 grouping——chunk→paper 聚合是产品必需，rerank 放哪层是实现细节。
"""

from citequest.retrieval.exact_match import build_phrase_filter, build_required_terms_filter, relax_slop
from citequest.retrieval.facets import ExactFacetSource, build_in_result_facets
from citequest.retrieval.filters import apply_retracted_blocklist, build_plan_filter
from citequest.retrieval.grouping import ParsedGroups, parse_groups, to_paper_hits
from citequest.retrieval.hybrid import HybridRetriever, RetrievalOutcome
from citequest.retrieval.rerank import Reranker, RerankOutcome
from citequest.retrieval.snippets import Snippet, build_snippet, highlight_terms

__all__ = [
    "ExactFacetSource",
    "HybridRetriever",
    "ParsedGroups",
    "RerankOutcome",
    "Reranker",
    "RetrievalOutcome",
    "Snippet",
    "apply_retracted_blocklist",
    "build_in_result_facets",
    "build_phrase_filter",
    "build_plan_filter",
    "build_required_terms_filter",
    "build_snippet",
    "highlight_terms",
    "parse_groups",
    "relax_slop",
    "to_paper_hits",
]
