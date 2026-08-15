"""索引参数（决策记录 D8，M2 §6）。

三类索引，各自的取值都有出处：

| 字段 | 索引 | 参数 |
|---|---|---|
| `dense` | HNSW / COSINE | `M=32`、`efConstruction=256`；查询期 `ef` **派生** |
| `sparse` | SPARSE_INVERTED_INDEX / BM25 | `inverted_index_algo=DAAT_MAXSCORE` |
| `paper_id` / `granularity` / `year` / `primary_category` | INVERTED | — |

**查询期 ef 必须是派生式，不能是常数**，这是本文件最重要的一件事：

1. 官方只给了两句——`ef` 默认值**等于 `limit`**、推荐区间 `[K, 10K]`。所以把 ef
   写死成 64 而上层要 180 条候选，意味着搜索宽度小于要返回的条数，召回会塌。
2. 改成「按 limit 派生」仍然改得不彻底：grouping search 把 `limit` 的语义变成了
   **组数**，而 ef 约束的是**扫出的 entity 数**。摘要级（1 chunk/篇）下两者恰好
   等价，全文级（约 25 chunk/篇）下就不等价了——同一篇论文的多个 chunk 在向量
   空间里高度聚集会连续命中，要凑满 60 个**不同**的 paper，底层必须扫远多于 60 个
   entity。
3. **这个失败模式最危险的地方是它不报错。** Milvus 不会说「凑不满」，只会返回更少
   的组；而 `strict_group_size=False`（本项目采用的值）恰恰是「优先满足组数、不
   保证组内条数」，问题会以「组数够但每组只有 1 条」的形式伪装成正常结果。

> **必须诚实标注**：grouping 场景下 ef 该怎么取，**官方没有任何指引**。这条派生式
> 是工程规则，`C=4` 是起点不是依据，待 M7 用 FLAT 暴力算 grouping ground truth、
> 扫 C ∈ {2,4,8,16} 画组召回曲线后定死。面试里必须这样讲，不能包装成官方做法。
"""

from __future__ import annotations

from typing import Any

from pymilvus import MilvusClient
from pymilvus.milvus_client.index import IndexParams

from citequest.config.milvus import MilvusSettings

__all__ = [
    "SCALAR_INDEX_FIELDS",
    "build_dense_search_params",
    "build_index_params",
    "build_sparse_search_params",
]

#: 建标量索引的字段。**只给真的会出现在 filter 表达式里的字段建**——倒排索引不是
#: 免费的（官方明确提示额外存储），给一个从不参与过滤的字段建索引是纯亏。
#: `license` / `source_dataset` 只随结果返回，不做过滤维度，所以不在列内。
SCALAR_INDEX_FIELDS: tuple[str, ...] = ("paper_id", "granularity", "year", "primary_category")


def build_index_params(milvus: MilvusSettings) -> IndexParams:
    """装配三类索引的建库参数。

    降本三档（mmap → HNSW_SQ+refine → DiskANN）在这里只体现前两档：mmap 是
    collection 属性不是索引参数，量化则是换 `index_type`。IVF_PQ **用不了**——
    它不在 grouping search 的官方支持索引列表内，而 chunk→paper 聚合就靠 group_by。
    这是「做完聚合裁定之后才发现的连锁约束」。
    """
    params = MilvusClient.prepare_index_params()

    dense_index_type = milvus.quantization or milvus.dense_index_type
    dense_params: dict[str, Any] = {
        "M": milvus.hnsw_m,
        "efConstruction": milvus.hnsw_ef_construction,
    }
    if milvus.quantization == "HNSW_SQ":
        # **必须开 refine**：同机制的公开实测（1M×768、VectorDBBench）显示纯量化
        # 召回从 95.2% 掉到 76.3%，加 SQ8 refine 才回到 94.9%。
        # refine_type 精度须高于 sq_type，这是官方对 refine 语义的要求。
        dense_params["refine"] = milvus.hnsw_sq_refine
        dense_params["refine_type"] = "FP16"
        dense_params["sq_type"] = "SQ8"

    params.add_index(
        field_name=milvus.dense_field,
        index_type=dense_index_type,
        index_name="idx_dense",
        metric_type=milvus.dense_metric_type,
        params=dense_params,
    )

    params.add_index(
        field_name=milvus.sparse_field,
        index_type=milvus.sparse_index_type,
        index_name="idx_sparse",
        # BM25 Function 的输出字段**必须**用 metric_type=BM25，这不是可选项。
        metric_type=milvus.sparse_metric_type,
        params={"inverted_index_algo": milvus.inverted_index_algo},
    )

    for field in SCALAR_INDEX_FIELDS:
        params.add_index(field_name=field, index_type="INVERTED", index_name=f"idx_{field}")

    return params


def build_dense_search_params(milvus: MilvusSettings, *, group_limit: int, group_size: int) -> dict[str, Any]:
    """dense 通道的查询期参数，`ef` 由 `(group_limit, group_size)` 派生。

    `ef = max(C × group_limit × group_size, 64)`，且断言 `ef ≥ group_limit × group_size`
    ——派生式与断言都在 `MilvusSettings.ef_for` 里，这里只负责组装成 Milvus 的
    `search_params` 形状，不重写一遍公式（重写就会有第二份正本）。

    搜索路（60×1）得 240，RAG 路（16×3）得 192。**连带代价必须承认**：ef 从写死的
    64 涨到 192–240，单次 ANN 延迟明显上升——这是把错的快换成对的慢。
    """
    return {
        "metric_type": milvus.dense_metric_type,
        "params": {"ef": milvus.ef_for(group_limit, group_size)},
    }


def build_sparse_search_params(milvus: MilvusSettings) -> dict[str, Any]:
    """sparse 通道的查询期参数。

    BM25 侧没有 ef 这类宽度旋钮——倒排的候选规模由词项本身决定。这里只固定
    metric_type，让「两路参数由同一处装配」这件事在代码里看得见：两路参数分散在
    两个文件里的直接后果，是有一天只有一路被改了。
    """
    return {"metric_type": milvus.sparse_metric_type}
