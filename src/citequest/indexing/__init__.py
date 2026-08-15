"""索引层（M2）：Milvus collection 的定义与写入。

**四项建库前不可逆的决定全部落在这里**（第四项占位符策略在 ingestion）：
analyzer 类型、`bm25_k1` / `bm25_b`、`text_exact` 是否存在。改任何一项 =
重建 collection + 重灌 147 万 chunk + 一次全量 TEI 重编码，6–10 小时量级。
所以它们必须在建库前用 1 万条抽样定死，不能「先不建、评测说需要再说」。

模块划分：

- `client.py`       —— MilvusClient 构造、连接管理、启动探针、
                       collection 生命周期与 `hybrid_search` 的真实调用
- `schema.py`       —— 字段定义：`text_en` / `text_exact` 双 analyzer、
                       `paper_id`、`dense`、`sparse`、标量字段、按年份 partition key
- `functions.py`    —— Milvus 内置 Function 声明：BM25 Function + TEI Embedding
                       Function（Data-in/Data-out，**客户端不做 embedding**）
- `index_params.py` —— HNSW(M=32, efConstruction=256) 与
                       SPARSE_INVERTED_INDEX(k1=1.2, b=0.75)
- `expressions.py`  —— 标量过滤 / TEXT_MATCH / PHRASE_MATCH(slop) 表达式构造。
                       放这里而不是 retrieval，是因为它的正确性完全依赖 schema 的
                       字段名与 analyzer 配置（短语必须打在 `text_exact` 上）
- `ann_requests.py` —— 两路 AnnSearchRequest、RRFRanker(k=60)、grouping 参数装配
- `loader.py`       —— 批量 upsert、checkpoint、幂等、先删后写与软删

参数一律从 `citequest.config` 读，本包**不写死任何数值**。
"""

from citequest.indexing.ann_requests import GroupingSpec, build_ann_requests, build_ranker, grouping_kwargs
from citequest.indexing.client import MilvusIndexClient
from citequest.indexing.expressions import (
    build_filter_expression,
    combine_expressions,
    phrase_match_expr,
    text_match_expr,
)
from citequest.indexing.functions import build_bm25_function, build_tei_embedding_function
from citequest.indexing.index_params import (
    SCALAR_INDEX_FIELDS,
    build_dense_search_params,
    build_index_params,
    build_sparse_search_params,
)
from citequest.indexing.loader import ChunkLoader, LoadReport, chunk_to_row
from citequest.indexing.schema import (
    CHUNK_ANCHOR_FIELDS,
    PAPER_OUTPUT_FIELDS,
    VARCHAR_LENGTHS,
    build_collection_schema,
    chunk_output_fields,
)

__all__ = [
    "CHUNK_ANCHOR_FIELDS",
    "PAPER_OUTPUT_FIELDS",
    "SCALAR_INDEX_FIELDS",
    "VARCHAR_LENGTHS",
    "ChunkLoader",
    "GroupingSpec",
    "LoadReport",
    "MilvusIndexClient",
    "build_ann_requests",
    "build_bm25_function",
    "build_collection_schema",
    "build_dense_search_params",
    "build_filter_expression",
    "build_index_params",
    "build_ranker",
    "build_sparse_search_params",
    "build_tei_embedding_function",
    "chunk_output_fields",
    "chunk_to_row",
    "combine_expressions",
    "grouping_kwargs",
    "phrase_match_expr",
    "text_match_expr",
]
