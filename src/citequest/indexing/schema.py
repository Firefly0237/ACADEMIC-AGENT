"""Collection schema 正本（M2 §5.1）。

⚠ **建库前不可逆的四项里有三项落在这个文件**：`text_en` 的 analyzer 类型、
BM25 的 k1/b（在 `functions.py`，由本文件挂载）、`text_exact` 字段是否存在。
第四项（占位符处理策略）在 ingestion 层。改任何一项 = 重建 collection + 重灌
147 万 chunk + 一次全量 TEI 重编码，**6–10 小时量级**。

在这个价格下，「先不建、评测说需要再加」不是省事，是赌博——所以 `text_exact`
从 P1 升级为必建，理由不是它更好，是**返工代价变了**。

三个必须解释的设计点（M2 §5.1）：

1. `text_en` 与 `text_exact` 存的是**同一段文本**（两份 VARCHAR + 两份倒排）。
   看起来冗余，但它们的 analyzer 目标相反，而 **Milvus 的 analyzer 绑定在字段上
   而不是查询上**——「一份文本两种切法」只能靠两个字段表达。代价是存储与写入放大，
   收益是精确匹配与语义召回互不牵制。
2. `inverted_index_algo=DAAT_MAXSCORE` 显式写出，但**不宣称它是默认值**
   （官方示例中出现，未明示是否默认）。显式写出的成本是零，赌它是默认的成本不是零。
3. **没有 `venue` / `citations` 参与排序**。引用量与时效不进单一分数，改为用户可
   切换的 `relevance / citations / date`——把不可比的量硬加权成一个分数，会让同一篇
   论文在不同 query 下拿到不同引用分，**排序不可复现也不可缓存**。
"""

from __future__ import annotations

from pymilvus import CollectionSchema, DataType, MilvusClient

from citequest.config.embedding import EmbeddingSettings
from citequest.config.milvus import MilvusSettings
from citequest.indexing.functions import build_bm25_function, build_tei_embedding_function

__all__ = [
    "CHUNK_ANCHOR_FIELDS",
    "PAPER_OUTPUT_FIELDS",
    "VARCHAR_LENGTHS",
    "build_collection_schema",
    "chunk_output_fields",
]

#: VARCHAR 长度（M2 §5.1）。口径统一是「450 token × 英文约 4.5 字符/token ≈ 2,025
#: 字符，向上取 2 的幂并留余量」→ 4096；ID/枚举类按天然上界取。
#: **【待实测】** 全量语料下 4096 是否够——截断会静默丢内容，这是 M2 §13 的第 10 项。
VARCHAR_LENGTHS: dict[str, int] = {
    "chunk_id": 64,
    "paper_id": 64,
    "granularity": 16,
    "primary_category": 32,
    "license": 32,
    "source_dataset": 32,
    "section": 128,
    "sec_number": 16,
    "sec_type": 32,
    "text_en": 4096,
    "text_exact": 4096,
}

#: 检索结果里除文本字段之外要取回的部分（字段名不随配置变化）。
CHUNK_ANCHOR_FIELDS: list[str] = [
    "chunk_id",
    "paper_id",
    "granularity",
    "section",
    "sec_number",
    "sec_type",
    "paragraph_idx",
    "year",
    "primary_category",
    "license",
    "source_dataset",
]

#: 论文级展示需要的字段子集（`/v1/papers` 与分面用）。
PAPER_OUTPUT_FIELDS: list[str] = ["paper_id", "year", "primary_category", "granularity", "license", "source_dataset"]


def chunk_output_fields(milvus: MilvusSettings) -> list[str]:
    """检索时 `output_fields` 的正本。

    两处刻意的省略，都是为了不把响应体撑大：
    - **不含 `text_exact`**——它与 `text_en` 内容完全相同，取回两份纯属浪费；
    - **不含两个向量字段**——dense 1024 维 float 取回来没有任何用途（RRF 已经在
      服务端算完），但每条结果会多 4 KB，60 组就是 240 KB 的无效载荷。

    文本字段名从配置读而不是写字面量：字段名一旦被改，写入端与读取端必须同时变，
    两处各写一份字面量就会出现「写进去了但取不出来」这类静默故障。
    """
    return [*CHUNK_ANCHOR_FIELDS, milvus.text_en_field]


def build_collection_schema(
    milvus: MilvusSettings,
    embedding: EmbeddingSettings,
) -> CollectionSchema:
    """构造完整 CollectionSchema：字段 + 两条 Function。

    `enable_dynamic_field=False` 是刻意的：动态字段会让「写错字段名」从一个立刻
    报错的问题，变成一个「数据写进去了但永远查不到」的问题。本项目的字段集是
    建库前定死的，没有动态字段的用武之地。
    """
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
        description="CiteQuest chunk collection：dense(TEI Function) + sparse(BM25 Function) + 双 analyzer 文本字段",
    )

    # ---- 主键与分组键 ----
    # chunk_id 由 f(paper_id, granularity, section 序号, 起止 paragraph_idx) 确定性构造，
    # 于是「按主键 upsert」直接等价于幂等写入（M1 §8.2）。
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=VARCHAR_LENGTHS["chunk_id"], is_primary=True)
    # paper_id 是 grouping search 的 group_by_field，也是引用锚点的根。
    schema.add_field("paper_id", DataType.VARCHAR, max_length=VARCHAR_LENGTHS["paper_id"])

    # ---- 标量：过滤与路由 ----
    # granularity 是整个 schema 里最重要的标量字段：方法级问题必须过滤到 fulltext，
    # 不过滤就会让模型拿 2024 年的摘要编出一个听起来合理的损失函数——
    # 这是路由层里唯一一个「不做就会静默产生幻觉」的分支。
    schema.add_field("granularity", DataType.VARCHAR, max_length=VARCHAR_LENGTHS["granularity"])
    # year 同时是 partition key：过滤是本语料的刚需不是可选项，而过滤选择率一低
    # HNSW 的图导航会退化成暴力扫描。按年份物理分区让带年份过滤命中分区而非全库。
    schema.add_field("year", DataType.INT16, is_partition_key=milvus.partition_key_field == "year")
    schema.add_field("primary_category", DataType.VARCHAR, max_length=VARCHAR_LENGTHS["primary_category"])
    # license 逐条带，不做全局混合声明——两层许可不同，汇编不是改编。
    schema.add_field("license", DataType.VARCHAR, max_length=VARCHAR_LENGTHS["license"])
    schema.add_field("source_dataset", DataType.VARCHAR, max_length=VARCHAR_LENGTHS["source_dataset"])

    # ---- 引用锚点三件套（不进倒排，随结果返回）----
    # 有了它们，引用能说「4.2 节第 3 段」而不是「这篇论文」。
    schema.add_field("section", DataType.VARCHAR, max_length=VARCHAR_LENGTHS["section"], nullable=True)
    schema.add_field("sec_number", DataType.VARCHAR, max_length=VARCHAR_LENGTHS["sec_number"], nullable=True)
    schema.add_field("sec_type", DataType.VARCHAR, max_length=VARCHAR_LENGTHS["sec_type"], nullable=True)
    schema.add_field("paragraph_idx", DataType.INT16, nullable=True)

    # ---- ⚠ 双 analyzer 文本字段 ----
    # text_en：主字段，english analyzer（standard tokenizer + lowercase/stemmer/stop），
    # 挂 BM25 Function 与 TEI Function，是**唯一**进融合的文本通道。
    schema.add_field(
        milvus.text_en_field,
        DataType.VARCHAR,
        max_length=VARCHAR_LENGTHS["text_en"],
        enable_analyzer=True,
        analyzer_params=milvus.text_en_analyzer_params,
    )
    # text_exact：whitespace + lowercase + alphanumonly，开 enable_match 建**带位置**
    # 的倒排，只服务 TEXT_MATCH 硬过滤与 PHRASE_MATCH(slop) 短语邻近。
    # **不挂 BM25 Function**——挂了它就变成第三条检索路，而融合必须保持
    # dense + text_en 两路，RRF 结构才不变。这是一处刻意的收敛。
    schema.add_field(
        milvus.text_exact_field,
        DataType.VARCHAR,
        max_length=VARCHAR_LENGTHS["text_exact"],
        enable_analyzer=True,
        enable_match=milvus.text_exact_enable_match,
        analyzer_params=milvus.text_exact_analyzer_params,
    )

    # ---- 向量字段：两者都是 Function 的**输出**，客户端永不直接写 ----
    schema.add_field(milvus.dense_field, DataType.FLOAT_VECTOR, dim=milvus.dense_dim)
    schema.add_field(milvus.sparse_field, DataType.SPARSE_FLOAT_VECTOR)

    # ---- 两条 Function 声明 ----
    schema.add_function(build_tei_embedding_function(milvus, embedding))
    schema.add_function(build_bm25_function(milvus))

    if milvus.text_exact_attach_bm25:  # pragma: no cover - 配置被改动时的显式失败
        raise ValueError(
            "text_exact_attach_bm25=True 会让 text_exact 变成第三条检索路，"
            "而融合层只承认 dense + text_en 两路（决策记录 D5）。要改必须先改决策记录"
        )
    return schema
