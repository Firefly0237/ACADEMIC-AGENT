"""Milvus 内置 Function 声明（决策记录 D3 / D4，M2 §5.2）。

**这两行是本模块的核心技术叙事**：

```
Function(type=TEXTEMBEDDING, provider=TEI, input=[text_en], output=[dense])
Function(type=BM25, input=[text_en], output=[sparse])
```

一个 Collection 定义把 dense 与 sparse 两条通道全部描述完，写入与检索共用同一份
声明，**中间没有任何自定义推理层可供质疑**。

- dense 走 **TEI 官方镜像 + Milvus TEI Embedding Function（Data-in/Data-out）**。
  客户端不做 embedding——「灌库用 A 模型、查询用 B 模型」这类不一致在结构上不可能
  发生。连锁后果有两个，都是好的：客户端拿不到 query embedding，于是 M3 的
  embedding 锚点路由失去了「反正本来就要算」的免费前提，M6 的 L3 embedding 缓存
  物理上不存在。
- sparse 走 **Milvus 内置 BM25 Function**，不是 learned sparse。这是把一个「没有
  正规部署路径的组件」从架构里删掉，而不是给它找一个勉强的部署方式。

> 两条 Function 的输入都是 `text_en`（原文字段）。dense 侧不受 analyzer 影响——
> TEI 拿到的是原始字符串，analyzer 只作用于倒排构建。这也是为什么 analyzer 选型
> 只影响 sparse 通道。
"""

from __future__ import annotations

from pymilvus import Function, FunctionType

from citequest.config.embedding import EmbeddingSettings
from citequest.config.milvus import MilvusSettings

__all__ = ["build_bm25_function", "build_tei_embedding_function"]


def build_bm25_function(milvus: MilvusSettings) -> Function:
    """`text_en` → `sparse` 的 BM25 Function。

    ⚠ **`bm25_k1` / `bm25_b` 是索引期参数，在 Function 定义时写死**——改它们要重建
    collection + 重灌 147 万 chunk + 一次全量 TEI 重编码，6–10 小时量级。它们是
    「建库前不可逆的四项」中的两项。

    取值都是官方默认（k1=1.2、b=0.75）。b 的**理由已经换过一次**：旧理由「摘要长度
    集中所以 b 无所谓」在全文语料下不成立（长度归一化第一次真正开始工作）；新理由是
    按 section 边界聚合到 450 token 的切块策略已经把长度方差约束在有界区间内，
    b 的杠杆两头都不大，偏离官方默认需要证据而我没有。
    """
    return Function(
        name="bm25_text_en",
        function_type=FunctionType.BM25,
        input_field_names=[milvus.text_en_field],
        output_field_names=[milvus.sparse_field],
        description="服务端 Tantivy 分词 + BM25 权重，输出 SPARSE_FLOAT_VECTOR",
        params={
            "bm25_k1": milvus.bm25_k1,
            "bm25_b": milvus.bm25_b,
        },
    )


def build_tei_embedding_function(milvus: MilvusSettings, embedding: EmbeddingSettings) -> Function:
    """`text_en` → `dense` 的 TEI Embedding Function（Data-in/Data-out）。

    参数取值全部来自决策记录 §4.3，其中两个必须显式写出：

    - `truncate="true"`：**官方默认是 false**，不显式开的话超长文本会直接报错而不是
      截断。切块上界是 450+80=530 token，理论上不会触发，但摘要层的异常长文本
      （见 `openalex.abstract` 的 `_MAX_POSITION`）会——报错会中断整批写入。
    - `truncation_direction="right"`：官方默认就是 right，显式写出的成本是零，
      赌它不变的成本不是零。

    **必须主动交代的不确定性**：TEI 官方 supported_models 表未逐条列出 `BAAI/bge-m3`，
    支持是按架构族给的（表中原文含 "XLM-RoBERTa models with absolute positions"，
    BGE-M3 正是 XLM-RoBERTa 底座）。落地第一步必须验 `/info` 与向量一致性，不一致
    就切 `intfloat/multilingual-e5-large-instruct`——同架构、同 1024 维、表内列名，
    **schema 一行不改**，e5 要求的前缀由 `ingestion_prompt` / `search_prompt` 承接。
    """
    params: dict[str, str | int] = {
        "provider": embedding.provider,
        "endpoint": embedding.endpoint,
        "truncate": "true" if embedding.truncate else "false",
        "truncation_direction": embedding.truncation_direction,
        "max_client_batch_size": embedding.max_client_batch_size,
    }
    if embedding.api_key:
        # 只有 TEI 带 --api-key 时才需要配 credential；否则 milvus.yaml **无需任何改动**
        # （官方 yaml 片段自带 `enable: true # no action required`）。
        params["credential"] = embedding.api_key
    if embedding.ingestion_prompt:
        params["ingestion_prompt"] = embedding.ingestion_prompt
    if embedding.search_prompt:
        params["search_prompt"] = embedding.search_prompt

    return Function(
        name="tei_embedding_text_en",
        function_type=FunctionType.TEXTEMBEDDING,
        input_field_names=[milvus.text_en_field],
        output_field_names=[milvus.dense_field],
        description=f"TEI({embedding.model_id}) 服务端编码，客户端不做 embedding",
        params=params,
    )
