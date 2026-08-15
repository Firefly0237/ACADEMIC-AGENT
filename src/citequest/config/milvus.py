"""Milvus 连接、collection schema 与索引参数（决策记录 §4.2）。

⚠ 标记的四项是**建库前不可逆**的：analyzer 类型、bm25_k1/bm25_b、text_exact
是否存在（第四项占位符策略在 `corpus.py`）。改任何一项 = 重建 collection +
重灌 147 万 chunk + 一次全量 TEI 重编码，6–10 小时量级。所以它们必须在建库前
用 1 万条抽样定死，而不是「先不建、评测说需要再说」。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["MilvusSettings"]


class MilvusSettings(BaseModel):
    """Milvus 2.6.x。底线抬到 2.6 是 TEI Embedding Function 的硬要求（决策记录 D4）。"""

    # ---- 连接 ----
    uri: str = Field(default="http://localhost:19530")
    token: str = Field(default="", description="留空表示本地无鉴权部署")
    db_name: str = Field(default="default")
    collection_name: str = Field(default="citequest_chunks")
    alias: str = Field(
        default="citequest_current",
        description="对外只用 alias，切库靠改 alias 指向。这是灰度切换与一键回滚的实现方式（K12）",
    )
    index_version: str = Field(
        default="idx_00000000_00",
        description="进 API 响应体、进缓存 key、进 facets_exact。换 index_version 即一键全量缓存失效",
    )
    shards_num: int = Field(default=2, ge=1, le=16, description="官方上限 16/collection")
    request_timeout_ms: int = Field(default=400, ge=1, description="单次调用超时。总 deadline 1.0s 含 1 次重试（§4.9）")

    # ---- dense 向量字段 ----
    dense_field: str = Field(default="dense")
    dense_dim: int = Field(default=1024, description="BGE-M3 输出维度")
    dense_metric_type: str = Field(default="COSINE")
    dense_index_type: str = Field(default="HNSW")
    hnsw_m: int = Field(
        default=32,
        ge=2,
        le=2048,
        description="官方范围 [2,2048]、默认 30、推荐 [5,100]；论文 arXiv:1603.09320 给的 reasonable range 是 5–48，高维高召回取大",
    )
    hnsw_ef_construction: int = Field(
        default=256,
        ge=1,
        description="官方默认 360、推荐 [50,500]。论文判据是「大到建图期召回接近 1」",
    )
    ef_coefficient_c: int = Field(
        default=4,
        ge=1,
        description="ef 派生式的系数。**自选起点，grouping 下 ef 官方无任何指引（U2）**，待 M7 组召回曲线标定 C ∈ {2,4,8,16}",
    )
    ef_floor: int = Field(default=64, ge=1, description="ef 下界。低于它没有意义——官方默认行为就是 ef=limit")

    # ---- ⚠ sparse：Milvus 内置 BM25 Function（不是 learned sparse）----
    sparse_field: str = Field(default="sparse")
    sparse_metric_type: str = Field(default="BM25", description="BM25 Function 输出字段必须用 BM25，不可改")
    sparse_index_type: str = Field(default="SPARSE_INVERTED_INDEX")
    bm25_k1: float = Field(
        default=1.2, ge=1.2, le=2.0, description="⚠ 官方默认，推荐区间 [1.2, 2.0]。索引期参数，改它要重建 collection"
    )
    bm25_b: float = Field(
        default=0.75,
        gt=0.0,
        lt=1.0,
        description=(
            "⚠ 官方默认。**理由已换**：全文语料下长度归一化第一次真正开始工作，"
            "但切块策略已把长度方差约束在有界区间内，b 的杠杆两头都不大，偏离默认需要证据而我没有（U6）"
        ),
    )
    inverted_index_algo: str = Field(
        default="DAAT_MAXSCORE",
        description="**部分核实**：官方示例中出现，未明示是否为默认值（V3）。显式写出，不宣称是默认",
    )

    # ---- ⚠ 两个 analyzed 文本字段 ----
    text_en_field: str = Field(default="text_en", description="主字段，挂 BM25 Function，进融合")
    text_en_analyzer_params: dict[str, Any] = Field(
        default_factory=lambda: {"type": "english"},
        description="⚠ english analyzer = standard tokenizer + lowercase/stemmer/stop。官方 decision guide 对 English-only 内容明确推荐 english 而非 standard",
    )
    text_exact_field: str = Field(default="text_exact")
    text_exact_analyzer_params: dict[str, Any] = Field(
        default_factory=lambda: {"tokenizer": "whitespace", "filter": ["lowercase", "alphanumonly"]},
        description="⚠ 官方对代码/技术内容的推荐配置",
    )
    text_exact_enable_match: bool = Field(
        default=True,
        description="建带位置的倒排，供 TEXT_MATCH 硬过滤与 PHRASE_MATCH(slop) 短语邻近",
    )
    text_exact_attach_bm25: bool = Field(
        default=False,
        description=(
            "**不挂 BM25 Function**。这是一处刻意收敛：挂了它就变成第三条检索路，"
            "而融合仍然只有 dense + text_en 两路，RRF 结构才保持不变（决策记录 D5）"
        ),
    )
    enable_multi_analyzer: bool = Field(
        default=False,
        description="不启用。该特性按文档的 language 字段分流，本项目文档侧纯英文，无分流对象",
    )
    explicit_analyzer_name_on_search: bool = Field(
        default=True,
        description="检索侧显式传 analyzer_name，不依赖隐式默认——隐式默认变了是静默的召回下降",
    )

    # ---- 标量字段与分区 ----
    scalar_fields: list[str] = Field(default_factory=lambda: ["granularity", "year", "primary_category", "license"])
    partition_key_field: str = Field(
        default="year",
        description="按年份物理分区。过滤是本语料刚需不是可选项，且低选择率过滤会让 HNSW 图导航退化成暴力扫描",
    )
    num_partitions: int = Field(default=16, ge=1)

    # ---- 降本三档（按代价从小到大，决策记录 §2.6）----
    mmap_enabled: bool = Field(
        default=False,
        description="档①：零召回损失，官方称 2CPU/8GB 下可多载约 4 倍数据；HNSW 降级可接受而 IVF 系列严重下降。与 DiskANN 互斥",
    )
    quantization: str | None = Field(
        default=None,
        description="档②：填 'HNSW_SQ' 时**必须** refine=true，且 refine_type 精度须高于 sq_type",
    )
    hnsw_sq_refine: bool = Field(default=True, description="强制 refine，不 refine 的量化召回掉幅不可接受")

    def ef_for(self, group_limit: int, group_size: int) -> int:
        """按 entity 预算而非组数派生查询期 ef（决策记录 §4.2）。

        `ef = max(C × group_limit × group_size, ef_floor)`，且必须满足
        `ef ≥ group_limit × group_size`。

        为什么必须派生而不能写死：ef 写死 64 配 limit=180 是真 bug——官方默认行为
        就是 ef=limit，写死 64 反而低于默认。搜索路（60×1）得 240，RAG 路（16×3）得 192。

        C 的取值**没有官方依据**（U2）：grouping search 文档既未枚举 group_by_field
        的字段类型，也未说明内部如何扩大候选来凑组，ef 与组数的关系官方完全没有指引。
        标定前只能说「我知道 ef 必须按 entity 预算派生」，不能给具体倍率。
        """
        if group_limit < 1 or group_size < 1:
            raise ValueError(f"group_limit({group_limit}) 与 group_size({group_size}) 必须为正")
        entity_budget = group_limit * group_size
        ef = max(self.ef_coefficient_c * entity_budget, self.ef_floor)
        # 这条断言是派生式自带的约束：ef 小于 entity 预算时 ANN 层根本凑不出这么多候选。
        assert ef >= entity_budget, f"ef({ef}) 必须 ≥ group_limit×group_size({entity_budget})"
        return ef
