"""全局 fixture。

一条纪律高于一切（全局指令）：**测试不复制也不重写业务逻辑**。这里只提供输入
（样例 chunk、假 Milvus 响应、被覆写的 Settings），断言写在各测试文件里，
业务路径必须完全复用生产代码。

样例数据刻意做成**两种粒度都有**：fulltext 与 abstract 的字段可空性不同
（摘要块的 section / sec_number / paragraph_idx 全为 None），只用一种粒度做测试，
引用锚点那条链路上的空值分支就永远测不到。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from citequest.config.settings import Settings, get_settings
from citequest.core.corpus import Attribution, Chunk, Paper
from citequest.core.enums import Granularity, LicenseId, SecType, SourceDataset
from citequest.core.search import RetrievedChunk


@pytest.fixture(autouse=True)
def _isolate_settings_cache() -> Iterator[None]:
    """每条用例前后都清掉 Settings 的进程内缓存。

    `get_settings` 用 lru_cache 做单例，不清理的话前一条用例通过 monkeypatch
    设的环境变量会泄漏到后面——表现为「单跑通过、全量跑失败」，是最难查的一类
    测试污染。
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """一份纯默认值的 Settings。

    **默认值本身就是被测对象**：决策记录 §4 的参数正本落在这些 default 上，
    所以不给它塞测试专用的值，否则「参数是否等于设计文档里那个数」这件事就没人验了。
    """
    return Settings()


@pytest.fixture
def attribution_fulltext() -> Attribution:
    """全文层署名。在线返回检索片段属于许可意义上的分发，必须带署名。"""
    return Attribution(
        text="unarXive open subset (Saier et al., JCDL 2023), CC BY-SA 4.0",
        source_url="https://doi.org/10.5281/zenodo.7752615",
        original_url="https://arxiv.org/abs/2103.00001",
    )


@pytest.fixture
def attribution_abstract() -> Attribution:
    """摘要层署名。CC0 与 CC-BY-SA-4.0 混在同一个响应里，所以必须逐条标注。"""
    return Attribution(
        text="OpenAlex, CC0",
        source_url="https://openalex.org",
        original_url=None,
    )


@pytest.fixture
def fulltext_chunk() -> Chunk:
    """全文层样例块：四元组引用锚点齐全。"""
    return Chunk(
        chunk_id="W2000000001:fulltext:s4:p12",
        paper_id="W2000000001",
        granularity=Granularity.FULLTEXT,
        text=(
            "We evaluate rotary position embedding against learned absolute positions "
            "on three long-context benchmarks, holding the training budget fixed."
        ),
        section="Experimental Setup",
        sec_number="4.2",
        sec_type=SecType.EXPERIMENT,
        paragraph_idx=12,
        year=2021,
        primary_category="cs.CL",
        license=LicenseId.CC_BY_SA_4_0,
        source_dataset=SourceDataset.UNARXIVE_OPEN_SUBSET,
        token_count=438,
        cited_open_alex_ids=["W1900000007"],
    )


@pytest.fixture
def abstract_chunk() -> Chunk:
    """摘要层样例块：section 三元全为 None——这不是缺陷，是粒度本身的诚实表达。"""
    return Chunk(
        chunk_id="W3000000002:abstract:s0:p0",
        paper_id="W3000000002",
        granularity=Granularity.ABSTRACT,
        text=(
            "We present a retrieval-augmented system for scientific literature search "
            "that grounds every generated sentence in cited passages."
        ),
        section=None,
        sec_number=None,
        sec_type=None,
        paragraph_idx=None,
        year=2024,
        primary_category="cs.IR",
        license=LicenseId.CC0,
        source_dataset=SourceDataset.OPENALEX,
        token_count=96,
    )


@pytest.fixture
def fulltext_paper(attribution_fulltext: Attribution) -> Paper:
    return Paper(
        paper_id="W2000000001",
        title="On Positional Encodings for Long-Context Transformers",
        year=2021,
        primary_category="cs.CL",
        authors=["Ada Lovelace", "Alan Turing"],
        venue="ACL",
        doi="10.18653/v1/2021.acl-long.1",
        arxiv_id="2103.00001",
        cited_by_count=421,
        granularity=Granularity.FULLTEXT,
        has_fulltext=True,
        license=LicenseId.CC_BY_SA_4_0,
        source_dataset=SourceDataset.UNARXIVE_OPEN_SUBSET,
        attribution=attribution_fulltext,
    )


@pytest.fixture
def abstract_paper(attribution_abstract: Attribution) -> Paper:
    return Paper(
        paper_id="W3000000002",
        title="CiteQuest: Grounded Retrieval for Scientific Literature",
        year=2024,
        primary_category="cs.IR",
        authors=["Grace Hopper"],
        venue=None,
        doi=None,
        arxiv_id="2403.00002",
        cited_by_count=7,
        granularity=Granularity.ABSTRACT,
        has_fulltext=False,
        license=LicenseId.CC0,
        source_dataset=SourceDataset.OPENALEX,
        attribution=attribution_abstract,
    )


@pytest.fixture
def fake_hybrid_search_hits() -> list[dict[str, object]]:
    """一份形状正确的 Milvus grouping hybrid_search 返回。

    形状照 pymilvus 实际返回的 `[{"id":..., "distance":..., "entity": {...}}]`，
    **不是**自己发明的结构——假数据的形状错了，基于它写的解析代码就是错的，
    而且这类错误会一直等到接真实 Milvus 才暴露。

    分组是刻意做成不均匀的（第一组 2 条、第二组 1 条）：`strict_group_size=False`
    的语义就是允许组内不满，用均匀假数据测不出补齐逻辑里的差一错误。
    """
    return [
        {
            "id": "W2000000001:fulltext:s4:p12",
            "distance": 0.0328,
            "entity": {
                "paper_id": "W2000000001",
                "granularity": "fulltext",
                "year": 2021,
                "primary_category": "cs.CL",
                "license": "CC-BY-SA-4.0",
            },
        },
        {
            "id": "W2000000001:fulltext:s2:p03",
            "distance": 0.0161,
            "entity": {
                "paper_id": "W2000000001",
                "granularity": "fulltext",
                "year": 2021,
                "primary_category": "cs.CL",
                "license": "CC-BY-SA-4.0",
            },
        },
        {
            "id": "W3000000002:abstract:s0:p0",
            "distance": 0.0159,
            "entity": {
                "paper_id": "W3000000002",
                "granularity": "abstract",
                "year": 2024,
                "primary_category": "cs.IR",
                "license": "CC0",
            },
        },
    ]


@pytest.fixture
def retrieved_fulltext_chunk(fulltext_chunk: Chunk) -> RetrievedChunk:
    """一条**已召回**的全文块（带两路名次与 RRF 分）。

    与 `fulltext_chunk` 分开而不是复用：`Chunk` 是灌库前的数据契约，
    `RetrievedChunk` 是检索期的契约，两者字段不同**是刻意的**——
    rrf_score / rerank_score / 两路名次只在检索期存在，把它们塞进 Chunk
    就等于让灌库侧多背四个永远为空的字段。
    """
    return RetrievedChunk(
        chunk_id=fulltext_chunk.chunk_id,
        paper_id=fulltext_chunk.paper_id,
        text=fulltext_chunk.text,
        granularity=fulltext_chunk.granularity,
        section=fulltext_chunk.section,
        sec_number=fulltext_chunk.sec_number,
        sec_type=fulltext_chunk.sec_type,
        paragraph_idx=fulltext_chunk.paragraph_idx,
        year=fulltext_chunk.year,
        primary_category=fulltext_chunk.primary_category,
        license=fulltext_chunk.license,
        source_dataset=fulltext_chunk.source_dataset,
        rrf_score=0.0328,
        rerank_score=0.91,
        dense_rank=1,
        sparse_rank=4,
    )


@pytest.fixture
def retrieved_abstract_chunk(abstract_chunk: Chunk) -> RetrievedChunk:
    """一条已召回的摘要块：**只被 dense 一路命中**（sparse_rank 为 None）。

    单路命中是常态而不是异常，`badcase` 的融合错判定正是靠这两个字段的对比——
    两路都填满的假数据会让那条分支永远测不到。
    """
    return RetrievedChunk(
        chunk_id=abstract_chunk.chunk_id,
        paper_id=abstract_chunk.paper_id,
        text=abstract_chunk.text,
        granularity=abstract_chunk.granularity,
        year=abstract_chunk.year,
        primary_category=abstract_chunk.primary_category,
        license=abstract_chunk.license,
        source_dataset=abstract_chunk.source_dataset,
        rrf_score=0.0159,
        rerank_score=0.42,
        dense_rank=3,
    )
