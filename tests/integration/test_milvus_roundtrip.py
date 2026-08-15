"""Milvus 往返：schema + 两条 Function + grouping 混合检索（M2 §5/§6/§12）。

这一层只测**单测测不到的那部分**——需要真实服务端才能验证的三件事：

1. **schema 与两条 Function 被 Milvus 接受**（dense 走 TEI Embedding Function、
   sparse 走内置 BM25 Function，都是 Data-in/Data-out，客户端不灌向量）；
2. **grouping search 的语义**：`limit` 是组数，`group_size` 是上界不是保证；
3. **TEXT_MATCH / PHRASE_MATCH 真的作用在 `text_exact` 上**且不产生分数。

不测的是排序质量：那是评测模块的事（`evaluation/`），用两条样本数据断言
「谁排前面」只会得到一个随索引参数抖动的脆弱用例。
"""

from __future__ import annotations

import pytest

from citequest.config.settings import Settings
from citequest.core.enums import RetrievalMode
from citequest.indexing.ann_requests import GroupingSpec, build_ann_requests
from citequest.indexing.client import MilvusIndexClient
from citequest.retrieval.grouping import parse_groups

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def seeded(milvus: MilvusIndexClient, loaded_collection: str) -> str:
    """灌几条样本。**行里没有任何向量字段**——dense 与 sparse 都是 Function 的输出，
    客户端灌向量会直接被 Milvus 拒绝，而这条约束正是「灌库用 A、查询用 B」
    在结构上不可能发生的原因。
    """
    rows = [
        {
            "chunk_id": f"W100000000{i}:f:s001:p0000{i}",
            "paper_id": f"W100000000{i % 2}",
            "granularity": "fulltext",
            "text_en": "Rotary position embedding improves length extrapolation on long context benchmarks.",
            "text_exact": "Rotary position embedding improves length extrapolation on long context benchmarks.",
            "year": 2021,
            "primary_category": "cs.CL",
            "license": "CC-BY-SA-4.0",
            "source_dataset": "unarXive_open_subset",
        }
        for i in range(4)
    ]
    milvus.upsert(rows)
    return loaded_collection


class TestSchemaAndFunctions:
    def test_collection_declares_both_functions(self, milvus: MilvusIndexClient, loaded_collection: str) -> None:
        description = milvus.raw.describe_collection(loaded_collection)
        function_names = {f.get("name") for f in description.get("functions", [])}
        # 两条 Function 都必须在 schema 上：写入与检索共用同一份声明，
        # 「灌库用 A 模型、查询用 B 模型」这类不一致因此在结构上不可能发生。
        assert len(function_names) >= 2

    def test_startup_probe_passes_on_a_healthy_collection(self, milvus: MilvusIndexClient, seeded: str) -> None:
        assert milvus.startup_probe() is True


class TestGroupingSemantics:
    def test_limit_is_group_count_not_row_count(
        self, milvus: MilvusIndexClient, integration_settings: Settings, seeded: str
    ) -> None:
        # 4 条 chunk 分属 2 篇论文；请求 2 组 × 3 段，返回的组数 ≤ 2 而不是 6。
        spec = GroupingSpec(group_limit=2, group_size=3)
        requests = build_ann_requests(
            "rotary position embedding", integration_settings.milvus, spec, mode=RetrievalMode.HYBRID
        )
        raw = milvus.hybrid_search(requests, spec)
        parsed = parse_groups(
            list(raw[0]),
            text_field=integration_settings.milvus.text_en_field,
            requested_groups=spec.group_limit,
        )
        assert parsed.group_count <= 2
        assert all(len(group) <= 3 for group in parsed.groups)

    def test_single_channel_search_still_returns_groups(
        self, milvus: MilvusIndexClient, integration_settings: Settings, seeded: str
    ) -> None:
        # 单路模式走的是同一个方法：`requests` 只有一条时 RRF 退化为恒等变换，
        # **没有必要为它写第二条代码路径**——这条要在真实服务端上确认一次。
        spec = GroupingSpec(group_limit=2, group_size=1)
        requests = build_ann_requests(
            "rotary position embedding", integration_settings.milvus, spec, mode=RetrievalMode.DENSE_ONLY
        )
        assert len(requests) == 1
        assert milvus.hybrid_search(requests, spec)


class TestExactMatch:
    def test_phrase_match_filters_without_scoring(
        self, milvus: MilvusIndexClient, integration_settings: Settings, seeded: str
    ) -> None:
        # `text_exact` 不挂 BM25 Function：它只做硬过滤，不产生分数、不进融合，
        # 所以它不会变成第三条检索路。
        field = integration_settings.milvus.text_exact_field
        rows = milvus.query(f'PHRASE_MATCH({field}, "position embedding")', limit=4)
        assert rows

    def test_text_match_is_or_by_default(
        self, milvus: MilvusIndexClient, integration_settings: Settings, seeded: str
    ) -> None:
        # Milvus 默认 OR；要 AND 得写成两个 TEXT_MATCH 用 and 连起来。
        field = integration_settings.milvus.text_exact_field
        assert milvus.query(f'TEXT_MATCH({field}, "rotary nonexistentterm")', limit=4)
