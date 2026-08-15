"""融合与 grouping 解析（M2 §12、决策记录 D6/D9）。

融合本身发生在**服务端**（Milvus 的 `RRFRanker(k=60)`），所以客户端侧可被断言的
是三件事，而这三件恰好是最容易写错的：

1. **请求装配**——两路 `AnnSearchRequest` 的参数、ef 的派生、grouping 的三个 kwarg；
2. **结果解析**——`rows → 组 → PaperHit`，包括组内不满与脏行的处置；
3. **多路合并**——HyDE 的两路按 chunk_id 去重、保留更好的那一条。

两条不可协商的约束在这里被守住：
**融合器唯一 RRFRanker(k=60)**（没有 WeightedRanker、没有权重字段）；
**RRF 分数无物理含义**，只允许留在内部字段 `rrf_score` 上，禁止做相关性阈值。
"""

from __future__ import annotations

import pytest

from citequest.config.settings import Settings
from citequest.core.enums import Granularity, LicenseId, RetrievalMode, SourceDataset
from citequest.core.search import RetrievedChunk
from citequest.indexing.ann_requests import GroupingSpec, build_ann_requests, build_ranker, grouping_kwargs
from citequest.retrieval.grouping import parse_groups, to_paper_hits
from citequest.retrieval.hybrid import merge_by_best_rank


@pytest.fixture
def settings() -> Settings:
    return Settings()


def _chunk(chunk_id: str, paper_id: str, rrf: float | None, **kwargs: object) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text="some retrieved text",
        granularity=Granularity.FULLTEXT,
        year=2021,
        license=LicenseId.CC_BY_SA_4_0,
        source_dataset=SourceDataset.UNARXIVE_OPEN_SUBSET,
        rrf_score=rrf,
        **kwargs,  # type: ignore[arg-type]
    )


class TestRankerAndRequests:
    """请求装配：RRF 是唯一融合器，ef 必须按 entity 预算派生。"""

    def test_ranker_is_rrf_with_k_60(self, settings: Settings) -> None:
        ranker = build_ranker(settings.retrieval)
        # pymilvus 的 RRFRanker 把 k 存在 `_k`；断言类型 + 参数，
        # 是因为「换成 WeightedRanker」这个改动本身不会报错，只会让排序悄悄变。
        assert type(ranker).__name__ == "RRFRanker"
        assert getattr(ranker, "_k", None) == settings.retrieval.rrf_k == 60

    def test_hybrid_mode_emits_two_channels(self, settings: Settings) -> None:
        spec = GroupingSpec.for_rag(settings.retrieval)
        requests = build_ann_requests("attention mechanisms", settings.milvus, spec, mode=RetrievalMode.HYBRID)
        assert len(requests) == 2

    def test_single_channel_modes_emit_one_request(self, settings: Settings) -> None:
        # 单路下 RRF 退化为恒等变换，**没有必要为它写第二条代码路径**——
        # 但请求数必须真的只有一条，否则「dense_only」只是个标签。
        spec = GroupingSpec.for_rag(settings.retrieval)
        for mode in (RetrievalMode.DENSE_ONLY, RetrievalMode.SPARSE_ONLY):
            assert len(build_ann_requests("查询", settings.milvus, spec, mode=mode)) == 1

    def test_grouping_kwargs_carry_all_three_parameters(self, settings: Settings) -> None:
        # 三个 kwarg 只能在顶层传（AnnSearchRequest 不支持它们）。少传
        # group_by_field 的表现是「结果变多了」而不是报错。
        kwargs = grouping_kwargs(GroupingSpec.for_search(settings.retrieval), settings.retrieval)
        assert kwargs["group_by_field"] == "paper_id"
        assert kwargs["group_size"] == 1
        assert kwargs["strict_group_size"] is False


class TestParseGroups:
    """结果解析：**本层不重排**，Milvus 已经在服务端排好序了。"""

    def test_rows_are_collapsed_by_paper_in_first_seen_order(
        self, fake_hybrid_search_hits: list[dict[str, object]], settings: Settings
    ) -> None:
        parsed = parse_groups(
            _with_text(fake_hybrid_search_hits, settings.milvus.text_en_field),
            text_field=settings.milvus.text_en_field,
            requested_groups=16,
        )
        assert parsed.group_count == 2
        # 第一组 2 条、第二组 1 条：`strict_group_size=False` 的语义就是允许组内不满。
        assert [len(g) for g in parsed.groups] == [2, 1]

    def test_fill_ratio_is_the_degrade_signal(
        self, fake_hybrid_search_hits: list[dict[str, object]], settings: Settings
    ) -> None:
        # 实际组数 / 请求组数 < 1 就是 M3 §3.5 档 1 降级的触发信号，
        # **不需要再拍一个「少于 8 组」的阈值**——凑不满预算本身就是事实。
        parsed = parse_groups(
            _with_text(fake_hybrid_search_hits, settings.milvus.text_en_field),
            text_field=settings.milvus.text_en_field,
            requested_groups=16,
        )
        assert parsed.fill_ratio == pytest.approx(2 / 16)

    def test_incomplete_row_is_dropped_not_raised(self, settings: Settings) -> None:
        # 一条 chunk 的字段写错只是少一条候选，而抛异常会让整个请求 500——
        # 检索是底线，它没有降级对象。
        rows = _with_text(
            [
                {
                    "id": "ok",
                    "distance": 0.03,
                    "entity": {
                        "chunk_id": "c1",
                        "paper_id": "W1",
                        "year": 2021,
                        "granularity": "fulltext",
                        "license": "CC-BY-SA-4.0",
                        "source_dataset": "unarXive_open_subset",
                    },
                },
                {
                    "id": "bad",
                    "distance": 0.02,
                    "entity": {
                        "chunk_id": "c2",
                        "paper_id": "W2",
                        "year": 2021,
                        "granularity": "not_a_granularity",
                        "license": "CC0",
                        "source_dataset": "openalex",
                    },
                },
            ],
            settings.milvus.text_en_field,
        )
        parsed = parse_groups(rows, text_field=settings.milvus.text_en_field, requested_groups=16)
        assert parsed.group_count == 1

    def test_rrf_score_stays_internal(
        self, fake_hybrid_search_hits: list[dict[str, object]], settings: Settings
    ) -> None:
        # distance 在 hybrid_search 下就是 RRF 分数：它进内部字段，
        # 而 PaperHit 上**没有** relevance_score 字段——不是忘了给，是不能给。
        parsed = parse_groups(
            _with_text(fake_hybrid_search_hits, settings.milvus.text_en_field),
            text_field=settings.milvus.text_en_field,
            requested_groups=16,
        )
        assert parsed.chunks[0].rrf_score == pytest.approx(0.0328)
        hits = to_paper_hits(parsed)
        assert "relevance_score" not in hits[0].model_dump()

    def test_paper_hit_falls_back_to_id_when_title_is_unknown(
        self, fake_hybrid_search_hits: list[dict[str, object]], settings: Settings
    ) -> None:
        # **宁可显示 ID 也不显示一个编出来的标题。**
        parsed = parse_groups(
            _with_text(fake_hybrid_search_hits, settings.milvus.text_en_field),
            text_field=settings.milvus.text_en_field,
            requested_groups=16,
        )
        hits = to_paper_hits(parsed)
        assert hits[0].title == hits[0].paper_id
        assert hits[0].has_fulltext is (hits[0].granularity is Granularity.FULLTEXT)


class TestMultiChannelMerge:
    """HyDE 的两路合并：按 chunk_id 去重、**保留 RRF 分更高的那条**。"""

    def test_keeps_the_better_ranked_copy(self) -> None:
        # 丢掉这个信息等于让 HyDE 的收益在去重时被抹掉。
        existing = _chunk("c1", "W1", 0.010)
        incoming = _chunk("c1", "W1", 0.031)
        assert merge_by_best_rank(existing, incoming) is incoming
        assert merge_by_best_rank(incoming, existing) is incoming

    def test_first_seen_wins_when_scores_tie(self) -> None:
        # 打平时保留先到的那条：原 query 恒在首位，它的名次优先——
        # 原 query 那一路存在的意义正是当锚点。
        existing = _chunk("c1", "W1", 0.02)
        incoming = _chunk("c1", "W1", 0.02)
        assert merge_by_best_rank(existing, incoming) is existing

    def test_missing_score_is_treated_as_worst(self) -> None:
        assert merge_by_best_rank(None, _chunk("c1", "W1", None)).chunk_id == "c1"
        assert merge_by_best_rank(_chunk("c1", "W1", None), _chunk("c1", "W1", 0.001)).rrf_score == 0.001


def _with_text(rows: list[dict[str, object]], text_field: str) -> list[dict[str, object]]:
    """给假 Milvus 行补上文本字段与 chunk_id。

    fixture 里的假数据形状照 pymilvus 实际返回构造，但 `parse_groups` 需要
    `chunk_id` 与文本字段——在测试里补齐而不是改 fixture，是因为那份 fixture
    同时被 e2e 用，两处需要的字段集不同。
    """
    out = []
    for row in rows:
        entity = dict(row.get("entity", {}))  # type: ignore[arg-type]
        entity.setdefault("chunk_id", row.get("id"))
        entity.setdefault(text_field, "retrieved passage text")
        entity.setdefault("source_dataset", "unarXive_open_subset" if entity.get("license") != "CC0" else "openalex")
        out.append({**row, "entity": entity})
    return out
