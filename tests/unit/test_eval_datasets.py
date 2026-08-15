"""评测集构造：库内过滤、过滤率、防泄漏、退场阈值（M7 §2、§3）。

**头条基准最大的单点风险在这里被程序化**：过滤到库内之后样本量未知，低于 3,000
条时头条退回 LitSearch。这条预案必须是一个会被真的执行的函数，而不是一句文档。

防泄漏也在这里守：marker 邻域必须删。它与索引侧「占位符删除而不是替换成被引标题」
是同一个决定的两面——少做任何一条，模型都能从输入里直接读出答案。
"""

from __future__ import annotations

from typing import Any

import pytest

from citequest.config.eval import EvalSettings
from citequest.core.enums import Granularity
from citequest.evaluation.datasets.citrec import (
    build_in_corpus_subset,
    headline_survives,
    normalize_openalex_id,
    strip_marker_neighborhood,
)
from citequest.evaluation.datasets.litsearch import bm25_sanity_check, build_litsearch_dataset


def _record(rid: str, label: str | None, text: str = "A passage that cites something.") -> dict[str, Any]:
    return {"_id": rid, "text": text, "marker": "[1]", "marker_offsets": [], "label": label}


class TestOpenAlexIdNormalization:
    """前缀与大小写不统一会让过滤率**凭空掉到接近零**，而它看起来像库覆盖率太低。"""

    @pytest.mark.parametrize(
        "raw", ["W2000000001", "https://openalex.org/W2000000001", "w2000000001", "openalex:W2000000001"]
    )
    def test_all_forms_normalize_to_the_bare_id(self, raw: str) -> None:
        assert normalize_openalex_id(raw) == "W2000000001"

    def test_unusable_label_returns_none_instead_of_raising(self) -> None:
        # 22.5 万条里出现少量脏 label 是常态；为一条脏数据中断整次构造没有好处。
        assert normalize_openalex_id(None) is None
        assert normalize_openalex_id("not-an-id") is None


class TestMarkerLeakage:
    """防泄漏第一条：删 marker 及 `marker_offsets` 邻域。"""

    def test_offsets_are_stripped(self) -> None:
        text = "RoPE was introduced by [CITE] and extended later."
        stripped = strip_marker_neighborhood(text, [(23, 29)])
        assert "[CITE]" not in stripped
        assert "RoPE was introduced by" in stripped

    def test_multiple_spans_do_not_shift_each_other(self) -> None:
        # 正序删除会让后一个区间的偏移全部失效，而偏移失效的表现是「删错了位置」
        # 而不是报错——一类会静默污染整个评测集的 bug。
        text = "AAA [X] BBB [Y] CCC"
        stripped = strip_marker_neighborhood(text, [(4, 7), (12, 15)])
        assert "[X]" not in stripped and "[Y]" not in stripped
        assert "AAA" in stripped and "BBB" in stripped and "CCC" in stripped

    def test_window_widens_the_cut(self) -> None:
        # 窗口开大会同时删掉真正的检索信号，所以默认 0；放宽是要写进报告的选择。
        text = "aaaa bbb [X] ccc dddd"
        assert "bbb" not in strip_marker_neighborhood(text, [(9, 12)], window=4)


class TestInCorpusSubset:
    """过滤率是**必报数字**，不是可以藏起来的中间步骤。"""

    def test_only_in_corpus_labels_survive(self) -> None:
        records = [_record("a", "W1"), _record("b", "W999"), _record("c", "W2")]
        dataset = build_in_corpus_subset(records, {"W1", "W2", "W3"}, EvalSettings())
        assert [q.query_id for q in dataset.queries] == ["a", "c"]
        assert dataset.qrels["a"] == {"W1": 1}

    def test_filter_rate_is_reported(self) -> None:
        records = [_record(str(i), "W1" if i < 3 else "W999") for i in range(10)]
        dataset = build_in_corpus_subset(records, {"W1"}, EvalSettings())
        stats = dataset.subset_stats
        assert stats is not None
        assert stats.total_records == 10
        assert stats.kept_records == 3
        assert stats.filter_rate == pytest.approx(0.3)
        assert stats.dropped_out_of_corpus == 7

    def test_queries_are_tagged_as_fulltext_layer(self) -> None:
        # citrec 是段落级的，只测得到全文层。粒度标在 query 上，
        # 让「分粒度报」在数据结构上就无法被绕过。
        dataset = build_in_corpus_subset([_record("a", "W1")], {"W1"}, EvalSettings())
        assert dataset.queries[0].layer is Granularity.FULLTEXT

    def test_empty_text_after_stripping_is_dropped(self) -> None:
        # 删完为空的段落不是「难样本」，是没有输入的样本——留着它等于给系统记一个
        # 必然的 miss，把指标压低一点点却说不清为什么。
        record = {"_id": "a", "text": "[X]", "marker_offsets": [[0, 3]], "label": "W1"}
        dataset = build_in_corpus_subset([record], {"W1"}, EvalSettings())
        assert dataset.queries == []

    def test_provenance_names_the_label_source(self) -> None:
        # 「谁给的标签」这一栏是标注池与被评系统是否解耦的唯一凭据。
        dataset = build_in_corpus_subset([_record("a", "W1")], {"W1"}, EvalSettings())
        assert "作者" in dataset.provenance.label_source


class TestHeadlineRetirement:
    """最大单点风险的预案：**低于 3,000 条就退场，不硬撑**。"""

    def test_small_subset_retires_the_headline_benchmark(self) -> None:
        records = [_record(str(i), "W1") for i in range(10)]
        dataset = build_in_corpus_subset(records, {"W1"}, EvalSettings())
        assert headline_survives(dataset, EvalSettings()) is False

    def test_threshold_comes_from_settings_not_from_a_literal(self) -> None:
        records = [_record(str(i), "W1") for i in range(10)]
        dataset = build_in_corpus_subset(records, {"W1"}, EvalSettings())
        assert headline_survives(dataset, EvalSettings(citrec_min_subset_size=5)) is True


class TestLitSearch:
    def test_queries_are_tagged_as_abstract_layer(self) -> None:
        records = [
            {"query": "how to search literature", "corpusids": [11, 22], "query_set": "inline", "specificity": 1}
        ]
        dataset = build_litsearch_dataset(records)
        assert dataset.queries[0].layer is Granularity.ABSTRACT
        assert dataset.qrels[dataset.queries[0].query_id] == {"11": 1, "22": 1}

    def test_query_without_qrels_is_skipped(self) -> None:
        # 没有 qrels 的 query 进评测就是一个恒定的 0 分，会把均值无缘无故拉低。
        assert build_litsearch_dataset([{"query": "orphan", "corpusids": []}]).queries == []

    def test_precision_limitation_is_carried_with_the_dataset(self) -> None:
        # 局限必须与数字同时讲，所以它挂在数据集上而不是写在某份文档里。
        dataset = build_litsearch_dataset([{"query": "q", "corpusids": [1]}])
        assert any("Precision" in note for note in dataset.provenance.notes)


class TestBm25SanityCheck:
    """**它叫 sanity check，不叫 reproduction**——三套 BM25 的分词器不同。"""

    def test_in_window_passes(self) -> None:
        result = bm25_sanity_check(41.0, 73.0, EvalSettings())
        assert result.passed is True
        assert result.wording == "sanity_check"

    def test_out_of_window_fails_and_keeps_the_reference(self) -> None:
        result = bm25_sanity_check(30.0, 73.0, EvalSettings())
        assert result.passed is False
        assert result.ndcg10_within_window is False
        # 参考值随结果一起带出去：它是「部分核实（单一来源）」，引用时必须说明。
        assert result.ndcg10_reference == 40.2
