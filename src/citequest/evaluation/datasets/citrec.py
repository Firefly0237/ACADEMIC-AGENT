"""头条基准：unarXive_citrec 库内子集（段落级，M7 §2）。

任务形态：给一段真实论文正文（其中某处引用了一篇文献），检索出**被引的那篇论文**。
label 是被引论文的 OpenAlex ID，**标签不是我给的，是论文作者给的**——这是这个
基准唯一但足够致命的优势（M7 §2.2）。

本模块只做三件事，每一件都对应一条必须主动讲出来的口径：

| 步 | 做什么 | 不做会怎样 |
|---|---|---|
| ① 库内过滤 | `label ∈ 本库 paper_id 集合` | 绝大多数被引文献不在库内，指标没有意义 |
| ② 过滤率统计 | 报 `SubsetStats.filter_rate` | 变成「我挑了个对我有利的子集」 |
| ③ 防泄漏 | 删 `marker` 及 `marker_offsets` 邻域 | 引用标记本身携带被引文献线索 |

**第 ③ 条只是防泄漏的一半**，另一半在索引侧：`{{cite:uuid}}` 占位符一律删除而
不是替换成被引论文标题（`ingestion.placeholders`）。若替换成标题，label 的标题
就直接躺在被检索文本里，**整个基准立刻作废**。两条是同一个决定的两面——
先定基准、再定数据处理策略，而不是先做完再找一个能证明自己的基准。

**最大单点风险写在 `headline_survives`**：过滤后不足 `citrec_min_subset_size`
（3,000）条时，bootstrap 95% CI 会宽到任何消融结论都不显著，头条退回 LitSearch。
这个判定是一个函数而不是一句文档，因为它必须在跑分前被真的执行一次。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import Any

from citequest.config.eval import EvalSettings
from citequest.core.enums import Granularity
from citequest.evaluation.datasets.schemas import (
    RELEVANT,
    EvalDataset,
    EvalQuery,
    Provenance,
    SubsetStats,
)
from citequest.observability.logging import get_logger

__all__ = [
    "CITREC_PROVENANCE",
    "build_in_corpus_subset",
    "headline_survives",
    "load_citrec_records",
    "normalize_openalex_id",
    "strip_marker_neighborhood",
]

_logger = get_logger(__name__)

#: OpenAlex ID 的两种常见写法：裸 `W2000000001` 与 URL 形态。库内 `paper_id`
#: 统一为去前缀形式（`core.corpus.Paper`），过滤前必须先归一化——**大小写与前缀
#: 不统一会让过滤率凭空掉到接近零，而它看起来像是「库覆盖率太低」**。
_OPENALEX_RE = re.compile(r"(W\d+)", re.IGNORECASE)

CITREC_PROVENANCE = Provenance(
    dataset_id="saier/unarXive_citrec",
    split="test",
    license="CC-BY-SA-4.0",
    label_source="论文原作者的真实引用行为（作者在写这一段时确实引了这篇）",
    citation="arXiv:2303.14957 (JCDL 2023)",
    notes=[
        "指标是在**库内子集**上算的，过滤率见 subset_stats.filter_rate",
        "每条 query 只有 1 个 label，Recall@100 在这个基准上等价于 Hit@100 / Success@100",
        "评测管线把 group_limit 加大到 100（线上搜索路是 60），这是评测与线上唯一允许的差异",
    ],
)


def normalize_openalex_id(raw: Any) -> str | None:
    """把任意形态的 OpenAlex ID 归一化成库内主键形态（`W` + 数字）。

    返回 None 表示这条 label 不可用。**不抛异常**：22.5 万条 test 里出现少量脏
    label 是常态，为一条脏数据中断整次评测集构造没有任何好处，计数进
    `dropped_no_label` 才是正确处置。
    """
    if raw is None:
        return None
    match = _OPENALEX_RE.search(str(raw))
    return match.group(1).upper() if match else None


def strip_marker_neighborhood(text: str, offsets: Iterable[tuple[int, int]], *, window: int = 0) -> str:
    """删除 `marker` 及 `marker_offsets` 邻域内容（防泄漏第一条，M7 §2.3）。

    :param offsets: `[(start, end), ...]`，来自数据集的 `marker_offsets` 字段。
    :param window: 向两侧各多删的字符数。默认 0 即只删标记本身；引用标记周围
        常有 "as shown by" 这类引导语，需要更保守时把窗口开大——**但窗口开大会
        同时删掉真正的检索信号**，所以默认取 0，放宽是一个要写进报告的选择。

    实现上按区间倒序删除：正序删会让后一个区间的偏移全部失效，而偏移失效的表现
    是「删错了位置」而不是报错——一类会静默污染整个评测集的 bug。
    """
    spans = sorted({(max(0, s - window), min(len(text), e + window)) for s, e in offsets}, reverse=True)
    out = text
    for start, end in spans:
        if start >= end:
            continue
        out = out[:start] + " " + out[end:]
    return re.sub(r"\s+", " ", out).strip()


def load_citrec_records(settings: EvalSettings) -> Iterator[dict[str, Any]]:
    """从 HuggingFace 拉 citrec 的 test split。

    **接真实实现**：正式实现是
    ``datasets.load_dataset(settings.citrec_dataset_id, split=settings.citrec_split)``
    然后逐条 yield（`_id` / `text` / `marker` / `marker_offsets` / `label`）。
    这里不预先 `list()`：test split 22.5 万条，全部读进内存只是为了过滤掉其中
    四分之三，没有必要——库内过滤是一个流式可做的操作。

    离线环境下 `datasets` 属于 eval 依赖组（`uv sync --group eval`），未安装时
    直接把这个事实抛出来，而不是返回一个空迭代器——**空数据集会让评测「成功地
    跑出 0 条样本」，那比报错难查得多**。
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise NotImplementedError(
            "接真实实现：需要 eval 依赖组（uv sync --group eval）才能加载 saier/unarXive_citrec"
        ) from exc

    dataset = load_dataset(settings.citrec_dataset_id, split=settings.citrec_split)
    _logger.info("citrec_loaded", dataset=settings.citrec_dataset_id, split=settings.citrec_split, rows=len(dataset))
    yield from dataset


def build_in_corpus_subset(
    records: Iterable[dict[str, Any]],
    corpus_paper_ids: set[str],
    settings: EvalSettings,
) -> EvalDataset:
    """构造库内子集：过滤 + 防泄漏 + 过滤率统计，一次遍历完成。

    :param corpus_paper_ids: 本库全部 paper_id（去前缀 OpenAlex ID）。由建库产物
        提供——**必须与在线检索用的是同一份索引**，否则过滤率就不是库内引用边
        覆盖率，而是一个无法解释的数。
    :param settings: `EvalSettings`，防泄漏开关与退场阈值都在里面。

    三条实现细节各自对应一个会静默出错的地方：

    1. **label 与库 ID 都归一化后再比**，否则前缀差异会让过滤率塌到接近零；
    2. **过滤后的 query 文本为空则丢弃**——marker 邻域删完只剩空白的段落进检索
       会让 BM25 产出空 sparse 向量并静默返回随机结果；
    3. **query_id 用数据集自带的 `_id`**，不自己编号：报告要能被别人按 id 复查。
    """
    normalized_corpus = {normalize_openalex_id(pid) or "" for pid in corpus_paper_ids}
    normalized_corpus.discard("")

    queries: list[EvalQuery] = []
    qrels: dict[str, dict[str, int]] = {}
    total = 0
    no_label = 0
    out_of_corpus = 0

    for record in records:
        total += 1
        label = normalize_openalex_id(record.get("label"))
        if label is None:
            no_label += 1
            continue
        if label not in normalized_corpus:
            out_of_corpus += 1
            continue

        text = str(record.get("text") or "")
        if settings.citrec_strip_marker_neighborhood:
            text = strip_marker_neighborhood(text, _marker_offsets(record))
        if not text:
            # 删完为空的段落不是「难样本」，是没有输入的样本。留着它等于给系统
            # 记一个必然的 miss，把指标压低一点点却说不清为什么。
            no_label += 1
            continue

        query_id = str(record.get("_id") or f"citrec-{total}")
        # citrec 的 query 是论文正文段落，只测得到全文层——这就是它与 LitSearch
        # 分工的全部理由，标在 query 上让「分粒度报」无法被绕过。
        queries.append(EvalQuery(query_id=query_id, text=text, layer=Granularity.FULLTEXT))
        qrels[query_id] = {label: RELEVANT}

    stats = SubsetStats(
        total_records=total,
        kept_records=len(queries),
        dropped_no_label=no_label,
        dropped_out_of_corpus=out_of_corpus,
        corpus_size=len(normalized_corpus),
    )
    _logger.info(
        "citrec_subset_built",
        total=total,
        kept=stats.kept_records,
        # 这个数字是必报项（M7 §2.3）：它就是库内引用边覆盖率的重测。
        filter_rate=round(stats.filter_rate, 4),
        min_required=settings.citrec_min_subset_size,
        survives=stats.kept_records >= settings.citrec_min_subset_size,
    )
    return EvalDataset(
        name="unarXive_citrec@in_corpus",
        queries=queries,
        qrels=qrels,
        provenance=CITREC_PROVENANCE,
        subset_stats=stats,
    )


def _marker_offsets(record: dict[str, Any]) -> list[tuple[int, int]]:
    """把数据集里的 `marker_offsets` 规整成 `(start, end)` 列表。

    上游给的是嵌套列表（一条段落可能有多个引用标记），偶尔会出现长度不为 2 的
    项。跳过这类项而不是报错：单条脏偏移的后果是少删一处标记，而中断构造的后果
    是整个头条基准跑不起来。
    """
    spans: list[tuple[int, int]] = []
    for item in record.get("marker_offsets") or []:
        pair = list(item) if isinstance(item, (list, tuple)) else []
        if len(pair) == 2:
            spans.append((int(pair[0]), int(pair[1])))
    return spans


def headline_survives(dataset: EvalDataset, settings: EvalSettings) -> bool:
    """头条基准是否还站得住（M7 §2.4，本轮最大单点风险）。

    触发退场的条件只有一条：库内子集 < `citrec_min_subset_size`（默认 3,000）。
    低于这个规模，bootstrap 95% CI 会宽到让任何消融结论都不显著——那时候再报
    数字就是在报噪声。

    **退场不是失败，是预案**：头条退回 LitSearch，citrec 降级为补充证据，同时
    明确承认「段落级能力」这一条只剩间接证据。调用方拿到 False 之后必须改报告
    口径，而不是继续跑然后把小样本的数字当结论。
    """
    kept = len(dataset.queries)
    survives = kept >= settings.citrec_min_subset_size
    if not survives:
        _logger.warning(
            "headline_benchmark_retired",
            kept=kept,
            threshold=settings.citrec_min_subset_size,
            fallback="LitSearch 升为头条，citrec 降级为补充证据",
        )
    return survives
