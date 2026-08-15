"""L1 检索层跑分：从数据集到一份可复现的报告（M7 §2、§6、§8.1）。

```
EvalDataset ──► RunProducer（四档配置之一）──► {query_id: [paper_id, ...]}
                                                    │
                                     metrics.per_query_metric（逐 query）
                                                    │
                          MetricReport ──► 分粒度切片 ──► report.write_report
```

**四档配置共用同一份 query、同一份 qrels、同一套代码，一次只变一项**（M7 §8.1）。
这条纪律在代码里的形状是：`RunProducer` 是唯一的变量，四档只改它的构造参数，
**不允许四档各写一个脚本**——差异会悄悄跑进实现里，最后比的不是配置。

**评测管线与线上管线唯一允许的差异**是 `group_limit` 加大到 100（线上搜索路
60），用来覆盖 Recall@100。这个差异由 `EvalConfigSnapshot.eval_group_limit`
带进报告，**显式声明而不是默认「跑的就是同一条管线」**。

**检索单位是 paper 不是 chunk**：两个基准的标签都是论文级的，而本系统靠
grouping 在服务端完成 chunk→paper 聚合——评测单位与产品单位天然一致，
不需要在评测侧再写一层聚合，那层聚合一旦与线上不同，指标就不代表线上。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from citequest.config.settings import Settings
from citequest.core.enums import Granularity, RetrievalMode
from citequest.evaluation.datasets.schemas import EvalDataset, EvalQuery
from citequest.evaluation.metrics import MetricReport, per_query_metric
from citequest.observability.logging import get_logger

__all__ = [
    "ABLATION_PRESETS",
    "AblationPreset",
    "PipelineRunProducer",
    "RunProducer",
    "evaluate_dataset",
    "split_by_layer",
]

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AblationPreset:
    """一档消融配置（M7 §8.1 的 C1–C4）。

    **四档只差三个布尔量**，这是刻意的：`retrieval_mode` 决定发几路，
    `rerank` 决定要不要精排。一档一个 dataclass 实例而不是一个 if 分支树，
    是为了让「一次只变一项」这件事在数据上可被检查（`differs_from` 就是干这个的）。
    """

    name: str
    retrieval_mode: RetrievalMode
    rerank: bool
    role: str

    def differs_from(self, other: AblationPreset) -> list[str]:
        """列出两档之间变了哪几项。**变了不止一项的对比不进报告。**"""
        changed = []
        if self.retrieval_mode is not other.retrieval_mode:
            changed.append("retrieval_mode")
        if self.rerank != other.rerank:
            changed.append("rerank")
        return changed


#: 四档的正本。名字与 `EvalSettings.ablation_configs` 一一对应——两处不一致时
#: CLI 会接受一个跑不出来的档位名，而报错要等到跑分中途才发生。
ABLATION_PRESETS: dict[str, AblationPreset] = {
    "bm25_only": AblationPreset(
        name="bm25_only",
        retrieval_mode=RetrievalMode.SPARSE_ONLY,
        rerank=False,
        role="主基线，且是唯一能与已发表数字对齐的配置（LitSearch BM25 sanity check）",
    ),
    "dense_only": AblationPreset(
        name="dense_only",
        retrieval_mode=RetrievalMode.DENSE_ONLY,
        rerank=False,
        role="消融项：稀疏通道的贡献",
    ),
    "hybrid": AblationPreset(
        name="hybrid",
        retrieval_mode=RetrievalMode.HYBRID,
        rerank=False,
        role="消融项：RRF 融合的贡献",
    ),
    "hybrid_rerank": AblationPreset(
        name="hybrid_rerank",
        retrieval_mode=RetrievalMode.HYBRID,
        rerank=True,
        role="完整配置",
    ),
}


class RunProducer(Protocol):
    """把一条 query 变成一个论文级排序结果。

    定义成 Protocol 而不是基类，是因为它有两种完全不同的实现：接真实 Milvus 的
    `PipelineRunProducer`，和单测里直接返回固定排序的桩。**评测代码本身必须能在
    没有任何外部服务的情况下被测**，否则「指标算得对不对」这件事就永远只能靠跑
    全量来验证。
    """

    def rank(self, query: EvalQuery, *, top_k: int) -> Sequence[str]: ...


class PipelineRunProducer:
    """走真实检索管线的 run 产出器。

    **接真实实现**：`rank` 里应当调
    ``await SearchPipeline.run(SearchCommand(q=query.text, page_size=..., ...))``
    并把 `SearchResult.papers` 的 `paper_id` 按序取出；`preset.rerank` 为 True 时
    走 RAG 路的 `AnswerPipeline` 检索段（rerank 之后的 8 条按分数序）。
    构造时注入 pipeline 而不是在这里建 Milvus 连接——评测脚本与 API 共用同一份
    装配（`pipelines.factory.build_services`），**评测跑的必须是线上那条管线**。

    当前骨架阶段直接抛 `NotImplementedError`：返回假排名会产出一份看起来正常、
    实际毫无意义的报告，而报告一旦落盘就会被引用。
    """

    def __init__(self, preset: AblationPreset, settings: Settings, *, search_pipeline: object | None = None) -> None:
        self._preset = preset
        self._settings = settings
        self._pipeline = search_pipeline
        _logger.info(
            "run_producer_configured",
            preset=preset.name,
            retrieval_mode=preset.retrieval_mode.value,
            rerank=preset.rerank,
            # 口径差异必须出现在跑分日志里，只写在报告里就没人在跑的时候注意到。
            group_limit=settings.retrieval.eval_group_limit,
        )

    def rank(self, query: EvalQuery, *, top_k: int) -> Sequence[str]:
        """一条 query → 论文级排序（paper_id 列表，长度 ≤ top_k）。"""
        if self._pipeline is None:
            raise NotImplementedError(
                "接真实实现：注入 pipelines.search.SearchPipeline，"
                f"用 eval_group_limit={self._settings.retrieval.eval_group_limit} 跑同一条线上管线"
            )
        raise NotImplementedError(
            f"接真实实现：调 SearchPipeline.run 并取 SearchResult.papers 的 paper_id（preset={self._preset.name}）"
        )


def evaluate_dataset(
    dataset: EvalDataset,
    producer: RunProducer,
    *,
    metrics: Sequence[str],
    preset_name: str,
    top_k: int,
) -> MetricReport:
    """跑完一个数据集，产出逐 query 指标。

    :param top_k: 取回多少篇。它必须 ≥ 最大的指标 k（Recall@100 → 100），
        否则指标会被检索深度而不是检索质量截断——**一个静默拉低分数的口径错误**。

    实现上刻意先把 run 全部收齐再算指标，而不是边跑边累加：per-query 值要进报告
    与显著性检验，攒不齐就没法做 win/loss/tie。
    """
    max_k = max(int(metric.split("@")[1]) for metric in metrics)
    if top_k < max_k:
        raise ValueError(f"top_k({top_k}) 小于指标要求的最大 k({max_k})，指标会被检索深度截断")

    runs: dict[str, Sequence[str]] = {}
    for query in dataset.queries:
        runs[query.query_id] = producer.rank(query, top_k=top_k)

    report = MetricReport(
        config_name=preset_name,
        dataset_name=dataset.name,
        per_query={metric: per_query_metric(runs, dataset.qrels, metric) for metric in metrics},
    )
    _logger.info(
        "retrieval_eval_done",
        dataset=dataset.name,
        preset=preset_name,
        queries=len(dataset.queries),
        **{m: round(v, 4) for m, v in report.means().items()},
    )
    return report


def split_by_layer(report: MetricReport, dataset: EvalDataset) -> dict[Granularity, MetricReport]:
    """按粒度切成两份报告（M7 §1.3：**禁止合并成一个数字**）。

    22 万条摘要会把 5 万篇全文的表现稀释掉，合并后的平均数谁都看不出问题。
    返回的字典里只保留实际有样本的层——citrec 只有 fulltext、LitSearch 只有
    abstract，硬塞一个空层进报告会渲染出一行 `0.0000`，那比不报更误导。
    """
    buckets: dict[Granularity, set[str]] = {}
    for query in dataset.queries:
        buckets.setdefault(query.layer, set()).add(query.query_id)
    return {layer: report.slice_by(ids) for layer, ids in buckets.items() if ids}
