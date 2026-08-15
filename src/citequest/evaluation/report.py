"""评测报告：配置快照 + 落盘（M7 §8.3 报告纪律、§14.3 配置版本化）。

**「指标掉了先查什么」的第一维永远是口径**（M7 §14.1）：评测集换了 split、指标 k
变了、两层粒度被合并、`group_limit` 与上次不同、judge 的 prompt 版本变了——这一维
最便宜也最常见。它能被排查的**前提**是每次评测的配置与结果一起落盘，缺任何一项
则该次结果不可比。

所以本模块的产物是两份东西，不是一份：

- `report.json` —— 机器可读，含 per-query 值，供后续消融与显著性检验复用；
- `report.md`   —— 人读的表，表头强制写出聚合口径、样本量、过滤率、group_limit。

五条报告纪律在 `render_markdown` 里逐条落地，缺一条这个消融就不可信：
均值 + bootstrap 95% CI / 相对已发表基线的 Δ / p 值（标明校正方法）/
per-query win-loss-tie 三元计数 / 分粒度分别报。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from citequest.config.settings import Settings
from citequest.evaluation.datasets.schemas import EvalDataset
from citequest.evaluation.metrics import MetricReport
from citequest.observability.logging import get_logger

__all__ = ["EvalConfigSnapshot", "render_markdown", "write_report"]

_logger = get_logger(__name__)


class EvalConfigSnapshot(BaseModel):
    """一次评测的完整配置快照。**它与结果一起落盘，缺项则结果不可比。**

    字段选择不是「能记的都记」，而是 M7 §14.1 三维归因里「口径」与「serving」
    两维各自的判定输入：口径维要能对比两次评测的数据集/指标/k/group_limit，
    serving 维要能对比 `index_version` / 模型版本 / prompt 版本。
    """

    model_config = ConfigDict(frozen=True)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dataset_id: str
    split: str
    metrics: list[str]
    aggregation: str = Field(default="macro")
    #: **评测唯一允许与线上不同的那一项**（M7 §2.5）。线上搜索路是 60，评测加大到
    #: 100 以覆盖 Recall@100。不写出来就是评测报告里最常见的隐性不一致。
    eval_group_limit: int
    group_size: int
    rrf_k: int
    index_version: str
    embedding_model: str
    reranker_model: str
    rerank_threshold: float | None = Field(default=None, description="None 表示未标定，此时不做阈值过滤")
    judge_model: str | None = None
    judge_prompt_version: str | None = None
    bootstrap_samples: int = 1_000
    significance_tests: list[str] = Field(default_factory=list)

    @classmethod
    def from_settings(cls, settings: Settings, *, dataset_id: str, split: str) -> EvalConfigSnapshot:
        """从 `Settings` 抓一份快照。

        走配置对象而不是让调用方逐个传参：参数正本只有一份（`config/`），
        评测报告里的数与线上跑的数必须来自同一个来源，**手抄一遍就是给漂移开门**。
        """
        return cls(
            dataset_id=dataset_id,
            split=split,
            metrics=list(settings.eval.metrics),
            aggregation=settings.eval.aggregation,
            eval_group_limit=settings.retrieval.eval_group_limit,
            group_size=settings.retrieval.rag_group_size,
            rrf_k=settings.retrieval.rrf_k,
            index_version=settings.milvus.index_version,
            embedding_model=settings.embedding.model_id,
            reranker_model=settings.rerank.model_id,
            rerank_threshold=settings.rerank.score_threshold,
            bootstrap_samples=settings.eval.bootstrap_samples,
            significance_tests=list(settings.eval.significance_tests),
        )


def render_markdown(
    report: MetricReport,
    snapshot: EvalConfigSnapshot,
    dataset: EvalDataset,
    *,
    intervals: dict[str, tuple[float, float]] | None = None,
) -> str:
    """渲染人读的报告。**表头承载的信息量比表格本身大。**

    `macro-averaged over queries` 这一句就能挡掉「是每个 query 算占比再求平均吗」
    这种追问；过滤率那一行把「我挑了个对我有利的子集」变成一条可追溯的证据链。

    :param intervals: `{指标: (下界, 上界)}` bootstrap 95% CI。传 None 时表格里
        写 `n/a` 而**不是留空**——留空看起来像是忘了算，写 n/a 是一个明确的状态。
    """
    intervals = intervals or {}
    lines = [
        f"# 评测报告 · {report.dataset_name} · {report.config_name}",
        "",
        f"- 生成时间：{snapshot.created_at.isoformat()}",
        f"- 数据集：`{snapshot.dataset_id}` split=`{snapshot.split}`（{dataset.provenance.license}）",
        f"- 标签来源：{dataset.provenance.label_source}",
        "- 聚合口径：**macro-averaged over queries**（按 query 平均，不是全局池化）",
        f"- 索引版本：`{snapshot.index_version}` · 编码 `{snapshot.embedding_model}` · rerank `{snapshot.reranker_model}`",
        f"- **评测 group_limit={snapshot.eval_group_limit}（线上搜索路是 60）**，这是评测与线上唯一允许的差异",
        "",
    ]

    if dataset.subset_stats is not None:
        stats = dataset.subset_stats
        lines += [
            "## 库内过滤（必报）",
            "",
            f"- 原始条数：{stats.total_records:,} → 库内子集：{stats.kept_records:,}",
            f"- **过滤率（= 库内引用边覆盖率）：{stats.filter_rate:.2%}**",
            f"- 库规模：{stats.corpus_size:,} 篇；label 缺失 {stats.dropped_no_label:,} / 不在库内 {stats.dropped_out_of_corpus:,}",
            "",
        ]

    lines += ["## 指标", "", "| 指标 | 均值 | bootstrap 95% CI | n |", "|---|---|---|---|"]
    for metric, value in report.means().items():
        low, high = intervals.get(metric, (float("nan"), float("nan")))
        ci = "n/a" if low != low else f"[{low:.4f}, {high:.4f}]"
        lines.append(f"| {metric} | {value:.4f} | {ci} | {report.query_count(metric)} |")

    lines += ["", "## 必须与数字同时讲的局限", ""]
    lines += [f"- {note}" for note in dataset.provenance.notes]
    return "\n".join(lines) + "\n"


def write_report(
    report: MetricReport,
    snapshot: EvalConfigSnapshot,
    dataset: EvalDataset,
    reports_dir: Path,
    *,
    intervals: dict[str, tuple[float, float]] | None = None,
) -> tuple[Path, Path]:
    """落盘 JSON + Markdown，返回两个路径。

    文件名带 `index_version` 与配置名：同一份代码在两个索引版本上跑出的报告不能
    互相覆盖——**覆盖掉的那一份正是「掉点之前长什么样」的唯一证据**。
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report.dataset_name}__{report.config_name}__{snapshot.index_version}"
    json_path = reports_dir / f"{stem}.json"
    md_path = reports_dir / f"{stem}.md"

    payload = {
        "snapshot": snapshot.model_dump(mode="json"),
        "dataset": {
            "name": dataset.name,
            "provenance": dataset.provenance.model_dump(mode="json"),
            "subset_stats": dataset.subset_stats.model_dump(mode="json") if dataset.subset_stats else None,
        },
        "means": report.means(),
        # per-query 必须进 JSON：显著性检验与 win/loss/tie 都要重新读它，
        # 只存均值等于让每一次对比都必须重跑一遍检索。
        "per_query": report.per_query,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report, snapshot, dataset, intervals=intervals), encoding="utf-8")
    _logger.info("eval_report_written", json=str(json_path), markdown=str(md_path))
    return json_path, md_path
