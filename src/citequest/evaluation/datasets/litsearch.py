"""第二基准：LitSearch（摘要级，M7 §3）。

| 项 | 值 | 口径 |
|---|---|---|
| query 数 | 597 | 【实测，MTEB descriptive_stats】 |
| corpus | 64,183 篇 title+abstract | 【实测】 |
| qrels | 639 条判定，平均 1.07 个相关文档/query | 【实测】 |
| 官方检索字段 | `title_abstract` | 【实测，官方仓库】 |

**为什么保留第二个基准**：语料是分层的，citrec 是段落级的（只测得到全文层），
LitSearch 是 title+abstract 级的（只测得到摘要层）。只跑 LitSearch，「换到全文
语料买到了段落级能力」就没有直接证据；只跑 citrec，占语料 81.5% 篇数的摘要层
完全没被测过。

**它用的是自己的 64,183 篇 corpus，不是本系统的 27 万篇**——所以它测的是
「同样的检索器在标准语料上的水平」，属于**横向可比性**证据；本系统语料上的表现
由 citrec 承担。这句话必须与数字同时出现，否则会被读成「系统在自己库上的成绩」。

**不报 Precision@k**：qrels 平均 1.07 个相关文档/query，Precision@10 的理论上限
只有 0.107，报出来只会把「系统很正常」误读成「系统很差」。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from citequest.config.eval import EvalSettings
from citequest.core.enums import Granularity
from citequest.evaluation.datasets.schemas import (
    RELEVANT,
    EvalDataset,
    EvalQuery,
    Provenance,
)
from citequest.observability.logging import get_logger

__all__ = [
    "LITSEARCH_PROVENANCE",
    "BaselineSanityCheck",
    "bm25_sanity_check",
    "build_litsearch_dataset",
    "load_litsearch_raw",
]

_logger = get_logger(__name__)

LITSEARCH_PROVENANCE = Provenance(
    dataset_id="princeton-nlp/LitSearch",
    split="query",
    license="MIT",
    label_source="数据集作者构造的 qrels（与本系统的检索结果完全无关）",
    citation="MTEB task: LitSearchRetrieval",
    notes=[
        "只有英文，且集中在 ML / NLP 子领域——不代表 CV / 系统 / 理论方向的检索质量",
        "qrels 平均 1.07 个相关文档/query，**因此不报 Precision@k**（理论上限约 0.107）",
        "corpus 是 LitSearch 自己的 64,183 篇，不是本系统的 27 万篇：它是横向可比性证据",
    ],
)


class BaselineSanityCheck(BaseModel):
    """BM25 基线自证的结果（M7 §3.4）。

    **这件事叫 sanity check，不叫 reproduction**——三套 BM25 实现的分词器不同
    （Milvus 内置 BM25 走 Tantivy 的 english analyzer、LitSearch 官方脚本用
    `rank-bm25`、BEIR 的公开数字来自 Anserini/Lucene），**精确复现在机制上就
    不可能**，所以只做量级一致性检查。字段名用 `wording` 把这一点固化下来。
    """

    model_config = ConfigDict(frozen=True)

    ndcg10: float
    recall100: float
    ndcg10_reference: float = Field(description="部分核实：单一来源（CASPER Table 2），引用时必须说明")
    recall100_reference: float
    ndcg10_within_window: bool
    recall100_within_window: bool
    wording: str = Field(default="sanity_check")

    @property
    def passed(self) -> bool:
        """两项都落窗内才算通过。

        **落窗外的处置是「先查索引与分词器，不许直接开始报提升数字」**——
        基线不成立时，所有相对提升都是在跟一个不知道对不对的东西比。
        """
        return self.ndcg10_within_window and self.recall100_within_window


def bm25_sanity_check(ndcg10: float, recall100: float, settings: EvalSettings) -> BaselineSanityCheck:
    """把实测的 BM25 基线与已发表数字比一次（M7 §3.4）。

    通过窗口 nDCG@10 40.2 ± 3、R@100 74.4 ± 4【设计】，依据是不同 BM25 实现之间
    的已知波动量级。**它不是一个精度指标，是一个「我的索引和分词器有没有装错」
    的探针**——这也是为什么它跑在报任何相对提升之前。

    落在窗口外又查不出原因时，诚实的写法是「我的基线偏弱/偏强 X 点，下面的相对
    提升请打折看」，而不是悄悄换一个让自己好看的基线。
    """
    ndcg_ok = abs(ndcg10 - settings.litsearch_bm25_ndcg10_reference) <= settings.litsearch_ndcg10_tolerance
    recall_ok = abs(recall100 - settings.litsearch_bm25_recall100_reference) <= settings.litsearch_recall100_tolerance
    result = BaselineSanityCheck(
        ndcg10=ndcg10,
        recall100=recall100,
        ndcg10_reference=settings.litsearch_bm25_ndcg10_reference,
        recall100_reference=settings.litsearch_bm25_recall100_reference,
        ndcg10_within_window=ndcg_ok,
        recall100_within_window=recall_ok,
        wording=settings.reproduction_wording,
    )
    if not result.passed:
        _logger.warning(
            "bm25_baseline_out_of_window",
            ndcg10=ndcg10,
            recall100=recall100,
            action="先查索引与分词器，不许直接开始报提升数字",
        )
    return result


def load_litsearch_raw(settings: EvalSettings) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """拉 LitSearch 的 query 与 corpus 两个 split。

    **接真实实现**：正式实现是两次
    ``datasets.load_dataset(settings.litsearch_dataset_id, name=...)``——官方仓库
    把 query 与 corpus 分成两个 config。corpus 侧只取 `settings.litsearch_retrieval_field`
    （`title_abstract`）**这一个字段**：官方检索字段就是它，换字段等于换任务，
    与已发表数字立刻不可比。
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise NotImplementedError(
            "接真实实现：需要 eval 依赖组（uv sync --group eval）才能加载 princeton-nlp/LitSearch"
        ) from exc

    queries = list(load_dataset(settings.litsearch_dataset_id, "query", split="full"))
    corpus = list(load_dataset(settings.litsearch_dataset_id, "corpus_clean", split="full"))
    _logger.info("litsearch_loaded", queries=len(queries), corpus=len(corpus))
    return queries, corpus


def build_litsearch_dataset(records: Iterable[dict[str, Any]]) -> EvalDataset:
    """把 LitSearch 的 query split 规整成 `EvalDataset`。

    `corpusids` 是一条 query 的相关文档列表（平均 1.07 个，最多 5）。这里**不做
    任何过滤**——与 citrec 不同，LitSearch 用它自己的 corpus，没有「库内子集」
    这回事，硬去过滤只会把一个可横向比较的基准改成一个自定义任务。
    """
    queries: list[EvalQuery] = []
    qrels: dict[str, dict[str, int]] = {}

    for index, record in enumerate(records):
        text = str(record.get("query") or "").strip()
        if not text:
            continue
        query_id = str(record.get("query_set") or "litsearch") + f"-{record.get('specificity', 0)}-{index}"
        relevant = [str(cid) for cid in (record.get("corpusids") or [])]
        if not relevant:
            # 没有 qrels 的 query 进了评测就是一个恒定的 0 分，会把均值无缘无故拉低。
            continue
        queries.append(EvalQuery(query_id=query_id, text=text, layer=Granularity.ABSTRACT))
        qrels[query_id] = dict.fromkeys(relevant, RELEVANT)

    judgements = sum(len(v) for v in qrels.values())
    _logger.info(
        "litsearch_dataset_built",
        queries=len(queries),
        judgements=judgements,
        # 密度要打进日志：Precision@k 不报的理由就是这个数，它变了结论也要跟着变。
        density=round(judgements / len(queries), 3) if queries else 0.0,
    )
    return EvalDataset(
        name="LitSearch",
        queries=queries,
        qrels=qrels,
        provenance=LITSEARCH_PROVENANCE,
    )
