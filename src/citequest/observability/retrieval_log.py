"""检索侧可观测：召回明细落盘（M7 §12）。**本模块是这一层的核心产物。**

> 「如何观察模型召回了哪些块」这个问题，答「看日志」是不及格的。
> **它需要一个具体产物。**

每一次检索请求，把候选集的完整明细落盘为一批 JSONL 记录，可按 `request_id` 回查。
每个候选 chunk 一行。

**这份表一举两得**：

| 用途 | 怎么用 |
|---|---|
| **排障工具** | 五类 badcase 的判定信号全部来自这张表（`badcase.py`） |
| **citation 验证依据** | `cited` 与 `in_context` 一比，就能直接算出引用是否落在真实进入上下文的 chunk 上——**离线的 citation precision 不需要重跑检索** |

**第二条是这个设计的真正价值**：引用质量的评测与检索的可观测，用的是同一份数据。
评测不是事后补的一套流程，它复用的就是线上每次请求都会产生的产物。

**代价写出来**：每次请求多一份 JSONL 写入与存储。这是 M7 §16 第 5 条取舍里
明确选择的那一边——选了「能下钻到单请求、能程序化归因五类 badcase」，
放弃了「更多的聚合指标」。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from citequest.core.answer import Evidence
from citequest.core.enums import Granularity, RetrievalMode
from citequest.core.search import RetrievedChunk
from citequest.observability.logging import get_logger

__all__ = ["RetrievalDetail", "RetrievalDetailWriter", "build_details"]

_logger = get_logger(__name__)


class RetrievalDetail(BaseModel):
    """一个候选 chunk 在这次检索里的完整轨迹（§12.1 字段表）。

    **三个布尔字段是三条判定分界线**，不是三个装饰：

    - `in_candidate` —— 检索错 / 融合错的分界线；
    - `in_context`   —— 截断错的判定依据；
    - `cited`        —— 引用错的判定依据。

    **两路的原始名次与分数都要留**：只留融合后的名次，就分不清「某一路根本没召回」
    与「召回了但被融合挤掉」——而这两件事的处置完全不同（前者查 ef 与切块，
    后者查该不该走单路）。
    """

    model_config = ConfigDict(frozen=True)

    request_id: str
    index_version: str
    chunk_id: str
    paper_id: str
    granularity: Granularity = Field(description="分粒度归因。fulltext 与 abstract 的失败模式不同")

    rank_dense: int | None = Field(default=None, ge=1, description="dense 通道原始名次。None = 这一路没召回它")
    score_dense: float | None = None
    rank_sparse: int | None = Field(default=None, ge=1, description="BM25 通道原始名次")
    score_sparse: float | None = None
    rank_fused: int | None = Field(default=None, ge=1)
    score_rrf: float | None = Field(
        default=None,
        description="**只进明细，不进 API 响应**（M6 §1.5）。RRF 分数无物理含义，禁止做阈值",
    )

    in_candidate: bool = Field(description="是否进入 grouping 之后的 ≤48 chunk")
    score_rerank: float | None = Field(default=None, ge=0.0, le=1.0, description="normalize=True 后的 [0,1]")
    in_context: bool = Field(default=False, description="是否进入最终的 8 条上下文")
    cited: bool = Field(default=False, description="是否被回答中的 [N] 引用")
    retrieval_mode: RetrievalMode = Field(description="降级归因：hybrid / dense_only / sparse_only")

    @property
    def recalled_by(self) -> list[str]:
        """哪几路召回了它。**融合错的第一判据**——某一路名次很靠前却掉出候选边界。"""
        channels = []
        if self.rank_dense is not None:
            channels.append("dense")
        if self.rank_sparse is not None:
            channels.append("sparse")
        return channels


def build_details(
    candidates: Sequence[RetrievedChunk],
    *,
    request_id: str,
    index_version: str,
    retrieval_mode: RetrievalMode,
    context_chunk_ids: Iterable[str] = (),
    evidences: Sequence[Evidence] = (),
    cited_citation_ids: Iterable[int] = (),
) -> list[RetrievalDetail]:
    """把一次检索的候选集组装成明细记录。

    :param context_chunk_ids: rerank 之后真正进 prompt 的那 ≤8 条。
    :param evidences: `[N]` 与 chunk 的映射。没有它就无法把引用编号还原成 chunk_id。
    :param cited_citation_ids: 回答里出现过的引用编号。**调用方必须用
        `rag.citations.citation_ids_in` 扫原始回答文本**，而不是读
        `AnswerSentence.citations`——后者已经过一次合法性过滤，用它算 `cited`
        会让「模型引了一个越界编号」这件事在明细里消失，**而那正是引用错这一类
        badcase 要看的东西**。

    参数取编号而不是直接调 `rag.citations` 扫文本，是一条**分层约束**：
    observability 是底座（core → observability → 其余全部），反向依赖 rag 会
    形成 `rag → llm → observability → rag` 的循环。
    """
    in_context = set(context_chunk_ids)
    chunk_of_citation = {e.citation_id: e.anchor.chunk_id for e in evidences}
    cited_chunks = {chunk_of_citation[cid] for cid in cited_citation_ids if cid in chunk_of_citation}

    details = [
        RetrievalDetail(
            request_id=request_id,
            index_version=index_version,
            chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            granularity=chunk.granularity,
            rank_dense=chunk.dense_rank,
            rank_sparse=chunk.sparse_rank,
            rank_fused=index,
            score_rrf=chunk.rrf_score,
            # 候选表里的每一条按定义都进了候选集；**不在候选里的 chunk 根本不会
            # 出现在这个列表里**——「检索错」那一类的判定因此是「查不到这一行」，
            # 由 badcase 层用库内 paper_id 集合来确认，不是这里的 False。
            in_candidate=True,
            score_rerank=chunk.rerank_score,
            in_context=chunk.chunk_id in in_context,
            cited=chunk.chunk_id in cited_chunks,
            retrieval_mode=retrieval_mode,
        )
        for index, chunk in enumerate(candidates, start=1)
    ]
    _logger.info(
        "retrieval_details_built",
        request_id=request_id,
        candidates=len(details),
        in_context=sum(1 for d in details if d.in_context),
        cited=sum(1 for d in details if d.cited),
    )
    return details


class RetrievalDetailWriter:
    """把明细追加进 JSONL。**按天分文件，按 `request_id` 回查。**

    为什么是 JSONL 而不是数据库：这份数据是**只追加、按 request_id 点查、偶尔全量
    扫一遍算 citation precision**——三种访问模式没有一种需要索引或事务。
    为它引一个数据库，是给一个不存在的问题付运维成本。

    写入失败**绝不影响业务**：可观测组件挂掉时正确的行为是丢掉这条记录，
    而不是把一个成功的请求变成 500。这条与 `spans._emit` 是同一条纪律。
    """

    def __init__(self, directory: Path, *, enabled: bool = True) -> None:
        self._dir = directory
        self._enabled = enabled

    def write(self, details: Sequence[RetrievalDetail], *, date_key: str) -> Path | None:
        """写一批明细，返回落盘路径。`enabled=False` 或空批次返回 None。

        :param date_key: 形如 `20260813`，用来分文件。由调用方给而不是这里取
            `now()`：评测复算时要能把明细写到与原请求相同的分区，取当前时间会
            让复算的产物散落到今天。
        """
        if not self._enabled or not details:
            return None
        path = self._dir / f"retrieval-{date_key}.jsonl"
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for detail in details:
                    handle.write(json.dumps(detail.model_dump(mode="json"), ensure_ascii=False) + "\n")
        except OSError as exc:
            # 这里**故意只捕获 OSError**：磁盘满 / 权限 / 路径问题都属于「丢掉这条
            # 记录」的范畴；而序列化错误是 bug，不该被这条兜底吞掉。
            _logger.warning("retrieval_detail_write_failed", error_type=type(exc).__name__, path=str(path))
            return None
        return path

    def read(self, path: Path, *, request_id: str | None = None) -> list[RetrievalDetail]:
        """回查。传 `request_id` 只取那一次请求的明细。

        离线算 citation precision 就走这个入口——**不需要重跑检索**，
        `cited` 与 `in_context` 两个字段已经在里面了（§12.2）。
        """
        rows = [
            RetrievalDetail.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [r for r in rows if request_id is None or r.request_id == request_id]
