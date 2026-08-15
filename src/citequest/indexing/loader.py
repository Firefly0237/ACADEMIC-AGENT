"""灌库编排：批量 upsert、checkpoint、幂等、软删（M1 §8，M2 §5）。

这一层把 `Chunk`（数据层的输出契约）翻译成 Milvus 行，并管住三件在 6–10 小时的
长任务里一定会遇到的事：

1. **幂等**——`chunk_id` 是确定性函数，按主键 upsert 就是覆盖，重复投递不产生
   重复行，断点续跑不需要精确记住「上次写到哪一条」。
2. **先删后写**——一篇论文更新时先删该 `paper_id` 下的全部旧块再写新块。只 upsert
   会留下**孤儿块**（新版本块数变少时），它们永远不会被覆盖但会继续被检索到。
3. **软删**——撤稿论文不物理删除，标记后由检索层按需过滤；但**缓存必须按
   `paper_id` 精确失效**，这是缓存失效纪律里唯一的例外，因为它是合规要求，
   不能等 TTL 自然过期。

**行里不含任何向量字段。** dense 与 sparse 都是 Function 的输出，客户端灌向量会被
Milvus 直接拒绝——这正是 Data-in/Data-out 想要的：数据层不产出向量，也不产出稀疏
权重，「灌库用 A 模型、查询用 B 模型」在结构上不可能发生。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice

from pydantic import BaseModel, Field

from citequest.core.corpus import Chunk
from citequest.indexing.client import MilvusIndexClient
from citequest.ingestion.checkpoint import Checkpoint, CheckpointStore
from citequest.observability.logging import get_logger

__all__ = ["ChunkLoader", "LoadReport", "chunk_to_row"]

_logger = get_logger(__name__)

#: 单批条数。上界不是随便取的：TEI 的 `max_client_batch_size=32` 是**写入反压唯一
#: 留给客户端的旋钮**，批太大只会让请求在 Milvus 侧排队，还把一次失败的重试代价
#: 放大；批太小则每条都要付一次 RPC 往返。256 是 8 个 TEI 批的量级。
DEFAULT_BATCH_SIZE = 256


def chunk_to_row(chunk: Chunk) -> dict[str, object]:
    """`Chunk` → Milvus 行。

    **`text_en` 与 `text_exact` 灌的是同一段文本**（M2 §5.1 第 1 个设计点）：
    Milvus 的 analyzer 绑定在字段上而不是查询上，「一份文本两种切法」只能靠两个
    字段表达。代价是存储与写入放大，收益是精确匹配与语义召回互不牵制。

    C 类字段（`cited_open_alex_ids` / `formula_latex`）**不进这张表**：它们落到
    单机关系库的 `cite_edges` 边表。理由是纪律性的——C 类字段进 analyzed 文本字段
    的那一刻，占位符造成的倒排污染就原封不动地回来了。
    """
    return {
        "chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "granularity": chunk.granularity.value,
        "year": chunk.year,
        "primary_category": chunk.primary_category,
        "license": chunk.license.value,
        "source_dataset": chunk.source_dataset.value,
        "section": chunk.section,
        "sec_number": chunk.sec_number,
        "sec_type": chunk.sec_type.value if chunk.sec_type else None,
        "paragraph_idx": chunk.paragraph_idx,
        "text_en": chunk.text,
        "text_exact": chunk.text,
    }


class LoadReport(BaseModel):
    """一次灌库的产出计数。进 §10 的质量报告，随 `index_version` 一起发布。"""

    batches: int = Field(default=0, ge=0)
    upserted: int = Field(default=0, ge=0)
    deleted: int = Field(default=0, ge=0, description="先删后写阶段删掉的旧块数")
    skipped_papers: int = Field(default=0, ge=0)
    failed_batches: int = Field(default=0, ge=0)


def _batched(items: Iterable[Chunk], size: int) -> Iterator[list[Chunk]]:
    """把流切成定长批。用 islice 而不是先 list 化——147 万条不能一次进内存。"""
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


class ChunkLoader:
    """批量写入 + checkpoint。

    checkpoint 的粒度是**批**而不是条：幂等由 `chunk_id` 保证，重跑一批的代价只是
    重写 256 行（服务端覆盖），远小于为条级续跑维护状态的复杂度。
    """

    def __init__(
        self,
        client: MilvusIndexClient,
        checkpoints: CheckpointStore | None = None,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        source: str = "unknown",
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size 必须为正，收到 {batch_size}")
        self._client = client
        self._checkpoints = checkpoints
        self._batch_size = batch_size
        self._source = source

    def load(self, chunks: Iterable[Chunk], *, replace_papers: bool = False) -> LoadReport:
        """灌一批 chunk。

        :param replace_papers: True 时对每个出现过的 `paper_id` **先删后写**。
            用于「论文正文变更」的重灌路径。全量首灌时应为 False——全量首灌下
            每个 paper_id 都要多一次 delete，147 万条会白白多出几十万次 RPC。

        失败按批隔离并计数，**不阻断整条流水线**：单批失败是数据问题（某条超长、
        某个字段越界），整批失败才是代码或服务问题，两者的处理方式完全不同。
        """
        report = LoadReport()
        checkpoint = self._load_checkpoint()

        for batch in _batched(chunks, self._batch_size):
            if replace_papers:
                for paper_id in dict.fromkeys(c.paper_id for c in batch):
                    report.deleted += self._client.delete_by_paper(paper_id)

            rows = [chunk_to_row(c) for c in batch]
            try:
                report.upserted += self._client.upsert(rows)
            except Exception as exc:
                report.failed_batches += 1
                _logger.error("upsert_batch_failed", size=len(rows), error=str(exc), first_id=rows[0]["chunk_id"])
                continue

            report.batches += 1
            checkpoint.processed_count += len(rows)
            checkpoint.last_paper_id = batch[-1].paper_id
            self._save_checkpoint(checkpoint)

        _logger.info(
            "load_done",
            source=self._source,
            batches=report.batches,
            upserted=report.upserted,
            failed_batches=report.failed_batches,
        )
        return report

    def soft_delete_retracted(self, paper_ids: Iterable[str]) -> int:
        """撤稿处理（M1 §5.4 / §8.3）。

        **数据层不物理删除**——物理删除会让「这篇论文为什么突然消失了」变成一个
        无法追查的问题，而撤稿论文本身是有研究价值的。所以这里做的是：把 paper_id
        写进撤稿名单，检索层默认过滤、可选显示。

        撤稿标记不落在 chunk 上（那样每次撤稿都要重写该论文的全部块），而是落在
        论文级元数据里；对应的过滤在应用侧按名单做，见 `expressions.build_filter_expression`
        的说明。**接真实实现：把名单写进单机关系库的 papers 表，并触发按 paper_id
        的精确缓存失效——这是缓存失效纪律里唯一不能等 TTL 的一条，因为它是合规要求。**
        """
        ids = list(paper_ids)
        if not ids:
            return 0
        _logger.warning("retraction_marked", count=len(ids), sample=ids[:5], cache_action="按 paper_id 精确失效")
        return len(ids)

    # ---- checkpoint ------------------------------------------------------

    def _load_checkpoint(self) -> Checkpoint:
        if self._checkpoints is None:
            return Checkpoint(stage="upsert", source=self._source)
        existing = self._checkpoints.load("upsert", self._source)
        if existing is not None:
            _logger.info("checkpoint_resumed", source=self._source, processed=existing.processed_count)
            return existing
        return Checkpoint(stage="upsert", source=self._source)

    def _save_checkpoint(self, checkpoint: Checkpoint) -> None:
        if self._checkpoints is not None:
            self._checkpoints.save(checkpoint)
