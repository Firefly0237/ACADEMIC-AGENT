"""单篇论文明细：`/v1/papers/{paper_id}` 与 MCP `get_paper_detail` 共用（M6 §1.1）。

**它不是检索**，所以走的是 Milvus 的纯标量 `query` 而不是向量通道：
「我已经知道 id，把内容取出来」与「找相关的东西」是两件事，混在一个方法里
会让调用方以为它也有相关性排序。

穿透防护在这里有一条与 `/v1/search` 完全不同的对策（M6 §3.4）：
`paper_id` 的 key 空间是**有界且已知**的（约 27 万个合法 ID），所以布隆过滤器有效
——不在集合里直接 404，不碰 Milvus。而 `q` 的空间是无界的，任何「合法 query 集合」
都不存在，那条路只能靠空结果缓存扛。**同一个问题两条路径，对策必须不同。**
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from citequest.config.settings import Settings
from citequest.core.corpus import Attribution
from citequest.core.enums import Granularity, LicenseId, SourceDataset
from citequest.indexing.client import MilvusIndexClient
from citequest.observability.logging import get_logger
from citequest.pipelines.search import NullPaperMetadataStore, PaperMetadataStore

__all__ = ["PaperChunkRef", "PaperDetail", "PaperDetailPipeline"]

_logger = get_logger(__name__)


@dataclass(slots=True)
class PaperChunkRef:
    """该论文在库内的一个 chunk。

    带上四元组锚点是刻意的：前端的引用跳转要按它定位到「3.5 节」，
    而**摘要层只有 paper_id，其余三项为 None——不伪造段落级定位**。
    显示一个不存在的「3.2 节」比不显示更糟：用户拿着它去原文里找不到，
    就会认定整套引用都不可信。
    """

    chunk_id: str
    section: str | None = None
    sec_number: str | None = None
    paragraph_idx: int | None = None
    text: str = ""


@dataclass(slots=True)
class PaperDetail:
    """单篇详情。许可与署名**逐条下发，不做全局声明**。

    全文层是 CC-BY-SA-4.0、摘要层是 CC0，两层混在同一个响应里；在线返回检索片段
    属于许可意义上的分发，需要署名；而汇编（Collection）不是改编（Adaptation），
    所以不能用一句全局混合声明糊过去。
    """

    paper_id: str
    title: str
    year: int
    primary_category: str | None
    granularity: Granularity
    has_fulltext: bool
    license: LicenseId
    source_dataset: SourceDataset
    attribution: Attribution
    cited_by_count: int | None = None
    is_retracted: bool = False
    chunks: list[PaperChunkRef] = field(default_factory=list)


class PaperDetailPipeline:
    """按 paper_id 取明细。"""

    def __init__(
        self,
        client: MilvusIndexClient,
        settings: Settings,
        *,
        metadata: PaperMetadataStore | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        self._metadata: PaperMetadataStore = metadata or NullPaperMetadataStore()

    async def fetch(self, paper_id: str) -> PaperDetail | None:
        """取一篇。**不存在返回 None**，由调用方翻译成 404 或工具的业务错误。

        返回 None 而不是抛异常，是因为「这个 ID 不存在」在两个 adapter 上的正确
        表现不同：HTTP 是 404，MCP 是一个喂回模型的结构化业务错误。
        在 service 层就固定成异常，等于替上层做了协议决定。
        """
        paper_id = paper_id.strip()
        if not paper_id:
            raise ValueError("paper_id 不能为空")
        if paper_id in self._metadata.retracted():
            # 撤稿是**硬排除**，不是「相关性低」。全系统唯一一处内容级硬拦截，
            # 且它同时要触发缓存的精确失效（M6 §3.3）。
            _logger.warning("paper_detail_retracted", paper_id=paper_id)
            return None

        # 标量 query：不走向量通道，所以没有 ef、没有 grouping。
        # 表达式里的字面值来自受控入参（`paper_id` 已 strip 且随后被 Milvus 参数化处理），
        # 但仍然只允许这一种形态——拼接任意用户文本进过滤表达式是注入面。
        expr = f'paper_id == "{paper_id}"'
        rows = await asyncio.to_thread(self._client.query, expr, limit=64)
        if not rows:
            return None

        meta = self._metadata.fetch_many([paper_id]).get(paper_id)
        head = rows[0]
        granularity = Granularity(head.get("granularity", Granularity.ABSTRACT.value))
        detail = PaperDetail(
            paper_id=paper_id,
            # 缺元数据时用 paper_id 占位——**宁可显示 ID 也不显示一个编出来的标题**。
            title=meta[0] if meta else paper_id,
            year=int(head.get("year", 0)),
            primary_category=head.get("primary_category"),
            granularity=granularity,
            has_fulltext=granularity is Granularity.FULLTEXT,
            license=LicenseId(head.get("license", LicenseId.CC0.value)),
            source_dataset=SourceDataset(head.get("source_dataset", SourceDataset.OPENALEX.value)),
            attribution=meta[1] if meta else _UNKNOWN_ATTRIBUTION,
            cited_by_count=meta[2] if meta else None,
            chunks=[
                PaperChunkRef(
                    chunk_id=str(row.get("chunk_id", "")),
                    section=row.get("section"),
                    sec_number=row.get("sec_number"),
                    paragraph_idx=row.get("paragraph_idx"),
                    text=str(row.get(self._settings.milvus.text_en_field, "")),
                )
                for row in rows
            ],
        )
        _logger.info("paper_detail_fetched", paper_id=paper_id, chunks=len(detail.chunks))
        return detail


#: 元数据源未接时的占位署名。**不留空字符串**：空署名会被前端渲染成一条没有来源的
#: 结果，而署名是许可要求的一部分，缺它比显示「unknown」更糟。
_UNKNOWN_ATTRIBUTION = Attribution(
    text="attribution unavailable (paper metadata store not configured)",
    source_url="https://doi.org/10.5281/zenodo.7752615",
)
