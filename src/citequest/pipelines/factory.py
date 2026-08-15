"""service 层的**唯一构造入口**：把各能力域装配成一组进程级单例。

放在 `pipelines/` 而不是 `api/`，是一条真实的分层约束而不是审美：
**MCP adapter 也要装配同一套 service**，而它没有 HTTP 概念。装配代码放在 api 层
就意味着 `mcp_server` 要 import fastapi 才能起一个 stdio server——那条依赖没有
任何道理，而且它会在「只跑 MCP、不跑 HTTP」的部署里凭空多装一堆包。

**所有重对象在这里建一次**（Milvus 连接、reranker、GPU 闸门）。
闸门必须是进程级单例这一条比性能更硬：`asyncio.Semaphore(12)` 每请求新建一个，
等于每个请求都拿到一份自己的 12 个额度——并发闸门直接失效，而且失效得悄无声息
（显存打穿之前一切正常）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from citequest.config.settings import Settings, get_settings
from citequest.indexing.client import MilvusIndexClient
from citequest.llm.chat import ChatClient
from citequest.llm.hhem import HhemClient
from citequest.llm.reranker import CrossEncoderReranker
from citequest.observability.logging import get_logger
from citequest.orchestration.gates import GpuGates
from citequest.orchestration.tool_specs import ToolContext, build_registry
from citequest.orchestration.tools import ToolRegistry
from citequest.pipelines.answer import AnswerPipeline
from citequest.pipelines.papers import PaperDetailPipeline
from citequest.pipelines.search import PaperMetadataStore, SearchPipeline
from citequest.query.decompose import QueryDecomposer
from citequest.query.hyde import HydeGenerator
from citequest.query.intent import QueryRouter
from citequest.rag.generator import AnswerGenerator
from citequest.rag.grounding import GroundingChecker
from citequest.retrieval.facets import ExactFacetSource
from citequest.retrieval.hybrid import HybridRetriever
from citequest.retrieval.rerank import Reranker

__all__ = ["Services", "build_services"]

_logger = get_logger(__name__)


@dataclass(slots=True)
class Services:
    """一次进程生命周期内的全部 service 单例。

    这个对象本身就是**装配图**：读它就知道这套服务由哪几块组成、谁依赖谁。
    `correction_graph` 可以是 None——**没有评估器就不该进循环**，
    而 LangGraph 图在没装配时保持 None 比装一个空图诚实。
    """

    settings: Settings
    milvus: MilvusIndexClient
    chat: ChatClient
    gates: GpuGates
    search: SearchPipeline
    answer: AnswerPipeline
    papers: PaperDetailPipeline
    tools: ToolRegistry
    correction_graph: Any | None = None

    async def aclose(self) -> None:
        """释放外部连接。

        顺序是有讲究的：先关 HTTP 客户端再关 Milvus——反过来的话，正在等 LLM
        响应的请求会在 Milvus 已经关掉之后才回来，然后在补写日志时拿到一个
        已关闭的连接。
        """
        await self.chat.aclose()
        self.milvus.close()


def build_services(
    settings: Settings | None = None,
    *,
    metadata: PaperMetadataStore | None = None,
    milvus: MilvusIndexClient | None = None,
) -> Services:
    """装配全部 service。**HTTP 与 MCP 两条入口共用这一个函数。**

    :param metadata: 论文级元数据源（DuckDB 读 parquet）。不传时用空实现，
        标题退化为 paper_id——**宁可显示 ID 也不显示一个编出来的标题**。
    :param milvus: 测试注入点。让 e2e 测试能在不起 Milvus 的情况下打通全链路，
        **不为测试在生产代码里加分支**。
    """
    settings = settings or get_settings()
    client = milvus or MilvusIndexClient(settings)
    chat = ChatClient(settings.generation)

    retriever = HybridRetriever(client, settings)
    reranker = Reranker(CrossEncoderReranker(settings.rerank), settings.rerank)
    checker = GroundingChecker(settings.grounding, hhem=HhemClient(settings.grounding))
    gates = GpuGates(settings.concurrency)

    search = SearchPipeline(
        QueryRouter(settings),
        retriever,
        settings,
        metadata=metadata,
        facets=ExactFacetSource(settings.milvus.index_version),
    )
    papers = PaperDetailPipeline(client, settings, metadata=metadata)
    answer = AnswerPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=AnswerGenerator(chat, settings.generation),
        checker=checker,
        hyde=HydeGenerator(chat, settings.rewrite),
        decomposer=QueryDecomposer(chat, settings.rewrite),
        gates=gates,
        settings=settings,
        # 纠正循环默认不装配：评估器与改写器尚未接真实实现，而
        # **没有信号就不该进循环**——盲目改写一次是纯延迟，而且它会把一个健康的
        # 检索结果换成一个未经验证的。
        # 接真实实现时在这里传 `build_correction_graph(CorrectionNodes(...), settings)`。
        correction_graph=None,
    )

    tools = build_registry(ToolContext(search=search, papers=papers))
    _logger.info(
        "services_built",
        collection=settings.milvus.collection_name,
        index_version=settings.milvus.index_version,
        correction_loop=settings.agent.correction_loop_enabled,
        tools=tools.names(),
    )
    return Services(
        settings=settings,
        milvus=client,
        chat=chat,
        gates=gates,
        search=search,
        answer=answer,
        papers=papers,
        tools=tools,
    )
