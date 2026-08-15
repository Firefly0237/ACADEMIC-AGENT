"""5 个只读幂等工具的**唯一定义处**（决策记录 D18、M5 §10.2）。

```
search_papers      发现 / 定位：这个方向有哪些工作      → paper 级
search_chunks      取证据：直接可引用的文本片段          → chunk 级（最贵）
get_paper_detail   单篇详情：元数据 + 库内 chunk 列表
structured_filter  纯结构化检索：年份 / 类目 / 粒度
expand_by_citation 引用上下文检索：别人在什么语境下引用 X
```

**加一个工具只改这一个文件**：注册表本身（`tools.py`）里没有任何按工具名分支的
代码，超时、重试、裁剪、配额四件事对所有工具是同一段逻辑，差异全部落在
`ToolSpec` 的字段值上。

`search_papers` 与 `search_chunks` 为什么必须拆成两个（工具粒度的判据是
**返回粒度决定工具边界**）：合成一个会让调用方混淆「我是在找论文还是在找证据」，
而这类混淆是工具选择错误的主要来源。代价是多一个工具、多一次选择判断，
所以它必须靠工具选择准确率这项过程指标来验证【待实测】。

**不做的**：把「文献综述」做成第 6 个工具。两条硬理由——① 它内部还有 LLM 调用，
调用方的全部预算护栏对它失效；② 它对外是个跑几十秒、中途不可观察也不可中断的
黑盒，违背 MCP model-controlled 的协议意图。这类流程该做成 Skill，让步骤对模型
可见、可被改写。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from citequest.core.enums import Granularity, SortMode
from citequest.observability.logging import get_logger
from citequest.orchestration.tools import ToolRegistry, ToolResult, ToolSpec
from citequest.pipelines.papers import PaperDetailPipeline
from citequest.pipelines.search import SearchCommand, SearchPipeline

__all__ = ["ToolContext", "build_registry"]

_logger = get_logger(__name__)

#: 观察层裁剪预算（M5 §11.2）。三档不同不是随手定的：
#: `search_chunks` 回 chunk 原文，是最贵的工具，所以它的预算最紧；
#: `search_papers` 只回标题/年份/片段，同样的 token 能装下更多条。
_BUDGET_PAPERS = 800
_BUDGET_CHUNKS = 800
_BUDGET_DETAIL = 600


@dataclass(slots=True)
class ToolContext:
    """工具的依赖注入口。

    **工具是薄壳**：它调的是 `pipelines` 里那份同样被 HTTP 用着的 service。
    两条入口各写一份编排，行为迟早静默不一致，而不一致的表现是「同一个问题在
    Claude 里和在网页里答案不同」——这种 bug 没人报得清楚。
    """

    search: SearchPipeline
    papers: PaperDetailPipeline


def build_registry(ctx: ToolContext) -> ToolRegistry:
    """装配注册表。**HTTP 与 MCP 共用同一份**。"""
    registry = ToolRegistry()

    async def search_papers(args: dict[str, Any]) -> ToolResult:
        """paper 级：发现 / 定位。只回标题、年份、片段，**不回完整摘要**。"""
        result = await ctx.search.run(_to_command(args, page_size=int(args.get("limit", 10))))
        if not result.papers:
            return ToolResult.empty()
        return ToolResult(
            ok=True,
            data={
                "results": [
                    {
                        "paper_id": p.paper_id,
                        "title": p.title,
                        "year": p.year,
                        "primary_category": p.primary_category,
                        "granularity": p.granularity.value,
                        # 片段而非全文：这是 paper 级工具与 chunk 级工具的边界所在。
                        "snippet": p.best_chunk.text[:200],
                        "license": p.license.value,
                    }
                    for p in result.papers
                ],
                "group_count": result.group_count,
                "retrieval_mode": result.meta.retrieval_mode.value,
            },
        )

    async def search_chunks(args: dict[str, Any]) -> ToolResult:
        """chunk 级：取可直接引用的证据。**最贵的工具，所以 schema 约束最严。**"""
        limit = min(int(args.get("limit", 8)), 16)
        result = await ctx.search.run(_to_command(args, page_size=min(limit, 20)))
        chunks = [chunk for paper in result.papers for chunk in paper.chunks][:limit]
        if not chunks:
            return ToolResult.empty()
        return ToolResult(
            ok=True,
            data={
                "results": [
                    {
                        "chunk_id": c.chunk_id,
                        "paper_id": c.paper_id,
                        "text": c.text,
                        # 引用锚点四元组：全文层能精确到节与段，摘要层只有 paper_id。
                        # **摘要层不伪造段落号**——显示一个不存在的「3.2 节」比不显示更糟。
                        "section": c.section,
                        "sec_number": c.sec_number,
                        "paragraph_idx": c.paragraph_idx,
                        "granularity": c.granularity.value,
                        "license": c.license.value,
                    }
                    for c in chunks
                ]
            },
        )

    async def get_paper_detail(args: dict[str, Any]) -> ToolResult:
        """单篇详情。**paper_id 不存在是业务错误**，结构化返回让模型自己改。

        走的是标量 query 而不是检索——「我已经知道 id」与「找相关的东西」是两件事。
        """
        detail = await ctx.papers.fetch(str(args.get("paper_id", "")))
        if detail is None:
            # 「指名要某一篇却没有」是业务错误而不是空结果：模型改得了
            # （换一个 ID，或先用 search_papers 定位）。空结果那条路对应的是
            # 「条件下没有匹配」，两者给模型的下一步动作不同。
            return ToolResult.param_error(
                "PAPER_NOT_FOUND", f"库内没有 paper_id={args.get('paper_id')}，可先用 search_papers 定位"
            )
        return ToolResult(
            ok=True,
            data={
                "paper_id": detail.paper_id,
                "title": detail.title,
                "year": detail.year,
                "primary_category": detail.primary_category,
                "cited_by_count": detail.cited_by_count,
                "granularity": detail.granularity.value,
                "has_fulltext": detail.has_fulltext,
                "license": detail.license.value,
                # 许可与署名**逐条下发，不做全局声明**：两层许可不同，
                # 汇编不是改编，不能用一句全局混合声明糊过去。
                "attribution": detail.attribution.model_dump(mode="json"),
                "chunks": [{"chunk_id": c.chunk_id, "section": c.section} for c in detail.chunks],
            },
        )

    async def structured_filter(args: dict[str, Any]) -> ToolResult:
        """纯结构化检索：不是独立链路，是「search + 一组非空 filter」。

        单列成一个工具只是为了让调用方能表达「我要的是筛选不是相关性排序」，
        以及让监控看得见结构化条件的占比。
        """
        command = _to_command(args, page_size=int(args.get("limit", 20)))
        if command.year_from is None and command.year_to is None and command.primary_category is None:
            raise ValueError("structured_filter 至少要给一个结构化条件（year_from / year_to / primary_category）")
        result = await ctx.search.run(command)
        if not result.papers:
            return ToolResult.empty()
        return ToolResult(
            ok=True,
            data={
                "results": [
                    {"paper_id": p.paper_id, "title": p.title, "year": p.year, "primary_category": p.primary_category}
                    for p in result.papers
                ],
                "facets_exact": result.facets_exact.model_dump(mode="json") if result.facets_exact else None,
            },
        )

    async def expand_by_citation(args: dict[str, Any]) -> ToolResult:
        """引用上下文检索：**别人在什么语境下引用 X**。

        > **能力边界，主动讲**：这是引用**上下文**检索，不是引用图遍历。
        > 本系统能回答「谁在什么语境下引用了 X」，**不能**回答「X 引了谁、
        > 引用链多深」——因为摘要层的引用关系字段在实测中大面积为空。
        > 这两件事听起来很近，实际是两种完全不同的数据可用性。

        **接真实实现**：正式实现走 `retrieval.filters.build_plan_filter` 的
        `cited_open_alex_id` 参数，把「正文里引用了该 ID」下推成 Milvus 标量过滤，
        再按 query 文本在这批 chunk 内做一次检索。这里先复用一次普通检索并把
        目标 ID 带进 query，**行为是近似的，所以必须标出来而不是装作已经实现**。
        """
        paper_id = str(args.get("paper_id", "")).strip()
        if not paper_id:
            raise ValueError("paper_id 不能为空")
        raise NotImplementedError(
            "接真实实现：引用上下文检索需把 cited_open_alex_id 下推为 Milvus 标量过滤"
            f"（retrieval.filters.build_plan_filter 已有该参数）；目标 paper_id={paper_id}"
        )

    registry.register(
        ToolSpec(
            name="search_papers",
            description="Find papers on a topic. Returns paper-level hits (title, year, snippet), not full text.",
            input_schema=_search_schema(with_limit_default=10),
            handler=search_papers,
            token_budget=_BUDGET_PAPERS,
            timeout_ms=1_000,
        )
    )
    registry.register(
        ToolSpec(
            name="search_chunks",
            description="Retrieve citable text chunks as evidence, with section-level anchors when full text exists.",
            input_schema=_search_schema(with_limit_default=8, limit_max=16),
            handler=search_chunks,
            token_budget=_BUDGET_CHUNKS,
            timeout_ms=1_500,
            # 最贵的工具，配额也最紧：一次返回 48 个 chunk 的文本，
            # token 成本远高于一次 get_paper_detail（M6 §4.5）。
            quota_per_session=15,
        )
    )
    registry.register(
        ToolSpec(
            name="get_paper_detail",
            description="Get metadata, license attribution and in-index chunk list for one paper id.",
            input_schema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "OpenAlex id without prefix, e.g. W2741809807"}
                },
                "required": ["paper_id"],
            },
            handler=get_paper_detail,
            token_budget=_BUDGET_DETAIL,
            timeout_ms=800,
        )
    )
    registry.register(
        ToolSpec(
            name="structured_filter",
            description="Filter papers by year range, arXiv category or granularity without semantic ranking.",
            input_schema=_search_schema(with_limit_default=20, require_query=False),
            handler=structured_filter,
            token_budget=_BUDGET_PAPERS,
            timeout_ms=1_000,
        )
    )
    registry.register(
        ToolSpec(
            name="expand_by_citation",
            description="Find in what context other papers cite a given paper (citation context, not graph traversal).",
            input_schema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string"},
                    "q": {"type": "string", "description": "Optional focus for the citing context"},
                },
                "required": ["paper_id"],
            },
            handler=expand_by_citation,
            token_budget=_BUDGET_CHUNKS,
            timeout_ms=1_500,
        )
    )
    _logger.info("tool_registry_built", tools=registry.names())
    return registry


def _search_schema(*, with_limit_default: int, limit_max: int = 20, require_query: bool = True) -> dict[str, Any]:
    """检索类工具的公共 schema。

    抽出来是因为四个检索类工具的入参有真实的共同结构（M6 §1.2 的那组参数），
    不是为了省几行——schema 各写各的必然漂移，而漂移的表现是同一个 `year_from`
    在两个工具里语义不同。
    """
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "q": {
                "type": "string",
                "maxLength": 500,
                "description": "Query text; >500 chars is rejected, not truncated",
            },
            "year_from": {"type": "integer", "minimum": 1900, "maximum": 2100},
            "year_to": {"type": "integer", "minimum": 1900, "maximum": 2100},
            "primary_category": {"type": "string", "description": "arXiv category such as cs.CL"},
            "granularity": {"type": "string", "enum": [g.value for g in Granularity]},
            "limit": {"type": "integer", "minimum": 1, "maximum": limit_max, "default": with_limit_default},
        },
    }
    if require_query:
        schema["required"] = ["q"]
    return schema


def _to_command(args: dict[str, Any], *, page_size: int) -> SearchCommand:
    """把工具入参翻译成 service 层命令。**校验在这里做，不在 handler 里散着做。**

    年份倒挂在这里就拦掉：它会生成一个恒假的过滤表达式，Milvus 返回空结果且不报错
    ——典型的「查不到东西但没人知道为什么」。抛 `ValueError` 会被注册表翻译成
    参数错误喂回模型，模型改得了。
    """
    q = str(args.get("q", "")).strip()
    year_from = args.get("year_from")
    year_to = args.get("year_to")
    if year_from is not None and year_to is not None and int(year_from) > int(year_to):
        raise ValueError(f"year_from({year_from}) 不能大于 year_to({year_to})")
    granularity = Granularity(args.get("granularity", Granularity.ANY.value))
    return SearchCommand(
        q=q,
        page=1,
        page_size=min(max(page_size, 1), 20),
        sort=SortMode.RELEVANCE,
        granularity=granularity,
        year_from=int(year_from) if year_from is not None else None,
        year_to=int(year_to) if year_to is not None else None,
        primary_category=args.get("primary_category"),
        request_id=str(args.get("request_id", "")),
    )
