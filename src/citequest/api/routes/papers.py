"""`GET /v1/papers/{paper_id}` 与 `GET /v1/facets`。

两个端点都不在检索链路上，共性是「按已知条件取确定的数据」而不是「找相关的东西」。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path

from citequest.api.deps import ServicesDep
from citequest.api.schemas.papers import FacetsResponse, PaperDetailResponse
from citequest.core.errors import InvalidParamError
from citequest.observability.logging import get_logger

__all__ = ["router"]

_logger = get_logger(__name__)

router = APIRouter(tags=["papers"])


@router.get("/v1/papers/{paper_id}", response_model=PaperDetailResponse)
async def paper_detail(
    services: ServicesDep,
    paper_id: Annotated[str, Path(max_length=64, pattern=r"^[A-Za-z0-9._\-/]+$")],
) -> PaperDetailResponse:
    """单篇详情。

    `paper_id` 的字符集在路径参数上就限死：它会被拼进 Milvus 的过滤表达式，
    不限制字符集就是一个表达式注入面。**限制放在最外层**，而不是等到拼表达式
    那一行才防——那时候值已经流经了日志与 trace。

    不存在时抛 `InvalidParamError`（→ 400）而不是 404：本项目的错误契约里没有
    404 这一档，前端按 `code` 分支而不是按状态码分支。**加一档状态码就要同时改
    前端状态机**，而这一个场景不值得。
    """
    detail = await services.papers.fetch(paper_id)
    if detail is None:
        raise InvalidParamError(f"paper_id={paper_id} 不存在或已撤稿")
    return PaperDetailResponse.from_domain(detail)


@router.get("/v1/facets", response_model=FacetsResponse)
async def facets(services: ServicesDep) -> FacetsResponse:
    """全库精确分面（年份直方图、领域论文总量），**与检索解耦**。

    挂了就返回 `null` 而不是报错：筛选器不显示精确计数是可以接受的降级，
    而把整个页面变成错误页不行。这一条对应降级矩阵里的 `no_exact_facets`。
    """
    exact = await services.search.exact_facets()
    _logger.info("facets_requested", available=exact is not None)
    return FacetsResponse(facets_exact=exact)
