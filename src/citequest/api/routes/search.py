"""`GET /v1/search`：纯检索，无 LLM、无 rerank、无 grounding 闸门。

**这是唯一能讲「亚百毫秒」的一条路**（首字节 p50 ≤100ms / p95 ≤250ms
【预算，非实测】）。路由函数里只有三件事：参数校验、调 `pipelines` 的一个函数、
把结果渲染成响应模型。**出现第四件事就说明业务逻辑漏到 HTTP 层了。**
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from citequest.api.deps import ServicesDep
from citequest.api.ratelimit import client_key, quota_headers
from citequest.api.schemas.search import SearchParams, SearchResponse
from citequest.observability.context import current_request_id
from citequest.pipelines.search import SearchCommand

__all__ = ["router"]

router = APIRouter(tags=["search"])


@router.get("/v1/search", response_model=SearchResponse)
async def search(
    request: Request,
    response: Response,
    services: ServicesDep,
    params: Annotated[SearchParams, Depends()],
) -> SearchResponse:
    """论文检索。

    限流用 `enforce`（超限 429 + 配额头），而不是像 AI 路那样静默降级：
    搜索路没有「降级对象」——它本身就是全系统的降级底线，再降就没有东西可给了。
    """
    limiter = request.app.state.search_limiter
    decision = limiter.enforce(client_key(request))
    response.headers.update(quota_headers(decision))

    result = await services.search.run(
        SearchCommand(
            q=params.q,
            page=params.page,
            page_size=params.page_size,
            sort=params.sort,
            granularity=params.granularity,
            year_from=params.year_from,
            year_to=params.year_to,
            primary_category=params.primary_category,
            request_id=current_request_id() or "",
        )
    )
    return SearchResponse.from_domain(result)
