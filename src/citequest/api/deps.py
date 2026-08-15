"""依赖注入：从 `app.state` 取 service 单例。

**这一层不构造任何东西。** 装配在 `pipelines.factory.build_services`——
放在那边而不是这里，是因为 MCP adapter 也要装配同一套 service，而它没有 HTTP
概念；装配代码留在 api 层就意味着 `mcp_server` 要 import fastapi 才能起一个
stdio server。

依赖函数只做「从 app.state 取出来」这一件事。用 `Depends` 每请求新建一个
`MilvusIndexClient` 会在每次请求上付一次连接建立的钱，而连接建立恰恰是那种
「平时看不出来、并发一上来就成为瓶颈」的开销；GPU 闸门更严重——每请求新建一个
`asyncio.Semaphore(12)` 等于每个请求都拿到自己的 12 个额度，
**并发闸门直接失效，而且失效得悄无声息**。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from citequest.config.settings import Settings
from citequest.orchestration.tools import ToolRegistry
from citequest.pipelines.answer import AnswerPipeline
from citequest.pipelines.factory import Services, build_services
from citequest.pipelines.papers import PaperDetailPipeline
from citequest.pipelines.search import SearchPipeline

__all__ = [
    "Services",
    "ServicesDep",
    "build_services",
    "get_answer_pipeline",
    "get_app_settings",
    "get_papers_pipeline",
    "get_search_pipeline",
    "get_services",
    "get_tool_registry",
]


def get_services(request: Request) -> Services:
    """从 `app.state` 取单例。**不在这里构造。**"""
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


def get_search_pipeline(services: ServicesDep) -> SearchPipeline:
    return services.search


def get_answer_pipeline(services: ServicesDep) -> AnswerPipeline:
    return services.answer


def get_papers_pipeline(services: ServicesDep) -> PaperDetailPipeline:
    return services.papers


def get_tool_registry(services: ServicesDep) -> ToolRegistry:
    return services.tools


def get_app_settings(services: ServicesDep) -> Settings:
    return services.settings
