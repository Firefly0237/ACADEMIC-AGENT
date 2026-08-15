"""FastAPI app 工厂、lifespan、中间件与路由挂载。

**服务层是 FastAPI + asyncio。** 一条 RAG 请求的墙钟时间是数秒，但这数秒里
应用进程几乎不消耗 CPU——它在等 Milvus 返回、等 TEI 编码、等 GPU 吐 token。
事件循环可以同时挂住成百上千条这样的连接，**真正的并发上限不在应用层，
而在后端 GPU 的吞吐**。所以扩容要扩 GPU 服务，加应用副本只会让排队队列变长。

这个模型有两个必须守住的前提，破一个就全盘失效：

1. **请求路径上不能有阻塞调用。** 一个同步的 `requests.get`、一次没走线程池的
   模型前向，就能让整个事件循环停摆，此时所有在途的 SSE 流一起卡住。
   本项目里 pymilvus 与 cross-encoder 都是同步的，所以它们一律走
   `asyncio.to_thread`（见 `pipelines/`）。
2. **CPU 密集的小动作也要盯着。** 长回答的 JSON 序列化、句边界正则、gzip 都是
   真 CPU。单次几毫秒，在几百条并发流下就是可观的事件循环延迟。

**默认单 worker**：并发闸门与 GPU 信号量是进程内状态，多 worker 会让
「rerank 并发 4」实际变成 4×worker 数，把显存账直接打穿。扩容靠多容器。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from citequest.api.deps import Services, build_services
from citequest.api.errors import install_exception_handlers
from citequest.api.middleware import RequestContextMiddleware
from citequest.api.ratelimit import limiter_from_settings
from citequest.api.routes import ALL_ROUTERS
from citequest.config.settings import Settings, get_settings
from citequest.observability.logging import configure_logging, get_logger

__all__ = ["app", "create_app"]

_logger = get_logger(__name__)


def create_app(settings: Settings | None = None, *, services: Services | None = None) -> FastAPI:
    """构造应用。

    :param services: 测试注入点。e2e 测试用 `TestClient` 打全链路时传一份用假
        Milvus 装配的 `Services`——**不为测试在生产代码里加分支**，
        注入一个参数比 `if testing:` 干净得多。
    """
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """启动装配、关闭释放。

        **重对象在这里建一次**：Milvus 连接、reranker、GPU 闸门、限流器。
        闸门与限流器必须是进程级单例——每请求新建一个 `asyncio.Semaphore(12)`
        等于每个请求都拿到自己的 12 个额度，并发闸门直接失效，
        而且失效得悄无声息（显存打穿之前一切正常）。
        """
        application.state.services = services or build_services(settings)
        application.state.search_limiter = limiter_from_settings(settings.api, ai_path=False)
        application.state.answer_limiter = limiter_from_settings(settings.api, ai_path=True)
        _logger.info(
            "app_started",
            env=settings.env,
            index_version=settings.milvus.index_version,
            # 这三个参数打进启动日志是刻意的：出现「线上表现和设计文档对不上」时，
            # 第一件事就是确认到底哪几项被环境变量改掉了。
            sem_generate=settings.concurrency.sem_generate,
            rate_limit_search=settings.api.rate_limit_search_per_minute,
            rate_limit_answer=settings.api.rate_limit_answer_per_minute,
        )
        try:
            yield
        finally:
            await application.state.services.aclose()
            _logger.info("app_stopped")

    application = FastAPI(
        title="CiteQuest",
        version="0.1.0",
        description="学术检索与 RAG：混合检索 + 论文级聚合 + 可验证引用",
        root_path=settings.api.root_path,
        lifespan=lifespan,
        # 默认的 422 校验错误格式会绕过统一错误信封，所以 `api.errors` 里
        # 专门装了一个 `RequestValidationError` 处理器把它翻译成 400。
    )

    application.add_middleware(RequestContextMiddleware, settings=settings.api)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        # 前端要读 request_id 才能提交反馈；跨域下不显式 expose，
        # 浏览器根本看不见这个响应头——反馈闭环会在最后一步断掉。
        expose_headers=["X-Request-Id", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
        allow_headers=["X-Request-Id", "Content-Type"],
    )

    install_exception_handlers(application)
    for router in ALL_ROUTERS:
        application.include_router(router)
    return application


#: uvicorn 的入口对象（`cli.serve` 用字符串路径 `citequest.api.main:app` 引用它，
#: 因为热重载要求 uvicorn 自己能重新导入模块）。
app = create_app()
