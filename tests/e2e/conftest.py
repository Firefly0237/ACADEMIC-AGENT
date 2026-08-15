"""e2e 层的装配：**用一个假 Milvus 打通全链路，不起任何容器**。

注入点是生产代码里本来就有的那两个（`build_services(milvus=...)` 与
`create_app(services=...)`）——**不为测试在生产代码里加分支**，
注入一个参数比 `if testing:` 干净得多。

假的只有 Milvus 这一层。路由、过滤表达式装配、grouping 解析、rerank 降级、
grounding 闸门、SSE 封装全部走真实实现，所以这一层测的是**真的链路**，
只是数据源是确定的。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from citequest.api.main import create_app
from citequest.config.settings import Settings
from citequest.indexing.client import MilvusIndexClient
from citequest.pipelines.factory import build_services

pytestmark = pytest.mark.e2e


class FakeRawMilvus:
    """`pymilvus.MilvusClient` 的替身。**只实现被真实调到的那几个方法。**

    形状照 pymilvus 的实际返回构造：`hybrid_search` 返回 `[[row, ...]]`
    （每条 query 一个列表），row 是 `{"id", "distance", "entity"}`。
    假数据的形状错了，基于它写的解析代码就是错的，而且这类错误会一直等到接真实
    Milvus 才暴露。
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.calls: list[dict[str, Any]] = []

    def hybrid_search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        # 把调用参数留下来：e2e 要断言的一件事正是「grouping 的三个 kwarg 真的传下去了」。
        self.calls.append(kwargs)
        return [self._rows]

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [row["entity"] for row in self._rows]

    def close(self) -> None:
        return None


@pytest.fixture
def settings() -> Settings:
    """e2e 用的配置：**只压缩超时预算，其余一律用默认值**。

    覆盖 `timeouts` 是必要的——生成与 HyDE 在骨架阶段必然失败，而失败前要等满
    线上预算（HyDE 2s + 首 token 6s），六条 SSE 用例加起来就是半分钟的空等。
    压缩的只有「等多久才认定失败」，**不动任何影响结果的参数**：路由、检索、
    融合、rerank、grounding 全部还是参数正本里的那份，否则这条 e2e 就不代表线上。
    """
    return Settings(timeouts={"hyde_ms": 50, "first_token_ms": 100, "correction_loop_ms": 100})  # type: ignore[arg-type]


@pytest.fixture
def milvus_rows(fake_hybrid_search_hits: list[dict[str, Any]], settings: Settings) -> list[dict[str, Any]]:
    """给假 Milvus 行补齐解析所需字段（chunk_id + 文本字段 + source_dataset）。"""
    text_field = settings.milvus.text_en_field
    rows: list[dict[str, Any]] = []
    for row in fake_hybrid_search_hits:
        entity = dict(row["entity"])  # type: ignore[arg-type]
        entity.setdefault("chunk_id", row["id"])
        entity.setdefault(text_field, "Rotary position embedding improves length extrapolation on long context.")
        entity.setdefault("source_dataset", "openalex" if entity.get("license") == "CC0" else "unarXive_open_subset")
        rows.append({**row, "entity": entity})
    return rows


@pytest.fixture
def fake_milvus(milvus_rows: list[dict[str, Any]]) -> FakeRawMilvus:
    return FakeRawMilvus(milvus_rows)


@pytest.fixture
def app(settings: Settings, fake_milvus: FakeRawMilvus) -> FastAPI:
    client = MilvusIndexClient(settings, client=fake_milvus)  # type: ignore[arg-type]
    services = build_services(settings, milvus=client)
    return create_app(settings, services=services)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    # `with` 是必须的：lifespan 只在进入上下文时才跑，而 `app.state.services`
    # 与两个限流器都是在 lifespan 里装配的——不进上下文就会拿到一个没有 state 的 app。
    with TestClient(app) as test_client:
        yield test_client
