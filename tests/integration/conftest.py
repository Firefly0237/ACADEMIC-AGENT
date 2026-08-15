"""集成层：**连本机真实的 Milvus 与 TEI**（`make up` 起的那两个容器）。

与 unit 的分界只有一条：**要不要外部服务**。所以这一层默认在 CI 上跳过
（`pytest -m "not integration"`），本地开发时用 `make test-integration` 跑。

跳过的方式是**探测连接再 skip**，不是靠环境变量开关：环境变量会漂——
有人本地设了忘了取消，于是集成测试在他机器上永远是绿的（因为根本没跑）。
探测失败给出的 skip 理由里带上 URI，让「为什么跳过了」一眼可见。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from citequest.config.settings import Settings
from citequest.indexing.client import MilvusIndexClient

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def integration_settings() -> Settings:
    """集成测试用的配置：**换一个 collection 名，不碰真库**。

    共用线上 collection 名的代价是一次误跑就可能 drop 掉 147 万 chunk
    ——那是 6–10 小时的全量重灌。名字里带 `_it_` 让它在 `list_collections`
    里一眼可辨。
    """
    return Settings(milvus={"collection_name": "citequest_it_chunks", "alias": "citequest_it"})  # type: ignore[arg-type]


@pytest.fixture(scope="session")
def milvus(integration_settings: Settings) -> Iterator[MilvusIndexClient]:
    """连真实 Milvus。连不上就整体跳过，不让它以失败的形式出现。

    「Milvus 没起」不是一次测试失败，是一个未满足的前置条件——把它报成失败会
    让红色 CI 变成噪声，而噪声一多，真正的失败就没人看了。
    """
    uri = integration_settings.milvus.uri
    try:
        client = MilvusIndexClient(integration_settings)
        client.raw.list_collections()
    except Exception as exc:  # pragma: no cover - 环境相关
        pytest.skip(f"Milvus 不可达（{uri}）：{type(exc).__name__}。用 `make up` 起本地依赖后再跑")

    yield client
    client.close()


@pytest.fixture(scope="session")
def loaded_collection(milvus: MilvusIndexClient, integration_settings: Settings) -> Iterator[str]:
    """建一个只含少量样本的临时 collection，跑完删掉。

    **建库路径必须走生产代码的 `create_collection`**：这条集成测试真正要验的不是
    「能不能插数据」，而是 schema + 两条 Function（BM25 与 TEI Embedding）+ 三类
    索引这一整套声明**在真实 Milvus 上是否被接受**——这件事在单测里无论如何都
    验不了，它正是这一层存在的全部理由。
    """
    name = integration_settings.milvus.collection_name
    milvus.create_collection(drop_existing=True)
    milvus.load()
    milvus.point_alias()
    yield name
    milvus.raw.drop_collection(name)
