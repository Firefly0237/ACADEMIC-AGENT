"""`/healthz` 存活与 `/readyz` 就绪（M6 §1.1 / §6.3）。

**匿名调用只返回存活信号，不泄露模型名、版本、endpoint 等配置**：健康检查端点
通常不鉴权，把配置摊在上面等于免费送一份内网拓扑。详情由
`api.healthz_expose_details` 控制，默认关闭。

存活与就绪必须分开：存活挂了要重启容器，就绪没好只是不接流量。合成一个的后果是
——Milvus 还在加载 collection 时，容器被反复重启。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response

from citequest.api.deps import ServicesDep
from citequest.observability.logging import get_logger

__all__ = ["router"]

_logger = get_logger(__name__)

router = APIRouter(tags=["ops"])


@router.get("/healthz")
async def healthz(services: ServicesDep) -> dict[str, Any]:
    """存活探针。**永远快、永远不碰下游。**

    碰下游的存活探针会把一次 Milvus 抖动放大成一次容器重启风暴——
    探针失败 → 重启 → 重新加载 collection → 更慢 → 再次探针失败。
    """
    body: dict[str, Any] = {"status": "ok"}
    if services.settings.api.healthz_expose_details:
        # 只在显式打开时暴露，且**只暴露运行期状态，不暴露 endpoint 与密钥**。
        body["gates"] = services.gates.snapshot()
        body["index_version"] = services.settings.milvus.index_version
    return body


@router.get("/readyz")
async def readyz(services: ServicesDep, response: Response) -> dict[str, Any]:
    """就绪探针：**启动探针通过才算就绪**（M6 §6.3）。

    启动探针是成本极低、价值极高的一项：用 3 条固定探针文本（纯英文 / 中文 /
    含数学符号）调一次编码，把结果与随索引发布的 `probe_vectors.json` 比对，
    余弦相似度低于 1−1e-5 就**拒绝就绪**（滚动发布自动停住）。

    为什么必须比对向量而不是比对模型名：TEI 升级、pooling 默认值变化、`truncate`
    配置漂移，都会静默改变输出向量，而模型名一个字都不会变。**向量变了 →
    新 query 的向量和库里的旧向量不在同一个空间 → 检索质量掉但不报错。**
    一次探针比对就能在启动阶段拦住它。
    """
    ok = services.milvus.startup_probe()
    if not ok:
        # 503 而不是 500：就绪未通过是「还不能接流量」，不是「服务坏了」。
        response.status_code = 503
        _logger.error("readiness_failed", reason="startup_probe")
    return {"status": "ready" if ok else "not_ready", "startup_probe": ok}
