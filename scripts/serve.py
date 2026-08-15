"""一键起服务：`python scripts/serve.py [--reload]`。

**默认单 worker，且多数情况下不该改**：并发闸门与 GPU 信号量是进程内状态
（rerank 4 / 生成 12 / HHEM 8 / 路由 12），多 worker 会让「rerank 并发 4」实际
变成 4×worker 数，把显存账直接打穿。要扩容就起多个容器，让闸门额度随实例数拆分。

起服务前依赖必须就位（`make up` 拉起 Milvus 2.6 + TEI）：`/readyz` 会跑启动探针，
探针不过就拒绝就绪——**滚动发布会自动停住**，而不是把一个检索质量已经塌掉的
实例放进流量里。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from citequest.cli import app

if __name__ == "__main__":
    app(["serve", *sys.argv[1:]])
