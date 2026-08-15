"""一键建库：`python scripts/build_index.py [--dry-run]`。

**薄壳**：真正的入口是 `citequest build-index`（`[project.scripts]` 注册的 CLI），
这个文件只是让「不装包也能跑一次」成为可能——运维在一台临时机器上 clone 下来
就想跑，让他先 `uv sync` 再 `uv run citequest ...` 是一道没必要的门槛。

**这是一条昂贵且不可逆的命令**：147 万 chunk 的全量重灌是 6–10 小时量级，而且
analyzer 类型、BM25 参数、`text_exact` 是否存在、占位符策略这四项一旦建库就改不动。
所以 `--dry-run` 不是可选的礼貌，是默认应该先跑一遍的步骤。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 未安装包时也能跑：把 src 加进 path。**只在脚本里做**——生产代码依赖这种
# path 拼接会让 src layout 的全部好处（防止误 import 开发副本）当场失效。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from citequest.cli import app

if __name__ == "__main__":
    # 直接把子命令名塞进 argv，其余参数（--granularity / --dry-run）原样透传。
    app(["build-index", *sys.argv[1:]])
