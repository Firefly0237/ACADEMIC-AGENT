"""一键跑评测：`python scripts/run_eval.py [--benchmark citrec|litsearch|all] [--config hybrid_rerank]`。

报告写到 `reports/`（JSON + Markdown 各一份，见 `evaluation.report`）。

**跑之前先确认一件事**：头条基准 unarXive_citrec 过滤到库内之后的样本量。
低于 `EvalSettings.citrec_min_subset_size`（3,000）时头条退回 LitSearch，
citrec 降级为补充证据——这条判定在 `evaluation.datasets.citrec.headline_survives`
里，是一个会被真的执行的函数，不是一句文档。

**评测跑的是加大到 `eval_group_limit=100` 的同一条线上管线**（线上搜索路是 60），
这个差异会随配置快照进报告，不是靠人记得写。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from citequest.cli import app

if __name__ == "__main__":
    app(["evaluate", *sys.argv[1:]])
