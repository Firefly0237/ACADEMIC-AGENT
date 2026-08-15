"""唯一的 LangGraph 部分：一条窄的检索纠正循环。

**定性必须先说清楚**：本项目主体是 workflow（routing + parallelization），
不是 agent。整个系统里只有这一处存在「模型决定下一步做什么」的循环，
所以不能宣称「做了 agent 系统」。

模块划分：

- `state.py`           —— 图状态定义（TypedDict）
- `nodes.py`           —— 评估检索质量 / 改写 / 重检索三个节点
- `correction_loop.py` —— 图装配，`recursion_limit=8`，循环上限 1 次

三条护栏都是**事先写死**的：

1. `recursion_limit=8` 显式设置。官方默认 1000，离安全边界差两个数量级——
   正因为默认值这么松，显式设才更必要。
2. 循环上限 1 次，单独设 5s 超时，**超时退回首轮结果不报错**。
3. 止损判据：纠正循环相对静态 pipeline 提升 < 5% → 只保留工具化与 MCP，
   砍掉循环。与「HyDE 跑不赢 direct 就下线」是同一套纪律。

检索质量三档（Correct / Ambiguous / Incorrect）**仅概念层引 CRAG**，机制层照
LangGraph 官方 agentic-rag 教程。必须声明的偏离：CRAG 的 Incorrect 分支是
web search，本项目不做——Incorrect 在这里只触发一次改写重检索。
"""

from citequest.graph.correction_loop import CorrectionOutcome, build_correction_graph, run_correction_loop
from citequest.graph.nodes import CorrectionNodes, GradeFn, RewriteFn, decide_after_grade
from citequest.graph.state import CorrectionState

__all__ = [
    "CorrectionNodes",
    "CorrectionOutcome",
    "CorrectionState",
    "GradeFn",
    "RewriteFn",
    "build_correction_graph",
    "decide_after_grade",
    "run_correction_loop",
]
