"""workflow 编排：把各能力域串成两条端到端链路。

- `search.py` —— 纯检索链路：query → 路由 → 并行双路检索 → RRF → grouping → 结果
- `answer.py` —— RAG 链路：检索 → rerank → 生成 → grounding → 流式输出

**这一层是 service 层**，也就是 API 与 MCP 共用的那一层（K11）。业务逻辑到此为止：
`api/` 与 `mcp_server/` 都只调这里，不直接碰 `retrieval` 与 `rag`。这条约束的
价值在 MCP 上最明显——两条入口各写一份编排，行为迟早会静默地不一致，而不一致的
表现是「同一个问题在 Claude 里和在网页里答案不同」，这种 bug 没人报得清楚。

编排形态是 **routing + parallelization 的静态 workflow**，不是 agent：
路由结果决定走哪几个分支，分支之间并行，汇合后继续——控制流写死在代码里。
唯一的例外是 `graph/` 里那条纠正循环。
"""

# `factory.py` **刻意不在这里再导出**：它是**组装根**，同时依赖 `orchestration`
# 与本包自身，而 `orchestration.tool_specs` 又要 import 本包的 pipeline 类。
# 把它放进 `__init__` 会形成包级循环导入（import citequest.orchestration →
# tool_specs → pipelines/__init__ → factory → orchestration.tool_specs 半初始化）。
# 组装根本来就该被显式 import：`from citequest.pipelines.factory import build_services`。
from citequest.pipelines.answer import AnswerCommand, AnswerPipeline
from citequest.pipelines.papers import PaperChunkRef, PaperDetail, PaperDetailPipeline
from citequest.pipelines.search import (
    NullPaperMetadataStore,
    PaperMetadataStore,
    SearchCommand,
    SearchPipeline,
)

__all__ = [
    "AnswerCommand",
    "AnswerPipeline",
    "NullPaperMetadataStore",
    "PaperChunkRef",
    "PaperDetail",
    "PaperDetailPipeline",
    "PaperMetadataStore",
    "SearchCommand",
    "SearchPipeline",
]
