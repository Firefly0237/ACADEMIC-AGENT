"""编排层的横切能力：工具注册表、会话状态、研究会话、冲突、衰减、GPU 闸门。

**这一层不是「agent」，命名上也不叫。** 主体编排（routing + parallelization 的
workflow）在 `pipelines/`，唯一那条 agentic 循环在 `graph/`；本包放的是**两者
共用、且 HTTP 与 MCP 两个 adapter 都要用**的东西：

| 模块 | 职责 | 设计文档 |
|---|---|---|
| `tools.py` | 工具注册表机制：数据结构而非 if-else，两类失败语义，返回裁剪 | M5 §12.2 / §13.2 |
| `tool_specs.py` | **5 个只读幂等工具的唯一定义处**——加工具只改这一个文件 | M5 §10.2 |
| `memory.py` | 会话内状态、压缩双闸门、增量叠加、原文保留 N 轮 | M5 §4 / §5 |
| `research.py` | 跨会话**研究会话**（不做用户画像记忆）、乐观锁、去重 | M5 §6 |
| `conflicts.py` | 四种冲突形态的四条裁决规则（C3 **不裁决**） | M5 §7 |
| `decay.py` | 时间衰减**只在记忆层**、同年同领域高引例外 | M5 §8 |
| `gates.py` | GPU 信号量 + 有界队列 + 排队超时 | M6 §5.2 |

放在这里而不是 `pipelines/` 的判据：**被两条以上链路共用，且本身不含链路顺序**。
一旦某个模块开始规定「先做 A 再做 B」，它就该搬去 `pipelines/`。
"""

from citequest.orchestration.conflicts import (
    ConflictGroup,
    ConflictKind,
    VersionMergeResult,
    detect_claim_conflicts,
    exclude_retracted,
    merge_study_versions,
)
from citequest.orchestration.decay import (
    CitationPercentileTable,
    DecayPolicy,
    RevalidationVerdict,
    assert_ranking_is_decay_free,
    needs_revalidation,
)
from citequest.orchestration.gates import GateBusy, GateName, GpuGates
from citequest.orchestration.memory import (
    CompressionResult,
    ConfirmedClaim,
    ExclusionSet,
    SessionState,
    SessionSummary,
    Turn,
    compress,
    estimate_tokens,
)
from citequest.orchestration.research import (
    InMemoryResearchStore,
    ResearchSession,
    ResearchSessionStore,
    StaleVersionError,
    filter_already_read,
    merge_summary_into_research,
)
from citequest.orchestration.tool_specs import ToolContext, build_registry
from citequest.orchestration.tools import (
    ToolCallLog,
    ToolErrorClass,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    clip_observation,
)

__all__ = [
    "CitationPercentileTable",
    "CompressionResult",
    "ConfirmedClaim",
    "ConflictGroup",
    "ConflictKind",
    "DecayPolicy",
    "ExclusionSet",
    "GateBusy",
    "GateName",
    "GpuGates",
    "InMemoryResearchStore",
    "ResearchSession",
    "ResearchSessionStore",
    "RevalidationVerdict",
    "SessionState",
    "SessionSummary",
    "StaleVersionError",
    "ToolCallLog",
    "ToolContext",
    "ToolErrorClass",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "Turn",
    "VersionMergeResult",
    "assert_ranking_is_decay_free",
    "build_registry",
    "clip_observation",
    "compress",
    "detect_claim_conflicts",
    "estimate_tokens",
    "exclude_retracted",
    "filter_already_read",
    "merge_study_versions",
    "merge_summary_into_research",
    "needs_revalidation",
]
