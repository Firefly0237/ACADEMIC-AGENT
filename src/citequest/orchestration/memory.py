"""会话内状态与上下文压缩（M5 §4、§5）。

**它是状态，不是 memory。** 会话内需要跨轮携带的东西是一个显式状态对象，跟着
thread 走、thread 结束就没了。真正配叫记忆的是跨会话那一层（`research.py`），
因为它有独立的生命周期、独立的主键、独立的冲突裁决规则。这个命名不是措辞洁癖：
叫它状态，「你的 memory 和一个 dict 有什么区别」这个问题就不成立；叫它 memory，
就得编出一个区别来。

三层容器，各有各的去处（M5 §5.3）——**这是整个压缩设计里成本收益最悬殊的一条**：

| 层 | 存什么 | 进 planner 上下文吗 | 保留多久 |
|---|---|---|---|
| 最近 N 轮原文 | 用户原话 + 回答摘要，N=2 | **进** | 滚动保留最近 2 轮 |
| 更早轮次 | 只留结构化摘要 | **进**（只有摘要） | 整个会话 |
| 原始 chunk 全文 | `evidence_pool` | **不进** | 整个会话，按 chunk_id 随时可取 |

账很直接：RAG 路单轮取回 ≤48 chunk × 450 token ≈ 21,600 token，重排后进生成
上下文 8 条 ≈ 3,600 token，而进会话上下文的观察摘要上限 800 token/轮。
**21,600 → 3,600 → 800，两次收敛（6× 与 4.5×）。** 原始 chunk 直接留在会话
上下文里的话，三轮就是 64,800 token——压缩再精巧也救不回来，因为问题不在压缩，
在于一开始就不该放进去。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from citequest.config.agent import MemorySettings
from citequest.core.enums import RetrievalMode
from citequest.core.search import RetrievedChunk
from citequest.observability.logging import get_logger

__all__ = [
    "CompressionResult",
    "ConfirmedClaim",
    "ExclusionSet",
    "SessionState",
    "SessionSummary",
    "Turn",
    "compress",
    "estimate_tokens",
]

_logger = get_logger(__name__)

#: 与 `rag.context` / `orchestration.tools` 同一套粗估口径。三处必须一致，
#: 否则「800 token 的观察摘要」在不同模块里是不同的量。
_CHARS_PER_TOKEN = 4.0

#: `open_questions` 与 `confirmed_claims` 的条数上限。超限按时间序淘汰最旧的——
#: 淘汰规则必须是确定性的，交给 LLM 挑「哪条不重要」会让摘要在压缩之间来回漂。
_MAX_CLAIMS = 40
_MAX_OPEN_QUESTIONS = 20


def estimate_tokens(text: str) -> int:
    """粗估 token 数。压缩闸门是一道闸不是一次计费，差 5% 不改变任何决策。"""
    return int(len(text) / _CHARS_PER_TOKEN) + 1


class ConfirmedClaim(BaseModel):
    """一条已确认的结论。

    **必须带 `cited`（引用边）与 `turn`（第几轮确认的）**：冲突裁决要能追到证据
    来源（`conflicts.py`），时间衰减要能追到确认时刻（`decay.py`）。
    一个不能追溯来源的「已确认结论」在学术场景里等于一句无出处的断言，不如不存。
    """

    claim: str
    cited: list[str] = Field(default_factory=list, description="支撑这条结论的 paper_id")
    turn: int = Field(ge=0)
    year_range: str | None = None
    confirmed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExclusionSet(BaseModel):
    """用户显式排除的条件。**确定性合并**，不经过 LLM。

    多轮追问里有相当一部分是**过滤条件的追加**而不是新的语义检索
    （「只看 2023 年之后的」）。把它识别出来走标量过滤，比让它进向量空间准确得多，
    而且零延迟——过滤条件下推到检索侧本来就要做。
    """

    year_min: int | None = None
    year_max: int | None = None
    exclude_papers: set[str] = Field(default_factory=set)
    exclude_categories: set[str] = Field(default_factory=set)

    def merged_with(self, other: ExclusionSet) -> ExclusionSet:
        """条件合并：区间取交、集合取并。**确定性操作，不会漂移。**"""
        return ExclusionSet(
            year_min=max(filter(None, [self.year_min, other.year_min]), default=None),
            year_max=min(filter(None, [self.year_max, other.year_max]), default=None),
            exclude_papers=self.exclude_papers | other.exclude_papers,
            exclude_categories=self.exclude_categories | other.exclude_categories,
        )


class SessionSummary(BaseModel):
    """压缩后的会话摘要（M5 §5.4）。

    **必须是结构化对象而不是一段自由文本。** 自由文本没有可做集合运算的部分，
    只能全交给 LLM，于是每一次压缩都是一次全字段的漂移机会。这里四个字段里
    只有 `narrative` 交给 LLM，前三个是集合运算——即使 narrative 漂了，
    「读过哪些论文、确认了什么、排除了什么」这三件学术检索里真正重要的事是精确的。
    """

    read_papers: set[str] = Field(default_factory=set, description="集合并集，确定性，不漂移")
    confirmed_claims: list[ConfirmedClaim] = Field(default_factory=list, description="追加，不重写，不漂移")
    excluded: ExclusionSet = Field(default_factory=ExclusionSet, description="条件合并，确定性，不漂移")
    open_questions: list[str] = Field(default_factory=list, description="追加 + 超限淘汰")
    narrative: str = Field(default="", description="≤400 token，**唯一交给 LLM 重写的字段，唯一会漂移的字段**")

    def token_estimate(self) -> int:
        return (
            estimate_tokens(self.narrative)
            + len(self.read_papers) * 4
            + estimate_tokens("".join(c.claim for c in self.confirmed_claims))
        )


class Turn(BaseModel):
    """一轮对话。

    `answer_digest` 是**观察摘要**不是完整回答（上限 800 token/轮）：原始 chunk
    从来就不进这里，它在 `evidence_pool` 里。
    """

    index: int = Field(ge=0)
    user_query: str
    answer_digest: str = Field(default="", description="≤800 token 的观察摘要")
    cited_papers: list[str] = Field(default_factory=list)
    retrieval_mode: RetrievalMode | None = None

    def token_estimate(self) -> int:
        # 单轮进入会话上下文的增量 ≈ 900 token（观察摘要上限 800 + query 与结构开销约 100），
        # 两个压缩闸门的取值都是从这个数推出来的。
        return estimate_tokens(self.user_query) + estimate_tokens(self.answer_digest) + 25


@dataclass(slots=True)
class CompressionResult:
    """一次压缩的记账（M5 §5.2）。

    `input_tokens` 必须记：增量叠加相对全量重算的收益是 O(n) vs O(n²)，
    而这个收益只有在累计输入 token 被记下来时才是**可验证**的，否则它只是一段推导。
    """

    summary: SessionSummary
    input_tokens: int
    incremental: bool
    compressed_turns: int
    narrative_rewritten: bool


class SessionState(BaseModel):
    """会话内的显式状态对象（M5 §4.1）。

    落在 LangGraph 官方的 thread 内持久化扩展点（checkpointer）上，不引入第三种
    存储抽象。**注意 checkpointer 后端尚未选型**（决策记录 U8）——在后端定下来
    之前，这一层只能说「设计上落在官方扩展点」，不能说已具备持久化能力。
    """

    model_config = {"arbitrary_types_allowed": True}

    thread_id: str
    turns: list[Turn] = Field(default_factory=list)
    summary: SessionSummary = Field(default_factory=SessionSummary)
    focus_papers: list[str] = Field(
        default_factory=list, description="最近被讨论的 paper_id，只进指代消解不进生成 prompt"
    )
    evidence_pool: dict[str, RetrievedChunk] = Field(
        default_factory=dict,
        description="本会话检索到的**原始 chunk 全文**。**永远不进 planner 上下文**，只在生成节点按 chunk_id 取用",
    )
    retrieval_mode_log: list[RetrievalMode] = Field(default_factory=list, description="只进可观测，不回灌模型")
    turns_since_compress: int = Field(default=0, ge=0)

    # ---- 写入 -----------------------------------------------------------

    def record_turn(self, turn: Turn, chunks: list[RetrievedChunk] | None = None) -> None:
        """记一轮。原始 chunk 进 `evidence_pool`，**不进 `turns`**。

        证据的完整性和上下文的经济性是两个目标，用两个容器分别满足，
        而不是用一个容器折中：生成节点要按 chunk_id 精确取原文做引用锚点
        （锚点细化到 paper_id / section / sec_number / paragraph_idx），
        摘要过的文本承担不了这个职责。
        """
        self.turns.append(turn)
        self.turns_since_compress += 1
        if turn.retrieval_mode is not None:
            self.retrieval_mode_log.append(turn.retrieval_mode)
        for paper_id in turn.cited_papers:
            if paper_id in self.focus_papers:
                self.focus_papers.remove(paper_id)
            self.focus_papers.insert(0, paper_id)
        for chunk in chunks or []:
            self.evidence_pool[chunk.chunk_id] = chunk

    # ---- 读取 -----------------------------------------------------------

    def planner_context(self, keep_raw_turns: int) -> dict[str, object]:
        """进 planner 上下文的东西——**只有摘要 + 最近 N 轮原文**。

        N=2 的推导：指代消解的先行词在自然语言里绝大多数落在最近一到两个话轮内。
        N=1 会漏掉「轮 1 提到论文 A，轮 2 岔开问了别的，轮 3 说『它』」这类跨一轮
        的指代；N=3 起收益迅速衰减而常驻成本线性上涨（每轮约 900 token）。
        **N=2 是「上一轮 + 上上一轮」的最小冗余**【设计】【待实测】。

        **这个方法是 `evidence_pool` 不外泄的唯一保证**：它是 planner 上下文的
        唯一构造入口，返回值里没有 `evidence_pool`，所以「原始 chunk 不进上下文」
        这条是结构上的，不是靠调用方自觉。
        """
        recent = self.turns[-keep_raw_turns:] if keep_raw_turns > 0 else []
        return {
            "summary": self.summary.model_dump(mode="json"),
            "recent_turns": [t.model_dump(mode="json") for t in recent],
            "focus_papers": self.focus_papers[:5],
        }

    def context_tokens(self) -> int:
        """当前会话上下文的累计 token，token 水位闸门用它。"""
        return sum(t.token_estimate() for t in self.turns) + self.summary.token_estimate()

    def should_compress(self, settings: MemorySettings) -> bool:
        """压缩触发：**轮数闸门 + token 水位闸门，任一命中即压**（M5 §5.1）。

        为什么必须是两个闸门而不是一个：

        - **只有轮数闸门** → 一次粘贴 3,000 token 的摘要，三轮就撑爆上下文，
          而计数器才走到 3；
        - **只有水位闸门** → 二十轮短问答（每轮 200 token）累计才 4,000，
          永远不触发压缩，但**上下文腐化已经发生**了——不是长度问题，
          是早期的无关内容一直在干扰当前判断。

        两个阈值（6 轮 / 8,000 token）都是【设计】值，推导可复算但未在真实多轮
        日志上标定（决策记录 U8 清单第 2 项）。
        """
        if self.turns_since_compress >= settings.compress_after_turns:
            return True
        return self.context_tokens() >= settings.compress_after_tokens


def compress(
    state: SessionState,
    settings: MemorySettings,
    *,
    narrative_writer: Callable[[str, list[Turn]], str] | None = None,
) -> CompressionResult:
    """压缩会话上下文：结构化字段做集合运算，只有 narrative 交给 LLM（M5 §5.2）。

    **增量叠加而不是全量重算。** 设每轮进上下文约 900 token、每 6 轮压一次、
    摘要上限 400 token：

    | 方案 | 第 n 次压缩的输入 | n 次累计 |
    |---|---|---|
    | 全量重算 | 6n × 900 = 5,400n | 2,700·n(n+1)，**O(n²)** |
    | 增量叠加 | 400 + 5,400 = **5,800（与 n 无关）** | 5,800n，**O(n)** |

    n=10（60 轮）时 297,000 vs 58,000，差 5.1×；n=20 时差到 9.8×。
    **选增量不是省了 5 倍，是把一个二次增长的成本换成了线性。**

    代价是误差累积：摘要的摘要会漂移。处理方式是把「可增量的部分」从「会漂移的
    部分」里切出去——前三个字段是集合运算（无损），只有 `narrative` 交给 LLM。

    :param narrative_writer: 注入的 LLM 重写函数 `(旧 narrative, 新增轮次) -> 新 narrative`。
        **接真实实现**：正式实现里它是一次 `ChatClient.complete` 调用，
        prompt 在 `rag/prompts/` 下按 §压缩单列一份，输出上限 `summary_max_tokens`。
        传 None 时退化为「保留旧 narrative 不重写」——这是**降级不是等价实现**，
        它让压缩在没有 LLM 的环境下仍然把三个确定性字段合并掉。
    """
    pending = state.turns[-state.turns_since_compress :] if state.turns_since_compress else []
    old = state.summary

    # ---- 确定性部分：集合并 / 条件合并 / 追加。**这三行永远不漂移。** ----
    read_papers = set(old.read_papers)
    for turn in pending:
        read_papers.update(turn.cited_papers)

    claims = list(old.confirmed_claims)[-_MAX_CLAIMS:]
    open_questions = list(old.open_questions)[-_MAX_OPEN_QUESTIONS:]

    # ---- 会漂移的部分：narrative，唯一交给 LLM 的字段 ----
    input_tokens = estimate_tokens(old.narrative) + sum(t.token_estimate() for t in pending)
    if not settings.incremental_compression:
        # 全量重算：读前 n 轮全部原文。留这条分支是为了让 M5 §5.2 的「增量 vs 重算」
        # 对照实验能真的跑起来——一个不能被关掉的优化，它的收益就没法被证明。
        input_tokens = sum(t.token_estimate() for t in state.turns)
        pending = list(state.turns)

    narrative = old.narrative
    rewritten = False
    if narrative_writer is not None and pending:
        narrative = narrative_writer(old.narrative, pending)
        rewritten = True
        max_chars = int(settings.summary_max_tokens * _CHARS_PER_TOKEN)
        if len(narrative) > max_chars:
            narrative = narrative[:max_chars]

    summary = SessionSummary(
        read_papers=read_papers,
        confirmed_claims=claims,
        excluded=old.excluded,
        open_questions=open_questions,
        narrative=narrative,
    )
    state.summary = summary
    compressed = state.turns_since_compress
    state.turns_since_compress = 0

    _logger.info(
        "session_compressed",
        thread_id=state.thread_id,
        incremental=settings.incremental_compression,
        compressed_turns=compressed,
        input_tokens=input_tokens,
        read_papers=len(read_papers),
        narrative_rewritten=rewritten,
    )
    return CompressionResult(
        summary=summary,
        input_tokens=input_tokens,
        incremental=settings.incremental_compression,
        compressed_turns=compressed,
        narrative_rewritten=rewritten,
    )
