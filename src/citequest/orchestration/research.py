"""跨会话：**研究会话，不是用户画像记忆**（M5 §6）。

| | 做 | 不做 |
|---|---|---|
| 名字 | **研究会话（research session）** | 用户画像记忆 |
| 主键 | `research_id`（用户显式创建或从上一会话续接） | 用户 ID |
| 存什么 | 已读论文集合、已确认结论及其引用边、排除条件、未决问题 | 「这个用户偏好某某方向」 |
| 生命周期 | 跟着一个**研究课题**走，可显式关闭 | 跟着人走，永久累积 |

做研究会话有理由，做用户画像没有：文献综述天然跨天进行——今天读了 15 篇，
明天接着读第 16 篇。这不是「记住这个人」，是「记住这个课题的进度」，
价值一句话说得清：**避免重复推荐已读论文、避免重复确认已确认结论、
让排除条件不用每天重说一遍。**

而用户画像在这个场景里收益为负，理由是具体的：学术检索的意图在会话之间几乎
不相关（今天查 MoE、明天查蛋白质结构），把昨天的兴趣带进今天是**污染而不是
个性化**；它会污染可复现性（同一条 query 在不同历史下返回不同结果，离线评测
数字就不代表线上行为）；它要求用户体系与持久个人数据，而本系统的边界是
**无登录的公开检索**。研究会话的收益反过来是可确定性测量的——重复推荐率是可数的。

**写入时机不是每轮**（M5 §6.2）：只在①一次压缩完成时（结构化字段刚做完集合运算，
是天然的一致点）②会话显式结束时。每轮写意味着一个 20 轮会话产生 20 次持久化写，
其中 18 次的增量只是几个 ID——那是纯写放大。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, Field

from citequest.core.search import PaperHit
from citequest.observability.logging import get_logger
from citequest.orchestration.memory import ConfirmedClaim, ExclusionSet, SessionSummary

__all__ = [
    "InMemoryResearchStore",
    "ResearchSession",
    "ResearchSessionStore",
    "StaleVersionError",
    "filter_already_read",
    "merge_summary_into_research",
]

_logger = get_logger(__name__)

_MAX_CLAIMS = 200


class StaleVersionError(Exception):
    """乐观锁冲突：读到的版本号已经不是最新的。

    调用方的正确反应是**重读 → 合并 → 重试**，不是报错给用户。之所以敢这么做，
    是因为 `read_papers` / `excluded` 这些字段的合并是**集合运算**，
    冲突合并是确定性的，不需要人或模型介入（M5 §14.5）。
    """


class ResearchSession(BaseModel):
    """一个研究课题的持久状态。

    `version` 是乐观锁的载体。**哪些字段能乐观锁、哪些必须单写者，分界线正好
    就是「能不能做集合运算」那条线**（M5 §14.5）：`read_papers` / `excluded` /
    `confirmed_claims` 可以并发合并；`narrative` 不能做集合合并，两个并发写只能
    有一个赢，所以它是单写者字段。同一个设计决定在两个地方各买到一次收益——
    这不是巧合，是把「可合并性」当成字段设计第一原则的结果。
    """

    research_id: str
    read_papers: set[str] = Field(default_factory=set)
    confirmed_claims: list[ConfirmedClaim] = Field(default_factory=list)
    excluded: ExclusionSet = Field(default_factory=ExclusionSet)
    open_questions: list[str] = Field(default_factory=list)
    narrative: str = Field(default="", description="单写者字段：不能做集合合并，并发写只能有一个赢")
    version: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    closed: bool = Field(default=False, description="用户显式关闭或课题结束。关闭后不再参与去重与条件继承")


def merge_summary_into_research(research: ResearchSession, summary: SessionSummary) -> ResearchSession:
    """把一次会话压缩的产物并进研究会话。**三个确定性合并 + 一个覆盖写。**

    `confirmed_claims` 用追加而不是重写：每条 claim 是独立记录，带 `cited` 与
    `turn`，重写会让「这条结论是哪一轮、凭什么确认的」这两条追溯信息一起丢掉。
    超限时按时间序淘汰最旧的——淘汰规则必须确定性，交给 LLM 挑会让它在两次
    合并之间来回漂。
    """
    merged_claims = [*research.confirmed_claims, *summary.confirmed_claims]
    merged_claims.sort(key=lambda c: c.confirmed_at)
    return research.model_copy(
        update={
            "read_papers": research.read_papers | summary.read_papers,
            "confirmed_claims": merged_claims[-_MAX_CLAIMS:],
            "excluded": research.excluded.merged_with(summary.excluded),
            "open_questions": list(dict.fromkeys([*research.open_questions, *summary.open_questions])),
            # narrative 是单写者字段：这里直接覆盖，并发安全由「只有压缩完成与会话
            # 结束两个写点」保证（写点少本身就是最有效的并发控制）。
            "narrative": summary.narrative or research.narrative,
            "version": research.version + 1,
            "updated_at": datetime.now(UTC),
        }
    )


def filter_already_read(hits: list[PaperHit], research: ResearchSession) -> tuple[list[PaperHit], list[PaperHit]]:
    """按已读集合把结果分成「新的」与「已读的」两堆。

    **不直接丢掉已读论文**，返回两堆让调用方决定：综述场景要的是「别再推荐一遍」，
    而「这篇你上周读过」本身也是有用的信息。直接过滤掉会让用户以为库里没有它，
    那是把一个去重优化变成了一个召回缺口。

    这条是研究会话里**唯一确定性可数**的收益（M5 §6.3）：已读论文在结果中的
    占比是可以直接统计的，不需要人工判断。
    """
    if research.closed:
        return hits, []
    fresh = [h for h in hits if h.paper_id not in research.read_papers]
    seen = [h for h in hits if h.paper_id in research.read_papers]
    if seen:
        _logger.info("research_dedup", research_id=research.research_id, fresh=len(fresh), already_read=len(seen))
    return fresh, seen


class ResearchSessionStore(Protocol):
    """持久化接口。

    **后端尚未选型**（决策记录 U8）：设计上与检索纠正循环的 checkpointer
    共用同一套持久化后端，不引入第二种存储。官方明确 InMemorySaver 只供开发测试、
    进程重启即丢，所以**在后端选定之前不能声称已具备持久化能力**。
    定义成 Protocol 就是为了让后端选型时只换一个实现类。
    """

    def load(self, research_id: str) -> ResearchSession | None: ...

    def save(self, session: ResearchSession, *, expected_version: int) -> ResearchSession: ...


class InMemoryResearchStore:
    """进程内实现：**只供开发与测试**，进程重启即丢。

    它存在的意义是让上层逻辑（合并、去重、条件继承、乐观锁重试）能在没有任何
    外部依赖的情况下被单测覆盖——这些逻辑才是这一层的价值所在，存储介质不是。
    """

    def __init__(self) -> None:
        self._data: dict[str, ResearchSession] = {}

    def load(self, research_id: str) -> ResearchSession | None:
        return self._data.get(research_id)

    def save(self, session: ResearchSession, *, expected_version: int) -> ResearchSession:
        """带版本校验的写入。冲突时抛 `StaleVersionError`，由调用方重读合并重试。"""
        current = self._data.get(session.research_id)
        if current is not None and current.version != expected_version:
            raise StaleVersionError(
                f"research_id={session.research_id} 版本冲突：期望 {expected_version}，实际 {current.version}"
            )
        self._data[session.research_id] = session
        _logger.info("research_saved", research_id=session.research_id, version=session.version)
        return session
