"""`/v1/answer` 的请求模型与 SSE 事件载荷（M6 §2.1）。

事件载荷单独建模，而不是直接把内部对象 dump 出去，理由和 §common 一样：
前端按这些字段写状态机，内部对象的每一次重构都会变成一次对外破坏性变更。

**`ai_answer_flagged` 只能降级、不能撤回**：它永远指向一个已经推送出去的 `seq`。
这条不变量在类型上是可见的——`AiAnswerFlagged` 里有 `seq` 却没有任何「删除」
或「替换」字段。token 一旦出网就收不回来，这不是实现不够好，是不可撤回性本身；
工业界对这一点有公开表态（NVIDIA NeMo Guardrails 的流式文档直接承认 rail 触发时
违规文本可能已经发送给用户，如何处理归调用方负责）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from citequest.api.schemas.common import PaperCard
from citequest.api.schemas.search import SearchParams
from citequest.core.enums import CitationSupport, DegradeReason, Granularity

__all__ = [
    "AiAnswerDelta",
    "AiAnswerDone",
    "AiAnswerError",
    "AiAnswerFlagged",
    "AiAnswerStart",
    "AnswerParams",
    "SearchResultsEvent",
    "SourceRef",
]


class AnswerParams(SearchParams):
    """与 `/v1/search` 共用同一组参数，**忽略 `page`**。

    继承而不是复制：两处参数校验规则漂开的表现是「同一个 query 在搜索页能搜、
    在问答页报 400」，而这类不一致没人报得清楚。
    `page` 字段仍然存在但不生效——**翻页请求不重新生成回答**，保留第 1 页已生成的。
    """

    language: str = Field(default="EN", pattern="^(EN|ZH)$", description="回答语言与 query 语言一致")


class SourceRef(BaseModel):
    """`ai_answer_start` 里的一条引用来源。

    带 `granularity` 与 `section` 是为了让前端的引用跳转能分粒度处理：
    全文层定位到具体段落并显示「3.5 节」，摘要层只定位到论文卡片，
    **不伪造段落级定位**。
    """

    citation_id: int = Field(ge=1)
    paper_id: str
    title: str
    year: int
    granularity: Granularity
    section: str | None = None
    sec_number: str | None = None


class AiAnswerStart(BaseModel):
    """首句**通过 grounding 闸门之后**才发。

    反过来说：没有 `ai_answer_start` 就直接 `ai_answer_error`，意味着首句没过闸、
    系统选择了完全不进流式——此时一个 token 都没推出去，所以「不展示」是可执行的。
    """

    sources: list[SourceRef] = Field(default_factory=list)
    pipeline: str = Field(default="search_plus_qa")
    answer_state: str = Field(default="normal")


class AiAnswerDelta(BaseModel):
    """每完成一个**句子**推一次。

    推送单位必须等于校验单位：按「攒 5–10 个 token 或 50ms」推送的话，
    半句话已经出网了，而这半句还没法校验——校验结果没有地方落地。
    句子是 grounding 的自然最小单位，这有两个独立来源（MiniCheck 官方 API 本身
    就是句子级；Anthropic Citations API 默认按句切块并明确不引用小于一句的片段），
    两者都不是为了迁就流式，而是事实核查这件事本身的粒度。
    """

    seq: int = Field(ge=1)
    text: str
    citations: list[int] = Field(default_factory=list)


class AiAnswerFlagged(BaseModel):
    """某句校验为弱支持时，**在该句已推送之后**发。前端标灰，不撤回。"""

    seq: int = Field(ge=1)
    reason: str
    strength: CitationSupport = CitationSupport.WEAK


class AiAnswerDone(BaseModel):
    """生成结束。

    `citation_support` 逐引用下发：在线只在首句跑一次 HHEM，其余边默认
    `unverified`——**`unverified` 不是失败，它是诚实取值**。

    **不给用户展示任何自造的 confidence 分档**：模型给出的支持强度只用来把弱支持
    的句子标灰，不换算成「87 分可信度」——那个数字没有校准依据，展示它等于用
    精确的外观包装一个没有校准的量。
    """

    citation_support: list[dict[str, object]] = Field(default_factory=list)
    degraded: list[DegradeReason] = Field(default_factory=list)
    sentence_count: int = 0
    took_ms: int = 0


class AiAnswerError(BaseModel):
    """AI 路失败但检索路成功。**前端只隐藏 AI 区域，保留搜索结果。**

    `fallback` 带的是「信息不足」回答的文本：它说清「为什么不足」而不只是「不足」
    ——「只有摘要粒度」「本库无 2023 年后全文层」是**可行动的信息**，
    「抱歉我不知道」不是。`answer_state=insufficient_evidence` 是**正常状态不是
    错误**，前端按正常卡片渲染并标注，不弹错误提示。
    """

    answer_state: str
    reason: str
    fallback: str | None = None


class SearchResultsEvent(BaseModel):
    """`search_results` 事件：与 `/v1/search` 的 `results` 同结构。

    它先于 AI 回答到达，这是刻意的时序：**用户在 ~70ms 就已经有东西看了**，
    AI 区域显示骨架屏。`cited_by_ai` 的标记靠 `ai_answer_start` 到达后**后补**，
    而不是等 AI 检索完再一起推——后者要牺牲整整一秒的首屏时间，
    换一个用户根本感知不到的「一次性正确」。
    """

    group_count: int
    papers: list[PaperCard]
