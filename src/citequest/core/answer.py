"""生成侧数据契约：Evidence / Citation / AnswerSentence / AnswerChunk / GroundingVerdict。

两条设计约束直接体现在类型上：

1. **判定沿已声明的 [N] 引用边**，不再对全部 chunk 取最大相似度（决策记录 D13）。
   所以 `GroundingVerdict` 的主键是 `(sentence_seq, citation_id)` 这条边，而不是句子本身。
2. **只降级不撤回**（D14）。`AnswerSentence` 一旦推送出去就不可撤销，
   后续只能通过 `ai_answer_flagged` 指回它的 seq 把支持强度调低。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from citequest.core.enums import CitationSupport, DegradeReason, Granularity, SecType

__all__ = [
    "AnswerChunk",
    "AnswerSentence",
    "Citation",
    "CitationAnchor",
    "Evidence",
    "GroundingVerdict",
]


class CitationAnchor(BaseModel):
    """引用锚点。全文块细化到 `(paper_id, section, sec_number, paragraph_idx)` 四元组
    （决策记录 §4.5）——这是换到全文语料买到的核心能力。

    摘要块只有 paper_id，其余三项为 None：这不是缺陷，是粒度本身的诚实表达。
    """

    model_config = ConfigDict(frozen=True)

    paper_id: str
    chunk_id: str
    section: str | None = None
    sec_number: str | None = None
    paragraph_idx: int | None = None


class Evidence(BaseModel):
    """喂进 prompt 的一条证据。

    `granularity` 必须逐块标注进 prompt（决策记录 §4.5）——不标，模型会把摘要
    当方法细节来编。这是整个生成层唯一一个「不做就会静默产生幻觉」的字段。
    """

    citation_id: int = Field(ge=1, description="prompt 里的 [N] 编号，从 1 开始连续分配")
    anchor: CitationAnchor
    text: str
    title: str
    year: int
    granularity: Granularity = Field(description="必须写进 prompt 的 Evidence 区，见 M4 §6")
    sec_type: SecType | None = None
    rerank_score: float | None = Field(default=None, ge=0.0, le=1.0)


class Citation(BaseModel):
    """回答中出现的一次 [N] 引用。"""

    model_config = ConfigDict(frozen=True)

    citation_id: int = Field(ge=1)
    anchor: CitationAnchor
    support: CitationSupport = CitationSupport.UNVERIFIED


class GroundingVerdict(BaseModel):
    """一条引用边的判定结果。

    在线只在**首句**跑一次 HHEM-2.1-Open（决策记录 §4.6）：逐句在线判定是一个
    默认关闭的开关，因为花在线延迟买一个 F1 只有 44.8–66.8、不敢用来拦截的
    软信号，必要性判据不过关（O22）。

    **硬 gate 只有一条**：[N] 越界 / 引用全部非法——确定性字符串校验，误判率
    结构性为零。HHEM 分数只降级、不拦截。
    """

    model_config = ConfigDict(frozen=True)

    sentence_seq: int = Field(ge=0)
    citation_id: int = Field(ge=1)
    support: CitationSupport
    hhem_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="HHEM 一致性分。None 表示未判定（HHEM 不可用或非首句），此时 support 应为 unverified",
    )
    reason: str | None = Field(default=None, description="降级原因文本，随 ai_answer_flagged 下发给前端")

    @model_validator(mode="after")
    def _unverified_has_no_score(self) -> GroundingVerdict:
        # 有分数却标 unverified，会让前端把「判过但弱」和「根本没判」混为一谈，
        # 而这两件事对用户的含义完全不同。
        if self.support is CitationSupport.UNVERIFIED and self.hhem_score is not None:
            raise ValueError("unverified 表示未判定，不应携带 hhem_score")
        return self


class AnswerSentence(BaseModel):
    """按句边界推送的一段回答（对应 SSE 的 `ai_answer_delta`）。

    放行粒度是**句子边界**而非 token 计数（决策记录 §4.6）。句边界识别必须处理
    缩写点号、小数点、以及紧跟句号的 [N] 标记——这三种情况都会让朴素的
    「按 . 切分」在真实学术文本上频繁切错。
    """

    seq: int = Field(ge=0, description="自增序号。ai_answer_flagged 用它指回一个已推送的句子")
    text: str
    citations: list[Citation] = Field(default_factory=list)
    is_first_sentence: bool = Field(
        default=False,
        description="首句是唯一被阻塞校验的句子——第一个 token 出网前是「不展示」唯一可执行的时刻（D14）",
    )


class AnswerChunk(BaseModel):
    """一次完整回答的终态（对应 SSE 的 `ai_answer_done`）。

    `confidence` 只在内部与日志中使用，**不作为自造分档暴露给用户**（K6）：
    对用户只展示「引用是否可点、可核对」。
    """

    request_id: str
    sentences: list[AnswerSentence]
    evidences: list[Evidence] = Field(description="本次回答的 Evidence 区全集，[N] 编号与之一一对应")
    citation_support: list[GroundingVerdict] = Field(default_factory=list)
    degraded: list[DegradeReason] = Field(default_factory=list)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    took_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _citations_within_evidence_range(self) -> AnswerChunk:
        # 这就是那条唯一的硬 gate（决策记录 §4.6「硬 gate 条件」）的数据层实现：
        # [N] 越界是确定性字符串校验，误判率结构性为零，所以敢用来拦截。
        valid = {e.citation_id for e in self.evidences}
        for sentence in self.sentences:
            for citation in sentence.citations:
                if citation.citation_id not in valid:
                    raise ValueError(
                        f"句 seq={sentence.seq} 引用了越界编号 [{citation.citation_id}]，"
                        f"Evidence 区只有 {sorted(valid)}"
                    )
        return self
