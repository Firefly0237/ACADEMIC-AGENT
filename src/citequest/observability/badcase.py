"""badcase 归因矩阵（M7 §13）。**归因能力比修复能力更重要。**

关键是**能把责任切到哪一层**。下面五类的判定信号全部可程序化，来源是
`retrieval_log` 的召回明细。

| 类别 | 判定信号 | 典型处置方向 |
|---|---|---|
| **① 检索错** | 正确证据的 paper **在库内**，但两路都没召回它（不在候选表里） | 提 ef；查 chunking 是否切碎了证据；查是否降级成 dense_only |
| **② 融合错** | 某一路原始名次很靠前，但 `rank_fused` 掉出候选边界 | 检查是否应走单路而非融合；确认 RRF 的 k 与两路候选深度 |
| **③ 截断错** | `in_candidate=true` 但 `in_context=false` | 阈值或 top-N 重标定；若集中在某一 granularity 则是粒度路由问题 |
| **④ 生成错** | `in_context=true` 但回答与之矛盾或未使用 | 粒度路由与 prompt 标注（摘要块被当方法细节用是最典型的一种） |
| **⑤ 引用错** | 句子被支持，但 `[N]` 指向的 chunk 不是真正支持它的那个 | 引用锚点生成方式；prompt 的引用格式约束 |

**判定顺序不能颠倒**（§13.2）：先怀疑生成、去调 prompt，是最常见也最浪费时间的
排障方式——因为生成层的问题有一半根源在上游，而上游的问题在明细表里是**一眼可见**的。

```
回答不对
  ├─ 正确证据在候选表里吗？ 否 → ①（再分：融合前没召回 / 融合后掉出 → ②）
  ├─ 进上下文了吗？        否 → ③
  ├─ 回答与上下文一致吗？  否 → ④
  └─ 引用指向对吗？        否 → ⑤
```
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from citequest.observability.logging import get_logger
from citequest.observability.retrieval_log import RetrievalDetail

__all__ = ["BadcaseClass", "BadcaseVerdict", "classify_badcase", "conflicting_evidence_present"]

_logger = get_logger(__name__)

#: 「某一路名次很靠前」的界线。§13.1 给的例子是 `rank_sparse ≤ 10`——**它是一个
#: 排障启发式，不是一个被标定过的阈值**，所以它的作用只是给人一个查看的起点，
#: 不参与任何在线决策。
NEAR_TOP_RANK = 10


class BadcaseClass(StrEnum):
    """五类归因 + 一个「没问题」。

    `NOT_A_BADCASE` 必须存在：分类函数被用在批量归因上，把「其实没问题」硬塞进
    五类里的任何一类，都会让那一类的计数变成噪声。
    """

    RETRIEVAL = "retrieval_miss"
    FUSION = "fusion_drop"
    TRUNCATION = "truncation_drop"
    GENERATION = "generation_error"
    CITATION = "citation_error"
    NOT_A_BADCASE = "not_a_badcase"


class BadcaseVerdict(BaseModel):
    """一次归因的结论 + 下一步该查什么。

    `next_action` 不是可有可无的文案：归因矩阵的价值在于**切完层之后知道去哪儿看**，
    只给一个类别名，排查的人还是得回去翻文档。
    """

    model_config = ConfigDict(frozen=True)

    verdict: BadcaseClass
    evidence_chunk_id: str | None = None
    signals: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    next_action: str = ""


def classify_badcase(
    details: Sequence[RetrievalDetail],
    *,
    gold_paper_id: str,
    gold_in_corpus: bool,
    answer_consistent: bool | None = None,
    citation_points_to_supporting_chunk: bool | None = None,
) -> BadcaseVerdict:
    """按 §13.2 的顺序自上而下切层。**顺序写死在函数里，不允许调用方指定。**

    :param gold_paper_id: 正确证据所在的论文。评测里来自 qrels，线上排障里来自
        人工确认。
    :param gold_in_corpus: 这篇论文**在不在库内**。这是判「检索错」的前置条件——
        不在库内就不是检索的问题，是语料覆盖的问题，两者的处置完全不同。
    :param answer_consistent: 回答是否与上下文一致（MiniCheck 判）。None 表示
        没跑生成侧判定，此时链路停在第 ③ 步，**不猜**。
    :param citation_points_to_supporting_chunk: `[N]` 是否指向真正支持它的 chunk。
        None 同上。
    """
    hits = [d for d in details if d.paper_id == gold_paper_id]

    # ---- 第一层：正确证据在候选表里吗 ----
    if not hits:
        if not gold_in_corpus:
            # 库里根本没有这篇。**这不是检索错**，把它算进检索错会让「提 ef」
            # 这个处置被反复尝试在一个 ef 永远解决不了的问题上。
            return BadcaseVerdict(
                verdict=BadcaseClass.NOT_A_BADCASE,
                signals={"gold_in_corpus": False},
                next_action="语料覆盖问题：扩语料或在报告里声明该问题超出库覆盖范围",
            )
        return BadcaseVerdict(
            verdict=BadcaseClass.RETRIEVAL,
            signals={"gold_in_corpus": True, "recalled": False},
            next_action=(
                "用 FLAT 暴力检索确认真实排名："
                "FLAT 能召回而 HNSW 不能 ⇒ ef 不足或索引参数问题（回到组召回曲线①）；"
                "FLAT 也召不回 ⇒ query 与 chunk 的语义/词面不匹配，查 chunking 与 retrieval_mode 降级"
            ),
        )

    best = max(hits, key=_candidate_quality)

    # ---- 第二层：是不是融合把它挤掉了 ----
    # 明细表里存在这一行说明它进了候选；**融合错的形态是「某一路很靠前但融合后
    # 掉出边界」**，所以判据是单路名次与 in_candidate 的矛盾，而不是名次本身高低。
    near_top = _near_top_channel(best)
    if not best.in_candidate and near_top is not None:
        return BadcaseVerdict(
            verdict=BadcaseClass.FUSION,
            evidence_chunk_id=best.chunk_id,
            signals={"channel": near_top, "rank_dense": best.rank_dense, "rank_sparse": best.rank_sparse},
            next_action=(
                "比对 rank_dense / rank_sparse / rank_fused 三个字段。"
                "中文 query 应走 dense 单路、引号短语应走 sparse 单路——"
                "需要通道偏置时的正确解法是**不融合**，不是把加权融合加回来"
            ),
        )

    # ---- 第三层：进上下文了吗 ----
    if not best.in_context:
        return BadcaseVerdict(
            verdict=BadcaseClass.TRUNCATION,
            evidence_chunk_id=best.chunk_id,
            signals={"score_rerank": best.score_rerank, "granularity": best.granularity.value},
            next_action=(
                "看 score_rerank 是低于阈值被过滤，还是排在第 9 位之后被 top-8 截掉。"
                "大量正确证据卡在 9–12 位 ⇒ 阈值或 top-N 需重标定；集中在某一 granularity ⇒ 粒度路由问题"
            ),
        )

    # ---- 第四层：回答与上下文一致吗 ----
    if answer_consistent is False:
        return BadcaseVerdict(
            verdict=BadcaseClass.GENERATION,
            evidence_chunk_id=best.chunk_id,
            signals={"in_context": True, "granularity": best.granularity.value},
            next_action=(
                "用 MiniCheck 对该句判蕴含；检查 prompt 里该块的 granularity 标注是否正确。"
                "**摘要块被当成方法细节使用是最典型的一种**"
            ),
        )

    # ---- 第五层：引用指向对吗 ----
    if citation_points_to_supporting_chunk is False:
        return BadcaseVerdict(
            verdict=BadcaseClass.CITATION,
            evidence_chunk_id=best.chunk_id,
            signals={"cited": best.cited},
            next_action=(
                "citation precision 低而 faithfulness 高 ⇒ 典型的引用错。"
                "查引用锚点的生成方式（细化到 paper_id/section/sec_number/paragraph_idx）与 prompt 的引用格式约束"
            ),
        )

    return BadcaseVerdict(
        verdict=BadcaseClass.NOT_A_BADCASE,
        evidence_chunk_id=best.chunk_id,
        signals={"in_context": True, "cited": best.cited},
        next_action="检索与生成两侧信号都正常；若用户仍不满意，属于需求理解层面，不在这五类里",
    )


def _candidate_quality(detail: RetrievalDetail) -> tuple[bool, bool, float]:
    """同一篇论文命中多个 chunk 时，取「走得最远」的那一条来归因。

    排序键是三段的：先看进没进上下文，再看进没进候选，最后看 rerank 分。
    **不按 rank_fused 取最靠前的**——一篇论文的某个块名次靠前但被截断、另一个块
    进了上下文时，按名次取会把一个「其实成功了」的样本判成截断错。
    """
    return (detail.in_context, detail.in_candidate, detail.score_rerank or 0.0)


def _near_top_channel(detail: RetrievalDetail) -> str | None:
    """哪一路把它排进了前 `NEAR_TOP_RANK` 名。都没有则返回 None。"""
    if detail.rank_sparse is not None and detail.rank_sparse <= NEAR_TOP_RANK:
        return "sparse"
    if detail.rank_dense is not None and detail.rank_dense <= NEAR_TOP_RANK:
        return "dense"
    return None


def conflicting_evidence_present(details: Sequence[RetrievalDetail], *, year_gap: int = 3) -> bool:
    """一类值得单独讲的 badcase：**证据冲突**（§13.3）。

    学术检索里有一种既真实又高频的情况：检索到的两篇论文对同一个问题给出相反结论
    （不同数据集、不同设定、不同年份）。

    **正确的系统行为是「既不编造一个统一结论，也不静默丢掉一方」**——并列陈述并
    分别标注来源与年份/设定。所以这个函数只回答「上下文里是否存在可能分歧的证据」，
    **不判断谁对**：裁决冲突需要 factuality 能力，而这个系统结构上没有
    （语料就是论文本身，论文是一手证据不是权威事实）。

    判据只用明细表里已有的元数据（粒度混合、跨多篇论文），**零额外模型调用**——
    既然结论是「不裁决」，检测出冲突之后也不会做任何不同的事，
    **检测一个不影响任何动作的信号是纯成本**。`year_gap` 参数留给调用方接论文级
    元数据后做更细的判定。
    """
    in_context = [d for d in details if d.in_context]
    if len({d.paper_id for d in in_context}) < 2:
        return False
    mixed_granularity = len({d.granularity for d in in_context}) > 1
    if mixed_granularity:
        _logger.info("mixed_granularity_context", chunks=len(in_context), year_gap_hint=year_gap)
    return mixed_granularity
