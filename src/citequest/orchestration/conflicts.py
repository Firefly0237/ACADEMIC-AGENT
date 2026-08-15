"""记忆与证据冲突：四种真实形态，四条互不相同的裁决规则（M5 §7）。

学术场景的冲突不是抽象的「新旧留哪个」，把四种混成一条规则是错的：

| # | 形态 | 裁决 |
|---|---|---|
| **C1** | 同一研究的 preprint 与正式版 | **归并**：同 DOI 视作同一研究，主键取 OpenAlex ID；版本优先正式版；**但引用时标注实际引的是哪一版** |
| **C2** | 撤稿论文 | **硬排除** + 缓存精确失效。全系统唯一一处内容级硬拦截 |
| **C3** | 不同论文结论互斥 | **不裁决。并列呈现 + 标注分歧维度 + 各方各给引用** |
| **C4** | 新旧综述观点更替 | **时间序不是裁决依据，是标注依据**：两条都保留，标出年份 |

C1 的主键说明：`paper_id` 统一是 OpenAlex ID，它是稳定标识符；而「这两条是不是
同一个研究的两个版本」用 **DOI** 判——标题会在投稿过程中被改，DOI 不会。
「版本优先」不等于「删掉旧版」：引用的是**库里那一条的原文**，所以旧版必须仍可
引用，只是要标出来。

C3 为什么必须不裁决（与通用问答 RAG 的关键分界）：
1. 两篇论文结论不同，在学术上常常两篇都对——条件不同（数据集/规模/评测协议）
   导致的表面冲突占大多数，裁决掉一方等于把条件信息一起裁掉；
2. **裁决所用的分数没有裁决资格**：重排分经 sigmoid 后是有界可比的排序分，
   不是概率，不做校准就不代表可信度；
3. 暴露分歧本身就是学术检索的正确输出。**假装共识是这个场景里最严重的一种幻觉**
   ——它比编造一个不存在的引用更难被发现。

**检测位置：重排之后、上下文组装之前**（M5 §7.2）。放更前面：重排前候选 ≤48 条，
两两比对是 O(n²)；放更后面：生成之后再检测就是事后过滤，而流式下已推出的句子
撤不回来，与 grounding 是同一个物理约束。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from citequest.core.answer import Evidence
from citequest.core.corpus import Paper
from citequest.core.search import PaperHit
from citequest.observability.logging import get_logger

__all__ = [
    "ConflictGroup",
    "ConflictKind",
    "VersionMergeResult",
    "detect_claim_conflicts",
    "exclude_retracted",
    "merge_study_versions",
]

_logger = get_logger(__name__)


class ConflictKind(StrEnum):
    """四种形态。取值进日志与评测报告，所以固定成枚举。"""

    VERSION = "version"  # C1
    RETRACTION = "retraction"  # C2
    CONTRADICTORY_CLAIM = "contradictory_claim"  # C3
    SUPERSEDED_VIEW = "superseded_view"  # C4


@dataclass(slots=True)
class VersionMergeResult:
    """C1 的归并产物。

    `alias_of` 保留「哪几条被并到了哪一条」，因为**引用时必须标注实际引的是哪一版**
    ——归并是为了不重复推荐同一个研究，不是为了抹掉版本差异。
    """

    kept: list[Paper]
    alias_of: dict[str, str] = field(default_factory=dict)
    merged_pairs: int = 0


@dataclass(slots=True)
class ConflictGroup:
    """C3 的呈现单元：**结构上无法表达「谁赢了」**。

    这里没有 `winner`、没有 `confidence`、没有 `score` 字段——不是忘了给，
    是不能给。一旦有这么个字段，早晚有人拿一个未经校准的排序分去裁决一个学术事实。
    """

    kind: ConflictKind
    sides: list[list[Evidence]]
    #: 分歧维度：**只标能从元数据或 chunk 里确定性读出来的**（年份 / 领域 /
    #: 粒度）。读不出来就只标「结论不一致」，不猜原因——猜错的原因比不给原因更糟。
    dimensions: list[str] = field(default_factory=list)


def merge_study_versions(papers: list[Paper]) -> VersionMergeResult:
    """C1：同 DOI 的多条记录归并成一条，保留别名映射。

    **误并率优先于漏并率**（M5 §7.3）：把两篇不同的论文并成一篇，比漏并一对严重
    得多——前者会让用户彻底看不到其中一篇，后者只是列表里多了一条。所以判据只用
    DOI 这个强标识，**不用标题相似度**：标题在投稿过程中会改，用它做归并键
    正是误并的主要来源。

    「版本优先取正式版」的可判定形态：`venue` 非空视作正式版（会议/期刊已收录），
    否则视作 preprint。这是一个**确定性可读的字段**，不是一次推断。
    """
    by_doi: dict[str, list[Paper]] = {}
    no_doi: list[Paper] = []
    for paper in papers:
        if paper.doi:
            by_doi.setdefault(paper.doi, []).append(paper)
        else:
            # 没有 DOI 就没有可靠的归并键。**保留而不是猜**——猜一个键去归并，
            # 买到的是误并率，而误并正是这一项里代价最高的错误。
            no_doi.append(paper)

    kept: list[Paper] = list(no_doi)
    alias_of: dict[str, str] = {}
    merged_pairs = 0
    for doi, group in by_doi.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        # 正式版优先；同类版本内按年份取新。两条都是确定性排序，不做模型判断。
        group.sort(key=lambda p: (p.venue is not None, p.year), reverse=True)
        primary = group[0]
        kept.append(primary)
        for other in group[1:]:
            alias_of[other.paper_id] = primary.paper_id
            merged_pairs += 1
        _logger.info("version_merged", doi=doi, primary=primary.paper_id, versions=len(group))

    return VersionMergeResult(kept=kept, alias_of=alias_of, merged_pairs=merged_pairs)


def exclude_retracted(
    hits: list[PaperHit],
    retracted: frozenset[str],
    *,
    invalidate_cache: Callable[[str], None] | None = None,
) -> tuple[list[PaperHit], list[str]]:
    """C2：撤稿论文**硬排除**，并触发按 paper_id 的精确缓存失效。

    撤稿不是「相关性低」，是「不该出现」。这是全系统唯一一处**内容级的硬拦截**
    ——其余所有质量判定都是软信号。

    **缓存失效不能等 TTL**（M6 §3.3）：TTL 30 分钟，而合规要求的是「立即」。
    实现上靠写缓存时同时维护的 `paper → key` 倒排集合，代价是 60 倍写放大
    （一条响应最多牵动 60 篇论文）。这是全系统唯一一处为合规而非为性能付出的写放大。

    :param invalidate_cache: 注入的失效回调。**接真实实现**：正式实现是
        `SMEMBERS cq:rev:{paper_id}` 取出关联 key 后 `UNLINK`（用 UNLINK 而不是
        DEL，后台回收不阻塞事件循环）。传 None 时只做结果过滤——
        那是降级不是等价实现，必须打点。
    """
    if not retracted:
        return hits, []
    kept = [h for h in hits if h.paper_id not in retracted]
    removed = [h.paper_id for h in hits if h.paper_id in retracted]
    for paper_id in removed:
        if invalidate_cache is not None:
            invalidate_cache(paper_id)
    if removed:
        _logger.warning(
            "retracted_excluded",
            removed=removed,
            cache_invalidated=invalidate_cache is not None,
        )
    return kept, removed


def detect_claim_conflicts(
    evidences: list[Evidence],
    *,
    judge: Callable[[str, str], str] | None = None,
) -> list[ConflictGroup]:
    """C3：在**同一批已重排的 8 条证据**内两两判逻辑关系。

    成本【估算】：8 条证据、只在同一子问题下的证据之间比对，实际比对对数远小于
    C(8,2)=28。这一步的机制与在线 grounding 判定同源（都是判两段文本的逻辑关系），
    所以**它的延迟量级与在线 grounding 那一次前向同档，不引入新的模型**。

    :param judge: 注入的蕴含判定函数 `(premise, hypothesis) -> "entail"|"contradict"|"neutral"`。
        **接真实实现**：正式实现复用 `llm.hhem.HhemClient` 的同一次前向能力
        （或离线 NLI 模型），不新引一个模型。传 None 时返回空列表并打点——
        **不猜冲突**：假阳性的「学界有分歧」比漏报更糟，它会凭空制造一个不存在的争议。

    > **已知边界，主动说**：跨句合成型的冲突（每句单独看都成立、合起来自相矛盾）
    > 这一层检测不到。这与 grounding 层的已知边界是同一个，只能靠离线层暴露。
    """
    if judge is None:
        _logger.info("conflict_detection_skipped", reason="no_judge", evidences=len(evidences))
        return []

    groups: list[ConflictGroup] = []
    for i, left in enumerate(evidences):
        for right in evidences[i + 1 :]:
            if left.anchor.paper_id == right.anchor.paper_id:
                # 同一篇论文内部的两段不算「不同论文结论互斥」。
                continue
            if judge(left.text, right.text) != "contradict":
                continue
            groups.append(
                ConflictGroup(
                    kind=ConflictKind.CONTRADICTORY_CLAIM,
                    sides=[[left], [right]],
                    dimensions=_deterministic_dimensions(left, right),
                )
            )
    if groups:
        _logger.info("claim_conflicts_detected", groups=len(groups))
    return groups


def _deterministic_dimensions(left: Evidence, right: Evidence) -> list[str]:
    """只标能确定性读出来的分歧维度。

    读不出来就只标「结论不一致」，**不猜原因**——猜错的原因会把用户引向一个
    不存在的问题，比不给原因更糟。数据集与模型规模这两个维度需要从 chunk 文本里
    抽取，属于不确定性推断，所以这里不做（**接真实实现**：若 M7 证明抽取准确率
    足够，可在此加入 dataset / scale 两维）。
    """
    dims: list[str] = []
    if left.year != right.year:
        dims.append(f"year:{left.year} vs {right.year}")
    if left.granularity is not right.granularity:
        dims.append(f"granularity:{left.granularity.value} vs {right.granularity.value}")
    if not dims:
        dims.append("结论不一致（可确定性读出的分歧维度为空）")
    return dims
