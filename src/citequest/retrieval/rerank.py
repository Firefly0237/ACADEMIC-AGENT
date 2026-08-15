"""应用侧精排：48 → 8（决策记录 D10 / §4.5、M4 §2.2）。

**为什么在应用侧而不是用 Milvus 内置 Model Ranker Function**：官方硬限制
「Model rankers cannot be used with grouping searches」。两个官方能力互斥时选
grouping——chunk→paper 聚合是产品必需，rerank 放哪层是实现细节。

**为什么输出取 8**：成熟系统喂给 LLM 的证据条数高度集中在 4–10
（PaperQA2 `answer_max_sources=5`、RAGFlow `top_n=6`、Dify `top_k=4`、
OpenScholar `--top_n=10`）。取**上沿**而不是中位数，是因为学术回答的可信度
**先取决于引了几篇不同的论文**，其次才是每篇挖多深；取 5 会系统性地把回答压成
「转述一两篇」。代价写出来：事实型窄问题里 8 条大概率有 3–4 条是噪声，
它们占 prefill、占中段位置，由 prompt 约束与引用边校验兜底。

**阈值不拍数**：`score_threshold` 默认 None（不过滤）。标定照 Cohere 官方
reranking best practices——30–50 条代表性 query → 标临界相关文档 → 过 reranker
取分 → 取均值。阈值绑模型版本，换模型必须重标并记版本号。

**降级链**（M6 §5.4）：rerank 超时或不可用 → **退回 RRF 名次直出**。
这条降级不影响可核对性：引用锚点与 [N] 的对应关系是确定性的，不依赖任何模型。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from citequest.config.retrieval import RerankSettings
from citequest.core.enums import DegradeReason
from citequest.core.search import RetrievedChunk
from citequest.llm.reranker import RerankerClient
from citequest.observability.logging import get_logger

__all__ = ["RerankOutcome", "Reranker"]

_logger = get_logger(__name__)


@dataclass(slots=True)
class RerankOutcome:
    """精排产物。

    `all_below_threshold` 单列出来是因为它是「信息不足」的四个触发口之一
    （M4 §7.1）：8 条候选全部低于标定阈值 → 走信息不足回答。
    把它藏在「结果为空」里会让下游分不清「没检索到」和「检索到但都不相关」，
    而这两件事给用户的提示完全不同。
    """

    chunks: list[RetrievedChunk]
    degraded: DegradeReason | None = None
    all_below_threshold: bool = False
    took_ms: int = 0
    dropped_by_threshold: int = 0
    trace: dict[str, int] = field(default_factory=dict)


class Reranker:
    """cross-encoder 精排的编排：截断输入 → 分批打分 → 排序 → 截断输出。

    它不直接持有模型，只持有 `RerankerClient`——降级路径（模型不可用）必须能在
    没有 GPU 的环境下被单测覆盖，而这条降级链是本模块最重要的行为之一。
    """

    def __init__(self, client: RerankerClient, settings: RerankSettings) -> None:
        self._client = client
        self._settings = settings

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        timeout_ms: int | None = None,
    ) -> RerankOutcome:
        """48 → 8。

        :param query: **必须是原始 query，不是 HyDE 产物、也不是 sub-query**
            （M4 §8.2 第 1 条）：最终要回答的是原始问题，rerank 的排序目标必须与
            生成目标一致。用 sub-query 排序会得到「对每个子问题都很好、拼起来偏题」
            的证据集。
        :param timeout_ms: 单次调用预算（`TimeoutSettings.rerank_ms` = 800ms）。
            超时不是失败，是**退回 RRF 名次直出**。
        """
        if not chunks:
            return RerankOutcome(chunks=[], all_below_threshold=False)

        # 输入截断：cross-encoder 是 N 次前向、延迟对 N 线性，所以输入条数是硬预算。
        # 超出的部分按**当前顺序**丢尾部——当前顺序就是 RRF 名次序，
        # 这是这一步唯一能用的排序依据。
        candidates = chunks[: self._settings.input_size]
        if len(chunks) > self._settings.input_size:
            _logger.info("rerank_input_truncated", got=len(chunks), kept=len(candidates))

        started = time.perf_counter()
        try:
            scores = self._client.score(query, [c.text for c in candidates])
        except NotImplementedError:
            # 骨架/未部署模型：这是**正常降级**而不是异常（M6 §5.4 rerank_unavailable）。
            return self._degrade(candidates, DegradeReason.RERANK_UNAVAILABLE, started)
        except Exception as exc:
            _logger.warning("rerank_failed", error_type=type(exc).__name__)
            return self._degrade(candidates, DegradeReason.RERANK_UNAVAILABLE, started)

        took_ms = int((time.perf_counter() - started) * 1000)
        if timeout_ms is not None and took_ms > timeout_ms:
            # 已经算完了才发现超预算：**分数照用**，但打降级点。
            # 丢掉一份已经付过代价的结果没有任何好处，而不打点会让超时不可见。
            _logger.warning("rerank_over_budget", took_ms=took_ms, budget_ms=timeout_ms)
            degraded: DegradeReason | None = DegradeReason.RERANK_TIMEOUT
        else:
            degraded = None

        scored = [c.model_copy(update={"rerank_score": s}) for c, s in zip(candidates, scores, strict=True)]
        # 唯一的排布规则：**按 rerank 分数降序**（M4 §2.3）。
        # 不做「最高分放开头、次高分放结尾、其余交替填中间」这类位置重排——
        # Lost in the Middle 是一个被观测到的现象，不是一个被验证过的排列算法。
        scored.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)

        threshold = self._settings.score_threshold
        kept = scored
        dropped = 0
        all_below = False
        if threshold is not None:
            kept = [c for c in scored if (c.rerank_score or 0.0) >= threshold]
            dropped = len(scored) - len(kept)
            all_below = not kept
            if all_below:
                # 全线低于阈值是「信息不足」的触发口之一，**不是空结果**：
                # 候选仍然要带出去，信息不足回答里要展示「最相关的是这些，但都不对题」。
                kept = scored[: self._settings.output_size]

        outcome = RerankOutcome(
            chunks=kept[: self._settings.output_size],
            degraded=degraded,
            all_below_threshold=all_below,
            took_ms=took_ms,
            dropped_by_threshold=dropped,
            trace={"input": len(candidates), "output": min(len(kept), self._settings.output_size)},
        )
        _logger.info(
            "reranked",
            input=len(candidates),
            output=len(outcome.chunks),
            took_ms=took_ms,
            threshold=threshold,
            all_below_threshold=all_below,
        )
        return outcome

    def _degrade(self, candidates: list[RetrievedChunk], reason: DegradeReason, started: float) -> RerankOutcome:
        """退回 RRF 名次直出。

        **不给 `rerank_score` 填一个假分数**：`rerank_score` 是全系统唯一可做阈值的
        相关性分数，填一个占位值会让下游的阈值判断作用在一个编出来的数上，
        而它不报错。留 None 的语义是清楚的——「这一批没有过精排」。
        """
        took_ms = int((time.perf_counter() - started) * 1000)
        kept = candidates[: self._settings.output_size]
        _logger.warning("rerank_degraded", reason=reason.value, kept=len(kept))
        return RerankOutcome(
            chunks=kept,
            degraded=reason,
            took_ms=took_ms,
            trace={"input": len(candidates), "output": len(kept)},
        )
