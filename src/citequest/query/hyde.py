"""HyDE：N=1 的假设文档生成（M3 §6.2、决策记录 D12 / §4.5）。

**它带着自己的死刑判据上线**（M3 §6.3，事先写死不是事后找补）：

> HyDE 必须在 unarXive_citrec 库内子集与 LitSearch 两个基准上跑赢 direct 基线。
> **跑不赢就砍掉，改为全链路 direct。**

理由是选它的证据全部来自别人的语料（原论文 arXiv:2212.10496、LlamaIndex 与
Haystack 的官方模块、ARAGOG 的第三方对照评测），而它在这里要收 **400ms 的过路费，
占 RAG 路首 token p50 预算的 41%**——这个价格必须由自己语料上的数字来批准。

适用性边界（同样来自原论文）：HyDE 打赢的是**无监督**检索器，对在域内监督微调过的
检索器并不占优。本项目的 BGE-M3 是通用预训练、未在 CS 语料上微调，所以前提成立；
**哪天真的微调了检索器，HyDE 的收益必须重新评估。**
"""

from __future__ import annotations

from dataclasses import dataclass

from citequest.config.rag import RewriteSettings
from citequest.core.enums import DegradeReason
from citequest.core.errors import UpstreamTimeoutError
from citequest.llm.chat import ChatClient, Deadline
from citequest.observability.logging import get_logger
from citequest.rag.prompts.hyde import build_hyde_messages

__all__ = ["HydeGenerator", "HydeResult"]

_logger = get_logger(__name__)


@dataclass(slots=True)
class HydeResult:
    """HyDE 产物。

    `queries` 是**要发出去的检索文本列表**，恒定包含原 query 且原 query 在首位——
    这一条不是可选项：HyDE 生成的是模型自己编的文本，可能把检索带偏；保留原 query
    那一路，最坏情况下融合结果退化成 direct 的排序，而不是被幻觉文档主导。
    """

    queries: list[str]
    hypothesis: str | None
    degraded: DegradeReason | None = None

    @property
    def used(self) -> bool:
        """本次是否真的用上了假设文档。进日志与消融口径——
        止损判据要对比 HyDE vs direct，而降级掉的那些请求实际上跑的就是 direct，
        混进 HyDE 组会把它的指标算高。"""
        return self.hypothesis is not None


class HydeGenerator:
    """一次 LLM 调用生成假设文档（N=1）。

    **不做 N>1**：论文用 temperature 1.0 采样 N=20 是为了求多样性，线上跑 20 次
    生成不可能。N=1 时高温只放大方差、没有多样性可言，所以温度降到 0.3。
    **这是明确的实现偏离，面试必须主动说，不能装作照搬了论文。**
    """

    def __init__(self, chat: ChatClient, settings: RewriteSettings) -> None:
        self._chat = chat
        self._settings = settings

    async def generate(self, query: str, *, deadline: Deadline | None = None) -> HydeResult:
        """生成假设文档并组装成两路检索文本。

        **止损开关**：`hyde_enabled=False` 时直接返回 direct 形态。这个开关存在的
        意义是让「跑不赢就砍」能在配置里一键执行，而不需要改代码——止损判据写死了，
        执行手段就必须是零成本的。

        超时降级：HyDE 失败**只是少一路检索**，不是请求失败（M4 §9.2 同构）。
        原 query 那一路永远在，所以最坏情况就是退化成 direct。
        """
        if not query.strip():
            raise ValueError("HyDE 的 query 不能为空")
        if not self._settings.hyde_enabled:
            return HydeResult(queries=[query], hypothesis=None)
        if self._settings.hyde_n != 1:  # pragma: no cover - 配置被改动时的显式失败
            raise ValueError(
                f"hyde_n={self._settings.hyde_n}：本设计锁死 N=1（决策记录 §4.5）。"
                "要改 N 必须先改决策记录并重估延迟预算——N 次采样就是 N 倍的 400ms"
            )

        try:
            hypothesis = await self._chat.complete(
                build_hyde_messages(query),
                temperature=self._settings.hyde_temperature,
                max_tokens=self._settings.hyde_max_tokens,
                deadline=deadline,
            )
        except (UpstreamTimeoutError, OSError) as exc:
            _logger.warning("hyde_degraded", error_type=type(exc).__name__)
            return HydeResult(queries=[query], hypothesis=None, degraded=DegradeReason.HYDE_TIMEOUT)

        hypothesis = hypothesis.strip()
        if not hypothesis:
            # 空产出与超时同处置：模型返回空串时若把它当检索文本发出去，
            # BM25 Function 会产出空 sparse 向量并静默返回随机结果。
            _logger.warning("hyde_empty_output")
            return HydeResult(queries=[query], hypothesis=None, degraded=DegradeReason.HYDE_TIMEOUT)

        # 原 query 在首位：下游融合按 RRF 只用名次，顺序本身不影响分数，
        # 但**通道归因**要靠这个顺序（哪一路贡献了哪些名次进日志）。
        return HydeResult(queries=[query, hypothesis], hypothesis=hypothesis)
