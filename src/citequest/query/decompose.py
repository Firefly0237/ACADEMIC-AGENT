"""decompose：一次拆分，不迭代（M3 §6.4、M4 §8、决策记录 D12）。

**形态锁死**：一次拆分 → 子问题**并行**检索 → 一次合成，上限 4。

不做迭代多跳的理由是数据层面的：self-ask / IRCoT 依赖「用前一跳的答案定位下一跳
文档」，而本库 81.5% 的论文只有摘要、库内引用边有限。**这条链在这个数据上没有
物质基础，做迭代就是在演**——多跑两轮，检索到的东西不会因为多了一跳而变好，
只会多两次 LLM 调用和一倍延迟。

「并行检索」不是可选优化，是这条路可行的前提：4 个子问题串行 = 4 倍检索延迟，
直接把 p95 打穿。这一点同时是编排层的定性依据——主链路是 workflow
（routing + parallelization），不是 agent。

**不做 sub-query 前置相似度过滤**（H9）：好的 sub-query 恰恰更具体更窄、
与原 query 的相似度天然更低，那个过滤器会优先杀掉最好的分解结果。
"""

from __future__ import annotations

from dataclasses import dataclass

from citequest.config.rag import RewriteSettings
from citequest.core.errors import UpstreamTimeoutError
from citequest.llm.chat import ChatClient, Deadline
from citequest.observability.logging import get_logger
from citequest.rag.prompts.decompose import build_decompose_messages

__all__ = ["DecomposeResult", "QueryDecomposer", "allocate_rerank_quota"]

_logger = get_logger(__name__)


@dataclass(slots=True)
class DecomposeResult:
    """拆分产物。

    `fell_back` 为 True 表示拆分失败并退回单 query 直检索——这是 M4 §9.2 降级矩阵
    里的一行（「decompose 拆分失败 → 退回 direct，正常回答但质量可能下降」），
    不是异常。
    """

    subqueries: list[str]
    fell_back: bool = False


class QueryDecomposer:
    """一次 LLM 调用完成拆分。**`decompose_iterations` 恒为 1**，配置层已用 `le=1` 锁死。"""

    def __init__(self, chat: ChatClient, settings: RewriteSettings) -> None:
        self._chat = chat
        self._settings = settings

    async def decompose(self, query: str, *, deadline: Deadline | None = None) -> DecomposeResult:
        """拆成 ≤4 个可独立回答的子问题。

        解析用**逐行**而不是 JSON：JSON 必须完整才能解析，坏一处整条作废；
        逐行文本的解析失败是局部的——坏掉的那行丢掉，其余照用。
        """
        if not query.strip():
            raise ValueError("decompose 的 query 不能为空")

        try:
            raw = await self._chat.complete(
                build_decompose_messages(query, max_subqueries=self._settings.decompose_max_subqueries),
                temperature=0.0,
                max_tokens=256,
                deadline=deadline,
            )
        except (UpstreamTimeoutError, OSError) as exc:
            _logger.warning("decompose_fallback", error_type=type(exc).__name__)
            return DecomposeResult(subqueries=[query], fell_back=True)

        subqueries = _parse_subqueries(raw, limit=self._settings.decompose_max_subqueries)
        if not subqueries:
            return DecomposeResult(subqueries=[query], fell_back=True)

        if self._settings.keep_original_query_as_fallback and query not in subqueries:
            # 原 query 保留做锚点，与 HyDE 那条同理：拆分产物可能整体偏题，
            # 而原 query 那一路保证最坏情况退化成 direct 而不是被拆错的子问题主导。
            subqueries.insert(0, query)
            subqueries = subqueries[: self._settings.decompose_max_subqueries]

        _logger.info("query_decomposed", n=len(subqueries))
        return DecomposeResult(subqueries=subqueries)


def _parse_subqueries(raw: str, *, limit: int) -> list[str]:
    """逐行解析并去重，保序。

    去掉常见的编号前缀（`1.` / `- ` / `* `）：prompt 里写了「no numbering」，
    但模型仍会漂——**对模型输出做防御性清洗，不指望 prompt 百分百被遵守**。
    """
    seen: set[str] = set()
    out: list[str] = []
    for line in raw.splitlines():
        text = line.strip().lstrip("-*•").strip()
        while text[:1].isdigit() and text[1:2] in (".", ")", "、"):
            text = text[2:].strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(text)
        if len(out) >= limit:
            break
    return out


def allocate_rerank_quota(subquery_count: int, *, rerank_input_size: int) -> list[int]:
    """把 rerank 输入配额在子问题之间**均分**（M4 §8.3）。

    这是做完并行合并之后才浮出来的连锁约束，**它没有出现在任何一份参考实现里**：
    单 query 路是 `16×3 = ≤48`，刚好落在 rerank 输入上限内；decompose 路 4 个子问题
    各跑一遍 → 合并去重后**最多 192 条，是上限的近 4 倍**。cross-encoder 对 N 线性，
    192 条会直接吃掉半个首 token 预算。

    **为什么是均分而不是按检索分数分配**：按分数分配会让「检索容易的子问题」吃掉
    全部配额，而**恰恰是检索困难的那个子问题最需要被看见**——它对应的往往是问题里
    最新、最窄的那一跳。均分保证每个子问题在 context 里都有代表。
    （同一个思路：STORM 给每个环节独立 token 预算而不是共享一个池子。）

    **用不完的配额由调用方回流**：某子问题去重后不足配额时，余额按 RRF 名次分给
    其他子问题——用名次不用分数，因为 RRF 分数无物理含义。

    :returns: 与子问题一一对应的配额列表，总和恒等于 `rerank_input_size`。
    """
    if subquery_count < 1:
        raise ValueError(f"subquery_count 必须为正，收到 {subquery_count}")
    if rerank_input_size < subquery_count:
        raise ValueError(
            f"rerank 输入上限({rerank_input_size})小于子问题数({subquery_count})，"
            "每个子问题至少要有 1 条代表，否则均分这条纪律就名存实亡"
        )
    base, remainder = divmod(rerank_input_size, subquery_count)
    # 余数分给前几个：前面的是模型先拆出来的，通常更贴近原问题的主干。
    return [base + (1 if i < remainder else 0) for i in range(subquery_count)]
