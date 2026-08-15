"""成本归因：三个维度回答「这次请求的钱花在哪一步」（M7 §11.4）。

| 维度 | 回答的问题 |
|---|---|
| **按 step 归因** | 哪一个环节最贵（通常是 HyDE 与生成） |
| **按 tool 归因** | 哪个工具的返回最占 token |
| **按 model 归因** | 小模型兜底与大模型生成之间的钱怎么分布 |

**`cost_estimate = tokens × 一张配置化的单价表`**（§11.3）。单价表是**配置**不是
常量：换模型改配置，不改代码。写死单价的后果是每次换模型都要发一次版本，
而没人会为了改一个价格发版本，于是成本账就永久地停留在上一个模型上。

**为什么三个维度都要**：只按 model 归因看不出「同一个模型在 HyDE 上花了四成
预算」；只按 step 归因看不出「某个工具的返回体撑大了 prompt」。HyDE 的存活条件
（跑不赢 direct 就砍）需要的正是 step 维度的数字。
"""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from citequest.observability.logging import get_logger
from citequest.observability.trace import SpanKind, SpanNode

__all__ = ["CostAttribution", "ModelPrice", "PriceTable", "attribute_cost"]

_logger = get_logger(__name__)


class ModelPrice(BaseModel):
    """一个模型的单价，单位：**每百万 token 的货币金额**。

    自建部署（本项目的 7B AWQ、reranker、HHEM 都是自托管）没有 API 账单，
    但仍然要填一个价：**折算的 GPU 摊销成本**。填 0 会让「哪一步最贵」这个问题
    的答案恒为「调外部 API 的那一步」，而本系统根本没有外部 API 调用——
    那样这三个维度就一个都用不上了。
    """

    model_config = ConfigDict(frozen=True)

    prompt_per_million: float = Field(ge=0.0)
    completion_per_million: float = Field(ge=0.0)
    cached_per_million: float = Field(default=0.0, ge=0.0, description="缓存命中的 prompt token 通常另计价")

    def cost_of(self, *, prompt: int, completion: int, cached: int = 0) -> float:
        return (
            prompt * self.prompt_per_million
            + completion * self.completion_per_million
            + cached * self.cached_per_million
        ) / 1_000_000


class PriceTable(BaseModel):
    """模型 → 单价。**它是配置，随环境注入，不进代码常量。**

    `default` 的存在是必要的：新接一个模型时忘了配价，正确行为是按默认价计一笔
    并告警，而不是静默计 0——**静默计 0 的表现是「成本没变」，而它其实涨了**。
    """

    prices: dict[str, ModelPrice] = Field(default_factory=dict)
    default: ModelPrice = Field(
        default=ModelPrice(prompt_per_million=0.0, completion_per_million=0.0),
        description="未配价模型的兜底。命中它会打 WARNING——不配价不是免费",
    )

    def price_of(self, model_id: str | None) -> ModelPrice:
        if model_id and model_id in self.prices:
            return self.prices[model_id]
        if model_id:
            _logger.warning("model_price_missing", model_id=model_id, hint="未配价不等于免费，请补 PriceTable")
        return self.default


class CostAttribution(BaseModel):
    """一次请求的成本三维分解。`total` 与三个维度的和必须相等。

    相等这件事不是自然成立的：一个 span 同时属于某个 step、某个 tool、某个 model，
    所以三份分解各自都应当等于总额。**任一维加不回总额，说明有 span 漏挂了父节点**
    ——那正是 span 树最容易出的一类错误。
    """

    model_config = ConfigDict(frozen=True)

    total: float = Field(ge=0.0)
    by_step: dict[str, float] = Field(default_factory=dict, description="按 PipelineStage：hyde / generate / ...")
    by_tool: dict[str, float] = Field(default_factory=dict, description="按 tool_call 的工具名")
    by_model: dict[str, float] = Field(default_factory=dict)
    total_tokens: int = Field(default=0, ge=0)

    def dominant_step(self) -> str | None:
        """最贵的那一步。HyDE 的存活条件（占 RAG 路 p50 预算四成）就靠这个数辩护。"""
        return max(self.by_step, key=lambda k: self.by_step[k], default=None)


def attribute_cost(nodes: list[SpanNode], table: PriceTable) -> CostAttribution:
    """按三个维度分解一次请求的成本，并把 `cost_estimate` 回填进每个 span。

    回填是刻意的：`cost_estimate` 是 span 的必记字段（§11.3），而算它需要单价表
    ——单价表在应用层，span 在观测层。让算好的值回流到 span 上，trace 后端里
    看到的每一条就自带成本，不必再关联一次。

    **按 tool 归因用最近的 `tool_call` 祖先**：一次工具调用内部可能套着若干次
    llm_call（工具自己要调模型），把这些子调用的钱算到工具头上才是「哪个工具最贵」
    的正确答案。
    """
    by_id = {n.span_id: n for n in nodes}
    by_step: dict[str, float] = defaultdict(float)
    by_tool: dict[str, float] = defaultdict(float)
    by_model: dict[str, float] = defaultdict(float)
    total = 0.0
    total_tokens = 0

    for node in nodes:
        if node.total_tokens == 0 and node.cached_tokens == 0:
            continue
        price = table.price_of(node.model_id)
        cost = price.cost_of(prompt=node.prompt_tokens, completion=node.completion_tokens, cached=node.cached_tokens)
        node.cost_estimate = cost
        total += cost
        total_tokens += node.total_tokens

        step = node.stage.value if node.stage else node.name
        by_step[step] += cost
        by_model[node.model_id or "unknown"] += cost
        tool = _nearest_tool(node, by_id)
        by_tool[tool or "no_tool"] += cost

    attribution = CostAttribution(
        total=total,
        by_step=dict(by_step),
        by_tool=dict(by_tool),
        by_model=dict(by_model),
        total_tokens=total_tokens,
    )
    _logger.info(
        "cost_attributed",
        total=round(total, 6),
        tokens=total_tokens,
        dominant_step=attribution.dominant_step(),
    )
    return attribution


def _nearest_tool(node: SpanNode, by_id: dict[str, SpanNode]) -> str | None:
    """沿 parent 链往上找最近的 `tool_call`。找不到返回 None。

    向上走而不是向下聚合：向下聚合要先建一棵真正的树，而这里只有一个扁平列表
    加 parent 指针——沿指针上溯是 O(深度)，而树最深只有四层。
    """
    current: SpanNode | None = node
    while current is not None:
        if current.kind is SpanKind.TOOL_CALL:
            return current.name
        current = by_id.get(current.parent_span_id or "")
    return None
