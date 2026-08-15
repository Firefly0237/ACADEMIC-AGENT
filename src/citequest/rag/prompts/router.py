"""Level 2 路由分类的 prompt（M3 §2.3）。

**一次调用同时产出四个字段**：`route` / `needs_fulltext` / `filters` / `rewrite_strategy`。
拆成多次调用只会让延迟乘以调用数，并引入四个判断互相矛盾的可能
（判成 SEARCH_ONLY 却又标 needs_fulltext=true）。
**一次调用、一个 schema、一次校验，输出不合 schema 就整体走默认值，不做部分采纳。**

温度固定 0.0（`GenerationSettings.router_temperature`）：路由必须可复现。
同一条 query 两次请求给出不同 route，会让缓存、监控和回归测试同时失效。

few-shot 示例与词表、阈值一起进 `ROUTER_CONFIG_VERSION`——**改动其中任何一项都触发
路由回归测试**（M3 §2.5）。这也是为什么示例写在这里而不是散在测试里。
"""

from __future__ import annotations

__all__ = ["ROUTER_FEW_SHOTS", "ROUTER_SYSTEM_PROMPT", "ROUTER_USER_TEMPLATE", "build_router_messages"]

from citequest.llm.chat import ChatMessage

ROUTER_SYSTEM_PROMPT = """You classify a query for an academic paper search engine over computer science papers.

Answer with one JSON object and nothing else:
{"route": "...", "needs_fulltext": true|false, "year_from": null|int, "year_to": null|int,
 "primary_category": null|"cs.XX", "rewrite_strategy": "direct"|"hyde"|"decompose"}

route is one of:
- SEARCH_ONLY: keyword-style lookup, no explanation needed.
- SEARCH_PLUS_QA: a question that deserves a short generated answer over the retrieved papers.
- SEARCH_PLUS_MULTIHOP: a comparison or a question with several independent parts.
- STRUCTURED: the query carries an explicit year or category constraint.

needs_fulltext is true when answering requires method-level or experiment-level evidence
(loss functions, hyperparameters, ablations, implementation details, reported numbers).
Abstracts physically do not contain those.
"""

#: few-shot 示例**只示范判据形态，不示范具体领域**——写死某个领域的例子会让模型
#: 把该领域的词面当成分类信号（M3 的 prompt 纪律）。
ROUTER_FEW_SHOTS: tuple[tuple[str, str], ...] = (
    (
        "graph neural network node classification",
        '{"route": "SEARCH_ONLY", "needs_fulltext": false, "year_from": null, "year_to": null,'
        ' "primary_category": null, "rewrite_strategy": "direct"}',
    ),
    (
        "why is layer norm placed before the residual connection",
        '{"route": "SEARCH_PLUS_QA", "needs_fulltext": true, "year_from": null, "year_to": null,'
        ' "primary_category": null, "rewrite_strategy": "hyde"}',
    ),
    (
        "trade-offs between two families of generative models for image synthesis",
        '{"route": "SEARCH_PLUS_MULTIHOP", "needs_fulltext": false, "year_from": null, "year_to": null,'
        ' "primary_category": null, "rewrite_strategy": "decompose"}',
    ),
)

ROUTER_USER_TEMPLATE = """Query: {query}

JSON:"""


def build_router_messages(query: str) -> list[ChatMessage]:
    """system + few-shot 对 + 待分类 query。

    few-shot 用 user/assistant 交替而不是全塞进 system：交替形态让模型看到的是
    「输入长这样、输出长这样」的对齐关系，而 system 区里的一段示例文本更容易被
    当成规则的一部分来解读。
    """
    messages = [ChatMessage(role="system", content=ROUTER_SYSTEM_PROMPT)]
    for shot_query, shot_answer in ROUTER_FEW_SHOTS:
        messages.append(ChatMessage(role="user", content=ROUTER_USER_TEMPLATE.format(query=shot_query)))
        messages.append(ChatMessage(role="assistant", content=shot_answer))
    messages.append(ChatMessage(role="user", content=ROUTER_USER_TEMPLATE.format(query=query)))
    return messages
