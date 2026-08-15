"""子问题拆分的 prompt（M3 §6.4、M4 §8、决策记录 D12）。

**形态锁死：一次拆分 → 子问题并行检索 → 一次合成，上限 4，不迭代。**

不做 self-ask / IRCoT 式迭代多跳的理由是数据层面的：那类方法依赖「前一跳的答案
定位下一跳的文档」，而本库里 81.5% 的论文只有摘要、库内引用边有限。
**这条链在这个数据上没有物质基础，做迭代就是在演。**

输出契约用**每行一条**而不是 JSON：JSON 需要完整才能解析，而「模型不按 JSON 输出」
本身是一个需要额外兜底的失败模式。逐行文本的解析失败是局部的——坏掉的那行丢掉，
其余照用；JSON 的解析失败是整体的。
"""

from __future__ import annotations

__all__ = ["DECOMPOSE_SYSTEM_PROMPT", "DECOMPOSE_USER_TEMPLATE", "build_decompose_messages"]

from citequest.llm.chat import ChatMessage

DECOMPOSE_SYSTEM_PROMPT = """You split a complex research question into independent sub-questions for a literature search engine.

Rules:
- Output at most {max_subqueries} sub-questions, one per line, no numbering, no extra text.
- Each sub-question must be answerable on its own, without the answer to any other sub-question.
- Keep the original technical terms verbatim. Do not generalise them.
- If the question is already atomic, output it unchanged as the single line.
"""

DECOMPOSE_USER_TEMPLATE = """Question: {query}

Sub-questions:"""


def build_decompose_messages(query: str, *, max_subqueries: int) -> list[ChatMessage]:
    """组装拆分消息。

    「保留原始技术术语、不做泛化」这条规则与 step-back 被否决的理由是同一条：
    抽象化会把 "RoPE" 抹成「位置编码方法」，而 "RoPE" 正是 BM25 侧最强的信号词，
    在 hybrid 检索上是净损失。**拆分不等于抽象**。

    「每个子问题必须能独立回答」直接对应「不迭代」这个形态：一旦子问题之间有依赖，
    就必须按顺序解、必须用前一跳的答案定位下一跳——那正是被否决的那条链。
    """
    if max_subqueries < 1:
        raise ValueError(f"max_subqueries 必须为正，收到 {max_subqueries}")
    return [
        ChatMessage(role="system", content=DECOMPOSE_SYSTEM_PROMPT.format(max_subqueries=max_subqueries)),
        ChatMessage(role="user", content=DECOMPOSE_USER_TEMPLATE.format(query=query)),
    ]
