"""HyDE 假设文档生成的 prompt（M3 §6.2、决策记录 D12）。

**机制**：让 LLM 先写一段假设性的学术段落，这段文本同时喂两条通道——dense 侧由
Milvus 服务端调 TEI 编码走 ANN，sparse 侧原文送 BM25 Function 走倒排；再与
**原始 query 的检索结果**一起用 `RRFRanker(k=60)` 融合。

**原 query 永远保留做锚点**，这不是可选项：HyDE 生成的是一段模型自己编的文本，
它可能把检索带偏；保留原 query 那一路，最坏情况下融合结果退化成 direct 的排序，
而不是被幻觉文档主导。

参数偏离必须主动说：论文（arXiv:2212.10496）采样 N=20、temperature 1.0，
本项目 N=1、temperature 0.3。**代价是单次采样方差大、收益一定低于论文报告值。**

prompt 里不写死任何具体 case 的内容——把当前这条 query 的细节写进提示词会让模型
过拟合到个例，而个例正是最不需要优化的那一类输入。
"""

from __future__ import annotations

__all__ = ["HYDE_SYSTEM_PROMPT", "HYDE_USER_TEMPLATE", "build_hyde_messages"]

from citequest.llm.chat import ChatMessage

#: 目标是**表面形态像论文段落**，不是事实正确——HyDE 的产物只进检索通道，
#: 永远不进生成 prompt（M4 §3.1 第 3 条），所以它编错事实不会污染回答。
HYDE_SYSTEM_PROMPT = """You write a short passage that looks exactly like an excerpt from a computer science paper.

Rules:
- Write one dense paragraph, no headings, no bullet points, no citations.
- Use the terminology and phrasing conventions of academic CS writing.
- Stay on the topic of the question. Do not answer it as a chatbot would.
- Do not mention that this passage is hypothetical.
"""

HYDE_USER_TEMPLATE = """Question: {query}

Write the passage:"""


def build_hyde_messages(query: str) -> list[ChatMessage]:
    """组装 HyDE 的两条消息。

    query 原文进 user 区、指令进 system 区，与生成层的三区分离同构——
    这样「检索内容/用户输入只出现在非指令区」在整个项目里是一条可检查的统一规则，
    而不是每个调用点各自决定。
    """
    if not query.strip():
        raise ValueError("HyDE 的 query 不能为空")
    return [
        ChatMessage(role="system", content=HYDE_SYSTEM_PROMPT),
        ChatMessage(role="user", content=HYDE_USER_TEMPLATE.format(query=query)),
    ]
