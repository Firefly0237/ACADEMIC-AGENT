"""外部模型服务的客户端封装。**只做协议适配，不含任何业务逻辑。**

- `chat.py`     —— 生成与路由模型（OpenAI 兼容端点）+ `Deadline`
- `reranker.py` —— bge-reranker-v2-m3，`normalize=True` 是对外的分数契约
- `hhem.py`     —— HHEM-2.1-Open 幻觉判定

**这里没有 embedding 客户端，这是设计而不是遗漏。** dense 编码走 TEI +
Milvus Embedding Function 的 Data-in/Data-out：写入只传原文、检索只传 query 原文，
Milvus 服务端自己去调 TEI。客户端**拿不到也不需要** query embedding。

这条路线有两个连锁后果，都记在这里免得后人重新发明：
1. M3 的 embedding 锚点路由失去了「反正本来就要算」的免费前提；
2. L3 embedding 缓存**物理上不存在**——不是砍掉的，是架构后果。

本包所有客户端都必须支持 deadline 传递（入口算绝对截止时间戳，每层发起下游调用
前先看剩余预算）。否则「每层 2s」会在四层链路上累积成 8s。
"""

from citequest.llm.chat import ChatClient, ChatMessage, Deadline
from citequest.llm.hhem import HhemClient
from citequest.llm.reranker import CrossEncoderReranker, RerankerClient

__all__ = [
    "ChatClient",
    "ChatMessage",
    "CrossEncoderReranker",
    "Deadline",
    "HhemClient",
    "RerankerClient",
]
