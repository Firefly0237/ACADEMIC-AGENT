"""配置层：决策记录 §4「参数正本表」在代码里的唯一落地处。

**这一层的纪律只有一条**：任何设计参数的默认值在这里有且只有一份。
环境变量可以覆盖部署差异（地址、密钥、开关），但**不改默认值**——
默认值就是设计文档里那个数，改它要同时改决策记录。

分组与决策记录小节一一对应：
- `corpus`     → §4.1 语料与数据处理
- `milvus`     → §4.2 索引与 schema（含四项建库前不可逆参数中的三项）
- `embedding`  → §4.3 推理服务（TEI + Milvus Function）
- `retrieval`  → §4.4 检索与融合
- `rag`        → §4.5 重排与生成、§4.6 grounding 与流式
- `agent`      → §4.5 LangGraph、§4.10 会话与记忆
- `api`        → §4.6 API、§4.7 延迟预算、§4.9 服务层工程账
- `eval`       → §4.8 评测
"""

from citequest.config.agent import AgentSettings, MemorySettings
from citequest.config.api import (
    ApiSettings,
    CacheSettings,
    ConcurrencySettings,
    LatencyBudget,
    TimeoutSettings,
)
from citequest.config.corpus import CorpusSettings
from citequest.config.embedding import EmbeddingSettings
from citequest.config.eval import EvalSettings
from citequest.config.milvus import MilvusSettings
from citequest.config.rag import (
    GenerationSettings,
    GroundingSettings,
    RewriteSettings,
    StreamingSettings,
)
from citequest.config.retrieval import RerankSettings, RetrievalSettings
from citequest.config.settings import Settings, get_settings

__all__ = [
    "AgentSettings",
    "ApiSettings",
    "CacheSettings",
    "ConcurrencySettings",
    "CorpusSettings",
    "EmbeddingSettings",
    "EvalSettings",
    "GenerationSettings",
    "GroundingSettings",
    "LatencyBudget",
    "MemorySettings",
    "MilvusSettings",
    "RerankSettings",
    "RetrievalSettings",
    "RewriteSettings",
    "Settings",
    "StreamingSettings",
    "TimeoutSettings",
    "get_settings",
]
