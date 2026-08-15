"""参数正本落地处：一个 Settings + 若干嵌套 BaseModel。

**这是整个骨架最重要的文件之一。** 决策记录 §4「参数正本表」的数值全部作为
字段 default 写在这一组模块里，成为代码里的唯一正本。环境变量可以覆盖，
但**不改默认值**——默认值就是设计文档里那个数。

环境变量写法（pydantic-settings 官方形态）：
- 前缀 `CITEQUEST_`，嵌套分隔符 `__`
- 例：`CITEQUEST_MILVUS__URI=http://milvus:19530`
- 例：`CITEQUEST_RETRIEVAL__RRF_K=60`

优先级：CLI > 构造参数 > 环境变量 > .env > 默认值。
仓库里只提交 `.env.example`，真实 `.env` 已进 .gitignore。
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """全局配置。分组与决策记录 §4 的小节一一对应。"""

    model_config = SettingsConfigDict(
        env_prefix="CITEQUEST_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        # 多环境靠「同一份 Settings 类 + 不同 .env / 环境变量注入」，
        # 不靠 if env == 'prod' 分支——分支会让「线上跑的到底是哪套参数」变成需要读代码才知道的事。
    )

    env: str = Field(default="local", description="仅用于日志与 span 的标签，不用来切换任何行为分支")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json", pattern="^(json|console)$", description="容器里用 json，本地开发用 console")

    corpus: CorpusSettings = Field(default_factory=CorpusSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    rewrite: RewriteSettings = Field(default_factory=RewriteSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    grounding: GroundingSettings = Field(default_factory=GroundingSettings)
    streaming: StreamingSettings = Field(default_factory=StreamingSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    concurrency: ConcurrencySettings = Field(default_factory=ConcurrencySettings)
    timeouts: TimeoutSettings = Field(default_factory=TimeoutSettings)
    latency_budget: LatencyBudget = Field(default_factory=LatencyBudget)
    eval: EvalSettings = Field(default_factory=EvalSettings)

    @model_validator(mode="after")
    def _cross_group_invariants(self) -> Settings:
        """跨组约束。

        这些约束单看每一组都是合法的，只有放在一起才暴露矛盾——而它们出错的
        表现形式全是「不报错、指标悄悄下滑」，所以必须在启动时一次性拦掉。
        """
        # 1. dense 维度必须三处一致：TEI 输出、Milvus 字段、fallback 模型。
        #    不一致的表现是灌库时报维度错，但如果只有 query 侧不一致，
        #    则是静默的召回崩塌。
        if self.embedding.dim != self.milvus.dense_dim:
            raise ValueError(f"embedding.dim({self.embedding.dim}) 与 milvus.dense_dim({self.milvus.dense_dim}) 不一致")

        # 2. rerank 输入不能超过 RAG 路带回的 chunk 数（详见 RerankSettings.validate_pipeline）。
        self.rerank.validate_pipeline(self.retrieval)

        # 3. 生成层 context 条数不能超过 rerank 输出条数。
        if self.generation.context_max_chunks > self.rerank.output_size:
            raise ValueError(
                f"generation.context_max_chunks({self.generation.context_max_chunks}) "
                f"超过 rerank.output_size({self.rerank.output_size})"
            )

        # 4. grounding 阈值要么两个都标定、要么都不标定。只标一个会让中间档语义未定义。
        has_supported = self.grounding.supported_threshold is not None
        has_weak = self.grounding.weak_threshold is not None
        if has_supported != has_weak:
            raise ValueError("grounding 的 supported_threshold 与 weak_threshold 必须同时标定或同时留空")
        if has_supported and has_weak:
            assert self.grounding.supported_threshold is not None
            assert self.grounding.weak_threshold is not None
            if self.grounding.supported_threshold <= self.grounding.weak_threshold:
                raise ValueError("supported_threshold 必须大于 weak_threshold")

        # 5. 排队超时必须短于该阶段的价值窗口：生成排 2s 后即使拿到额度也必破 2.5s 的 p95。
        if self.concurrency.queue_timeout_generate_ms >= self.latency_budget.rag_first_token_p95_ms:
            raise ValueError(
                f"生成排队超时({self.concurrency.queue_timeout_generate_ms}ms) 必须短于 "
                f"RAG 首 token p95 预算({self.latency_budget.rag_first_token_p95_ms}ms)"
            )

        return self

    @property
    def search_ef(self) -> int:
        """搜索路查询期 ef（60 组 × 1 → 240）。"""
        return self.milvus.ef_for(self.retrieval.search_group_limit, self.retrieval.search_group_size)

    @property
    def rag_ef(self) -> int:
        """RAG 路查询期 ef（16 组 × 3 → 192）。"""
        return self.milvus.ef_for(self.retrieval.rag_group_limit, self.retrieval.rag_group_size)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例。

    用 lru_cache 而不是模块级常量，是为了让测试能通过 `get_settings.cache_clear()`
    在改完环境变量后拿到一份新的 Settings，而不必重载模块。
    """
    return Settings()
