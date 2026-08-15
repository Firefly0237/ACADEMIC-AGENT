"""查询改写、生成、grounding 与流式参数（决策记录 §4.5 / §4.6）。

HyDE 是这里唯一一个**带存活条件**的组件：它收 400ms 过路费，占 RAG 路 p50 预算的
四成，所以它必须在 citrec 库内子集与 LitSearch 上跑赢 direct，跑不赢就砍。
把这个条件写进配置的注释里，是为了让「事先写死止损判据」这条纪律可执行。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["GenerationSettings", "GroundingSettings", "RewriteSettings", "StreamingSettings"]


class RewriteSettings(BaseModel):
    """查询改写（决策记录 D12）。只剩 HyDE 与 decompose 两件事。

    已删除：multi-query（ARAGOG 实测降低 retrieval precision）、rewrite；
    step-back 只进离线消融（抽象化会把 'RoPE' 抹成「位置编码方法」，正好杀掉
    BM25 的强信号，在 hybrid 上是净损失）。
    """

    hyde_enabled: bool = Field(default=True, description="RAG 路默认策略。存活条件见本模块 docstring")
    hyde_n: int = Field(
        default=1,
        ge=1,
        description="**明确偏离论文默认值 20，面试必须主动说**。代价是方差大于论文设定",
    )
    hyde_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="论文用 1.0 是为配合 N=20 求多样性；N=1 时高温只放大方差",
    )
    hyde_max_tokens: int = Field(default=128, ge=1, description="对齐语料形态，过长会引入无关术语稀释信号")

    decompose_max_subqueries: int = Field(default=4, ge=1, description="context 预算约束")
    decompose_iterations: int = Field(
        default=1,
        ge=1,
        le=1,
        description="**不迭代**。语料与库内引用边撑不起 self-ask 式多跳链，做迭代就是在演（O23）",
    )
    keep_original_query_as_fallback: bool = Field(
        default=True,
        description="不做 sub-query 前置相似度过滤——好的 sub-query 恰恰更具体更窄、相似度天然更低，那个过滤器会优先杀掉最好的分解结果（H9）",
    )


class GenerationSettings(BaseModel):
    """答案生成（M4）。"""

    model_id: str = Field(default="Qwen2.5-7B-Instruct-AWQ", description="本地 7B AWQ，显存账按 4.0 GB 计")
    base_url: str = Field(default="http://localhost:8000/v1", description="OpenAI 兼容端点")
    api_key: str = Field(default="EMPTY")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    context_max_chunks: int = Field(default=8, ge=1, description="rerank 输出的 8 条。≤48 chunk → rerank → 8 条")
    annotate_granularity_in_prompt: bool = Field(
        default=True,
        description="**prompt 里必须逐块标注 granularity**，否则模型会把摘要当方法细节来编。这是生成层唯一「不做就静默产生幻觉」的开关",
    )
    router_model_id: str = Field(default="Qwen2.5-1.5B-Instruct", description="路由兜底用的小模型，1.5B–3B 量级即可")
    router_temperature: float = Field(
        default=0.0,
        description="路由必须可复现。同一条 query 两次给出不同 route 会让缓存、监控和回归测试同时失效",
    )
    prompt_version: str = Field(
        default="v1", description="进日志与 span（K12）。改 prompt 不改版本号 = 线上指标变动无法归因"
    )


class GroundingSettings(BaseModel):
    """引用校验与幻觉软信号（决策记录 §4.6）。

    分层是这样的：**在线只做确定性硬校验 + 首句一次 HHEM 软信号，打分整体移到离线**。
    硬 gate 只有 [N] 越界 / 引用全部非法两条——确定性字符串校验，误判率结构性为零，
    所以敢用来拦截；HHEM 只降级不拦截。
    """

    hhem_model_id: str = Field(
        default="vectara/hallucination_evaluation_model",
        description="HHEM-2.1-Open，flan-t5-base 底座、0.1B、F32 下 <600MB、Apache-2.0、无上下文长度上限",
    )
    online_first_sentence_only: bool = Field(
        default=True,
        description="在线只在首句跑一次。逐句在线判定是默认关闭的开关——花在线延迟买一个不敢用来拦截的软信号，必要性判据不过关（O22）",
    )
    online_per_edge_enabled: bool = Field(default=False, description="逐引用边在线判定，默认关闭")
    english_only: bool = Field(
        default=True,
        description="HHEM 仅英文。中文回答**只做确定性引用校验**，跨语言需商用版 HHEM-2.3",
    )
    supported_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="**待 M7 在 30–50 条标定集上按 balanced accuracy 调，不沿用余弦时代的 0.7/0.5**。None 表示未标定，此时一律 unverified",
    )
    weak_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    hard_gate_on_citation_out_of_range: bool = Field(
        default=True,
        description="唯一的硬 gate。[N] 越界或引用全部非法 → 首句不过闸，完全不进流式，直接返回「信息不足」",
    )
    keep_previous_sentence_as_context: bool = Field(
        default=True,
        description="判定上下文保留前一句，对应 NeMo 的 context_size 滑窗作用（官方示例 chunk_size 200 / context_size 50）",
    )


class StreamingSettings(BaseModel):
    """SSE 流式策略（决策记录 D14）。

    这一组参数的来历值得单独说：原设计同时规定「逐 token 推送」和「不合格不展示」，
    这两条**物理上不可同时执行**——token 出网就收不回来了。解法是把「不展示」收缩到
    第一个 token 出网前这个唯一可执行的时刻。
    """

    first_sentence_gate_enabled: bool = Field(
        default=True,
        description="首句阻塞校验，过闸才发 ai_answer_start；不过闸完全不进流式",
    )
    push_by_sentence: bool = Field(
        default=True,
        description="首句之后按**句边界**推送而非 token。句边界识别须处理缩写点号、小数点、紧跟句号的 [N] 标记",
    )
    downgrade_only_never_retract: bool = Field(
        default=True,
        description=(
            "先推后标，只降级不撤回。工业先例：NeMo Guardrails 官方承认违规文本可能已送达、"
            "「It's the responsibility of the caller to manage this situation」"
        ),
    )
    sse_total_timeout_ms: int = Field(default=60_000, ge=1, description="流总长上限")
    heartbeat_interval_ms: int = Field(default=15_000, ge=1, description="防中间代理按空闲超时断连")
