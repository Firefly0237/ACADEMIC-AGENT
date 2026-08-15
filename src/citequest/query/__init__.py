"""查询理解与改写（M3）：只做路由判定、HyDE、decompose 三件事。

产物是 `citequest.core.query_plan.QueryPlan`——M3 唯一的对外契约，
下游只读它、不再回头看原始 query。

模块划分（阶段顺序即文件顺序）：

- `lexicons.py`    —— 词表与类目映射正本 + `ROUTER_CONFIG_VERSION`。改词表必须换版本号
- `normalize.py`   —— 阶段 A：NFKC 归一化、零宽清理、信号提取（p50 1ms 预算）
- `extraction.py`  —— 结构化条件抽取与剥离：年份 / 类目 / 引号短语 → `QueryFilters`
- `granularity.py` —— **唯一「不做就会静默产生幻觉」的分支**：三种处置 + 档 1 降级
- `intent.py`      —— 两级路由与 `QueryPlan` 组装
- `level2.py`      —— 小 LLM 兜底分类器：一次调用出四个字段
- `hyde.py`        —— HyDE(N=1, temp=0.3, max_tokens=128)，带死刑判据
- `decompose.py`   —— 一次性子问题拆分，**不迭代**，上限 4 条 + rerank 配额均分

已删除的：multi-query（ARAGOG 实测降低 retrieval precision）、rewrite、
拼写纠错。step-back 只进离线消融——抽象化会把 "RoPE" 抹成「位置编码方法」，
正好杀掉 BM25 的强信号，在 hybrid 上是净损失。

**HyDE 带存活条件**：它收约 400ms 过路费，占 RAG 路 p50 预算四成，所以必须在
citrec 库内子集与 LitSearch 上跑赢 direct，跑不赢就下线。这条止损判据是事先
写死的，不是事后找补。
"""

from citequest.query.decompose import DecomposeResult, QueryDecomposer, allocate_rerank_quota
from citequest.query.extraction import ExtractionResult, extract_filters
from citequest.query.granularity import (
    GranularityDecision,
    decide_granularity,
    fallback_on_insufficient_groups,
)
from citequest.query.hyde import HydeGenerator, HydeResult
from citequest.query.intent import Level2Output, QueryRouter, RoutingOutcome, level1_route
from citequest.query.level2 import LlmLevel2Classifier
from citequest.query.lexicons import ROUTER_CONFIG_VERSION
from citequest.query.normalize import extract_signals, normalize_query

__all__ = [
    "ROUTER_CONFIG_VERSION",
    "DecomposeResult",
    "ExtractionResult",
    "GranularityDecision",
    "HydeGenerator",
    "HydeResult",
    "Level2Output",
    "LlmLevel2Classifier",
    "QueryDecomposer",
    "QueryRouter",
    "RoutingOutcome",
    "allocate_rerank_quota",
    "decide_granularity",
    "extract_filters",
    "extract_signals",
    "fallback_on_insufficient_groups",
    "level1_route",
    "normalize_query",
]
