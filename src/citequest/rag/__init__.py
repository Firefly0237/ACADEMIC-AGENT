"""生成层（M4/M5）：把检索结果变成带引用、可流式、可判定的答案。

模块划分：

- `prompts/`     —— 各类 prompt 常量与模板，与代码分离便于单独 review
- `context.py`   —— context assembly：token 预算、排布（按 rerank 分数降序，不洗牌）、
                    证据块格式化（元数据头 + `<<< >>>` 定界符）
- `citations.py` —— [N] 引用编号的分配、解析与回填；锚点是四元组
                    `(paper_id, section, sec_number, paragraph_idx)`
- `sentences.py` —— 句边界识别：缩写点号、小数点、紧跟句号的 [N] 三个真实的坑
- `generator.py` —— 调用 LLM 生成答案 + 「信息不足」回答
- `grounding.py` —— 沿已声明的 [N] 引用边判定 + 在线首句一次 HHEM-2.1-Open
- `streaming.py` —— 首句阻塞闸门 → 之后按句推送 → 只降级不撤回的状态机

这一层有两条约束是**物理性的**，不是偏好：

1. 「逐 token 推送」和「不合格不展示」不可同时执行——token 出网就收不回来了。
   解法是把「不展示」收缩到第一个 token 出网前这唯一可执行的时刻（首句闸门）。
2. 硬 gate 只有「[N] 越界 / 引用全部非法」两条，因为它们是确定性字符串校验，
   误判率结构性为零，所以敢用来拦截。HHEM 是软信号，只降级不拦截。

`generation.annotate_granularity_in_prompt` 是本层唯一一个「不做就静默产生幻觉」
的开关：不标粒度，模型会把摘要当方法细节来编。
"""

from citequest.rag.citations import (
    allocate_citation_ids,
    bind_anchors,
    citation_ids_in,
    parse_citations,
    unsupported_sentences,
)
from citequest.rag.context import ContextBudget, ContextPack, assemble_context, format_evidence_block
from citequest.rag.generator import AnswerGenerator, GenerationRequest, build_insufficient_answer
from citequest.rag.grounding import GateResult, GroundingChecker, SentenceGrounding
from citequest.rag.sentences import Sentence, iter_complete_sentences, split_sentences
from citequest.rag.streaming import AnswerStreamer, StreamEvent, StreamState

__all__ = [
    "AnswerGenerator",
    "AnswerStreamer",
    "ContextBudget",
    "ContextPack",
    "GateResult",
    "GenerationRequest",
    "GroundingChecker",
    "Sentence",
    "SentenceGrounding",
    "StreamEvent",
    "StreamState",
    "allocate_citation_ids",
    "assemble_context",
    "bind_anchors",
    "build_insufficient_answer",
    "citation_ids_in",
    "format_evidence_block",
    "iter_complete_sentences",
    "parse_citations",
    "split_sentences",
    "unsupported_sentences",
]
