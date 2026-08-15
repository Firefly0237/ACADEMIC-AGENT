"""prompt 常量与模板。

单独成包而不是散在各调用点，有两个具体理由：

1. prompt 是本项目**改动最频繁**的部分，集中放置才能让「改了哪句」在 diff 里
   一眼看清；
2. `generation.prompt_version` 要随 prompt 一起改。改 prompt 不改版本号，
   线上指标的变动就永远无法归因——这是可观测性上最难补的一类窟窿。

按用途分文件（与设计文档章节对齐）：

- `hyde.py`      —— HyDE 假设文档生成（M3 §6.2）
- `decompose.py` —— 子问题拆分（M3 §6.4）
- `router.py`    —— Level 2 路由分类，一次调用出四个字段（M3 §2.3）
- `answer.py`    —— 带 [N] 引用的答案生成（M4 §3），含 granularity 逐块标注
- `evaluator.py` —— 检索质量三档判定 Correct/Ambiguous/Incorrect（纠正循环用，
                    由 `graph/` 提供，本包不重复定义）

**纪律**：prompt 里不写死任何具体 case 的内容。把当前这一条 query 的细节写进
提示词，会让模型过拟合到个例，而个例正是最不需要优化的那一类输入。
"""

from citequest.rag.prompts.answer import (
    ANSWER_SYSTEM_PROMPT,
    EVIDENCE_SECTION_HEADER,
    INSUFFICIENT_EVIDENCE_TEMPLATE_EN,
    INSUFFICIENT_EVIDENCE_TEMPLATE_ZH,
    USER_SECTION_TEMPLATE,
)
from citequest.rag.prompts.decompose import build_decompose_messages
from citequest.rag.prompts.hyde import build_hyde_messages
from citequest.rag.prompts.router import build_router_messages

__all__ = [
    "ANSWER_SYSTEM_PROMPT",
    "EVIDENCE_SECTION_HEADER",
    "INSUFFICIENT_EVIDENCE_TEMPLATE_EN",
    "INSUFFICIENT_EVIDENCE_TEMPLATE_ZH",
    "USER_SECTION_TEMPLATE",
    "build_decompose_messages",
    "build_hyde_messages",
    "build_router_messages",
]
