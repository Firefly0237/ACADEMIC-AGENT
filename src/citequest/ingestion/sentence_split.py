"""长段落的回退档（M1 §6.8）。

R1 说「永不在段落中间切」，但存在极端情况：**单个段落本身就超过 450 token**
（长证明、长实验描述）。这时 R1 与 R3 直接冲突，必须有一个确定的回退，否则会
产出一个远超上界的块。

回退规则：退化到句子边界切分，按「段落分隔符 → 句子分词 → 中英混合标点 → 词」
的多级降级执行——这是 LlamaIndex `SentenceSplitter` 的既有做法，**同构照抄，不自造**。

**这一档仍然 overlap = 0**，理由与 §6.7 一致：切点落在句子边界上，不需要靠重叠去
救断句（LangChain/PaperQA2 的 5% 与 LlamaIndex 的 20% 之间的差异正说明重叠比例是
在补救切点质量）。

**【待实测】这一档的触发率。** 高于个位数百分比说明段落长度分布与预期不符，
450 这个上界要重新评估——所以它是 §10 统计脚本的必测项，不是一个可以「先跑起来
再说」的分支。
"""

from __future__ import annotations

import re

from citequest.ingestion.tokenization import TokenCounter

__all__ = ["TextPiece", "split_long_text"]

#: 句子边界与从句边界。中英标点都收，因为语料里存在中英混排的作者信息与引文。
_SENTENCE_END_RE = re.compile(r"(?<=[.!?。！？])\s+")
_CLAUSE_END_RE = re.compile(r"(?<=[,;:，；：])\s+")

#: `(片段文本, 起字符偏移, 止字符偏移)`。偏移是给引用边分配用的——丢了偏移，
#: 「别人在什么语境下引用 X」在这些片段上就会出现假阳性。
TextPiece = tuple[str, int, int]


def _split_units(text: str) -> list[str]:
    """降级取最小可用切分单位。

    句子切不开（一整段没有句号，长公式说明里很常见）就退到从句，再切不开就退到
    词——**必须保证最后一定切得开**，否则回退档自己会产出超界块，而这一档存在的
    唯一目的就是防止超界块。
    """
    for pattern in (_SENTENCE_END_RE, _CLAUSE_END_RE):
        parts = [p.strip() for p in pattern.split(text) if p.strip()]
        if len(parts) > 1:
            return parts
    return text.split()


def split_long_text(text: str, target_tokens: int, counter: TokenCounter) -> list[TextPiece]:
    """把超长段落切成不超过 `target_tokens` 的片段，返回带字符偏移的三元组。

    累加逻辑与 `chunking` 的段落聚合一致——**先判超再封口**，不是吸完再判断；
    反过来写会让每个片段都恰好越界一个单位，450 就成了下界而不是上界。
    """
    if target_tokens < 1:
        raise ValueError(f"target_tokens 必须为正，收到 {target_tokens}")

    units = _split_units(text)
    pieces: list[TextPiece] = []
    buf: list[str] = []
    buf_tokens = 0
    buf_start = 0
    cursor = 0

    for unit in units:
        start = text.find(unit, cursor)
        if start < 0:  # pragma: no cover - 防御：切分不应改写内容
            start = cursor
        end = start + len(unit)
        cursor = end
        unit_tokens = counter.count(unit)
        if buf and buf_tokens + unit_tokens > target_tokens:
            pieces.append((" ".join(buf), buf_start, start))
            buf = []
            buf_tokens = 0
        if not buf:
            buf_start = start
        buf.append(unit)
        buf_tokens += unit_tokens

    if buf:
        pieces.append((" ".join(buf), buf_start, len(text)))
    return pieces
