"""句边界识别（M4 §6.3、决策记录 §4.6「放行粒度」）。

**放行粒度是句子，不是 token 计数**，而且这个选择有两个独立来源，不是为了绕开
流式与后校验的矛盾硬切的：

- MiniCheck 官方 API 本身就是句子级：`MiniCheck(document, sentence) → [0,1]`，
  文档明确要求多句 claim 必须先切句；
- Anthropic Citations API 默认按句切块，并明确说明**不引用小于一句的片段**。

**推送粒度必须与校验粒度一致**——按 token 或按固定字节推送，会出现「半句已推、
校验结果无处落地」的状态。

三个真实的坑（读文档读不出来，M4 §6.3）：

| 坑 | 例子 | 处理 |
|---|---|---|
| 缩写点号 | "et al." / "e.g." / "Fig. 3" | 缩写白名单 |
| 小数点 | "提升了 2.1 个点" | 点号两侧均为数字则不切 |
| 紧跟句号的引用标记 | "……SOTA。[3]" 与 "……SOTA [3]。" 两种写法都会出现 | 尾随的 `[N]` 归到**前一句** |

第三个坑最隐蔽：**它不会报错**，只会让引用边系统性地偏移一句，而每一句看起来
都「有引用」——整条 grounding 判定就错了对象。

**这不是 `ingestion.sentence_split` 的复用场景**：那一个是切块阶段的长段落回退档，
按 token 预算切、允许退化到从句与词；这一个是流式放行的边界识别，
必须只在真句号处切，且要处理引用标记。目标相反，共用一份实现只会让两边都别扭。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["SENTENCE_ABBREVIATIONS", "Sentence", "iter_complete_sentences", "split_sentences"]

#: 缩写白名单。学术文本里这几个出现频率极高，切错一次就会把一句话劈成两半，
#: 而**半句话会同时破坏引用边和句子级校验的放行单位**。
SENTENCE_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "et al.",
        "e.g.",
        "i.e.",
        "cf.",
        "vs.",
        "etc.",
        "fig.",
        "figs.",
        "tab.",
        "eq.",
        "sec.",
        "ref.",
        "refs.",
        "approx.",
        "resp.",
        "no.",
        "vol.",
        "pp.",
        "al.",
    }
)

#: 句末标点。中英都收——回答语言与 query 一致，中文回答用的是全角句号。
_TERMINATORS = ".!?。！？"

#: 尾随引用标记：句末标点之后紧跟的一串 `[N]`。**归前一句**。
_TRAILING_CITATION_RE = re.compile(r"^\s*(?:\[\d+\])+")

#: **半个**引用标记（`[` 或 `[1`，编号还没写完）。流式下这一状态必须被识别出来，
#: 否则「句号后紧跟 [N]」这个坑会以另一种形式复活：句子在 `[` 到达的那一刻被判为
#: 封口并推送出去，而它的引用还在后面——引用边照样偏移一句。
_PARTIAL_CITATION_RE = re.compile(r"^\s*(?:\[\d+\])*\[\d*$")

#: 整个残片本身就只是引用标记（`[`、`[4`、`[4]`）。它在切句结果里会表现为
#: **一个独立的残句**，从而让它前面那句看起来已经封口——这是同一个坑的第三种形态。
_CITATION_FRAGMENT_RE = re.compile(r"(?:\[\d+\])*\[?\d*\]?")


@dataclass(slots=True, frozen=True)
class Sentence:
    """一句话及其在原文里的字符区间。

    带偏移是为了让流式层能把「已推送到哪里」表达成一个位置而不是一份副本——
    只降级不撤回意味着已推的内容必须能被后续事件按 seq 指回来，
    而 seq 与偏移是同一件事的两种表达。
    """

    text: str
    start: int
    end: int


def _ends_with_abbreviation(text: str, dot_index: int) -> bool:
    """判断 `text[dot_index]` 这个点号是否属于缩写。

    向前取最多 8 个字符做后缀匹配：白名单里最长的是 "approx."（7 字符），
    留一位余量。**不做全词回溯**——回溯到空格边界会让 "Fig." 前面若是标点时判空。
    """
    window = text[max(0, dot_index - 8) : dot_index + 1].lower()
    return any(window.endswith(abbr) for abbr in SENTENCE_ABBREVIATIONS)


def _is_decimal_point(text: str, dot_index: int) -> bool:
    """点号两侧均为数字 → 小数点，不切。"""
    if text[dot_index] != ".":
        return False
    before = text[dot_index - 1] if dot_index > 0 else ""
    after = text[dot_index + 1] if dot_index + 1 < len(text) else ""
    return before.isdigit() and after.isdigit()


def _boundary_end(text: str, index: int) -> int | None:
    """给定一个句末标点的下标，返回这一句真正的结束位置；不是边界则返回 None。"""
    char = text[index]
    if char not in _TERMINATORS:
        return None
    if char == "." and (_is_decimal_point(text, index) or _ends_with_abbreviation(text, index)):
        return None

    end = index + 1
    # 连续标点（"?!"、"。」"）一并吞掉，避免下一句以标点开头。
    while end < len(text) and text[end] in _TERMINATORS:
        end += 1
    # 尾随的 [N] 归到前一句——这是三个坑里最隐蔽的一个。
    trailing = _TRAILING_CITATION_RE.match(text[end:])
    if trailing is not None:
        end += trailing.end()
    return end


def split_sentences(text: str) -> list[Sentence]:
    """把一段完整文本切成句子。

    末尾**没有句末标点的残句同样返回**：生成结束时最后一句常常没有句号
    （被 `max_tokens` 截断、或模型就是这么写的），丢掉它等于丢掉一段已生成的内容，
    而超时/截断语义全系统只有一种——**已产生内容一律保留**。
    """
    sentences: list[Sentence] = []
    cursor = 0
    index = 0
    while index < len(text):
        end = _boundary_end(text, index)
        if end is None:
            index += 1
            continue
        chunk = text[cursor:end].strip()
        if chunk:
            offset = cursor + (len(text[cursor:end]) - len(text[cursor:end].lstrip()))
            sentences.append(Sentence(text=chunk, start=offset, end=end))
        cursor = end
        index = end

    tail = text[cursor:].strip()
    if tail:
        offset = cursor + (len(text[cursor:]) - len(text[cursor:].lstrip()))
        sentences.append(Sentence(text=tail, start=offset, end=len(text)))
    return sentences


def iter_complete_sentences(buffer: str) -> tuple[list[Sentence], str]:
    """流式专用：从缓冲区里取出**已完整**的句子，返回 `(句子列表, 剩余缓冲)`。

    与 `split_sentences` 的关键差别：**残句不返回，留在缓冲里**。
    流式下一个还没写完的句子不能推送——推了就收不回来了，而它可能正好是那句
    要被 grounding 判定的话。这是「只降级不撤回」这条语义在缓冲层的落点。

    还有一个必须留残余的具体理由：句末标点后可能紧跟 `[3]`。此刻若立即放行，
    引用边就会挂到下一句上。所以**句末标点后如果缓冲已经到头，就再等一轮**——
    等下一个 token 到达才能确定那个 `[` 是不是引用标记的开头。
    """
    if not buffer:
        return [], ""

    sentences = split_sentences(buffer)
    if not sentences:
        return [], buffer

    last = sentences[-1]
    if len(sentences) >= 2 and _CITATION_FRAGMENT_RE.fullmatch(last.text):
        # 缓冲末尾是一个**孤立的引用残片**：它属于前一句，所以前一句同样不能放行。
        # 不处理这一种，就会出现「句子先推、它的 [N] 后到」——引用边偏移一句，
        # 而且每一句看起来都有引用，不报错。
        return sentences[:-2], buffer[sentences[-2].start :]
    tail = buffer[last.end :]
    # **只有当最后一句之后已经出现了非空白、且不是引用标记的内容时，它才算封口。**
    # 这一条同时解决两件事：残句不推（token 出网就收不回来），
    # 以及句末标点后可能还没到达的 `[N]` 不会被挂到下一句上。
    closed = (
        bool(tail.strip()) and _TRAILING_CITATION_RE.match(tail) is None and _PARTIAL_CITATION_RE.match(tail) is None
    )
    if closed:
        return sentences, tail
    return sentences[:-1], buffer[last.start :]
