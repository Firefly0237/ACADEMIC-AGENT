"""S2 · LaTeX 解析噪声（M1 §5.2）。

上游数据集已经完成 LaTeX → 结构的转换，但转换不是无损的，残留有固定几类。
这里只处理**有明确形态**的四类，不做通用 LaTeX 解析器——通用解析器正是
「自建」的典型形态，而 unarXive 已经把这件事做完并发表了（JCDL 2023）。

四类里最值得说的是**断词连字符**：`atten-\ntion` 不还原会直接伤 BM25，因为
`atten` 和 `tion` 都不是词，english analyzer 的 stemmer 也救不回来——它们会作为
两个只出现少数几次的垃圾 token 进倒排，而原本该命中的 `attention` 反而丢了一次
词频。这是「清洗不做会静默降低召回」而不是「不做只是不好看」的一类。

表格与图注碎片**刻意不特殊处理**，交给 `chunking` 的 80 token 最小长度规则吸收：
为它写识别规则的误伤率高于收益，而最小长度规则本来就要跑。
"""

from __future__ import annotations

import html
import re

from pydantic import BaseModel, Field

__all__ = ["CleanTextResult", "clean_latex_noise"]

#: 未展开的行内命令：`\textbf{attention}` → `attention`。
#: 只吃一层花括号——嵌套命令在正文里罕见，而贪婪匹配会把整段吞掉。
_INLINE_CMD_RE = re.compile(r"\\(?:textbf|textit|emph|texttt|textsc|mathrm|mathbf|text|underline)\{([^{}]*)\}")
#: 剩余的孤立控制序列（`\qquad`、`\noindent`），去掉命令本身即可。
_BARE_CMD_RE = re.compile(r"\\[a-zA-Z]+\*?")
#: 断词连字符：行尾 `-` + 换行 + 下一行首词。
_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z])-\n([a-z])")
#: 残留 HTML 标签（`<sub>2</sub>`）：去标签保留内容。
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>")
_INLINE_SPACE_RE = re.compile(r"[ \t]+")


class CleanTextResult(BaseModel):
    """清洗产物 + 各类噪声的命中计数。

    计数字段进 §10 的质量报告：某一类噪声的命中率突然变化，意味着上游数据集版本
    变了，而这件事**不会报错**——它只会让某一批块的文本质量悄悄下降。
    """

    text: str
    inline_command_hits: int = Field(default=0, ge=0)
    hyphen_break_hits: int = Field(default=0, ge=0)
    html_entity_hits: int = Field(default=0, ge=0)


def clean_latex_noise(text: str) -> CleanTextResult:
    """按固定顺序处理四类残留噪声。

    顺序不是随意的：
    1. **断词还原必须最先做**，它依赖行尾换行符；后面任何一步压缩空白都会毁掉线索。
    2. HTML 实体解码在去标签之前——`&lt;sub&gt;` 解码后才是一个可识别的标签。
    3. 行内命令展开在裸命令删除之前，否则 `\\textbf{x}` 会先被删成 `{x}`，
       花括号留在正文里。
    """
    if not text:
        return CleanTextResult(text="")

    hyphen_hits = len(_HYPHEN_BREAK_RE.findall(text))
    working = _HYPHEN_BREAK_RE.sub(r"\1\2", text)

    unescaped = html.unescape(working)
    entity_hits = 1 if unescaped != working else 0
    working = _HTML_TAG_RE.sub(" ", unescaped)

    cmd_hits = len(_INLINE_CMD_RE.findall(working))
    working = _INLINE_CMD_RE.sub(r"\1", working)
    working = _BARE_CMD_RE.sub(" ", working)

    working = _INLINE_SPACE_RE.sub(" ", working)
    # 只清行内空白，**不碰换行**：段落分隔要留给 placeholders.normalize_unicode_and_whitespace
    # 统一处理，两处都改会让「段落边界在哪一步丢的」变成一个需要二分查找的问题。
    working = "\n".join(line.strip() for line in working.split("\n")).strip()

    return CleanTextResult(
        text=working,
        inline_command_hits=cmd_hits,
        hyphen_break_hits=hyphen_hits,
        html_entity_hits=entity_hits,
    )
