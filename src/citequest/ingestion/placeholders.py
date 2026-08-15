"""S1 · 占位符处理（M1 §5.1）——**强制前置步骤，不是可选项**。

unarXive 正文里公式与引用被替换成 `{{formula:uuid}}` / `{{cite:uuid}}`，真实内容
放在记录的其他字段。这里做的是「删除文本、保留结构」：

- 文本侧：两类占位符一律删除，并清理删除后留下的孤立标点与连续空白；
- 结构侧：公式 LaTeX 与被引论文 `open_alex_id` 抢救到**不参与检索**的 metadata。

**为什么是删而不是留**（机制层，必须能扛追问）：主分析器是 english（standard
tokenizer），它会把 `{{formula:a1b2...}}` 切成 `formula` 加一串十六进制碎片。
147 万 chunk 规模下这是几十万个只出现一次的垃圾 token，后果有两层——放大倒排，
以及**污染 avgdl 从而扭曲每一个正常词的 BM25 分数**（长度归一化项是
`1 - b + b × |d| / avgdl`，分母被污染等于所有词的分数都被系统性改变）。

**为什么不把 `{{cite:uuid}}` 替换成被引论文标题**（决定性理由，不是偏好）：头条
评测基准 unarXive_citrec 的 label 就是被引论文的 OpenAlex ID，注入标题等于把标签
泄漏进被检索的文本，基准立刻作废。

代价写在明面上：**放弃公式内容的可检索性**。M7 用小样本做 delete vs 保留 LaTeX
的 A/B，给这个取舍一个数字，而不是只给理由。
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field

__all__ = [
    "PlaceholderStripResult",
    "count_residual_placeholders",
    "normalize_unicode_and_whitespace",
    "strip_placeholders",
]

#: 占位符正则。uuid 段放宽到 `[0-9a-zA-Z-]`——unarXive 各版本的 key 形态不完全一致，
#: 正则写太紧的后果是静默漏删（漏删就等于上面那两层污染原样回来）。
_FORMULA_RE = re.compile(r"\{\{formula:[0-9a-zA-Z-]+\}\}")
_CITE_RE = re.compile(r"\{\{cite:[0-9a-zA-Z-]+\}\}")
#: 残留检测用的宽形态：任何 `{{...}}`。这是常驻监控项（M1 §5.1「残留检测」）。
_ANY_PLACEHOLDER_RE = re.compile(r"\{\{[^{}]{0,128}\}\}")

#: 删除占位符后典型的孤立标点：`( , )` / `[ ]` / ` ,` / ` ;` / 连续逗号。
_ORPHAN_BRACKET_RE = re.compile(r"[(\[]\s*[,;]?\s*[)\]]")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,;.:!?])")
_REPEATED_PUNCT_RE = re.compile(r"([,;])(\s*[,;])+")
#: 零宽字符：whitespace tokenizer 下它们会造出幽灵 token（M1 §5.3）。
_ZERO_WIDTH_RE = re.compile("[\\u200b\\u200c\\u200d\\ufeff]")
_INLINE_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{2,}")


class PlaceholderStripResult(BaseModel):
    """一段文本的占位符清洗产物。

    三个计数字段不是调试用的：它们要汇总成 §10 的「占位符残留率」质量指标，
    而残留率是常驻监控项——uuid 形态一旦有变体，漏删是静默的。
    """

    text: str = Field(description="已删除占位符、已清理孤立标点的文本")
    cite_keys: list[str] = Field(
        default_factory=list, description="按出现顺序的 {{cite:key}} 的 key，用于回查 bib_entries"
    )
    formula_keys: list[str] = Field(default_factory=list, description="按出现顺序的 {{formula:key}} 的 key")
    formula_count: int = Field(default=0, ge=0)
    cite_count: int = Field(default=0, ge=0)
    residual_count: int = Field(
        default=0, ge=0, description="清洗后仍匹配 `{{...}}` 的数量。非 0 说明正则漏配，属于必须告警的情况"
    )


def _extract_keys(pattern: re.Pattern[str], text: str, prefix: str) -> list[str]:
    """抽出占位符里的 key（`{{cite:abc}}` → `abc`），保持出现顺序。

    保持顺序是有意义的：`formula_latex` 按块内出现顺序保存（M1 §6.11 C 类字段），
    顺序丢了就无法把第 N 个公式对回原文位置。
    """
    head = len(prefix) + 2  # "{{" + prefix
    return [m.group(0)[head:-2] for m in pattern.finditer(text)]


def strip_placeholders(text: str) -> PlaceholderStripResult:
    """删除两类占位符并抢救出它们的 key。

    顺序是固定的：先抽 key（此时占位符还在）→ 再删文本 → 再清孤立标点。
    抽 key 必须在删之前，否则信息就真的丢了——本模块的目的是把信息从**被索引的
    文本**里挪到**不被索引的字段**里，不是把信息删掉。
    """
    formula_keys = _extract_keys(_FORMULA_RE, text, "formula:")
    cite_keys = _extract_keys(_CITE_RE, text, "cite:")

    cleaned = _FORMULA_RE.sub(" ", text)
    cleaned = _CITE_RE.sub(" ", cleaned)

    # 删除后留下的形态：`(, )`、`[ ]`、`word ,`、`a ,, b`。
    # 不清理会直接进 english analyzer，虽然标点会被 tokenizer 丢掉，
    # 但它们会先污染句子边界识别（§6.8 长段回退档要按句子切）。
    cleaned = _ORPHAN_BRACKET_RE.sub(" ", cleaned)
    cleaned = _REPEATED_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _INLINE_SPACE_RE.sub(" ", cleaned).strip()

    return PlaceholderStripResult(
        text=cleaned,
        cite_keys=cite_keys,
        formula_keys=formula_keys,
        formula_count=len(formula_keys),
        cite_count=len(cite_keys),
        residual_count=count_residual_placeholders(cleaned),
    )


def count_residual_placeholders(text: str) -> int:
    """统计清洗后仍存在的 `{{...}}`。汇总即 §10 的占位符残留率。"""
    return len(_ANY_PLACEHOLDER_RE.findall(text))


def normalize_unicode_and_whitespace(text: str) -> str:
    """S3 · 编码与空白标准化（M1 §5.3）。

    **顺序是硬约束**：NFKC 必须在占位符删除**之后**跑。某些兼容字符折叠会改变
    花括号形态（如全角 `｛`），先折叠会让 `strip_placeholders` 的正则漏配——
    这正是本函数不与 `strip_placeholders` 合并成一个「清洗」函数的原因。

    保留段落分隔（空行）同样是硬约束：段落边界是切块的原子单位（R1），
    在这一步被抹掉，`chunking` 就再也拿不回来了。
    """
    # NFKC：全角字母 Ａ→A、带圈数字 ①→1 等兼容字符折叠。
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _INLINE_SPACE_RE.sub(" ", normalized)
    # 连续空行压成一个——空行是段落分隔符，个数不携带信息。
    normalized = _BLANK_LINES_RE.sub("\n\n", normalized)
    return normalized.strip()
