"""Context assembly：token 预算、排布与格式化（M4 §2）。

**预算不是被窗口卡住的**（M4 §2.1）：48 条 chunk 塞进 32K 窗口在物理上完全放得下
（48×450 ≈ 21.6K token），但仍然只放 8 条。真正的约束有三条，全都与窗口无关：

1. **证据条数的行业共识 4–10**：PaperQA2 `answer_max_sources=5`、RAGFlow `top_n=6`、
   Dify `top_k=4`、OpenScholar `--top_n=10`；
2. **中间位置衰减**（Lost in the Middle, arXiv:2307.03172）：条数是这个效应最直接的旋钮；
3. **首 token 延迟**：prefill 与输入长度近似线性，而首 token p50 预算只有 1.0s。

所以正确的讲法是：**按 8K 窗口做预算，但即使换成 128K 窗口也不放宽**——
放宽的收益是「多塞几条边缘证据」，代价是中段衰减 + prefill 变长 + 幻觉面变大。

**排布规则只有一条**：按 rerank 分数降序，从上到下。**不做位置洗牌**——
Lost in the Middle 是一个被观测到的**现象**，不是一个被验证过的**排列算法**；
把现象包装成自创算法是最容易在追问第二层塌掉的地方。要做位置优化的唯一正规路径
是 LangChain 官方的 `LongContextReorder` + 自己评测集上的 A/B 数字，列为 M7 可选消融。
"""

from __future__ import annotations

from dataclasses import dataclass

from citequest.config.rag import GenerationSettings
from citequest.core.answer import Evidence
from citequest.core.enums import Granularity
from citequest.observability.logging import get_logger
from citequest.rag.prompts.answer import EVIDENCE_SECTION_HEADER

__all__ = ["ContextBudget", "ContextPack", "assemble_context", "format_evidence_block"]

_logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class ContextBudget:
    """token 预算（M4 §2.5）。以 8K 窗口为基准，32K 下取值不变。

    | 组成 | token |
    |---|---|
    | system prompt（角色 + 5 条规则 + 引用格式） | ≈250 |
    | 8 个元数据头 | ≈240（30×8） |
    | 8 个证据块正文 | 1,760～3,600 |
    | 用户 query | ≈50 |
    | **合计** | **≈2,300～4,140，期望 ≈3,100** |

    这个期望值不是凑出来的：延迟预算表第 6 行「首句生成（含 prompt 预填充 ~3k token）」
    独立写着同一个量级——**两处推导相互印证，如果这里算出 6K 说明有一处口径错了**。
    """

    max_context_tokens: int = 4_096
    """硬上限。上界 4,140 已经比期望值高 35%，会把首 token p50 预算吃穿，所以设一道确定性的闸。"""

    defensive_input_limit_tokens: int = 16_384
    """防御性入口上限 = 4× 预算。形态照 PaperQA2 的 `page_size_limit`（注释写明是
    128k 上下文的 10 倍）——**防御上限就该设成远大于正常值的量级，它防的是畸形输入
    不是正常波动**。"""

    chars_per_token: float = 4.0
    """粗估系数。**这里刻意不引真分词器**：context 预算是一道闸不是一次计费，
    差 5% 不改变任何决策，而为它引 tokenizer 会让生成层多一个模型依赖，
    且不同后端的分词还不一致。**接真实实现：若 M7 实测 chunk 长度分布偏离，
    这里改成按后端 tokenizer 计数。**"""


@dataclass(slots=True)
class ContextPack:
    """组装产物。

    `truncated` 与 `dropped` 必须打点（M4 §2.5 末段）：丢块直接影响引用的可用范围，
    而它在响应里的表现只是「引用数变少了」——不打点就没人知道发生过。
    """

    evidences: list[Evidence]
    evidence_text: str
    estimated_tokens: int
    truncated: bool = False
    dropped: int = 0
    has_abstract_evidence: bool = False


def _estimate_tokens(text: str, chars_per_token: float) -> int:
    return int(len(text) / chars_per_token) + 1


def format_evidence_block(evidence: Evidence) -> str:
    """一条证据在 prompt 里的形态（M4 §2.4）。

    ```
    [3] Attention Is All You Need (2017)
        granularity=fulltext | section=3.2 Multi-Head Attention | para=2
        <<<
        ...原文...
        >>>
    ```

    每个字段都有一个「不放会怎样」：

    - `[N]` 编号：不放则模型自己发明引用格式，后校验无法用确定性字符串匹配去验，
      **硬 gate 直接失效**；
    - title + 年份：不放则模型把年份编错，或无法识别两条证据是「新旧更替」而非
      「互相矛盾」——年份同时是证据冲突那一节的判据来源；
    - **`granularity`**：不放则模型拿摘要回答方法细节，**编出一个合理但不存在的做法**；
    - `section`/`para`：不放则引用只能到论文级，「可核对」这个承诺打对折；
    - `<<< >>>`：**指令与数据的物理分区**，是注入防护的载体。

    元数据头成本约 30 token/块 × 8 = 240 token，占 prompt 预算约 8%——
    这是**为了 grounding 可核对而主动付的税**，不是浪费。
    """
    meta = [f"granularity={evidence.granularity.value}"]
    # 摘要块**直接省略** section 而不是填 N/A：填 N/A 会让模型把「摘要」
    # 和「全文里某个未知章节」混淆。
    if evidence.anchor.sec_number or evidence.anchor.section:
        number, title = evidence.anchor.sec_number, evidence.anchor.section
        # unarXive 的 section 标题里常常已经含编号（"3.2 Multi-Head Attention"），
        # 无条件拼接会得到 "3.2 3.2 Multi-Head Attention"——重复的编号会让模型
        # 在行文里把它当成两个不同的定位。
        label = title if (title and number and title.startswith(number)) else " ".join(x for x in (number, title) if x)
        meta.append(f"section={label}")
    if evidence.anchor.paragraph_idx is not None:
        meta.append(f"para={evidence.anchor.paragraph_idx}")

    return (
        f"[{evidence.citation_id}] {evidence.title} ({evidence.year})\n"
        f"    {' | '.join(meta)}\n"
        f"    <<<\n{evidence.text}\n    >>>"
    )


def assemble_context(
    evidences: list[Evidence],
    settings: GenerationSettings,
    budget: ContextBudget | None = None,
) -> ContextPack:
    """按预算装配 Evidence 区。

    **超限时从 rerank 分数最低的整块开始丢，丢到不超限为止**，并且
    **禁止半块截断**（M4 §2.5 硬约束）：半块会切断引用锚点，并产出一段
    「看起来完整、实际被腰斩」的证据——模型无法察觉，用户点进去也对不上。
    **宁可少一条完整证据，不要一条残块。**

    入参顺序即 rerank 降序（`allocate_citation_ids` 已按此分配编号），
    所以「丢最低分」等价于「从尾部丢」。
    """
    budget = budget or ContextBudget()
    if not evidences:
        return ContextPack(evidences=[], evidence_text="", estimated_tokens=0)
    if not settings.annotate_granularity_in_prompt:  # pragma: no cover - 配置被改动时的显式失败
        raise ValueError(
            "annotate_granularity_in_prompt=False 会让模型把摘要当方法细节来编"
            "（M4 §1.2、决策记录 §4.5）。这是生成层唯一「不做就静默产生幻觉」的开关，不允许关"
        )

    kept = evidences[: settings.context_max_chunks]
    dropped = len(evidences) - len(kept)

    blocks = [format_evidence_block(e) for e in kept]
    text = EVIDENCE_SECTION_HEADER + "\n" + "\n\n".join(blocks)
    tokens = _estimate_tokens(text, budget.chars_per_token)

    truncated = False
    while tokens > budget.max_context_tokens and len(kept) > 1:
        # 整块丢，从尾部（分数最低）开始。留最后一条不丢——一条证据都没有的
        # 「回答」在本系统里不存在：它会走信息不足那条路，而不是硬挤出一个无引用回答。
        kept.pop()
        dropped += 1
        truncated = True
        blocks = [format_evidence_block(e) for e in kept]
        text = EVIDENCE_SECTION_HEADER + "\n" + "\n\n".join(blocks)
        tokens = _estimate_tokens(text, budget.chars_per_token)

    if truncated:
        _logger.warning("context_truncated", dropped=dropped, kept=len(kept), tokens=tokens)

    return ContextPack(
        evidences=kept,
        evidence_text=text,
        estimated_tokens=tokens,
        truncated=truncated,
        dropped=dropped,
        has_abstract_evidence=any(e.granularity is Granularity.ABSTRACT for e in kept),
    )
