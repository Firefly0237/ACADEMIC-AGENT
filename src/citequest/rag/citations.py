"""[N] 引用编号的分配、解析与回填（M4 §3.3、§5.1）。

**`[N]` 这个选择的真正价值不在格式，在它让硬 gate 成立**：编号是系统分配的有限
集合，于是「引用是否合法」就是一次集合成员判定——**没有任何模型参与，误判率
结构性为零**。整套引用可信体系建立在这个性质上。

被否掉的三种写法各自的失败点：

| 候选 | 结论 |
|---|---|
| `(作者, 年份)` | 同一作者同年多篇会撞；模型可能写出证据里根本没有的作者年份组合，**校验要做模糊匹配，硬 gate 立刻退化成软信号** |
| DOI / paper_id 内联 | 长串 ID 占 token、模型容易抄错一位、用户读不下去 |
| 结构化输出（JSON 带 citations 数组） | JSON 必须完整才能解析，与按句放行的流式模型直接冲突 |

**多引用写死一种写法**：`[1][3]` 而不是 `[1,3]`——前者能用同一条正则一次扫出所有
引用边，后者要多一层拆分逻辑，且模型会在两种写法之间漂。
"""

from __future__ import annotations

import re

from citequest.core.answer import Citation, CitationAnchor, Evidence
from citequest.core.enums import CitationSupport
from citequest.core.search import RetrievedChunk

__all__ = [
    "CITATION_RE",
    "allocate_citation_ids",
    "bind_anchors",
    "citation_ids_in",
    "parse_citations",
    "unsupported_sentences",
]

#: 扫引用边的唯一正则。**同时接受 `[1][3]` 与被 prompt 禁止的 `[1,3]`**——
#: 校验必须能看见模型的违规写法，看不见就等于默认它没发生。
#: 逗号形式在解析后会被记为一次格式违规（见 `parse_citations`）。
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def allocate_citation_ids(chunks: list[RetrievedChunk]) -> list[Evidence]:
    """rerank 后的 chunk → 带 `[N]` 编号的 Evidence 列表。

    **编号由系统分配，从 1 开始连续**，模型只负责填。不让模型自取编号是注入防护的
    第 3 道：模型无法凭空造出一个合法引用边。

    顺序即 rerank 分数降序（调用方已排好），**这里不重排**：编号顺序与 prompt 里
    的出现顺序一致，读者从回答跳到 Evidence 区时不需要在一堆乱序编号里找。

    `granularity` 逐条带上——这是全链路唯一「不写就会静默产幻」的字段：
    摘要块必须告诉模型「这里只有结论，没有实现细节」。
    """
    evidences: list[Evidence] = []
    for index, chunk in enumerate(chunks, start=1):
        evidences.append(
            Evidence(
                citation_id=index,
                anchor=CitationAnchor(
                    paper_id=chunk.paper_id,
                    chunk_id=chunk.chunk_id,
                    # 摘要块的 section/para 直接为 None 而不是填 "N/A"：
                    # 填 N/A 会让模型把「摘要」与「全文里某个未知章节」混淆。
                    section=chunk.section,
                    sec_number=chunk.sec_number,
                    paragraph_idx=chunk.paragraph_idx,
                ),
                text=chunk.text,
                # title 由论文级元数据侧回填；此处先用 paper_id 占位——
                # **宁可显示 ID 也不显示一个编出来的标题**。
                title=chunk.paper_id,
                year=chunk.year,
                granularity=chunk.granularity,
                sec_type=chunk.sec_type,
                rerank_score=chunk.rerank_score,
            )
        )
    return evidences


def citation_ids_in(sentence: str) -> list[int]:
    """扫一句话里的全部引用编号，按出现顺序、去重。

    去重是必要的：`"...[3] 并且 ...[3]"` 是同一条引用边，判两次只会让 grounding
    的分母虚高，而 citation precision 的分母定义是**边数**不是标记出现次数。
    """
    found: list[int] = []
    for match in CITATION_RE.finditer(sentence):
        for part in match.group(1).split(","):
            value = int(part.strip())
            if value not in found:
                found.append(value)
    return found


def parse_citations(
    sentence: str,
    valid_ids: set[int],
) -> tuple[list[Citation], list[int], bool]:
    """解析一句话的引用边。

    :returns: `(合法引用, 越界编号, 是否用了逗号写法)`。
        **越界编号单独返回而不是丢弃**——它是唯一敢用来硬拦截的信号
        （M4 §5.0：确定性硬校验，误判率结构性为零），丢掉它硬 gate 就没有输入了。

    `anchor` 在这里留空由调用方按 Evidence 回填：Citation 的锚点必须与 Evidence
    的锚点是同一份数据，在两个地方各构造一次迟早会漂。
    """
    comma_form = any("," in m.group(1) for m in CITATION_RE.finditer(sentence))
    ids = citation_ids_in(sentence)
    legal = [i for i in ids if i in valid_ids]
    out_of_range = [i for i in ids if i not in valid_ids]
    citations = [
        Citation(
            citation_id=i,
            anchor=CitationAnchor(paper_id="", chunk_id=""),
            support=CitationSupport.UNVERIFIED,
        )
        for i in legal
    ]
    return citations, out_of_range, comma_form


def bind_anchors(citations: list[Citation], evidences: list[Evidence]) -> list[Citation]:
    """把 Evidence 的锚点回填进 Citation。

    单列一步而不是在解析时就查表，是为了让解析函数保持纯字符串操作、能被单测
    直接断言；而锚点绑定需要 Evidence 上下文，属于组装期的事。
    """
    by_id = {e.citation_id: e for e in evidences}
    bound: list[Citation] = []
    for citation in citations:
        evidence = by_id.get(citation.citation_id)
        if evidence is None:  # pragma: no cover - 合法 id 必在 Evidence 里
            continue
        bound.append(citation.model_copy(update={"anchor": evidence.anchor}))
    return bound


def unsupported_sentences(sentences: list[str]) -> list[int]:
    """返回**一条引用边都没有**的句子下标。

    这是沿引用边判定这个定义的**已知代价**（M4 §5.2 末段）：完全不标引用的编造句
    在这个框架下没有引用边可判。补法不是把它藏起来，而是**把它变成一个数字**——
    「无引用句占比」是一个可观测指标，进 M7 回归。

    注意它不区分「关键论断句」与「过渡句」：判断一句话是不是关键论断需要一个
    我没有标注数据去标定的判别器，而按全部句子算的占比是一个稳定、可比较的口径。
    """
    return [i for i, s in enumerate(sentences) if not citation_ids_in(s)]
