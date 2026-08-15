"""语料域数据契约：Paper / Chunk / Attribution。

这是数据层（M1）的**输出**，也是索引层（M2）的**输入**。
一条纪律写在类型里：数据层不产出向量，也不产出稀疏权重（M1 开篇）——
文本原样交给 Milvus，由服务端 Function 完成分词与编码，
「灌库用 A 模型、查询用 B 模型」这类不一致在结构上不可能发生。

字段清单来自决策记录 §4.1「统一输出 schema」与 M1 §7。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from citequest.core.enums import Granularity, LicenseId, SecType, SourceDataset

__all__ = ["Attribution", "Chunk", "CiteEdge", "Paper"]


class Attribution(BaseModel):
    """署名信息。在线返回检索片段属于许可意义上的分发，必须带署名（决策记录 D2 第 5 条）。

    逐条下发而非全局声明：全文层 CC-BY-SA-4.0、摘要层 CC0 混在同一个响应里，
    而汇编（Collection）不是改编（Adaptation），一句全局混合声明糊不过去。
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(
        description="人类可读的署名串，例如 'unarXive open subset (Saier et al., JCDL 2023), CC BY-SA 4.0'"
    )
    source_url: str = Field(description="数据集本身的稳定标识，全文层是 Zenodo DOI")
    original_url: str | None = Field(default=None, description="链回原文（arXiv abs 页）。摘要层可能没有")


class Paper(BaseModel):
    """论文级元数据。

    `paper_id` 统一为去前缀的 OpenAlex ID（`W` + 数字）。全文层通过 arXiv ID 反查
    对齐；对齐不上的记录保留 arXiv ID 作为主键并标记，**不丢弃**（M1 §5.5）。
    """

    paper_id: str = Field(description="主键。OpenAlex ID 去前缀形式（W…），对齐失败时为 arXiv ID")
    title: str
    year: int = Field(ge=1900, le=2100, description="发表年。范围校验见 M1 §5.5")
    primary_category: str | None = Field(default=None, description="arXiv 类目体系，如 cs.CL")
    authors: list[str] = Field(default_factory=list, description="保留原始 display_name，不做缩写统一（无权威映射表）")
    venue: str | None = None
    doi: str | None = Field(default=None, description="去 https://doi.org/ 前缀并全小写")
    arxiv_id: str | None = None
    cited_by_count: int | None = Field(default=None, ge=0)
    granularity: Granularity = Field(description="这篇论文在库内的最高粒度")
    has_fulltext: bool = Field(description="冗余字段，等价于 granularity == FULLTEXT；API 响应直接下发，前端不做推导")
    license: LicenseId
    source_dataset: SourceDataset
    attribution: Attribution
    is_retracted: bool = Field(
        default=False,
        description="撤稿不物理删除，只打标记按需过滤（M1 §5.4）；但缓存必须按 paper_id 精确失效——这是失效纪律唯一的例外",
    )

    @field_validator("paper_id")
    @classmethod
    def _paper_id_not_blank(cls, v: str) -> str:
        # paper_id 是 grouping search 的分组键，空值会让整组结果归并到一起，
        # 属于会静默产生错误结果而不是报错的那类 bug，所以在入口拦掉。
        if not v.strip():
            raise ValueError("paper_id 不能为空")
        return v


class Chunk(BaseModel):
    """检索的最小单位，也是引用锚点的载体。

    引用锚点是 `(paper_id, section, sec_number, paragraph_idx)` 四元组——这是换到
    全文语料买到的核心能力，也正是 M1 §6.3 拒绝跨 section 聚合的原因：跨 section
    的块没有唯一的 section，引用锚点直接失效。

    `text` 是**已清洗**的文本：`{{formula:uuid}}` / `{{cite:uuid}}` 占位符在进索引
    前一律删除（决策记录 D5）。删而不是替换成被引论文标题，理由是决定性的——
    头条基准 unarXive_citrec 的 label 就是被引论文的 OpenAlex ID，注入被引标题
    等于把标签泄漏进索引文本，基准立刻作废（H17）。
    """

    chunk_id: str = Field(
        description="主键，由 f(paper_id, granularity, section 序号, 块内起止 paragraph_idx) 确定性构造（M1 §8.1），重跑切块结果一致"
    )
    paper_id: str = Field(description="grouping search 的分组键 group_by_field")
    granularity: Granularity
    text: str = Field(description="已删除占位符的正文；同时灌进 text_en（挂 BM25）与 text_exact（只做 MATCH）两个字段")
    section: str | None = Field(default=None, description="章节标题原文。摘要层为 None")
    sec_number: str | None = Field(default=None, description="章节编号，如 '4.2'")
    sec_type: SecType | None = Field(default=None, description="IMRaD 类型，由 unarXive 给出，不做标题正则识别")
    paragraph_idx: int | None = Field(default=None, ge=0, description="块内起始段落序号，引用锚点第四元")
    year: int = Field(ge=1900, le=2100, description="标量过滤字段，同时是 partition key")
    primary_category: str | None = None
    license: LicenseId
    source_dataset: SourceDataset
    token_count: int | None = Field(
        default=None,
        ge=0,
        description="切块时的实测 token 数。目标 450、最小 80（决策记录 §4.1）；落盘是为了 U1「chunk/paper=25 未实测」能被回填",
    )
    cited_open_alex_ids: list[str] = Field(
        default_factory=list,
        description="本块内 {{cite:uuid}} 解析出的被引 OpenAlex ID。**不参与检索**的 metadata 字段，只喂 cite_edges 边表",
    )
    formula_latex: list[str] = Field(
        default_factory=list,
        description="被删除的公式 LaTeX 原文。同样不参与检索——代价是放弃公式内容可检索性（决策记录 D5）",
    )

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, v: str) -> str:
        # 空文本 chunk 会被 BM25 Function 正常接收并产生一个空 sparse 向量，
        # 既不报错也永远不会被命中，属于静默的数据腐化。
        if not v.strip():
            raise ValueError("chunk.text 不能为空（占位符清洗后为空的块应在切块阶段丢弃）")
        return v


class CiteEdge(BaseModel):
    """引用边。落一张 `cite_edges(chunk_id, cited_open_alex_id)` 边表进单机关系库（M1 §7.4 方案 B）。

    引用边来自 unarXive 的 `bib_entries[*].ids.open_alex_id`，**不用** OpenAlex 的
    `referenced_works`——实测 arXiv-primary CS 2023 抽样 200 篇里 94.0% 的
    `referenced_works_count` 为 0，那不是覆盖率低，是数据不存在（决策记录 §2.2）。
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    paper_id: str
    cited_open_alex_id: str
