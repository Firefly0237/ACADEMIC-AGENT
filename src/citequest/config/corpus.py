"""语料与数据处理参数（决策记录 §4.1）。

这一组里有**四项建库前不可逆**的参数中的一项：占位符处理策略。改了就是重灌
147 万 chunk + 一次全量 TEI 重编码，6–10 小时量级。另外三项在 `milvus.py`
（analyzer 类型、bm25 参数、text_exact 是否存在）。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

__all__ = ["CorpusSettings"]


class CorpusSettings(BaseModel):
    """语料定义与切块参数。数值全部取自决策记录 §4.1，不许在别处另写一份。"""

    # ---- 全文层：unarXive open subset（CC-BY-SA-4.0，2019–2022）----
    fulltext_zenodo_doi: str = Field(
        default="10.5281/zenodo.7752615",
        description="unarXive open subset 的 Zenodo DOI。tar.xz 4,844,918,096 字节（Zenodo API 实测）",
    )
    fulltext_category_prefix: str = Field(default="cs.", description="metadata.categories 的 primary 前缀过滤")
    fulltext_year_from: int = Field(
        default=2019, description="提交年下界。若解压后 CS 篇数不足 5 万，放宽到 2017（U11）"
    )
    fulltext_year_to: int = Field(default=2022, description="全文层只到 2022——这是换源的三条代价之一，不藏")
    fulltext_target_papers: int = Field(
        default=50_000,
        description="目标篇数。可用上限估约 9.2 万，**以解压后实测为准**（U11）；超出则从最新年份往回截",
    )

    # ---- 摘要层：OpenAlex（CC0，2023–2025）----
    openalex_field_id: str = Field(default="fields/17", description="primary_topic.field.id 过滤，即 Computer Science")
    openalex_require_arxiv_primary_source: bool = Field(default=True)
    abstract_year_from: int = Field(default=2023)
    abstract_year_to: int = Field(default=2025)
    abstract_papers: int = Field(default=219_676, description="用户实测 67,007 + 75,538 + 77,131")

    # ---- 切块（M1 §6）----
    chunk_target_tokens: int = Field(default=450, ge=1, description="按 section 边界聚合段落的目标长度")
    chunk_min_tokens: int = Field(default=80, ge=1, description="低于此值并入同 section 相邻段")
    chunk_respect_section_boundary: bool = Field(
        default=True,
        description="按 section 硬停不跨界。跨 section 的块没有唯一 section，引用锚点直接失效（M1 §6.3）——这条不是可调项",
    )
    tokenizer_model_id: str = Field(
        default="BAAI/bge-m3",
        description="计 token 用与 dense 编码器同一个分词器，否则 450 token 的预算与实际编码长度对不上",
    )

    # ---- 占位符（⚠ 建库前不可逆）----
    strip_placeholders: bool = Field(
        default=True,
        description=(
            "{{formula:uuid}} / {{cite:uuid}} 进索引前一律删除。两个理由都是硬的："
            "(1) english analyzer 会把它们切成几十万个只出现一次的 uuid 垃圾 token，"
            "放大倒排并污染 avgdl 从而扭曲每个正常词的 BM25 分数；"
            "(2) 删而不是替换成被引标题——替换等于把 unarXive_citrec 的 label 泄漏进索引文本，头条基准立刻作废（H17）"
        ),
    )
    formula_placeholder_pattern: str = Field(default=r"\{\{formula:[0-9a-fA-F-]+\}\}")
    cite_placeholder_pattern: str = Field(default=r"\{\{cite:[0-9a-fA-F-]+\}\}")
    keep_formula_latex_in_metadata: bool = Field(
        default=True,
        description="公式 LaTeX 保留在不参与检索的 metadata 字段。代价是放弃公式内容可检索性，M7 用小样本做 delete vs 保留的 A/B",
    )

    # ---- 数据目录（cookiecutter-data-science 四层）----
    data_root: Path = Field(default=Path("data"))
    raw_dir: Path = Field(default=Path("data/raw"), description="原始不可变数据：unarXive tar.xz、OpenAlex 快照")
    interim_dir: Path = Field(default=Path("data/interim"), description="清洗后的 section")
    processed_dir: Path = Field(default=Path("data/processed"), description="最终 450-token chunk，即灌库输入")
    external_dir: Path = Field(default=Path("data/external"), description="第三方评测集：citrec / LitSearch")

    # ---- 估算值：必须能被实测回填（U1）----
    estimated_chunks_per_paper: int = Field(
        default=25,
        description="【估算，M1 必须实测回填】它同时决定 147 万 chunk 的规模账、6.5 GB 的内存账、以及 ef 派生式里 C 的合理量级",
    )

    @property
    def estimated_total_chunks(self) -> int:
        """约 147 万 = 全文层 125 万 + 摘要层 22 万。

        这个数不是拿来炫耀规模的，是拿来给「哪些参数必须建库前定死」定价的：
        147 万 chunk 全量重灌是 6–10 小时量级。
        """
        return self.fulltext_target_papers * self.estimated_chunks_per_paper + self.abstract_papers
