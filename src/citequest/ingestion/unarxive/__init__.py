"""全文层：unarXive open subset（CC-BY-SA-4.0，cs.*，2019–2022）。

源：Zenodo DOI `10.5281/zenodo.7752615`，tar.xz 约 4.84 GB。

这一层是「换源」买到的核心能力：section 级正文让引用锚点能细化到
`(paper_id, section, sec_number, paragraph_idx)` 四元组，而摘要层只到 paper_id。

三件事必须在这里做对，且都是**建库前不可逆**的：

1. 按 section 边界**硬停**聚合，不跨 section。跨 section 的块没有唯一 section，
   引用锚点直接失效（M1 §6.3）——这不是可调项。
2. `{{formula:uuid}}` / `{{cite:uuid}}` 进索引前**删除**而非替换成被引标题。
   替换等于把 unarXive_citrec 的 label 泄漏进索引文本，头条基准立刻作废。
3. `bib_entries[*].ids.open_alex_id` 落到 `cite_edges` 边表，**不参与检索**。
   不用 OpenAlex 的 `referenced_works`——实测 arXiv-primary CS 2023 抽样 200 篇里
   94.0% 的 `referenced_works_count` 为 0，那不是覆盖率低，是数据不存在。
"""

from citequest.ingestion.unarxive.download import EXPECTED_ARCHIVE_BYTES, ZenodoFile, download_unarxive, verify_file
from citequest.ingestion.unarxive.pipeline import FulltextBuildResult, UnarxivePipeline
from citequest.ingestion.unarxive.reader import FilterStats, UnarxiveRecord, iter_records, passes_filters

__all__ = [
    "EXPECTED_ARCHIVE_BYTES",
    "FilterStats",
    "FulltextBuildResult",
    "UnarxivePipeline",
    "UnarxiveRecord",
    "ZenodoFile",
    "download_unarxive",
    "iter_records",
    "passes_filters",
    "verify_file",
]
