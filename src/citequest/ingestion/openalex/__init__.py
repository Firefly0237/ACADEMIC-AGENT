"""摘要层：OpenAlex（CC0，Computer Science，2023–2025），219,676 篇。

筛选条件：`primary_topic.field.id=fields/17` + arXiv 为 primary source
+ 发表日期落在 2023-01-01 ~ 2025-12-31。逐年实测量：67,007 / 75,538 / 77,131。

这一层的价值是**时效**：全文层只到 2022，没有它，2023 年后的工作在库里完全不存在。
代价同样明确且必须主动讲——摘要没有 section，方法级问题它答不了。所以
`granularity` 是刻意暴露给调用方的能力边界，不是内部实现细节：路由层不按粒度
分流，模型就会拿摘要编方法细节（决策记录 §2.7）。

OpenAlex 的 `abstract_inverted_index` 需要还原成正文——这是本包第一件要做的事。
"""

from citequest.ingestion.openalex.abstract import AbstractRebuildResult, rebuild_abstract
from citequest.ingestion.openalex.client import ARXIV_SOURCE_ID, OpenAlexClient, build_works_filter
from citequest.ingestion.openalex.pipeline import AbstractBuildResult, OpenAlexPipeline

__all__ = [
    "ARXIV_SOURCE_ID",
    "AbstractBuildResult",
    "AbstractRebuildResult",
    "OpenAlexClient",
    "OpenAlexPipeline",
    "build_works_filter",
    "rebuild_abstract",
]
