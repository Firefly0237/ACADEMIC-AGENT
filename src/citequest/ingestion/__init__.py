"""语料获取与切块（M1）：从原始数据到可灌库的 chunk。

**这一层不产出向量，也不产出稀疏权重**（M1 开篇的第一条纪律）。文本原样交给
Milvus，由服务端 Function 完成分词与编码。这样「灌库用 A 模型、查询用 B 模型」
这类不一致在结构上不可能发生——它是本项目最贵的一类静默故障。

模块划分：

- `unarxive/` —— 全文层：解压 tar.xz、按 `metadata.categories` 筛 cs.*、
                 2019–2022、`body_text` 非空；按 section 边界聚合到 450 token
- `openalex/` —— 摘要层：`primary_topic.field.id=fields/17` + arXiv primary source
                 + 2023–2025，每篇 1 chunk
- `chunking.py` —— section 边界硬停的段落聚合（本层最重的一段逻辑）
- `chunk_models.py` —— 切块的输入契约 / 统计契约 / `chunk_id` 确定性构造
- `sentence_split.py` —— 单段超过 450 token 时的多级降级回退档（§6.8）
- `placeholders.py` —— `{{formula:uuid}}` / `{{cite:uuid}}` 的删除与旁路保存（S1/S3）
- `cleaning.py` —— LaTeX 解析噪声（S2）
- `dedup.py` —— 两级去重（S4）
- `normalize.py` —— 字段标准化与逐条署名（S5）
- `tokenization.py` —— token 计数，纪律是「必须与编码器同一个分词器」
- `checkpoint.py` —— 断点续跑：原子写 + JSONL 半行截断

输出契约直接用 `citequest.core.corpus` 的 `Paper` / `Chunk` / `CiteEdge`，
本包**不再定义第二套数据模型**。
"""

from citequest.ingestion.checkpoint import Checkpoint, CheckpointStore, truncate_incomplete_jsonl_tail
from citequest.ingestion.chunk_models import (
    ChunkingStats,
    SourceParagraph,
    build_chunk_id,
    build_cite_edges,
)
from citequest.ingestion.chunking import SectionAwareChunker
from citequest.ingestion.cleaning import CleanTextResult, clean_latex_noise
from citequest.ingestion.dedup import DedupReport, dedup_exact, dedup_fuzzy, normalize_title
from citequest.ingestion.normalize import (
    LAYER_DEFAULTS,
    UNARXIVE_ATTRIBUTION,
    normalize_doi,
    normalize_openalex_id,
    resolve_paper_id,
    validate_year,
)
from citequest.ingestion.placeholders import (
    PlaceholderStripResult,
    count_residual_placeholders,
    normalize_unicode_and_whitespace,
    strip_placeholders,
)
from citequest.ingestion.sentence_split import split_long_text
from citequest.ingestion.tokenization import TokenCounter, build_token_counter

__all__ = [
    "LAYER_DEFAULTS",
    "UNARXIVE_ATTRIBUTION",
    "Checkpoint",
    "CheckpointStore",
    "ChunkingStats",
    "CleanTextResult",
    "DedupReport",
    "PlaceholderStripResult",
    "SectionAwareChunker",
    "SourceParagraph",
    "TokenCounter",
    "build_chunk_id",
    "build_cite_edges",
    "build_token_counter",
    "clean_latex_noise",
    "count_residual_placeholders",
    "dedup_exact",
    "dedup_fuzzy",
    "normalize_doi",
    "normalize_openalex_id",
    "normalize_title",
    "normalize_unicode_and_whitespace",
    "resolve_paper_id",
    "split_long_text",
    "strip_placeholders",
    "truncate_incomplete_jsonl_tail",
    "validate_year",
]
