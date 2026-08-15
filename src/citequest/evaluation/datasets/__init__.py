"""评测数据集适配器。

- `schemas.py`   —— `EvalDataset` / `EvalQuery` / `Provenance` / `SubsetStats` 契约
- `citrec.py`    —— unarXive_citrec test split（225,348 条）→ **库内子集**
- `litsearch.py` —— LitSearch（query 597 / corpus 64,183 title+abstract）+ BM25 自证

两个基准各对应一层语料，这是刻意的：citrec 是段落级的（只测得到全文层），
LitSearch 是 title+abstract 的（只测得到摘要层）。

**citrec 的两条防泄漏措施是同一个决定的两面**：
(a) 这里删除 marker 及 `marker_offsets` 邻域；
(b) 索引侧已删除 `{{cite:uuid}}` 而不是替换成被引标题。
少做任何一条，模型都能从输入里直接读出答案。

**退场阈值写死在配置里**：库内子集 < 约 3,000 条 → 头条基准退回 LitSearch，
citrec 降级为补充证据（`citrec.headline_survives`）。低于这个规模，bootstrap 95% CI
会宽到任何消融结论都不显著——那时候再报数字就是在报噪声。

LitSearch 的 qrels 密度是 1.07 个相关文档/query，所以 **Precision@10 上限只有
0.1 左右**，报告里必须解释，否则看起来像是模型很差。

`unarXive_imrad_clf` **不作检索指标**，只在建库期校验 chunk 的 `sec_type` 分布与
原始分布是否一致，所以它属于数据质量监控（M7 §10.3）而不是这个子包。
"""

from citequest.evaluation.datasets.citrec import (
    CITREC_PROVENANCE,
    build_in_corpus_subset,
    headline_survives,
    load_citrec_records,
    normalize_openalex_id,
    strip_marker_neighborhood,
)
from citequest.evaluation.datasets.litsearch import (
    LITSEARCH_PROVENANCE,
    BaselineSanityCheck,
    bm25_sanity_check,
    build_litsearch_dataset,
    load_litsearch_raw,
)
from citequest.evaluation.datasets.schemas import (
    RELEVANT,
    EvalDataset,
    EvalQuery,
    Provenance,
    SubsetStats,
)

__all__ = [
    "CITREC_PROVENANCE",
    "LITSEARCH_PROVENANCE",
    "RELEVANT",
    "BaselineSanityCheck",
    "EvalDataset",
    "EvalQuery",
    "Provenance",
    "SubsetStats",
    "bm25_sanity_check",
    "build_in_corpus_subset",
    "build_litsearch_dataset",
    "headline_survives",
    "load_citrec_records",
    "load_litsearch_raw",
    "normalize_openalex_id",
    "strip_marker_neighborhood",
]
