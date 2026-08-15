"""分面：两类口径，分别用不同机制（决策记录 D11、M6 §1.3）。

**这不是性能问题，是正确性问题。** 候选集只有几十组，拿它统计年份直方图得到的是
「**相关文档**的年份分布」，不是「**符合筛选条件的论文**的年份分布」——而前端会把
它渲染成筛选器计数，用户当成后者。**把近似值当精确值展示**是这一条被拆成两类的
全部理由。

| 类 | 数据源 | 契约 |
|---|---|---|
| 结果内分面 | grouping 后的 paper 级结果，零额外查询 | 必带 `based_on="current_result_set"` + `sample_size` |
| 全库精确分面 | 单机关系库（DuckDB 读 parquet）跑 GROUP BY，或建库时离线算成随 `index_version` 发布的静态 JSON | 必带 `index_version` |

**不引入 Postgres**（D11）：年份直方图/领域总量是只读、低基数、随索引版本变化的
聚合，27 万行在单机关系库上是毫秒级——不需要新服务。这与「不新增有状态服务」
否决 Elasticsearch 是同一条判据，**同一个代价一次否决一次接受才是最容易被抓的不一致**。

**venue 的全库精确 top-10 不做**（O24）：「全库哪个会议论文最多」对论文检索没有
真实决策价值。
"""

from __future__ import annotations

from collections import Counter

from citequest.core.search import ExactFacets, InResultFacets, PaperHit
from citequest.observability.logging import get_logger

__all__ = ["ExactFacetSource", "build_in_result_facets"]

_logger = get_logger(__name__)


def build_in_result_facets(papers: list[PaperHit]) -> InResultFacets:
    """结果内分布。**零额外查询**——它就是对已有结果的一次计数。

    `sample_size` 统计的是**组数（论文数）而不是 chunk 数**：分面是给用户按论文
    筛选用的，用 chunk 数会让全文层论文（每篇约 25 块）在分布里被放大 25 倍，
    得到一个「2019–2022 占绝对多数」的假象。

    维度只做 `primary_category` 与 `granularity` 两个。**`granularity` 这一维是本
    项目特有的**：它把「这批结果里有多少篇只有摘要」直接摆给用户看，
    是能力边界的可执行表达（M3 §3.4 第 2 条实现约束）。
    """
    categories = Counter(p.primary_category for p in papers if p.primary_category)
    granularities = Counter(p.granularity.value for p in papers)
    return InResultFacets(
        sample_size=len(papers),
        primary_category=dict(categories),
        granularity=dict(granularities),
    )


class ExactFacetSource:
    """全库精确分面的取数口。

    形态是**两选一**，由部署决定（D11）：
    1. DuckDB 直接读论文级 parquet 跑 `GROUP BY`，条件用用户的过滤条件；
    2. 建库时离线算成一份随 `index_version` 发布的静态 JSON，查询时按条件切片。

    第 2 种更省，代价是新鲜度延迟——靠换 `index_version` 一起更新。
    **诚实标注**：DuckDB 在 27 万篇元数据上的 GROUP BY 延迟【未实测】，
    判断依据是常识性推断。若实测不达标，正确补救是改成静态 JSON，
    **而不是回头引入 Postgres**。
    """

    def __init__(self, index_version: str, *, parquet_path: str | None = None) -> None:
        self._index_version = index_version
        self._parquet_path = parquet_path

    def fetch(
        self, *, year_from: int | None = None, year_to: int | None = None, primary_category: str | None = None
    ) -> ExactFacets:
        """按用户过滤条件取全库精确聚合。

        **接真实实现**：这里应当是一次 DuckDB 查询——

            SELECT year, COUNT(*) FROM read_parquet(?) WHERE <条件> GROUP BY year

        连接与 SQL 拼装留在这一层，是因为**条件必须与 Milvus 那边下推的条件一致**：
        两边条件不一致时，分面计数会和结果集对不上，而这类偏差不报错。
        M7 有一条校验就是抽样对比两者偏差，量化「不拆会错多少」。
        """
        if self._parquet_path is None:
            # 未配置数据源时返回空聚合而不是抛异常：分面是**可选**的响应字段
            # （`facets_exact: ExactFacets | None`），缺它不影响检索结果本身。
            _logger.info("exact_facets_unavailable", index_version=self._index_version)
            return ExactFacets(index_version=self._index_version)
        raise NotImplementedError(
            "接真实实现：DuckDB 读论文级 parquet 跑 GROUP BY（决策记录 D11）。"
            f"条件 year∈[{year_from},{year_to}] primary_category={primary_category}；"
            "备选是建库时离线算成随 index_version 发布的静态 JSON"
        )
