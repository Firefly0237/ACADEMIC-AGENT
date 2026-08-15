"""检索指标：nDCG / MRR / Recall / MAP（M7 §6）。

**本模块的每个定义都写到「能被追问算法」的程度。** 评测被击穿的地方通常不是指标
选得不对，而是说不清自己是怎么算的。

三条一次性说清、之后不再重复的口径：

| 口径 | 取值 | 为什么 |
|---|---|---|
| **聚合** | **按 query 平均（macro）** | 每条 query 权重相同，与「用户每次搜索的体验同样重要」一致。全局池化会让相关文档多的 query 主导整个指标 |
| **二值化阈值** | **不适用** | 两个基准的 qrels 本来就是二值的（citrec 唯一 label / LitSearch 639 条判定），所以不存在阈值。这一条要主动说，因为它是 nDCG 最常见的追问点 |
| **MRR 截断** | 前 k 位内无相关文档记 **0**，不是 1/(k+1) | 两种口径都有人用，不声明就无法比较 |

二值 rel 下 `gain = 2^rel − 1 = 1`，DCG 退化为纯位置折损 `Σ 1/log2(i+1)`。

**与 ranx 的关系**：显著性检验与多系统对比走 ranx（`ablation.py`），本模块负责
**逐 query 的指标值**——因为 win/loss/tie 三元计数与 bootstrap CI 都需要 per-query
数组，而不是一个均值。两处口径必须一致，`ablation.check_against_ranx` 就是用来
守这条的：同一份 run 两边算出来不一致，是本模块有 bug，不是 ranx。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from citequest.evaluation.datasets.schemas import RELEVANT

__all__ = [
    "METRIC_FUNCTIONS",
    "MetricReport",
    "average_precision",
    "macro_average",
    "ndcg_at_k",
    "parse_metric",
    "per_query_metric",
    "recall_at_k",
    "reciprocal_rank_at_k",
]

#: 一次检索的排序结果：doc_id 按名次从好到差。**不带分数**——所有指标只依赖名次，
#: 传分数进来只会诱使调用方在中途重排一次，而重排口径与线上就不一致了。
type Ranking = Sequence[str]
#: 一条 query 的判定：`{doc_id: rel}`。rel < RELEVANT 视为不相关。
type Judgements = dict[str, int]


def _relevant_set(judgements: Judgements) -> set[str]:
    return {doc for doc, rel in judgements.items() if rel >= RELEVANT}


def ndcg_at_k(ranking: Ranking, judgements: Judgements, k: int) -> float:
    """nDCG@k。

    ```
    DCG@k  = Σ_{i=1..k} (2^rel_i − 1) / log2(i + 1)   # 二值下 = Σ 1/log2(i+1)
    IDCG@k = 把该 query 的全部已判定相关文档排到最前面时的 DCG@k
    ```

    **单相关文档时的退化形态必须能讲**：IDCG@k = 1/log2(2) = 1，于是
    `nDCG@k = 1/log2(rank+1)`（rank ≤ k 时）。这时它与 MRR 高度相关但**不等价**
    ——log 折损比倒数折损平缓，所以「你 nDCG 和 MRR 为什么走势一样」的正确答案
    是「因为单相关文档，但曲线形状不同」。
    """
    relevant = _relevant_set(judgements)
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, doc in enumerate(ranking[:k]) if doc in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def reciprocal_rank_at_k(ranking: Ranking, judgements: Judgements, k: int) -> float:
    """RR@k：第一个相关文档的倒数名次；前 k 位内没有则记 **0**。

    截断语义是本项目声明过的那一种：第一个相关文档排在第 k+1 位及以后，该 query
    记 0，**而不是记 1/(k+1)**。
    """
    relevant = _relevant_set(judgements)
    for index, doc in enumerate(ranking[:k], start=1):
        if doc in relevant:
            return 1.0 / index
    return 0.0


def recall_at_k(ranking: Ranking, judgements: Judgements, k: int) -> float:
    """Recall@k = |top-k 中的相关文档| / |该 query 的全部**已判定**相关文档|。

    **citrec 上的退化必须主动交代**：每条 query 只有 1 个 label，分母恒为 1，
    所以 Recall@100 在那个基准上等价于 Hit@100 / Success@100（命中与否，0 或 1）。
    报告里写作 `Recall@100 (= Hit@100, single-relevant)`——把它叫「召回率」而不
    解释，就是在制造一个会被戳穿的印象。

    「已判定」三个字是分母的全部口径：未进入判定集的文档一律按不相关处理。
    """
    relevant = _relevant_set(judgements)
    if not relevant:
        return 0.0
    hit = sum(1 for doc in ranking[:k] if doc in relevant)
    return hit / len(relevant)


def average_precision(ranking: Ranking, judgements: Judgements, k: int) -> float:
    """AP@k。**本项目不把 MAP 作为对外主指标**（M7 §5.1）。

    理由是它的分母与 Recall 同病：`|该 query 的全部相关文档|` 在任何不完整判定下
    都会被高估，所以**不能只砍掉 Recall 保留 MAP**——它们要么一起用（在正确的
    pooling 协议下），要么一起不作为主口径。这里实现它，只为 smoke test 集内部
    做版本间的相对比较。
    """
    relevant = _relevant_set(judgements)
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, doc in enumerate(ranking[:k], start=1):
        if doc in relevant:
            hits += 1
            precision_sum += hits / index
    return precision_sum / len(relevant)


#: 指标名 → 计算函数。名字形如 `ndcg@10`，与 `EvalSettings.metrics` 里的字面值
#: 以及 ranx 的指标名保持一致——**三处用同一套字面值**，否则报告表头与实际算的
#: 东西会在某次重构后悄悄错位。
METRIC_FUNCTIONS = {
    "ndcg": ndcg_at_k,
    "mrr": reciprocal_rank_at_k,
    "recall": recall_at_k,
    "map": average_precision,
}


def parse_metric(name: str) -> tuple[str, int]:
    """`"ndcg@10"` → `("ndcg", 10)`。不合法的名字直接抛错。

    宽容解析在这里是有害的：把 `ndcg10` 静默当成 `ndcg@10` 意味着某天有人写错
    指标名，报告照样跑出一个数——而那个数的 k 是猜出来的。
    """
    head, _, tail = name.lower().partition("@")
    if head not in METRIC_FUNCTIONS or not tail.isdigit():
        raise ValueError(f"指标名必须形如 'ndcg@10'，且指标属于 {sorted(METRIC_FUNCTIONS)}；收到 {name!r}")
    return head, int(tail)


def per_query_metric(
    runs: dict[str, Ranking],
    qrels: dict[str, Judgements],
    metric: str,
) -> dict[str, float]:
    """逐 query 算一个指标，返回 `{query_id: value}`。

    **以 qrels 的 query 集合为准，不是以 run 的**：系统一条结果都没返回的 query
    必须计 0 分进平均，从 run 侧遍历会让这些 query 凭空消失——**分母缩小，指标
    上升，而且没有任何报错**。这是评测代码最经典的一个自欺。
    """
    head, k = parse_metric(metric)
    func = METRIC_FUNCTIONS[head]
    return {qid: func(runs.get(qid, []), judgements, k) for qid, judgements in qrels.items()}


def macro_average(values: dict[str, float]) -> float:
    """按 query 平均：`(1/|Q|) · Σ metric(q)`。

    单列一个函数而不是内联 `sum/len`，是为了让「macro」这个口径在代码里有一个
    可被搜索到的名字——报告表头写 `macro-averaged over queries`，读代码的人要
    能一眼找到它对应哪一行。
    """
    return sum(values.values()) / len(values) if values else 0.0


class MetricReport(BaseModel):
    """一次跑分的完整结果。**per-query 值必须保留**。

    只留均值的报告是不可用的：bootstrap CI 需要重采样 per-query 数组，
    win/loss/tie 三元计数需要两个配置的 per-query 值逐条比对，而这两样是
    M7 §8.3 报告纪律里的第 1 条和第 4 条。**少了它们，这个消融就不可信。**
    """

    model_config = ConfigDict(frozen=True)

    config_name: str = Field(description="消融档位：bm25_only / dense_only / hybrid / hybrid_rerank")
    dataset_name: str
    aggregation: str = Field(default="macro", description="报告表头必须原样打出 'macro-averaged over queries'")
    per_query: dict[str, dict[str, float]] = Field(description="{指标名: {query_id: 值}}")

    def mean(self, metric: str) -> float:
        return macro_average(self.per_query[metric])

    def means(self) -> dict[str, float]:
        return {metric: macro_average(values) for metric, values in self.per_query.items()}

    def query_count(self, metric: str) -> int:
        """样本量。它必须与均值同时出现——一个没有 n 的均值无法判断可信度。"""
        return len(self.per_query[metric])

    def slice_by(self, query_ids: set[str]) -> MetricReport:
        """按 query 子集切片，用于**分粒度分别报**（M7 §1.3，禁止合并）。

        22 万条摘要会把 5 万篇全文的表现稀释掉，合并后的平均数谁都看不出问题。
        切片保留 `config_name` 与口径，只换样本集——**不重算**，因为逐 query 值
        与样本集无关，重算只会引入两条不同的代码路径。
        """
        return self.model_copy(
            update={
                "per_query": {
                    metric: {qid: v for qid, v in values.items() if qid in query_ids}
                    for metric, values in self.per_query.items()
                }
            }
        )
