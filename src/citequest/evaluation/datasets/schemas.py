"""评测数据集的统一契约（M7 §2、§3、§6）。

**一份数据集不是「query + qrels」两个字典就够了**：M7 §1.1 的原则要求每个对外
报出的数字都能回答「样本是什么 / 谁给的标签 / 怎么聚合 / 和谁比」。前两问的答案
必须跟着数据集本身走，否则报告里就只剩一个孤零零的分数。所以 `EvalDataset` 上
挂了 `provenance`——它会原样进报告头，不是装饰。

**qrels 一律二值**（M7 §6.2）：citrec 的 label 是唯一一个被引论文的 OpenAlex ID，
LitSearch 的 639 条判定也是二值的。**本项目的主口径里不存在二值化阈值**，
`RELEVANT` 常量写在这里就是为了让「rel 分几级、阈值定在哪」这个追问有一个
指得出来的答案。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from citequest.core.enums import Granularity

__all__ = [
    "RELEVANT",
    "EvalDataset",
    "EvalQuery",
    "Provenance",
    "SubsetStats",
]

#: 二值 qrels 里「相关」的取值。gain = 2^rel − 1 = 1，DCG 因此退化为纯位置折损
#: （M7 §6.2）。它是常量而不是配置项——配置项意味着它可调，而它不可调。
RELEVANT = 1


class Provenance(BaseModel):
    """数据集的口径来源。**它进报告头，不进日志**。

    `label_source` 是这套评测最关键的一句话：citrec 写的是「论文原作者的真实引用
    行为」，自建标注集只能写「系统作者本人」。把这一栏留空的数据集不允许进指标表。
    """

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(description="HuggingFace 数据集 ID，报告里要能被别人原样复跑")
    split: str
    license: str
    label_source: str = Field(description="谁给的标签。这一栏是「标注池与被评系统是否解耦」的唯一凭据")
    citation: str | None = Field(default=None, description="论文或 DOI，让外人可核")
    notes: list[str] = Field(default_factory=list, description="必须与数字同时讲的局限，逐条进报告")


class EvalQuery(BaseModel):
    """一条评测 query。

    `layer` 不是可选的：两个基准各测一层语料，**fulltext 与 abstract 的指标必须
    分开报、禁止合并**（M7 §1.3）。把粒度挂在 query 上而不是靠调用方记住，
    是让「分粒度报」这件事在数据结构上就无法绕过。
    """

    model_config = ConfigDict(frozen=True)

    query_id: str
    text: str = Field(description="已做过防泄漏处理的检索输入。citrec 这里是删掉 marker 邻域后的段落")
    layer: Granularity = Field(description="这条 query 测的是哪一层语料")


class SubsetStats(BaseModel):
    """库内过滤的统计。**过滤率是必报数字，不是中间步骤**（M7 §2.3）。

    把它报出来，评测就从「我挑了个对我有利的子集」变成一条可追溯的证据链：
    库覆盖率是多少 → 因此可评样本是多少 → 因此这个指标代表什么。藏起来才是问题。
    """

    model_config = ConfigDict(frozen=True)

    total_records: int = Field(ge=0, description="过滤前的原始条数")
    kept_records: int = Field(ge=0, description="label ∈ 本库 OpenAlex ID 集合的条数")
    dropped_no_label: int = Field(default=0, ge=0, description="label 缺失或格式非法")
    dropped_out_of_corpus: int = Field(default=0, ge=0, description="label 不在本库内")
    corpus_size: int = Field(default=0, ge=0, description="用来做过滤的本库 paper_id 集合大小")

    @property
    def filter_rate(self) -> float:
        """库内引用边覆盖率 = kept / total。此前在另一个语料切片上实测 25.3%【实测】，
        换到「27 万篇库 vs 22.5 万条 test」之后必须重测【待实测】。"""
        return self.kept_records / self.total_records if self.total_records else 0.0


class EvalDataset(BaseModel):
    """一份可跑的评测集：query 列表 + 二值 qrels + 口径来源。

    `qrels` 的形态是 `{query_id: {doc_id: 1}}`——与 ranx 的 `Qrels.from_dict`
    直接对齐，不在中间再造一层结构。doc_id 用 **paper_id**（论文级）而不是
    chunk_id：两个基准的标签都是论文级的，而本系统靠 grouping 在服务端完成
    chunk→paper 聚合，评测单位与产品单位因此天然一致。
    """

    name: str
    queries: list[EvalQuery]
    qrels: dict[str, dict[str, int]]
    provenance: Provenance
    subset_stats: SubsetStats | None = Field(
        default=None, description="库内过滤的统计。只有 citrec 有；LitSearch 用自己的 corpus，不做库内过滤"
    )

    _layers: dict[str, Granularity] = PrivateAttr(default_factory=dict)

    def model_post_init(self, _context: object, /) -> None:
        # 建一次索引而不是每次 layer_of 都遍历 queries：报告要按粒度切片，
        # 22 万条 query × 每条一次线性扫描是一个纯粹白付的平方级开销。
        self._layers = {q.query_id: q.layer for q in self.queries}

    def layer_of(self, query_id: str) -> Granularity:
        """这条 query 属于哪一层语料。分粒度报表按它切片。"""
        return self._layers[query_id]

    def relevant_count(self, query_id: str) -> int:
        """该 query 的**已判定**相关文档数，即 Recall 与 MAP 的分母。

        「已判定」三个字是关键（M7 §6.3）：未进入判定集的文档一律按不相关处理。
        公开基准下这是数据集作者的选择，本项目只需声明沿用。
        """
        return sum(1 for rel in self.qrels.get(query_id, {}).values() if rel >= RELEVANT)
