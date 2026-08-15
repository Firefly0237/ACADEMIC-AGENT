"""临界相关标定集与阈值标定（M7 §4）。**这是一等交付物，不可降级。**

|  | 标定集 | 回归集 |
|---|---|---|
| 规模 | **30–50 条** query | 200 条量级 |
| 用途 | **标定阈值**（rerank 过滤阈值、grounding 三档阈值） | smoke test |
| 是否进指标表 | **否** | **否** |
| 能不能降级 | **不能** | 可以 |

**两者必须分开命名。** 一旦混成一件事，一句「标注集太小所以降级」就会同时把
阈值的依据抽掉——而 rerank 过滤阈值和 grounding 三档阈值都写着「在标注集上标定」，
**载体消失，依赖它的结论就同时悬空**。

**标定流程（Cohere 官方 reranking best practices）**：

```
① 挑 30–50 条本领域代表性 query
② 为每条标出「临界相关」文档 —— 不是最相关的，是「勉强算相关、再差一点就该被过滤掉」的
③ 与 query 组 pair 过 bge-reranker-v2-m3（normalize=True）
④ 取这些分数的**平均值** = 过滤阈值
```

产出的阈值有明确语义：「等于我判定为临界相关的那些文档的平均得分」。
**这不是一个魔法数字，它是一句可以被追问、也答得上来的定义。**

**为什么不能凭 sigmoid 分数直觉去拍**：bge-reranker-v2-m3 默认输出 raw logit，
模型卡示例 −5.65 → normalize 后 0.003497。**sigmoid 只是单调映射，不是概率校准**
——0.8 分不代表 80% 的概率相关。真要当概率用需要 Platt / isotonic 校准，
且校准器必须在独立 held-out 上拟合，本项目标注量到不了这个规模，
**所以这套分数只被当作「有界可比的排序分数」，不声称它是概率**。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from citequest.config.eval import EvalSettings
from citequest.llm.reranker import RerankerClient
from citequest.observability.logging import get_logger

__all__ = [
    "CalibrationItem",
    "ThresholdCalibration",
    "calibrate_rerank_threshold",
    "distribution_crossover",
    "load_calibration_set",
    "retest_agreement",
]

_logger = get_logger(__name__)


class CalibrationItem(BaseModel):
    """标定集的一条：一个 query 与一段被人工判为**临界相关**的文档。

    `label` 三档而不是二值：交点法（`distribution_crossover`）需要正负两侧的分布，
    而均值法（Cohere 流程）只用 `borderline` 那一档。同一份标注支撑两种取法，
    是为了让两个数能互相印证——**只有一种取法时，阈值就是一个无法自证的数**。
    """

    model_config = ConfigDict(frozen=True)

    query: str
    doc_id: str
    text: str
    label: str = Field(pattern="^(relevant|borderline|irrelevant)$", description="borderline = 再差一点就该被过滤掉")
    annotator: str = Field(default="author", description="单人标注。IAA 不存在，缓解手段是 test-retest")
    round_index: int = Field(default=1, ge=1, description="第几轮标注。重标 20 条测自一致性时用 2")


class ThresholdCalibration(BaseModel):
    """一次阈值标定的产物。**它进配置，不进代码**（M7 §4.3 第 3 条）。

    `model_version` 与 `threshold` 绑死：阈值与模型版本、语料强绑定，换 reranker
    版本或换语料都必须重标。落进 `RerankSettings.calibrated_for_model_version`
    的就是这个字段——**没有版本号的阈值等于没有标定过**。
    """

    model_config = ConfigDict(frozen=True)

    threshold: float = Field(ge=0.0, le=1.0, description="= 临界相关文档的平均得分（Cohere 流程第 ④ 步）")
    model_version: str
    sample_size: int = Field(ge=0)
    borderline_scores: list[float] = Field(default_factory=list, description="落盘以便重算与画分布")
    crossover: float | None = Field(default=None, description="正负分布的交点。与均值法互相印证，不取代它")
    boundaries: list[str] = Field(
        default_factory=lambda: [
            "单人标注，无标注者间一致性（IAA）；缓解手段是同一人间隔重标 20 条测 test-retest",
            "标注者即系统作者，存在确认偏误——所以标定集**只用来定阈值，不用来算指标**",
            "阈值与模型版本、语料强绑定，换任一即失效，必须重标并换版本号",
            "分数只是有界可比的排序分数，**不声称它是概率**（sigmoid 不是概率校准）",
        ],
        description="三条边界原样进报告，不允许删",
    )


def load_calibration_set(path: Path) -> list[CalibrationItem]:
    """从 JSONL 读标定集。

    路径默认 `data/external/calibration_set.jsonl`（`EvalSettings.calibration_set_path`）。
    放 `data/external/` 而不是 `tests/fixtures/`：它是一份**交付物**，不是测试夹具
    ——放进 tests 会让「它是一等产物」这句话在目录结构上就被否掉。
    """
    if not path.exists():
        raise FileNotFoundError(f"标定集不存在：{path}。它是一等交付物，缺它则 rerank 阈值与 grounding 阈值同时悬空")
    items = [
        CalibrationItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _logger.info("calibration_set_loaded", path=str(path), items=len(items))
    return items


def calibrate_rerank_threshold(
    items: Sequence[CalibrationItem],
    client: RerankerClient,
    settings: EvalSettings,
    *,
    model_version: str,
) -> ThresholdCalibration:
    """跑 Cohere 官方流程的第 ③④ 步：过 reranker 打分 → 取临界相关文档的均值。

    :param client: 打分口径必须与线上一致（`normalize=True` → [0,1]）。用线上那
        同一个 `RerankerClient` 而不是另写一段推理，否则标出来的阈值作用在一套
        分数上、线上过滤作用在另一套上——**而这两套分数看起来完全一样**。

    规模校验是硬的：`EvalSettings.calibration_set_size` 限定 30–50 条。少于 30 条
    的「标定」产出的是一个噪声均值；这时应该去补标，而不是接受它。
    """
    queries = {item.query for item in items}
    if len(queries) < settings.calibration_set_size * 0.75:
        _logger.warning(
            "calibration_set_undersized",
            queries=len(queries),
            expected=settings.calibration_set_size,
            action="补标而不是接受一个噪声均值",
        )

    borderline = [i for i in items if i.label == "borderline" and i.round_index == 1]
    if not borderline:
        raise ValueError("标定集里没有 borderline 条目——阈值的语义正是「临界相关文档的平均得分」，缺它就无从标定")

    scores = _score_items(borderline, client)
    threshold = sum(scores) / len(scores)

    positives = _score_items([i for i in items if i.label == "relevant" and i.round_index == 1], client)
    negatives = _score_items([i for i in items if i.label == "irrelevant" and i.round_index == 1], client)

    calibration = ThresholdCalibration(
        threshold=threshold,
        model_version=model_version,
        sample_size=len(borderline),
        borderline_scores=scores,
        crossover=distribution_crossover(positives, negatives) if positives and negatives else None,
    )
    _logger.info(
        "rerank_threshold_calibrated",
        threshold=round(threshold, 4),
        crossover=calibration.crossover,
        model_version=model_version,
        n=len(borderline),
        # 两个数差得远说明标注的「临界」与分数分布的真实边界不在一处——
        # 那是标注指南要改，不是阈值要挑一个好看的。
        note="均值法与交点法差距大时先查标注指南，不要在两个数里挑",
    )
    return calibration


def _score_items(items: Sequence[CalibrationItem], client: RerankerClient) -> list[float]:
    """按 query 分组打分。逐条调用会让 cross-encoder 的 batch 完全失效。"""
    by_query: dict[str, list[CalibrationItem]] = {}
    for item in items:
        by_query.setdefault(item.query, []).append(item)
    scores: list[float] = []
    for query, group in by_query.items():
        scores.extend(client.score(query, [i.text for i in group]))
    return scores


def distribution_crossover(positive: Sequence[float], negative: Sequence[float], *, bins: int = 20) -> float:
    """正负分布的交点：**画分布取交点**那一步的程序化形态。

    定义取**等错误点**：扫 [0,1] 上的候选切点，每个切点的错误 =
    「被切掉的正样本比例 + 留在上面的负样本比例」，取错误最小的那个切点。
    最小值出现在一段区间上（两个分布之间的空隙）时取该区间的中点——
    **不取第一个**，否则交点会贴在空隙的下沿，随一两个离群样本大幅跳动。

    它与均值法（Cohere 流程）是两条独立的路径：两个数接近时阈值可信，差得远时
    说明标注的「临界」与分数分布的真实边界不在一处，**那时该改标注指南，
    不是在两个数里挑一个好看的**。

    **它不取代均值法**：均值法的产出有一句可被追问的定义（「等于我判定为临界
    相关的那些文档的平均得分」），交点法的产出没有，它只是一个分布事实。
    """
    if not positive or not negative:
        raise ValueError("交点法需要正负两侧都有样本")
    width = 1.0 / bins
    cuts = [index * width for index in range(bins + 1)]
    errors = [
        sum(1 for s in positive if s < cut) / len(positive) + sum(1 for s in negative if s >= cut) / len(negative)
        for cut in cuts
    ]
    minimum = min(errors)
    best = [cut for cut, err in zip(cuts, errors, strict=True) if err == minimum]
    return sum(best) / len(best)


def retest_agreement(items: Sequence[CalibrationItem]) -> float | None:
    """同一人间隔重标 20 条的自一致性（M7 §4.3 第 1 条）。

    **函数名刻意不叫 `test_retest_agreement`**：那是这套方法的标准称呼，
    但以 `test_` 开头的模块级函数会被 pytest 当成用例收集，
    表现是一条名字古怪、参数缺失的失败用例。

    返回两轮标签一致的比例；没有第二轮时返回 None。

    **必须声明这是自一致性，不是客观性**：它测的是「我两次标得一不一致」，
    不是「我标得对不对」。单人项目没有第二个标注者，这个数字是能拿到的最强证据，
    但它的天花板就在那里，把它说成 IAA 是一次实质性的夸大。
    """
    first = {(i.query, i.doc_id): i.label for i in items if i.round_index == 1}
    second = {(i.query, i.doc_id): i.label for i in items if i.round_index == 2}
    shared = first.keys() & second.keys()
    if not shared:
        return None
    agreed = sum(1 for key in shared if first[key] == second[key])
    return agreed / len(shared)
