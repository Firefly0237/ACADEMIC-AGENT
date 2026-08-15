"""需要真实索引才跑得起来的评测入口。

`evaluation/` 下其余模块（`metrics` / `ablation` / `calibration` /
`generation_eval`）是**纯函数或可注入桩的编排**，不连外部服务也能被单测覆盖；
这个子包放的是另一类东西——**每一条都必须连一个真实的 Milvus 才有意义**：

- `curves.py` —— 四条必测曲线（组召回 / recall-QPS / 内存-召回 / 过滤选择率）

按「能不能在没有外部服务的单测里被断言」来分包，而不是按「是不是评测」来分：
前者决定了 CI 上哪些跑、哪些跳过，后者决定不了任何事。

跑分本身的入口不在这里，在包顶层：`retrieval_eval.evaluate_dataset`（L1）与
`generation_eval`（L2）。它们通过 `RunProducer` / `EntailmentJudge` 接受注入，
所以既能接真实管线，也能在单测里接桩——**跑分编排的正确性不该只能靠跑全量来验证**。
"""

from citequest.evaluation.runners.curves import CurvePoint, CurveResult, group_recall_curve, recall_qps_curve

__all__ = ["CurvePoint", "CurveResult", "group_recall_curve", "recall_qps_curve"]
