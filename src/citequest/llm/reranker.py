"""bge-reranker-v2-m3 客户端（决策记录 D10 / §4.5）。**只做协议适配。**

三条写进接口的契约：

1. **对外统一 `normalize=True`**（sigmoid → [0,1]），内部才是 raw logit。
   统一口径的理由很具体：0.3 这个阈值按 sigmoid 反推 ≈ logit −0.85，
   等于把模型判为**倾向不相关**的 chunk 计为有效证据。
2. **sigmoid 只是单调映射，0.8 分不代表 80% 概率相关。** 要当概率用得按
   scikit-learn 的概率校准来（Platt 小样本 / isotonic 官方建议 >~1000 样本，
   且校准器必须在独立 held-out 上拟合），本项目标注量到不了，
   **所以只当有界可比的排序分数，不声称它是概率**。
3. **延迟机制明确、数字未实测**：cross-encoder 是 N 次前向、对 N 线性，
   ≈ `ceil(N/batch) × 单批前向时间`。公开数字口径不一且条件不明，**不引用**。

部署在应用侧而不是用 Milvus 内置 Model Ranker Function，理由是官方硬限制
「Model rankers cannot be used with grouping searches」——两个官方能力互斥时选
grouping，因为 chunk→paper 聚合是产品必需，rerank 放哪层是实现细节。
"""

from __future__ import annotations

from citequest.config.retrieval import RerankSettings
from citequest.observability.logging import get_logger

__all__ = ["CrossEncoderReranker", "RerankerClient"]

_logger = get_logger(__name__)


class RerankerClient:
    """cross-encoder 打分的最小接口。

    定义成一个类而不是 Protocol，是为了让降级路径（模型未加载时）有一个明确的
    实现载体——「reranker 不可用」在本系统里是一条**正常降级链**（退回 RRF 名次
    直出），不是异常。
    """

    def score(self, query: str, documents: list[str]) -> list[float]:
        """返回与 `documents` 等长的相关性分数，取值域 [0,1]（已 sigmoid）。"""
        raise NotImplementedError("请使用 CrossEncoderReranker 或在测试里注入桩实现")


class CrossEncoderReranker(RerankerClient):
    """本地 `BAAI/bge-reranker-v2-m3`（约 568M，FP16 约 1.1 GB）。

    模型在**首次打分时才加载**而不是 `__init__` 里：API 进程与评测脚本共用这个类，
    而评测脚本可能只是想跑 BM25 基线——启动就吃 1.1GB 显存会让「跑一条不需要
    rerank 的基线」变成必须有 GPU。
    """

    def __init__(self, settings: RerankSettings) -> None:
        self._settings = settings
        self._model: object | None = None

    def _ensure_model(self) -> object:
        """惰性加载。

        **接真实实现**：这里应当是
        ``FlagReranker(self._settings.model_id, use_fp16=self._settings.use_fp16)``
        （FlagEmbedding）或 sentence-transformers 的 `CrossEncoder`。
        两者的 `normalize` 语义不同——FlagEmbedding 的
        `compute_score(pairs, normalize=True)` 直接过 sigmoid，
        sentence-transformers 需要显式传 `activation_fn`，**换实现时必须重新确认
        分数是不是已经在 [0,1]**，否则阈值会静默作用在 raw logit 上。
        """
        if self._model is None:
            raise NotImplementedError(
                "接真实实现：加载 BAAI/bge-reranker-v2-m3（FlagReranker(use_fp16=True)）。"
                "骨架阶段由调用方注入桩实现或走 rerank 降级链"
            )
        return self._model

    def score(self, query: str, documents: list[str]) -> list[float]:
        """按 `batch_size` 分批前向打分。

        分批不是优化而是显存约束：48 pair 一次前向的峰值显存与 batch 大小成正比，
        `batch_size=32` 意味着 48 条走 2 批——这也是延迟预算里
        「fp16、batch 32 → 2 批前向」那一行的来源（决策记录 §4.7 第 5 行）。
        """
        if not documents:
            return []
        if not query.strip():
            raise ValueError("rerank 的 query 不能为空——空 query 会让 cross-encoder 打出无意义的排序")

        model = self._ensure_model()
        scores: list[float] = []
        batch = self._settings.batch_size
        for start in range(0, len(documents), batch):
            window = documents[start : start + batch]
            pairs = [(query, doc) for doc in window]
            # 真实调用形态：model.compute_score(pairs, normalize=self._settings.normalize)
            raw = getattr(model, "compute_score")(pairs, normalize=self._settings.normalize)  # noqa: B009
            scores.extend(float(s) for s in (raw if isinstance(raw, list) else [raw]))

        if len(scores) != len(documents):  # pragma: no cover - 防御：批次拼接错位
            raise RuntimeError(f"rerank 分数条数({len(scores)})与文档条数({len(documents)})不一致")
        _logger.debug("rerank_scored", n=len(documents), batches=(len(documents) + batch - 1) // batch)
        return scores
