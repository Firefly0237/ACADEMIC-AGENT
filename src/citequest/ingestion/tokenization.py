"""token 计数（M1 §6.12）。

**这一层存在的唯一理由是一条硬纪律**：计 token 必须用与 dense 编码器一致的
分词器，不能用 `len(text.split())`，也不能用「字符数除以 4」。分词器不一致时，
统计出来的 450 和索引里实际编码的 450 不是同一个数——切块预算会系统性偏移，
而这件事不报错，只表现为「块比预期长/短」，等到发现时已经灌完 147 万 chunk。

所以本模块提供两个实现，并且**刻意让近似实现难用**：
- `HuggingFaceTokenCounter`：真实分词器，建库路径唯一允许用的实现；
- `WhitespaceTokenCounter`：近似，仅供单元测试构造确定性输入，构造时必须显式
  传 `acknowledge_approximation=True`。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from citequest.observability.logging import get_logger

__all__ = [
    "HuggingFaceTokenCounter",
    "TokenCounter",
    "WhitespaceTokenCounter",
    "build_token_counter",
]

_logger = get_logger(__name__)


@runtime_checkable
class TokenCounter(Protocol):
    """切块层依赖的最小接口。

    只要一个方法——切块算法不需要 token id，只需要长度。把接口收到这么窄，
    是为了让「换编码模型」只影响这一个实现类，而不是把 transformers 的类型
    渗透到 `chunking.py` 里去。
    """

    def count(self, text: str) -> int:
        """返回 `text` 的 token 数（不含特殊 token）。"""
        ...


class HuggingFaceTokenCounter:
    """基于 HF `AutoTokenizer` 的真实计数（建库路径唯一允许的实现）。

    分词器**延迟加载**：`transformers` 在 `[ingest]` / `[models]` 这两个可选依赖组
    里，API 容器不装它。模块导入期就加载会让 API 进程凭空多几百 MB 且启动变慢，
    而 API 进程根本不切块。
    """

    def __init__(self, model_id: str, *, add_special_tokens: bool = False) -> None:
        if not model_id.strip():
            raise ValueError("model_id 不能为空")
        self._model_id = model_id
        # 不算特殊 token：450 是**正文**预算。把 [CLS]/[SEP] 算进来会让每个块的
        # 有效内容少 2 个 token，147 万块累计就是一笔无意义的系统性损耗。
        self._add_special_tokens = add_special_tokens
        self._tokenizer: object | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _ensure_tokenizer(self) -> object:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:  # pragma: no cover - 依赖缺失路径
                raise RuntimeError(
                    f"计 token 需要 transformers（可选依赖组 [ingest]/[models]）：uv sync --extra ingest。原始错误：{exc}"
                ) from exc
            _logger.info("tokenizer_loading", model_id=self._model_id)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        return self._tokenizer

    def count(self, text: str) -> int:
        """真实编码一次取长度。

        没有做缓存：切块过程中每个段落只会被计一次长度（`chunking` 里的累加是加
        整数，不是重新计数），加缓存只会白白占内存。
        """
        if not text:
            return 0
        tokenizer = self._ensure_tokenizer()
        encode = getattr(tokenizer, "encode", None)
        if encode is None:  # pragma: no cover - 防御性
            raise RuntimeError(f"{self._model_id} 的 tokenizer 没有 encode 方法")
        token_ids: list[int] = encode(text, add_special_tokens=self._add_special_tokens)
        return len(token_ids)


class WhitespaceTokenCounter:
    """空白切分的近似计数。**禁止用于建库**。

    它的存在是为了让 `chunking.py` 的单元测试能给出确定性、可手算的输入长度，
    而不必在 CI 上下载 2 GB 的分词器。构造时必须显式承认它是近似值——这个参数
    看起来啰嗦，但它把「误用」从一次静默的数据事故变成了一行显眼的代码。
    """

    def __init__(self, *, acknowledge_approximation: bool) -> None:
        if not acknowledge_approximation:
            raise ValueError(
                "WhitespaceTokenCounter 是近似计数，与编码器分词结果不一致，"
                "只允许在测试里用；请显式传 acknowledge_approximation=True"
            )

    def count(self, text: str) -> int:
        return len(text.split())


def build_token_counter(model_id: str) -> TokenCounter:
    """按配置构造计数器。切块入口只认这个工厂，不直接 new 实现类。

    :param model_id: 取自 `CorpusSettings.tokenizer_model_id`，默认与 dense 编码器
        同为 `BAAI/bge-m3`——这正是本模块开头那条纪律的落地点。
    """
    return HuggingFaceTokenCounter(model_id)
