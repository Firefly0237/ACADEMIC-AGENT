"""断点续跑 checkpoint（M1 §8.2）。

灌库是一个 6–10 小时量级的过程，中断是常态不是异常。这里的机制只有三条，
每一条都对应一个真实会发生的失败：

1. **原子写**（临时文件 + `os.replace`）。直接覆写的失败形态是「进程在写 checkpoint
   的瞬间被杀，文件只写了一半」，恢复时读到半个 JSON，于是从头再跑。
2. **JSONL 续写前先截断不完整的最后一行**。追加写被中断会留下半行，下次直接
   `append` 会把两个半行粘成一个语法上合法但内容错乱的记录。
3. **checkpoint 只记进度，不记内容**。幂等由 `chunk_id` 的确定性构造保证
   （M1 §8.2），所以 checkpoint 不需要精确到「上次写到哪一条」——重叠部分会被
   按主键 upsert 吸收掉。这条让 checkpoint 的写入频率可以放得很低。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from citequest.observability.logging import get_logger

__all__ = ["Checkpoint", "CheckpointStore", "truncate_incomplete_jsonl_tail"]

_logger = get_logger(__name__)


class Checkpoint(BaseModel):
    """一次灌库任务的进度快照。

    `stage` 与 `source` 一起构成任务标识：清洗、切块、灌库三个阶段各自有独立的
    checkpoint，因为它们的重跑代价差了一个数量级——切块可以重跑，全量 TEI 重编码
    不可以。
    """

    stage: str = Field(description="阶段名，如 parse / chunk / upsert")
    source: str = Field(description="数据源标识，如 unarxive / openalex-2024")
    processed_count: int = Field(default=0, ge=0)
    last_paper_id: str | None = Field(default=None, description="最后一条成功处理的 paper_id，用于人工核对")
    last_cursor: str | None = Field(
        default=None,
        description="源侧游标（OpenAlex 的 cursor / tar 成员名）。它是唯一能让**拉取**也断点续跑的字段——processed_count 只够让写入续跑",
    )
    failed_count: int = Field(default=0, ge=0, description="按篇隔离的失败数。失败计数不丢弃，它是数据质量指标")


class CheckpointStore:
    """按 `(stage, source)` 存取 checkpoint 的文件后端。

    刻意用文件而不是数据库：本项目已用「不新增有状态服务」这条判据否掉了外挂检索
    引擎与独立关系库，为一个进度文件引入服务就是双标。
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, stage: str, source: str) -> Path:
        safe_source = source.replace("/", "_").replace("\\", "_")
        return self._root / f"{stage}__{safe_source}.checkpoint.json"

    def load(self, stage: str, source: str) -> Checkpoint | None:
        """读取。文件损坏时返回 None 并告警，而不是抛异常。

        损坏就从头跑是安全的——幂等由 `chunk_id` 保证。但**必须告警**：静默从头跑
        的表现是「这个任务怎么每次都跑这么久」，那是一个很难被联想到 checkpoint 的现象。
        """
        path = self._path(stage, source)
        if not path.exists():
            return None
        try:
            return Checkpoint.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            _logger.warning("checkpoint_corrupted", path=str(path), error=str(exc))
            return None

    def save(self, checkpoint: Checkpoint) -> None:
        """原子写：先写同目录临时文件，再 `os.replace`。

        临时文件必须与目标**同目录**——跨文件系统的 replace 不是原子操作，
        而 `tempfile.gettempdir()` 在容器里经常与数据卷不在同一个挂载点上。
        """
        path = self._path(checkpoint.stage, checkpoint.source)
        payload = checkpoint.model_dump_json(indent=2)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                # fsync 之后再 replace：没有 fsync 时断电会得到一个「replace 成功
                # 但内容还在页缓存里」的空文件，比没有 checkpoint 更糟。
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise


def truncate_incomplete_jsonl_tail(path: Path) -> int:
    """截断 JSONL 文件末尾不完整的一行，返回截掉的字节数。

    续写产物（chunks.jsonl）前必须调用。判据是「最后一行能不能被 JSON 解析」而不是
    「结不结尾于换行符」：写入被中断在 `\\n` 之前和之后都可能发生，只看换行符会漏掉
    「写了完整一行但缺换行」这种其实无需处理的情况，也会漏掉「换行写进去了但前面的
    内容被截断」这种真正的坏数据。
    """
    if not path.exists() or path.stat().st_size == 0:
        return 0
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    # 末尾恰好以换行结束时，split 会产生一个空尾元素，那说明上一行是完整的。
    if lines and lines[-1] == b"":
        return 0
    tail = lines[-1]
    try:
        json.loads(tail.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        keep = len(raw) - len(tail)
        with path.open("r+b") as handle:
            handle.truncate(keep)
        _logger.warning("jsonl_tail_truncated", path=str(path), dropped_bytes=len(tail))
        return len(tail)
    return 0
