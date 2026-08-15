"""全文层获取：从 Zenodo 拉 unarXive open subset（M1 §3.2）。

源是一个 **4,844,918,096 字节**的 tar.xz（Zenodo API 实测），DOI
`10.5281/zenodo.7752615`。无审批、无账号、零付费、单机可解——这是它通过「可实现性」
这道关的原因（决策记录 §2.3）。

三件事必须做对，都是「不做就会在第 5 小时崩掉」的那类：

1. **断点续传**（HTTP Range）。4.84 GB 在家用网络上是几十分钟到几小时，一次
   连接中断就要重来是不可接受的。
2. **校验**。Zenodo 的 record 元数据给出 size 与 md5，下完必须比对——截断的
   tar.xz 在解压到第 N 个成员时才报错，而那时已经过去很久了。
3. **不重复下载**。已存在且校验通过就直接返回，让「重跑建库脚本」是安全的。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from citequest.observability.logging import get_logger

__all__ = ["ZenodoFile", "download_unarxive", "fetch_record_files", "verify_file"]

_logger = get_logger(__name__)

#: Zenodo 的公开 REST API。DOI 尾号即 record id。
_ZENODO_RECORD_URL = "https://zenodo.org/api/records/{record_id}"
#: 决策记录 §4.1：这个尺寸是实测值，用它做「下完了没有」的第一道判据。
EXPECTED_ARCHIVE_BYTES = 4_844_918_096
#: 8 MB 一块。太小会让 4.84 GB 产生几十万次 write 调用，太大在断线时浪费的重传更多。
_CHUNK_BYTES = 8 * 1024 * 1024


class ZenodoFile(BaseModel):
    """Zenodo record 里的一个文件条目。"""

    key: str = Field(description="文件名")
    size: int = Field(ge=0)
    checksum: str = Field(description="形如 `md5:xxxx`，Zenodo 的原始格式")
    link: str = Field(description="下载直链")

    @property
    def md5(self) -> str | None:
        prefix = "md5:"
        return self.checksum[len(prefix) :] if self.checksum.startswith(prefix) else None


def fetch_record_files(record_id: str, *, timeout_s: float = 30.0) -> list[ZenodoFile]:
    """查 record 元数据，拿到文件列表与校验和。

    **不把下载直链写死在代码里**：Zenodo 的直链带版本路径，硬编码一次就意味着
    数据集出新版时代码会静默下到旧版。DOI 才是稳定标识（决策记录 §2.1）。
    """
    url = _ZENODO_RECORD_URL.format(record_id=record_id)
    response = httpx.get(url, timeout=timeout_s, follow_redirects=True)
    response.raise_for_status()
    payload = response.json()
    files = payload.get("files", [])
    return [
        ZenodoFile(
            key=item["key"],
            size=int(item.get("size", 0)),
            checksum=str(item.get("checksum", "")),
            link=item["links"]["self"],
        )
        for item in files
    ]


def verify_file(path: Path, *, expected_size: int | None = None, expected_md5: str | None = None) -> bool:
    """尺寸 + md5 双重校验。

    先比尺寸再算 md5：尺寸比对是 O(1) 的，而 4.84 GB 的 md5 要读一遍整个文件。
    绝大多数失败（中断、磁盘满）在尺寸这一步就能判掉。
    """
    if not path.exists():
        return False
    actual_size = path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        _logger.warning("archive_size_mismatch", path=str(path), actual=actual_size, expected=expected_size)
        return False
    if expected_md5 is None:
        return True

    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK_BYTES):
            digest.update(block)
    ok = digest.hexdigest() == expected_md5
    if not ok:
        _logger.error("archive_md5_mismatch", path=str(path), actual=digest.hexdigest(), expected=expected_md5)
    return ok


def download_unarxive(
    target_dir: Path,
    *,
    record_id: str = "7752615",
    file_key: str | None = None,
    timeout_s: float = 60.0,
) -> Path:
    """下载（或复用）unarXive open subset 的归档文件，返回本地路径。

    断点续传的实现是标准的 HTTP `Range: bytes=N-`：本地已有 N 字节就从 N 续传，
    服务端返回 206 才追加；返回 200 说明它不支持 Range，此时必须**从头覆写**而不是
    追加——把一个完整响应追加到半个文件后面，得到的是一个尺寸看起来更大、内容
    完全错乱的归档，而 tar.xz 要解压到中途才会报错。

    :param file_key: 指定文件名；缺省取 record 里最大的那个文件（open subset 是单文件）。
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    files = fetch_record_files(record_id, timeout_s=timeout_s)
    if not files:
        raise RuntimeError(f"Zenodo record {record_id} 没有可下载文件")

    entry = next((f for f in files if f.key == file_key), None) if file_key else max(files, key=lambda f: f.size)
    if entry is None:
        raise RuntimeError(f"Zenodo record {record_id} 中没有文件 {file_key}")

    dest = target_dir / entry.key
    if verify_file(dest, expected_size=entry.size, expected_md5=entry.md5):
        _logger.info("archive_already_present", path=str(dest), size=entry.size)
        return dest

    offset = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    _logger.info("archive_download_start", url=entry.link, offset=offset, total=entry.size)

    with httpx.stream("GET", entry.link, headers=headers, timeout=timeout_s, follow_redirects=True) as response:
        response.raise_for_status()
        resumed = response.status_code == 206
        mode = "ab" if resumed and offset else "wb"
        if offset and not resumed:
            _logger.warning("range_not_supported_restarting", url=entry.link)
        with dest.open(mode) as handle:
            for block in response.iter_bytes(_CHUNK_BYTES):
                handle.write(block)

    if not verify_file(dest, expected_size=entry.size, expected_md5=entry.md5):
        raise RuntimeError(f"{dest} 校验失败：尺寸或 md5 与 Zenodo 元数据不一致，请删除后重下")
    _logger.info("archive_download_done", path=str(dest), size=entry.size)
    return dest
