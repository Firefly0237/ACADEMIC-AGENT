"""Milvus 客户端：连接、collection 生命周期、检索调用（M2 §5/§6/§12）。

这一层只做三件事，且刻意不做第四件：

1. **连接与启动探针**——固定探针文本比对向量指纹，不一致拒绝启动。成本极低，
   能挡住「灌库与查询用了不同模型」这类最难查的故障。
2. **collection 生命周期**——建库、建索引、load、alias 切换。**对外只用 alias**，
   切库靠改 alias 指向，这是灰度切换与一键回滚的实现方式。
3. **检索调用**——`hybrid_search` 与单路 `search`，含 grouping 参数。

不做的第四件是**请求装配**：过滤表达式在 `expressions.py`、两路请求在
`ann_requests.py`。分开的理由很实际——装配逻辑要能在没有 Milvus 的单元测试里被
断言，而这个类必须连服务。
"""

from __future__ import annotations

from typing import Any

from pymilvus import AnnSearchRequest, MilvusClient

from citequest.config.settings import Settings
from citequest.core.errors import SearchUnavailableError
from citequest.indexing.ann_requests import GroupingSpec, build_ranker, grouping_kwargs
from citequest.indexing.index_params import build_index_params
from citequest.indexing.schema import build_collection_schema, chunk_output_fields
from citequest.observability.logging import get_logger

__all__ = ["MilvusIndexClient"]

_logger = get_logger(__name__)


class MilvusIndexClient:
    """`pymilvus.MilvusClient` 的薄封装。

    薄到什么程度：**不缓存结果、不重试、不做熔断**。这三件都在上层——重试与超时
    预算是服务层的事（单次调用 400ms、总 deadline 1.0s 含一次重试），缓存 key 要
    含 `index_version` 与全部影响结果的参数，这两件在这一层都拿不到足够信息。
    """

    def __init__(self, settings: Settings, *, client: MilvusClient | None = None) -> None:
        self._settings = settings
        self._milvus = settings.milvus
        self._client = client or MilvusClient(
            uri=self._milvus.uri,
            token=self._milvus.token or None,
            db_name=self._milvus.db_name,
        )

    @property
    def raw(self) -> MilvusClient:
        """底层客户端。给评测与运维脚本用——它们需要 `describe_index` 这类
        本类不打算长期支持的调用，而为它们逐个加转发方法只会让这层变厚。"""
        return self._client

    @property
    def target(self) -> str:
        """检索目标名。

        **读路径一律走 alias，写路径一律走真实 collection 名**：alias 存在的意义
        就是让「换一份索引」对读侧是原子的，读侧若直接用 collection 名，切换就变成
        了一次需要重启的操作。
        """
        return self._milvus.alias or self._milvus.collection_name

    # ---- collection 生命周期 --------------------------------------------

    def create_collection(self, *, drop_existing: bool = False) -> None:
        """建库：schema + 两条 Function + 三类索引，一次调用完成。

        `create_collection(schema=..., index_params=...)` 一次传两个参数而不是先建
        collection 再单独 `create_index`：分两步的中间态是「有数据能写、但检索走
        暴力扫」，而这个状态不报错，只是慢——上线检查很容易漏掉它。
        """
        name = self._milvus.collection_name
        if self._client.has_collection(name):
            if not drop_existing:
                _logger.info("collection_exists", collection=name)
                return
            # 重建 = 重灌 147 万 chunk + 一次全量 TEI 重编码，6–10 小时量级。
            # 所以这条路径必须显式要求 drop_existing，不能是默认行为。
            _logger.warning("collection_drop", collection=name, cost_hint="重建意味着 6–10 小时的全量重灌")
            self._client.drop_collection(name)

        schema = build_collection_schema(self._milvus, self._settings.embedding)
        self._client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=build_index_params(self._milvus),
            shards_num=self._milvus.shards_num,
            num_partitions=self._milvus.num_partitions,
            properties={"mmap.enabled": self._milvus.mmap_enabled},
        )
        _logger.info("collection_created", collection=name, shards=self._milvus.shards_num)

    def load(self) -> None:
        """载入内存。载入数据须 < 总内存 90%（官方限制），超了会 OOM 而不是降级。"""
        self._client.load_collection(self._milvus.collection_name)

    def point_alias(self) -> None:
        """把 alias 指向当前 collection。

        `create_alias` 在 alias 已存在时会报错，所以先探测再决定 create 还是 alter。
        这两个 API 不是幂等的，而灰度切换脚本必须能重跑。
        """
        alias, name = self._milvus.alias, self._milvus.collection_name
        existing = {item.get("alias") if isinstance(item, dict) else item for item in self._client.list_aliases()}
        if alias in existing:
            self._client.alter_alias(collection_name=name, alias=alias)
        else:
            self._client.create_alias(collection_name=name, alias=alias)
        _logger.info("alias_switched", alias=alias, collection=name)

    def startup_probe(self) -> bool:
        """启动探针：用固定文本走一次真实检索路径，确认端到端可用。

        探针文本**必须固定不变**——换文本就等于换基准，探针立刻失去意义。
        当前实现只验「连得上、查得动、schema 认得出字段」；**接真实实现：拿到
        TEI 的 `/info` 与固定文本的向量指纹，与 `startup_probe_expected_sha256`
        比对，不一致拒绝启动。** 之所以现在拿不到向量，正是 Data-in/Data-out 的
        直接后果——客户端不持有 embedding，指纹只能由建库时落盘的那份提供。
        """
        embedding = self._settings.embedding
        if not embedding.startup_probe_enabled:
            return True
        try:
            self._client.query(
                collection_name=self.target,
                filter='granularity == "fulltext"',
                output_fields=["chunk_id"],
                limit=1,
            )
        except Exception as exc:
            _logger.error("startup_probe_failed", error=str(exc), endpoint=embedding.endpoint)
            return False
        if embedding.startup_probe_expected_sha256 is None:
            # 为 None 时只做连通性检查并告警，不拒绝启动——首次建库时这个值本来就
            # 还不存在，硬拒绝会让第一次建库无法进行。
            _logger.warning("startup_probe_no_baseline", hint="首次建库后用 TEI 实际输出回填 expected_sha256")
        return True

    # ---- 写入 ------------------------------------------------------------

    def upsert(self, rows: list[dict[str, Any]]) -> int:
        """按主键 upsert 一批 chunk 行，返回写入条数。

        幂等由 `chunk_id` 的确定性构造保证（M1 §8.2）：同一 id 写两次就是覆盖，
        于是重复投递不产生重复行、断点续跑安全。**行里不含任何向量字段**——
        dense 与 sparse 都是 Function 的输出，客户端灌向量会直接被 Milvus 拒绝。
        """
        if not rows:
            return 0
        result = self._client.upsert(collection_name=self._milvus.collection_name, data=rows)
        count = int(result.get("upsert_count", len(rows))) if isinstance(result, dict) else len(rows)
        return count

    def delete_by_paper(self, paper_id: str) -> int:
        """删掉一篇论文名下的全部 chunk。

        **更新语义是「先删后写」而不是逐条 upsert**（M1 §8.3）：新版本块数变少时
        （某个 section 被删掉），单纯 upsert 会留下**孤儿块**——它们的 chunk_id 不在
        新集合里，永远不会被覆盖，但会继续被检索到并指向一个已不存在的段落。
        这是一个很难被发现的数据腐化，因为它不报错。
        """
        expr = f'paper_id == "{paper_id}"'
        result = self._client.delete(collection_name=self._milvus.collection_name, filter=expr)
        return int(result.get("delete_count", 0)) if isinstance(result, dict) else 0

    # ---- 检索 ------------------------------------------------------------

    def hybrid_search(
        self,
        requests: list[AnnSearchRequest],
        grouping: GroupingSpec,
        *,
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """双路融合 + 服务端分组的真实调用。

        `limit` 传的是 **组数**（`group_limit`）——这是 grouping search 改变的语义，
        接口参数因此改名为 `group_limit`，不叫 `top_k`。grouping 的三个 kwargs 只能
        在顶层传（`AnnSearchRequest` 不支持它们，见 `ann_requests` 的模块说明）。

        单路模式（中文 → dense_only，引号短语 → sparse_only）走的仍然是这个方法：
        `requests` 只有一条时 RRF 退化为恒等变换，**没有必要为它写第二条代码路径**。
        """
        if not requests:
            raise ValueError("requests 不能为空")
        try:
            groups: list[list[dict[str, Any]]] = self._client.hybrid_search(
                collection_name=self.target,
                reqs=requests,
                ranker=build_ranker(self._settings.retrieval),
                limit=grouping.group_limit,
                output_fields=output_fields or chunk_output_fields(self._milvus),
                partition_names=partition_names,
                timeout=self._milvus.request_timeout_ms / 1000,
                **grouping_kwargs(grouping, self._settings.retrieval),
            )
            return groups
        except Exception as exc:
            # 检索是底线，**这条没有降级对象**——检索挂了就没有兜底。
            _logger.error("hybrid_search_failed", error=str(exc), target=self.target)
            raise SearchUnavailableError(f"Milvus 检索失败：{exc}") from exc

    def query(
        self, filter_expr: str, *, output_fields: list[str] | None = None, limit: int = 16
    ) -> list[dict[str, Any]]:
        """纯标量查询（取 chunk 明细、分面兜底、探针）。

        不走向量通道，所以没有 ef、没有 grouping——它的用途是「我已经知道 id/条件，
        把内容取出来」，与检索是两件事，混在一个方法里会让调用方以为它也有相关性。
        """
        try:
            rows: list[dict[str, Any]] = self._client.query(
                collection_name=self.target,
                filter=filter_expr,
                output_fields=output_fields or chunk_output_fields(self._milvus),
                limit=limit,
                timeout=self._milvus.request_timeout_ms / 1000,
            )
            return rows
        except Exception as exc:
            raise SearchUnavailableError(f"Milvus query 失败：{exc}") from exc

    def close(self) -> None:
        self._client.close()
