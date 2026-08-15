"""dense 编码服务参数（决策记录 §4.3）。

**铁律**：本项目没有任何自写推理服务。dense 编码走 TEI 官方镜像 + Milvus 内置
TEI Embedding Function 的 Data-in/Data-out——写入只传原文、检索只传 query 原文，
Milvus 在写入期和检索期分别去调 TEI。

这条路线的连锁后果有两个，都是好的：客户端拿不到 query embedding，于是
M3 的 Level-2 embedding 锚点路由失去了「反正本来就要算」的免费前提（O15），
M6 的 L3 embedding 缓存也物理上不存在（O6）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["EmbeddingSettings"]


class EmbeddingSettings(BaseModel):
    """TEI 服务与 Milvus TEI Function 的参数。"""

    provider: str = Field(default="TEI", description="Milvus Embedding Function 的 provider 名，官方 provider 列表内")
    endpoint: str = Field(
        default="http://tei:80", description="TEI 服务地址。Milvus 容器要能解析它，所以用 compose 服务名而非 localhost"
    )
    model_id: str = Field(
        default="BAAI/bge-m3",
        description=(
            "**部分核实**：TEI 官方 supported_models 表未逐条列出 bge-m3，"
            "支持是按架构族给的（表中原文含 'XLM-RoBERTa models with absolute positions'，BGE-M3 正是 XLM-RoBERTa 底座）"
        ),
    )
    fallback_model_id: str = Field(
        default="intfloat/multilingual-e5-large-instruct",
        description=(
            "替换项：表内逐条列名、MTEB 排名 7、560M、XLM-RoBERTa、dim 同为 1024，**schema 一行不改**。"
            "e5 要求的 'passage: '/'query: ' 前缀由下面两个 prompt 字段承接（U5）"
        ),
    )
    dim: int = Field(default=1024, description="必须与 MilvusSettings.dense_dim 一致，切换 fallback 也不变")
    truncate: bool = Field(default=True, description="**官方默认 false，必须显式开**——否则超长文本直接报错而不是截断")
    truncation_direction: str = Field(default="right", pattern="^(left|right)$", description="官方默认 right，显式写出")
    max_client_batch_size: int = Field(default=32, ge=1, description="官方默认 32。这是写入反压唯一留给客户端的旋钮")
    ingestion_prompt: str | None = Field(default=None, description="切到 e5 时填 'passage: '")
    search_prompt: str | None = Field(default=None, description="切到 e5 时填 'query: '")
    api_key: str = Field(
        default="", description="TEI 带 --api-key 时才需要；否则 milvus.yaml 无需任何改动（官方 provider 默认已开）"
    )

    # ---- 启动探针（K12）----
    startup_probe_enabled: bool = Field(
        default=True,
        description="固定探针文本比对向量 hash，不一致拒绝启动。成本极低，能挡住「灌库与查询用了不同模型」这类最难查的故障",
    )
    startup_probe_text: str = Field(
        default="citequest startup probe: transformer positional encoding",
        description="探针文本必须固定不变——换文本就等于换基准，探针立刻失去意义",
    )
    startup_probe_expected_sha256: str | None = Field(
        default=None,
        description="首次建库时用 TEI 实际输出填入。为 None 时探针只做连通性检查并告警，不拒绝启动",
    )

    # ---- 离线专用（禁止进在线 QPS 路径）----
    offline_bgem3_enabled: bool = Field(
        default=False,
        description=(
            "pymilvus[model] 的 BGEM3EmbeddingFunction **仅离线**用途："
            "learned sparse 的离线 A/B、评测脚本。进程内模型封装，无 dynamic batching、"
            "不可独立扩缩容、与调用方共享 GPU 与 GIL，**禁止进在线路径**"
        ),
    )
