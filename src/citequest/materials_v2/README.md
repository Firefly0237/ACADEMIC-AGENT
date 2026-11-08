# 材料处理

本组件负责上传材料的文本处理、解析调度、分块和原文定位。

| 模块 | 职责 |
| --- | --- |
| `api.py`、`services.py` | 材料接口与领域操作 |
| `processor.py`、`worker.py` | 解析任务处理 |
| `adapter.py`、`isolation.py` | 解析器适配与进程隔离 |
| `text.py`、`provenance.py` | 文本表示及来源位置 |
| `chunking.py` | 语义拆分、token窗口与位置映射 |
| `references.py`、`reference_fetch.py` | 文献引用解析与来源获取 |
| `models.py` | 材料、解析表示和片段模型 |

处理结果供`material_evidence_v3`读取，用于检索与RAG引用。系统关系见[架构](../../../docs/architecture.md)。
