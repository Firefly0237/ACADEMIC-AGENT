# Changelog

本文件记录本项目所有值得注意的变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.1.0] - 2026-08-13

首个版本。

### Added

- **两层语料构建管线**：unarXive open subset 全文层（`cs.*`，2019–2022，CC-BY-SA-4.0）与
  OpenAlex 摘要层（Computer Science，2023–2025，CC0），合计约 27 万篇 / 约 147 万 chunk。
  全文层按 section 边界硬停聚合段落到 450 token，占位符在进索引前删除。
- **Milvus 2.6 索引层**：一个 collection 上声明 dense 与 sparse 两个 Function——
  dense 走 TEI Embedding Function 的 Data-in/Data-out（BGE-M3，1024 维，HNSW `M=32` / `efConstruction=256`），
  sparse 走内置 BM25 Function（`k1=1.2`，`b=0.75`，`SPARSE_INVERTED_INDEX`）。
  另有 `text_exact` 字段支持 `TEXT_MATCH` 与 `PHRASE_MATCH(slop)`。查询期 `ef` 按 entity 预算派生。
- **混合检索**：唯一融合器 `RRFRanker(k=60)`；中文 query 走 dense 单路、带引号短语走 sparse 单路，
  两者都不融合。Milvus 原生 grouping search 按 `paper_id` 聚合，搜索路 60 组 × 1，RAG 路 16 组 × 3。
- **应用侧精排**：`bge-reranker-v2-m3`，48 → 8，对外统一 `normalize=True`。
- **查询理解与路由**：两级判定（规则快筛 + 小 LLM 兜底），四个分支——意图、粒度、语言、引号短语；
  改写只保留 HyDE（N=1）与 decompose（一次拆分不迭代）。
- **RAG 生成与引用**：引用锚点为 `(paper_id, section, sec_number, paragraph_idx)` 四元组，
  摘要层不伪造段落级定位；grounding 沿已声明的引用边逐边判定，在线只在首句跑一次 HHEM-2.1-Open。
- **流式契约**：首句阻塞闸门 → 过闸才发 `ai_answer_start` → 之后按句边界推送 → 只降级不撤回。
  整条 SSE 流恒为 HTTP 200，失败以事件下发。
- **LangGraph 检索纠正循环**：`recursion_limit=8`，循环上限 1 次，超时退回首轮结果不报错。
- **HTTP API**：`/v1/search`、`/v1/answer`（SSE）、`/v1/papers/{paper_id}`、`/v1/facets`、
  `/v1/feedback`、`/healthz`、`/readyz`。逐条下发 `license` / `source_dataset` / `attribution`。
- **MCP server**：官方 Python SDK，stdio 传输，5 个只读幂等工具
  （`search_papers` / `search_chunks` / `get_paper_detail` / `structured_filter` / `expand_by_citation`），
  与 HTTP 共用同一份 service 层。
- **工程护栏**：分类不共享的 GPU 信号量与有界队列、逐层传递的 deadline、
  统一的超时语义（已产生内容一律保留）、按 `index_version` 的一键缓存失效、撤稿论文精确失效。
- **可观测性**：structlog 结构化日志、`request_id` 全链路注入、span 与成本记录，
  以及三个静默失败监控项（组数达成率、降级占比、`dense_only` 占比）。
- **评测模块**：unarXive_citrec 库内子集（Recall@100 / MRR@10）与 LitSearch（nDCG@10 / Recall@k）
  分层报告；四档消融、bootstrap 置信区间、配对 t 与 Fisher randomization 双跑、Tukey HSD 校正、
  per-query win/loss/tie 三元计数。
- **配置正本**：pydantic-settings 落地全部设计参数，五条启动时跨组校验，`citequest show-config` 可核对生效值。
- **文档**：README、[架构详解](docs/architecture.md)、[配置说明](docs/configuration.md)、
  [开发指南](docs/development.md)、贡献指南与 `CITATION.cff`。

### Notes

- 本仓库不分发任何语料原文、处理后的 chunk 或索引快照，只提供构建脚本。
  代码许可为 Apache-2.0，与两层语料各自的许可分开。
- README 与文档中出现的所有延迟均为**设计预算，非实测**；`latency_budget.measured` 为 `False`
  时禁止在对外材料中引用具体毫秒数。rerank 与 grounding 阈值默认未标定（`None`），
  未标定时引用支持强度一律为 `unverified`。

[Unreleased]: https://github.com/citequest/citequest/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/citequest/citequest/releases/tag/v0.1.0
