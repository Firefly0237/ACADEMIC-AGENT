# 配置说明

参数正本在 `src/citequest/config/`：**每个设计数值在代码里有且只有一份默认值**，
本文是它的可读版本。环境变量只用来覆盖部署差异（地址、密钥、开关），**不用来改设计参数**——
要改设计参数，改 `config/` 里的 `default` 并同步设计文档，`tests/unit/test_settings_defaults.py` 拦的就是这件事。

## 怎么配

环境变量形态是 pydantic-settings 的官方写法：前缀 `CITEQUEST_`，嵌套用双下划线 `__`。

```bash
CITEQUEST_MILVUS__URI=http://milvus:19530
CITEQUEST_RETRIEVAL__RRF_K=60
```

优先级：**CLI > 构造参数 > 环境变量 > `.env` > 默认值**。仓库里只提交 [`.env.example`](../.env.example)，真实 `.env` 已进 `.gitignore`。

核对当前实际生效的全部参数：

```bash
uv run citequest show-config
```

这条命令存在的理由很具体：参数正本在代码里，但环境变量可以覆盖。
出现「线上表现和设计文档对不上」时，第一件事是确认到底哪几项被环境变量改掉了——靠读代码查不出来。

**「依据」列的四档**：`官方`（有官方文档明文）/ `论文`（有论文明文）/ `自选`（工程取值，必须给理由）/ `估算`（推算值，未实测）。

## ⚠ 建库前不可逆的四项

这四项写进 collection 就固化。改任何一项 = 重建 collection + 重灌全量 chunk + 一次全量 TEI 重编码，
**6–10 小时量级**。在这个价格下，「先不建、评测说需要再说」不是省事而是赌博。

| 参数 | 默认值 | 位置 |
|---|---|---|
| `milvus.text_en_analyzer_params` | `{"type": "english"}` | `config/milvus.py` |
| `milvus.bm25_k1` / `milvus.bm25_b` | `1.2` / `0.75` | `config/milvus.py` |
| `milvus.text_exact_field` 是否存在 | 存在，`enable_match=True` | `config/milvus.py` |
| `corpus.strip_placeholders` | `True` | `config/corpus.py` |

## 全局

| 参数 | 环境变量 | 默认值 | 依据 |
|---|---|---|---|
| `env` | `CITEQUEST_ENV` | `local` | 自选。**仅用于日志与 span 的标签，不用来切换任何行为分支**——`if env == "prod"` 会让「线上跑的到底是哪套参数」变成需要读代码才知道的事 |
| `debug` | `CITEQUEST_DEBUG` | `false` | — |
| `log_level` | `CITEQUEST_LOG_LEVEL` | `INFO` | — |
| `log_format` | `CITEQUEST_LOG_FORMAT` | `json` | 自选。容器里用 `json`，本地开发用 `console` |

## 语料与切块（`corpus`）

| 参数 | 默认值 | 依据 |
|---|---|---|
| `fulltext_zenodo_doi` | `10.5281/zenodo.7752615` | 官方。unarXive open subset，tar.xz 4,844,918,096 字节 |
| `fulltext_category_prefix` | `cs.` | 自选。按 `metadata.categories` 的 primary 过滤 |
| `fulltext_year_from` / `fulltext_year_to` | `2019` / `2022` | 自选。全文层只到 2022 是换源的代价之一，不藏 |
| `fulltext_target_papers` | `50000` | 自选。可用上限估约 9.2 万，以解压后实测为准 |
| `openalex_field_id` | `fields/17` | 官方。`primary_topic.field.id`，即 Computer Science |
| `abstract_year_from` / `abstract_year_to` | `2023` / `2025` | 自选 |
| `abstract_papers` | `219676` | 实测（67,007 + 75,538 + 77,131） |
| `chunk_target_tokens` | `450` | 自选。按 section 边界聚合段落的目标长度 |
| `chunk_min_tokens` | `80` | 自选。低于此值并入同 section 相邻段 |
| `chunk_respect_section_boundary` | `True` | 自选。**不是可调项**：跨 section 的块没有唯一 section，引用锚点直接失效 |
| `tokenizer_model_id` | `BAAI/bge-m3` | 自选。与 dense 编码器同一个分词器，否则 450 token 的预算与实际编码长度对不上 |
| ⚠ `strip_placeholders` | `True` | 自选。理由两条且都是硬的：防倒排污染、防 citrec label 泄漏 |
| `keep_formula_latex_in_metadata` | `True` | 自选。保留在不参与检索的字段，代价是放弃公式内容可检索性 |
| `estimated_chunks_per_paper` | `25` | **估算，必须实测回填**。它同时决定 147 万 chunk 的规模账、6.5 GB 的内存账，以及 `ef` 派生式里 `C` 的合理量级 |
| `raw_dir` / `interim_dir` / `processed_dir` / `external_dir` | `data/{raw,interim,processed,external}` | 自选（cookiecutter-data-science 四层） |

## Milvus（`milvus`）

### 连接

| 参数 | 环境变量 | 默认值 | 依据 |
|---|---|---|---|
| `uri` | `CITEQUEST_MILVUS__URI` | `http://localhost:19530` | — |
| `token` | `CITEQUEST_MILVUS__TOKEN` | `""` | 留空表示本地无鉴权部署 |
| `collection_name` | `CITEQUEST_MILVUS__COLLECTION_NAME` | `citequest_chunks` | — |
| `alias` | `CITEQUEST_MILVUS__ALIAS` | `citequest_current` | 自选。**对外只用 alias**，切库靠改 alias 指向——灰度切换与一键回滚都在这一行上 |
| `index_version` | `CITEQUEST_MILVUS__INDEX_VERSION` | `idx_00000000_00` | 自选。进响应体、进缓存 key、进 `facets_exact`。换它即一键全量缓存失效 |
| `shards_num` | — | `2` | 官方上限 16 / collection |
| `request_timeout_ms` | — | `400` | 自选。总 deadline 1.0 s 含 1 次重试 |

### dense 通道

| 参数 | 默认值 | 依据 |
|---|---|---|
| `dense_dim` | `1024` | 官方。BGE-M3 输出维度 |
| `dense_metric_type` | `COSINE` | 官方用法 |
| `dense_index_type` | `HNSW` | 官方 |
| `hnsw_m` | `32` | 官方范围 [2,2048]、默认 30、推荐 [5,100]；论文给的 reasonable range 是 5–48，高维高召回取大 |
| `hnsw_ef_construction` | `256` | 官方默认 360、推荐 [50,500]；论文判据是「大到建图期召回接近 1」 |
| `ef_coefficient_c` | `4` | **自选，无官方依据**。`ef = max(C × group_limit × group_size, ef_floor)`；grouping 下 `ef` 官方完全没有指引，待组召回曲线扫 `C ∈ {2,4,8,16}` 标定 |
| `ef_floor` | `64` | 自选。低于它没有意义——官方默认行为就是 `ef = limit` |

派生结果：搜索路（60 × 1）`ef = 240`，RAG 路（16 × 3）`ef = 192`。
派生式自带断言 `ef ≥ group_limit × group_size`：`ef` 小于 entity 预算时 ANN 层根本凑不出这么多候选。

### sparse 通道与文本字段

| 参数 | 默认值 | 依据 |
|---|---|---|
| `sparse_metric_type` | `BM25` | 官方。BM25 Function 输出字段必须用，不可改 |
| `sparse_index_type` | `SPARSE_INVERTED_INDEX` | 官方 |
| ⚠ `bm25_k1` | `1.2` | 官方默认，推荐区间 [1.2, 2.0]。索引期参数 |
| ⚠ `bm25_b` | `0.75` | 官方默认。理由：切块策略已把长度方差约束在有界区间内，`b` 的杠杆两头都不大，偏离默认需要证据 |
| `inverted_index_algo` | `DAAT_MAXSCORE` | **部分核实**：官方示例中出现，未明示是否为默认值。显式写出，不宣称是默认 |
| ⚠ `text_en_analyzer_params` | `{"type": "english"}` | 官方。decision guide 对 English-only 内容明确推荐 `english` 而非 `standard` |
| ⚠ `text_exact_analyzer_params` | `{"tokenizer": "whitespace", "filter": ["lowercase", "alphanumonly"]}` | 官方对代码 / 技术内容的推荐配置 |
| `text_exact_enable_match` | `True` | 官方。建带位置的倒排，供 `TEXT_MATCH` 与 `PHRASE_MATCH(slop)` |
| `text_exact_attach_bm25` | `False` | 自选。**刻意收敛**：挂了它就变成第三条检索路，融合必须保持两路，RRF 结构才不变 |
| `enable_multi_analyzer` | `False` | 官方。该特性按文档的 `language` 字段分流，本项目文档侧纯英文，无分流对象 |
| `explicit_analyzer_name_on_search` | `True` | 官方。检索侧显式传 `analyzer_name`——隐式默认变了是静默的召回下降 |

### 标量、分区与降本

| 参数 | 默认值 | 依据 |
|---|---|---|
| `scalar_fields` | `["granularity", "year", "primary_category", "license"]` | 自选 |
| `partition_key_field` | `year` | 自选。过滤是本语料刚需；低选择率过滤会让 HNSW 图导航退化成暴力扫描 |
| `num_partitions` | `16` | 自选 |
| `mmap_enabled` | `False` | 官方（降本档①）。零召回损失，官方称 2 CPU / 8 GB 下可多载约 4 倍数据；与 DiskANN 互斥 |
| `quantization` | `None` | 官方（降本档②）。填 `HNSW_SQ` 时**必须** `refine=true`，且 `refine_type` 精度须高于 `sq_type` |
| `hnsw_sq_refine` | `True` | 官方。不 refine 的量化召回掉幅不可接受 |

## dense 编码服务（`embedding`）

**铁律：本项目没有任何自写推理服务。** 编码走 TEI 官方镜像 + Milvus 内置 TEI Embedding Function 的 Data-in/Data-out。

| 参数 | 环境变量 | 默认值 | 依据 |
|---|---|---|---|
| `provider` | — | `TEI` | 官方 provider 列表内 |
| `endpoint` | `CITEQUEST_EMBEDDING__ENDPOINT` | `http://tei:80` | **填的是 Milvus 容器视角的地址**：去调 TEI 的是 Milvus 服务端，不是本进程。写 `localhost` 会在 Milvus 容器内解析到它自己 |
| `model_id` | `CITEQUEST_EMBEDDING__MODEL_ID` | `BAAI/bge-m3` | **部分核实**：TEI 官方 supported_models 表未逐条列出 bge-m3，支持是按 XLM-RoBERTa 架构族给的 |
| `fallback_model_id` | — | `intfloat/multilingual-e5-large-instruct` | 官方。表内逐条列名、同为 1024 维、**schema 一行不改**；e5 要求的前缀由下面两个 prompt 字段承接 |
| `dim` | — | `1024` | 必须与 `milvus.dense_dim` 一致，切 fallback 也不变（启动时跨组校验） |
| `truncate` | — | `True` | 官方默认 `false`，**必须显式开**——否则超长文本直接报错而不是截断 |
| `truncation_direction` | — | `right` | 官方默认 right，显式写出 |
| `max_client_batch_size` | — | `32` | 官方默认。这是写入反压唯一留给客户端的旋钮，须与 TEI 启动参数一致，否则大的一边会在小的一边报 413 |
| `ingestion_prompt` / `search_prompt` | — | `None` | 切到 e5 时分别填 `"passage: "` / `"query: "` |
| `startup_probe_enabled` | — | `True` | 自选。固定探针文本比对向量 hash，不一致拒绝启动。成本极低，能挡住「灌库与查询用了不同模型」这类最难查的故障 |
| `startup_probe_text` | — | 固定串 | **必须固定不变**——换文本就等于换基准，探针立刻失去意义 |
| `startup_probe_expected_sha256` | `CITEQUEST_EMBEDDING__STARTUP_PROBE_EXPECTED_SHA256` | `None` | 首次建库时用 TEI 实际输出填入。为 `None` 时探针只做连通性检查并告警，不拒绝启动 |
| `offline_bgem3_enabled` | — | `False` | 自选。`pymilvus[model]` 的 `BGEM3EmbeddingFunction` **仅离线**用途（learned sparse 的离线 A/B、评测脚本），**禁止进在线路径** |

## 检索与融合（`retrieval`）

| 参数 | 默认值 | 依据 |
|---|---|---|
| `rrf_k` | `60` | 官方默认 60、范围 (0,16384)、推荐 [10,100]；Cormack SIGIR'09 的敏感性数据显示 [10,100] 几乎平坦。**唯一融合器，没有 WeightedRanker、没有权重字段** |
| `group_by_field` | `paper_id` | 官方。chunk → paper 聚合靠它 |
| `strict_group_size` | `False` | 官方警告 `True` 在数据分布不均时降性能，而混合粒度语料就是典型的分布不均 |
| `search_group_limit` / `search_group_size` | `60` / `1` | 自选。对应浅分页 3 页 × 20 篇 |
| `rag_group_limit` / `rag_group_size` | `16` / `3` | 自选。16 × 3 = 48 ≤ rerank 输入上限；`group_size ∈ {2,3,4}` 是必做消融 |
| `eval_group_limit` | `100` | 自选。覆盖 Recall@100。**报告里必须显式声明**评测用的是加大 limit 的同一条管线，否则口径与线上 60 不一致 |
| `max_page` | `3` | 自选。翻页不重新检索，从缓存的 60 组里切片 |
| `page_size` | `20` | 自选 |
| `phrase_match_default_slop` | `0` | 官方。0 = 严格相邻同序；2 时容许词序反转 |
| `text_match_default_logic` | `OR` | 官方默认 OR；要 AND 得写成 `TEXT_MATCH(t,'a') and TEXT_MATCH(t,'b')` |
| `default_sort` | `relevance` | 自选。引用量与时效**不进单一相关性分数**，改为可切换的 `relevance` / `citations` / `date` |

> **接口硬约束**：RRF 分数无物理含义（`k=60` 时上界约 0.033），禁止任何上游拿它做相关性阈值。
> 阈值统一由 rerank 层给。API 响应模型里干脆没有 `rrf_score` 字段。

## 精排（`rerank`）

| 参数 | 默认值 | 依据 |
|---|---|---|
| `model_id` | `BAAI/bge-reranker-v2-m3` | 官方。约 568M，FP16 约 1.1 GB |
| `normalize` | `True` | 自选。对外统一 sigmoid → [0,1]；内部是 raw logit。0.3 按 sigmoid 反推约 logit −0.85，等于把倾向不相关的 chunk 计为有效证据 |
| `input_size` / `output_size` | `48` / `8` | 自选（约 6x）。公开区间：Anthropic 150→20 = 7.5x；LlamaIndex 10→3 ≈ 3.3x |
| `batch_size` | `32` | 自选。48 pair 分 2 批前向 |
| `use_fp16` | `True` | — |
| `enabled_on_search_path` | `False` | 自选。搜索路不开 rerank。连带后果必须主动承认：**RAG 路的指标不能给搜索质量背书** |
| `score_threshold` | `None` | 自选。**不拍数**。标定照 Cohere 官方流程：30–50 条代表性 query → 标临界相关文档 → 过 reranker 取分 → 取均值 |
| `calibrated_for_model_version` | `None` | 自选。阈值绑模型版本，换模型必须重标并记版本号 |

启动时跨组校验：`input_size ≤ rag_group_limit × rag_group_size`，且 `output_size ≤ input_size`。
这两个是独立配置项，任何一个被单独改动都会让 rerank 静默地少收候选——不报错，只是指标悄悄下滑。

## 查询改写（`rewrite`）

| 参数 | 环境变量 | 默认值 | 依据 |
|---|---|---|---|
| `hyde_enabled` | `CITEQUEST_REWRITE__HYDE_ENABLED` | `true` | 论文（HyDE, arXiv:2212.10496）。**带存活条件**：必须在 citrec 库内子集与 LitSearch 上跑赢 direct，跑不赢就下线 |
| `hyde_n` | — | `1` | **自选，明确偏离论文默认值 20**，代价是方差更大 |
| `hyde_temperature` | — | `0.3` | 自选。论文用 1.0 是为配合 N=20 求多样性；N=1 时高温只放大方差 |
| `hyde_max_tokens` | — | `128` | 自选。对齐语料形态，过长会引入无关术语稀释信号 |
| `decompose_max_subqueries` | — | `4` | 自选。context 预算约束 |
| `decompose_iterations` | — | `1` | 自选。**不迭代**：语料与库内引用边撑不起 self-ask 式多跳链 |
| `keep_original_query_as_fallback` | — | `True` | 自选。不做 sub-query 前置相似度过滤——好的 sub-query 恰恰更具体更窄、相似度天然更低，那个过滤器会优先杀掉最好的分解结果 |

## 生成（`generation`）

| 参数 | 环境变量 | 默认值 | 依据 |
|---|---|---|---|
| `model_id` | `CITEQUEST_GENERATION__MODEL_ID` | `Qwen2.5-7B-Instruct-AWQ` | 自选。显存账按 4.0 GB 计 |
| `base_url` | `CITEQUEST_GENERATION__BASE_URL` | `http://localhost:8000/v1` | OpenAI 兼容端点 |
| `temperature` | — | `0.2` | 自选 |
| `max_tokens` | — | `1024` | 自选 |
| `context_max_chunks` | — | `8` | 自选。≤48 chunk → rerank → 8 条（启动时校验 ≤ `rerank.output_size`） |
| `annotate_granularity_in_prompt` | — | `True` | 自选。**prompt 里必须逐块标注 `granularity`**，否则模型会把摘要当方法细节来编——生成层唯一「不做就静默产生幻觉」的开关 |
| `router_model_id` | — | `Qwen2.5-1.5B-Instruct` | 自选。路由兜底小模型，1.5B–3B 量级即可 |
| `router_temperature` | — | `0.0` | 自选。路由必须可复现：同一条 query 两次给出不同 route 会让缓存、监控与回归测试同时失效 |
| `prompt_version` | `CITEQUEST_GENERATION__PROMPT_VERSION` | `v1` | 自选。进日志与 span。改 prompt 不改版本号 = 线上指标变动无法归因 |

## grounding（`grounding`）

| 参数 | 默认值 | 依据 |
|---|---|---|
| `hhem_model_id` | `vectara/hallucination_evaluation_model` | 官方。HHEM-2.1-Open，flan-t5-base 底座、0.1B、Apache-2.0、无上下文长度上限 |
| `online_first_sentence_only` | `True` | 自选。**在线只在首句跑一次**。必要性判据：花在线延迟买一个不敢用来拦截的软信号，不过关 |
| `online_per_edge_enabled` | `False` | 自选。逐引用边在线判定是默认关闭的开关 |
| `english_only` | `True` | 官方。HHEM 仅英文；中文回答只做确定性引用校验 |
| `supported_threshold` / `weak_threshold` | `None` / `None` | 自选。**待标定，不沿用余弦时代的旧阈值**。`None` 表示未标定，此时一律 `unverified` |
| `hard_gate_on_citation_out_of_range` | `True` | 自选。**唯一的硬 gate**：`[N]` 越界或引用全部非法 → 首句不过闸、完全不进流式 |
| `keep_previous_sentence_as_context` | `True` | 自选。判定上下文保留前一句，对应 NeMo 的 `context_size` 滑窗作用 |

启动时跨组校验：两个阈值必须同时标定或同时留空，且 `supported_threshold > weak_threshold`。
只标一个会让中间档语义未定义。

## 流式（`streaming`）

| 参数 | 默认值 | 依据 |
|---|---|---|
| `first_sentence_gate_enabled` | `True` | 自选。首句阻塞校验，过闸才发 `ai_answer_start`；不过闸完全不进流式 |
| `push_by_sentence` | `True` | 论文 / 官方。按**句边界**推送而非 token；句边界识别须处理缩写点号、小数点、紧跟句号的 `[N]` 标记 |
| `downgrade_only_never_retract` | `True` | 工业先例（NeMo Guardrails 官方承认违规文本可能已送达，归调用方处理） |
| `sse_total_timeout_ms` | `60000` | 自选 |
| `heartbeat_interval_ms` | `15000` | 自选。防中间代理按空闲超时断连；与 `X-Accel-Buffering: no` 是两件事（一个防断连、一个防攒批） |

## 编排与记忆（`agent` / `memory`）

| 参数 | 环境变量 | 默认值 | 依据 |
|---|---|---|---|
| `agent.correction_loop_enabled` | `CITEQUEST_AGENT__CORRECTION_LOOP_ENABLED` | `true` | — |
| `agent.recursion_limit` | — | `8` | 自选，**显式设置**。官方默认 1000，离安全边界差两个数量级。8 = 1 轮纠正的 4 步 × 一倍余量 |
| `agent.max_correction_rounds` | — | `1` | 自选。**超时退回首轮结果不报错** |
| `agent.correction_loop_timeout_ms` | — | `5000` | 自选 |
| `agent.checkpointer_backend` | `CITEQUEST_AGENT__CHECKPOINTER_BACKEND` | `memory` | 官方明确进程内 saver 仅供开发测试。**非 `memory` 之外的取值意味着声称具备持久化能力** |
| `agent.stop_loss_min_improvement` | — | `0.05` | 自选。**事先写死的止损判据**：纠正循环提升 < 5% → 砍掉循环，只保留工具化与 MCP |
| `agent.parallel_tool_calls_max` | — | `3` | 自选。与 decompose 上限同量级 |
| `agent.observation_max_tokens` | — | `800` | 自选。观察层裁剪：21,600 → 3,600 → 800 两次收敛；不裁剪则三轮 64,800 token |
| `agent.no_progress_rounds_to_stop` | — | `2` | 自选。连续 N 轮新增有效 chunk 为 0 即止损 |
| `memory.enabled` | `CITEQUEST_MEMORY__ENABLED` | `false` | 自选。**只做研究会话（跟课题走），不做用户画像记忆（跟人走）** |
| `memory.compress_after_turns` | — | `6` | 自选。单轮增量约 900 token → 6 轮 ≈ 5,400，覆盖一个完整探索段落 |
| `memory.compress_after_tokens` | — | `8000` | 自选。比轮数闸门略松一档，专治「单轮特别长」 |
| `memory.summary_max_tokens` | — | `400` | 自选。仅压缩 narrative 字段；结构化字段做集合运算，因此不漂移 |
| `memory.keep_raw_turns` | — | `2` | 自选。指代先行词绝大多数落在最近 1–2 轮 |
| `memory.incremental_compression` | — | `True` | 估算。增量叠加 O(n) vs 全量重算 O(n²) |

> `memory` 这一组的口径是全表最严的一档：**无一项来自已定参数，全部为新设计值，只有自己的推导，没有外部出处。**
> 收进配置是为了闭合「查不到的数值不许写」这条规矩，不是为它们背书。

## API 与限流（`api`）

| 参数 | 环境变量 | 默认值 | 依据 |
|---|---|---|---|
| `host` / `port` | `CITEQUEST_API__HOST` / `__PORT` | `0.0.0.0` / `8080` | — |
| `cors_allow_origins` | `CITEQUEST_API__CORS_ALLOW_ORIGINS` | `["http://localhost:5173"]` | — |
| `query_max_length` | — | `500` | 自选。**超出返回 400 而不是静默截断**：截断会让用户看到的输入与实际执行的 query 不一致，且截断后的 query 会以截断形态进缓存 key |
| `accept_client_request_id` | — | `True` | 自选。沿用客户端 `X-Request-Id` 便于跨服务串联，但**必须校验字符集与长度再落日志**——它会进日志和 trace，不校验就是一个日志注入面 |
| `healthz_expose_details` | — | `False` | 自选。匿名调用只返回存活信号，不泄露模型名、版本、endpoint |
| `rate_limit_search_per_minute` | — | `60` | 自选 |
| `rate_limit_answer_per_minute` | — | `10` | 自选。AI 路更贵，配额单列；**超限静默降级成纯搜索而不是 429** |

## 缓存（`cache`）

| 参数 | 默认值 | 依据 |
|---|---|---|
| `l1_enabled` / `l1_ttl_seconds` | `True` / `900` | 自选。L1 = 全量响应缓存 |
| `l2_enabled` / `l2_ttl_seconds` | `True` / `3600` | 自选。L2 = 热门 query 的 paper 级结果 |
| `key_includes_index_version` | `True` | 自选。key 必须含 `index_version` + **全部影响结果的参数**。不含 `top_k` 之类参数是正确性缺陷而不是命中率问题 |
| `retracted_paper_exact_invalidation` | `True` | 自选。撤稿论文按 `paper_id` 精确失效——TTL 之外唯一的例外，因为它是合规要求 |

**没有 L3 embedding 缓存**：Data-in/Data-out 之后编码在服务端发生，客户端根本没有 embedding 可缓存。
这不是砍掉的，是架构后果。

## 并发闸门与超时（`concurrency` / `timeouts`）

各类**不共享额度**：共享额度会让秒级的生成任务饿死几十毫秒的编码任务。

| 阶段 | 信号量 | 队列 | 排队超时 | 超时后的动作 |
|---|---|---|---|---|
| rerank | `4` | `8` | `500 ms` | 退回 RRF 名次直出 |
| 生成 | `12` | `12` | `2000 ms` | 只给搜索结果 |
| HHEM | `8` | `8` | `200 ms` | 只保确定性引用校验 |
| 路由 | `12` | `16` | `300 ms` | 走默认路由 |

`sem_generate = 12 = min(显存上限 60, 延迟 SLO 上限 12)`：只算显存会得出「能开 60」，然后 20 并发时首 token 全线崩。

单次调用超时，取法是**该阶段 p95 预算 × 2，向上取到易记刻度**：

| 参数 | 默认值 |
|---|---|
| `router_ms` | `800` |
| `milvus_ms` / `milvus_total_deadline_ms` | `400` / `1000`（含 1 次重试） |
| `hyde_ms` | `2000` |
| `rerank_ms` | `800` |
| `first_sentence_gate_ms` | `200` |
| `first_token_ms` | `6000` |
| `correction_loop_ms` | `5000` |
| `sse_total_ms` | `60000` |
| `propagate_deadline` | `True`（逐层传递绝对截止时间戳，否则「每层 2 s」在四层链路上累积成 8 s） |

启动时跨组校验：生成的排队超时必须短于 RAG 首 token 的 p95 预算——排队 2 s 之后即使拿到额度，也必破预算。

## 延迟预算（`latency_budget`）

| 参数 | 默认值 |
|---|---|
| `search_p50_ms` / `search_p95_ms` | `100` / `250` |
| `rag_first_token_p50_ms` / `rag_first_token_p95_ms` | `1000` / `2500` |
| `rag_with_correction_p95_ms` | `4000` |
| `measured` | `False` |

> **全表无一行实测。** 这些值只用于告警阈值与回归基线，不用于对外宣称。
> `measured` 为 `False` 之前，禁止在任何对外材料里引用具体毫秒数。

## 评测（`eval`）

| 参数 | 默认值 | 依据 |
|---|---|---|
| `citrec_dataset_id` / `citrec_split` | `saier/unarXive_citrec` / `test` | 官方。label = 被引论文 OpenAlex ID |
| `citrec_min_subset_size` | `3000` | 自选。**头条基准退场阈值**：过滤到库内后低于此规模 → 头条退回 LitSearch，citrec 降级为补充证据 |
| `citrec_report_filter_rate` | `True` | 自选。**必报数字**：label 落在本库内的比例。这是证据链不是遮掩 |
| `citrec_strip_marker_neighborhood` | `True` | 自选。防泄漏第一条（第二条在索引侧：删除占位符而非替换成标题） |
| `litsearch_dataset_id` | `princeton-nlp/LitSearch` | 官方。query 597 / corpus 64,183 |
| `litsearch_retrieval_field` | `title_abstract` | 官方检索字段 |
| `litsearch_bm25_ndcg10_reference` / `_recall100_reference` | `40.2` / `74.4` | **部分核实：单一来源**，引用时必须说明 |
| `litsearch_*_tolerance` | `3.0` / `4.0` | 自选。超窗先查索引与分词器，**不许直接开始报提升数字** |
| `reproduction_wording` | `sanity_check` | 自选。只做量级一致性检查，**叫 sanity check 不叫 reproduction** |
| `metrics` | `recall@100, mrr@10, ndcg@10, recall@5, recall@20` | 自选 |
| `aggregation` | `macro` | 自选。按 query 平均，不是全局池化 |
| `bootstrap_samples` / `confidence_level` | `1000` / `0.95` | 惯例 |
| `significance_tests` | `paired_t` + `fisher_randomization` | 论文（Smucker CIKM'07）。**双跑**，不一致以 randomization 为准并标注；不跑 Wilcoxon / sign |
| `multiple_comparison_correction` | `tukey_hsd` | 论文 |
| `report_per_query_win_loss_tie` | `True` | 自选。**少了三元计数这个消融就不可信** |
| `report_split_by_granularity` | `True` | 自选。fulltext 与 abstract 必须分开报，禁止合并 |
| `ablation_configs` | `bm25_only, dense_only, hybrid, hybrid_rerank` | 自选。共用同一份 query、qrels、代码，一次只变一项 |
| `ab_learned_sparse_vs_bm25` | `True` | 自选。**必做**——这是「为什么敢放弃 learned sparse」唯一的数字来源 |
| `ab_hyde_vs_direct` | `True` | 自选。HyDE 的存活条件 |
| `calibration_set_size` / `calibration_set_path` | `40` / `data/external/calibration_set.jsonl` | 官方流程（Cohere）。30–50 条「临界相关」query，与回归集分开命名 |
| `reports_dir` | `reports` | — |

## 启动时的跨组校验

单看每一组都合法、只有放在一起才暴露矛盾的约束，全部在 `Settings` 的 model validator 里一次性拦掉。
它们出错的表现形式都是「不报错、指标悄悄下滑」：

1. `embedding.dim == milvus.dense_dim`——不一致时若只有 query 侧错，是静默的召回崩塌；
2. `rerank.input_size ≤ rag_group_limit × rag_group_size`，且 `output_size ≤ input_size`；
3. `generation.context_max_chunks ≤ rerank.output_size`；
4. grounding 的两个阈值同时标定或同时留空，且 `supported > weak`；
5. `concurrency.queue_timeout_generate_ms < latency_budget.rag_first_token_p95_ms`。
