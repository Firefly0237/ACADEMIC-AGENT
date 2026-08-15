# TEI 服务说明

TEI 是本项目**唯一**的 dense 编码服务。铁律：不写自己的推理服务，
dense 编码一律走 TEI + Milvus 内置 Embedding Function 的 Data-in/Data-out
（决策记录 D4）。所以这个目录里没有任何代码，只有部署参数说明。

## 镜像 tag 必须按 GPU 架构选

TEI 不发「一个镜像通吃」的 tag，tag 前缀就是 CUDA compute capability：

| GPU | tag 前缀 | 示例 |
|---|---|---|
| Ada Lovelace（RTX 4090 / L4 / L40S） | `89` | `ghcr.io/huggingface/text-embeddings-inference:89-1.9` |
| Ampere 80（A100） | `80` | `…:80-1.9` |
| Ampere 86（A10 / RTX 3090） | `86` | `…:86-1.9` |
| Turing（T4 / RTX 2000 系） | `turing` | `…:turing-1.9`（实验性） |
| Hopper（H100） | `hopper` | `…:hopper-1.9` |
| CPU | `cpu` | `…:cpu-1.9` |

**compute capability < 7.5（V100、GTX 10 系）TEI 不支持**——这不是性能问题，
是直接起不来。那种机器上只能退到 `cpu` 镜像，同时把延迟预算整表作废重估。

tag 通过 compose 的 `TEI_IMAGE_TAG` 环境变量注入，默认 `89-1.9`。

## 模型

默认 `BAAI/bge-m3`（dim 1024）。

**这是一处「部分核实」**：TEI 官方 supported_models 表没有逐条列出 bge-m3，
支持是按架构族给的（表中原文含 "XLM-RoBERTa models with absolute positions"，
BGE-M3 正是 XLM-RoBERTa 底座）。所以首次部署必须验证两件事：

1. `curl http://localhost:8081/info` 能返回模型信息；
2. 输出向量维度确实是 1024。

任一不成立，切到替换项 `intfloat/multilingual-e5-large-instruct`
——表内逐条列名、同为 XLM-RoBERTa、dim 同样是 1024，**Milvus schema 一行不用改**。
代价只有一处：e5 要求 `"passage: "` / `"query: "` 前缀，由 Milvus TEI Function 的
`ingestion_prompt` / `search_prompt` 承接（配置项已在 `config/embedding.py` 预留）。

## 权重缓存

`./volumes/tei-cache:/data` 必须挂出来。bge-m3 权重约 2 GB，不挂就是每次重建容器重下一次。

## 与 Milvus 侧参数的对齐

| TEI 启动参数 | Milvus TEI Function 参数 | 说明 |
|---|---|---|
| `--max-client-batch-size=32` | `max_client_batch_size=32` | 官方默认值。两边不一致时大的那边会被小的那边直接 413 |
| （无对应） | `truncate="true"` | **官方默认 false，必须显式开**；否则超长文本是报错而不是截断 |
| （无对应） | `truncation_direction="right"` | 官方默认就是 right，显式写出以免默认值变更 |
