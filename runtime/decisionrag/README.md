# DecisionRAG 概率导航服务

使用作者公开的 Laya 0.4.22，对已授权文献候选和原文片段进行两级概率选择。Agent通过MCP调用私有CPU服务，返回候选分数、入选引用和实际推理回执。

## 配置文件

| 文件 | 用途 |
| --- | --- |
| [gateway-linux-laya2.example.json](gateway-linux-laya2.example.json) | `Linux-CPU/2`网关，默认端口18032，单序列8192 token |
| [task-laya2.example.json](task-laya2.example.json) | `task/2`与`remote/2`，有界分批导航 |
| [gateway-linux.example.json](gateway-linux.example.json)、[task.example.json](task.example.json) | 独立`laya1`兼容配置，端口18031，单序列1024 token |
| [laya2-policy.example.json](../s08/laya2-policy.example.json) | API的新任务profile选择 |
| [citequest-decisionrag.service](citequest-decisionrag.service) | systemd服务配置 |

`laya2`显式使用模型编码器的8192位置容量；checkpoint的原生默认`max_len=1024`仍保留。每次请求最多8个候选，逐行预检完整共享上下文与问题，要求无截断、无丢弃源token。`laya1`保持原有协议和资源配置，不随新配置自动升级。

## 分批导航与额度

文献层与片段层分别装入完整候选，不切碎或截断单个候选。超过单次容量时，每批先独立评分，按固定批内名额生成shortlist，再将shortlist放入同一共享上下文重新评分，最终排序只使用这次共同评分。不同筛选批次的概率不直接混排。单批可直接进入最终评分；单候选或共同shortlist放不下时返回容量错误。

[task-laya2.example.json](task-laya2.example.json)提供以下参数：

| 参数 | 样例值与含义 |
| --- | --- |
| `document_keep`、`passage_keep` | 保留2篇文献、5个片段 |
| `max_documents`、`max_passages` | 整次导航最多20篇文献、合计20个片段 |
| `shortlist_count` | 每层共同评分最多8个候选 |
| `max_batches_per_level` | 每层最多4个筛选批次 |
| `maximum_forward_calls` | 两层合计最多10次实际forward |
| `max_shared_bytes` | 单次共享输入最多32768字节 |
| `active_seconds` | 导航最多25秒，并受所属Task剩余时间约束 |
| `local_input_token_limit` | 各次forward、各候选行输入token累计最多131072 |

导航仅处理已检索并获准的初始证据frontier；它不代表全文覆盖或蕴含验证。分批路径不声明检索质量、延迟或基准效果提升。

## 启动服务

1. 准备Python 3.12 CPU运行环境，以及固定版本的PyTorch、Transformers和Laya依赖。
2. 将模型、Laya wheel和资产回执放到配置中的`assets`目录。
3. 复制所选协议的网关样例为`/etc/citequest/decisionrag-gateway.json`，填写实际资产回执SHA、私网监听地址和Bearer密钥文件。
4. 为服务用户准备`cache_root`，从仓库根目录启动：

```bash
PYTHONPATH=src:. python -m scripts.s14.serve_decisionrag \
  --config /etc/citequest/decisionrag-gateway.json \
  --execute-local-cpu
```

也可以修改并安装所附systemd配置：

```bash
sudo cp runtime/decisionrag/citequest-decisionrag.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now citequest-decisionrag
```

服务使用`POST /v1/decision`，`laya2`协议为`citequest-decisionrag-probability/2`，`laya1`为`/1`。请求包含`preflight`或`score`操作、候选文本、请求摘要和`remaining_ms`。预检不执行forward；每次成功评分由模型hook确认恰好一次forward，并核对官方tokenizer的完整输入计数。

网关父进程执行物理deadline，超时会终止模型子进程并保持故障状态，等待显式重启。回执中的`trace`、`selection_steps`、`local_forward_admissions`和`local_usage`保留批次、共同评分候选、请求摘要、模型绑定、实际forward/token数及耗时。结果未知时总数为`null`，同时保留已完成的已知用量；不会记作零次调用。`external_API_cash_ncny=0`只表示本地评分未调用外部API，物理计算费用仍未计量。

## 接入Agent

1. API的`task_profile_policy`采用[laya2 policy样例](../s08/laya2-policy.example.json)：QA使用`qa-v9`、`qa_content_protocol: "isolated1"`及`evidence_navigation_protocol: "laya2"`，保留该profile所需的其他协议字段。
2. 将[task-laya2.example.json](task-laya2.example.json)保存为部署目录的`decisionrag-task.json`，填写Agent实际可访问的网关地址和Bearer密钥文件引用。按[worker配置结构](../s08/worker-decisionrag.example.json)设置`decisionrag_config_file`，将该文件**最终原始字节**的SHA-256填入`decisionrag_config_sha256`；修改地址或额度后同步更新摘要。
3. API选择的导航协议、worker装载的任务版本和网关协议必须一致；worker/MCP会校验文件摘要与Task的协议绑定。保留`laya1`时使用原[API样例](../s08/api-decisionrag.example.json)、`task/1`和`Linux-CPU/1`网关，配置摘要单独计算。

导航结果通过任务的`evidence_navigation`返回，工作站可展开查看文献→片段的选择过程，并打开入选引用。
