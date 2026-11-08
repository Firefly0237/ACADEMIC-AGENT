# DecisionRAG 概率导航服务

`laya1` 使用作者公开的 Laya 模型，对文献候选和原文片段进行两级概率选择。Agent通过MCP调用服务，结果包含候选分数、入选引用和推理用量。

## 配置文件

| 文件 | 用途 |
| --- | --- |
| [gateway-linux.example.json](gateway-linux.example.json) | Linux概率服务地址、模型资产、回执和缓存目录 |
| [task.example.json](task.example.json) | Agent访问地址、Bearer密钥引用和导航参数 |
| [citequest-decisionrag.service](citequest-decisionrag.service) | systemd服务配置 |

## 启动服务

1. 准备Python 3.12 CPU运行环境，以及固定版本的PyTorch、Transformers和Laya依赖。
2. 将模型、Laya wheel和资产回执放到配置中的`assets`目录。
3. 复制网关样例为`/etc/citequest/decisionrag-gateway.json`，填写实际资产回执SHA、监听地址和Bearer密钥文件。
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

服务使用`POST /v1/decision`，协议为`citequest-decisionrag-probability/1`。请求包含`preflight`或`score`操作、候选文本和请求摘要。

## 接入Agent

使用[导航API样例](../s08/api-decisionrag.example.json)与[导航worker样例](../s08/worker-decisionrag.example.json)。

将任务样例保存为部署目录中的`decisionrag-task.json`，填写服务的实际可访问地址和密钥文件。随后在worker配置中设置`decisionrag_config_file`和该文件的`decisionrag_config_sha256`。API配置中的QA任务使用`evidence_navigation_protocol: "laya1"`。

| 任务参数 | 含义 |
| --- | --- |
| `document_keep`、`passage_keep` | 两级导航的入选数量 |
| `max_documents`、`max_passages` | 候选集合大小 |
| `max_shared_bytes` | 完整共享输入的字节额度 |
| `active_seconds` | 单次导航时限 |
| `local_input_token_limit` | 本地推理输入token额度 |

导航结果通过任务的`evidence_navigation`返回，工作站可展开查看文献→片段的选择过程，并打开入选引用。
