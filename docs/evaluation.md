# 记录式 Agent 评估

评估工具读取已执行任务的真实账本，以固定用例检查业务结果和工具轨迹，同时汇报时延、费用及覆盖情况。评估与在线 Agent 分开运行，使用 `runtime/agent-evals` 的独立 Linux 依赖锁。

## 导出任务记录

准备评估专用的 PostgreSQL 只读账号和至少 32 字节的私有 HMAC key 文件，然后显式指定 owner 和任务：

```bash
PYTHONPATH=src python -m citequest.evaluation.cli export \
  --business-dsn-file /run/secrets/evaluation-dsn \
  --owner-id <owner-uuid> --task-id <task-uuid> \
  --hmac-key-file /run/secrets/evaluation-hmac \
  --output output/evaluation/recorded.json
```

一次导出使用只读 `REPEATABLE READ` 事务，所有查询均限定 owner 和显式 task IDs。工具顺序来自真实 `attempt_reserved` 事件；只有存在 `sent_at` 的工具 attempt 进入轨迹。模型原始请求、回复、用户问题和数据库秘密不写入导出文件。工具名、参数、身份、引用和结果版本通过分域 HMAC 绑定，整份导出另有完整性 HMAC。

## 编写用例

[用例 Schema](../contracts/v3/evaluation/cases.schema.json) 定义固定字段；[示例](../contracts/v3/evaluation/synthetic-case.example.json) 标为 `synthetic`，用于说明结构。真实用例使用导出的 `command_hmac`，`expected_result` 只列需要核对的公开结果字段，例如：

```json
{"publication_status":"validated_ai_review","school_section_count":14,"command_outcome":"completed"}
```

拒绝输入的用例可以要求 `{"command_outcome":"rejected"}`。原始输入和预期工具参数保留在私有用例源中，用相同 key 调用 `Protector.tool(name, arguments)` 生成比较值；每个预期物理调用独立列一项。预期值应来自用例设计，不从被测轨迹复制出正确答案。

`require_trajectory_match` 决定业务成功是否要求工具顺序与参数匹配；`require_literal_citations` 决定是否要求历史引用逐字匹配。`exclude_if` 只能显式列 `cancelled`、`unknown_outcome` 或 `missing_citation_evidence`，每次排除会单独计数。

## 执行评估

```bash
PYTHONPATH=src python -m citequest.evaluation.cli evaluate \
  --recorded output/evaluation/recorded.json \
  --cases /run/evaluation/cases.json \
  --hmac-key-file /run/secrets/evaluation-hmac \
  --output output/evaluation/report.json
```

- 工具轨迹调用官方 AgentEvals `create_trajectory_match_evaluator(trajectory_match_mode="strict", tool_args_match_mode="exact")`。一个实际 attempt 对应一个消息，以保留调用先后关系。
- 业务结果使用 OpenEvals `exact_match` 比较用例明确指定的结果投影。任务状态本身不代表业务成功；已知失败或拒绝不能被成功率分母静默遗漏。
- 时延分别记录排队、单次发送至结算、命令整体时间。费用分别保留已知费用、预留和未知；存在未知或未结预留时实际总额为 `null`。
- 引用逐字检查针对任务已存证据正文；语义支持和当前授权单独标记。记录评估不会把历史文本匹配当作当前授权或语义事实证明。
- 报告列出总记录、显式用例、缺失记录、缺少 gold、排除、不完整和已评分数量，并显示已评分分母与成功率。缺少实际记录或 gold 时输出 `no_data` 和 `null`。

评估在 `langsmith.tracing_context(enabled=False)` 中调用确定性比较器，无需模型或远程评估服务。依赖来源：[AgentEvals](https://github.com/langchain-ai/agentevals)、[OpenEvals](https://github.com/langchain-ai/openevals)、[LangSmith tracing context](https://docs.langchain.com/langsmith/trace-without-env-vars)。
