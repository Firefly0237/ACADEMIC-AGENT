# Source updates

Linux source broker 使用独立 `updater` 镜像角色。发布作业使用包含 Milvus SDK 的 `indexer` 角色。配置示例中的 UUID 用部署环境实际记录替换，密钥通过只读文件挂载。

## Source broker

```bash
python -m citequest.updates_v2.broker --config /run/core-config/source-broker.json
```

Broker 自动轮询 PostgreSQL 中所属 publisher 的固定更新窗口。`--once` 执行一个窗口。单实例部署共用持久 POSIX 锁目录；源 CAS 挂载到 `/opt/citequest/data/runtime-v2-app/blobs`。原始响应写入 CAS，恢复 journal 写入 PostgreSQL。SIGTERM 停止新调度并排空当前请求。

首次切换 Linux 网络策略时，排空旧 broker 后执行相同命令并添加 `--adopt-network-policy --once`。OAI 窗口使用 `protocol_profile: "oai-fixed-window/1"`，起止时间为 UTC 零点，结束边界不包含在窗口内；opaque token、请求累计、重启次数和成功 epoch 持久保存。

## Publication jobs

```bash
python -m citequest.updates_v2.publication_cli --config /run/core-config/publication.json
```

通过配置 `phase` 顺序执行：

1. `observed`：读取窗口完成状态与原观察 seal。
2. `prepare`：基于完整旧 generation 合并有界观察、剔除来源隔离，创建不可变 snapshot 与新 generation。
3. `build`：使用既有 tokenizer、原生 BM25/dense 索引和完整回读路径，按执行时间窗继续同一 generation。
4. `publish`：使用 `expected_epoch`、稳定 `intent_id`、真实 alias 回读及 SQL 提交发布。
5. `recover`：按 SQL authority 修复同一 intent 的 alias 状态。

输出状态分别为 `observed`、`prepared`、`published`；`search_published` 仅在 SQL active generation 确认后为 true。`prepared/status=blocked` 保留准确的维护门与已有进度。业务读请求固定 SQL generation 和具体 collection，已有 reader lease 与 Task 引用继续使用原版本。

GPU backend 的 binding/adoption 使用独立迁移身份和显式维护输入：

```bash
python -m citequest.updates_v2.backend_maintenance --config /run/core-config/backend-binding.json
python -m citequest.updates_v2.backend_maintenance --config /run/core-config/backend-adoption.json
```

`bind` 绑定真实准备完成的 generation manifest 与固定 backend profile。`adopt` 还需要实际标准质量报告、原已采用质量记录、指向当前 generation/manifest/backend/report 的明确 approval 文件和完整 adoption JSON；文件 SHA 与数据库原门控共同验收。采用现成标准 backend 质量时，记录仍明确归属标准集合。

运行身份 `cq_s03_app` 对新增 receipt/snapshot/member 表拥有 SELECT/INSERT；历史记录由数据库追加约束保护。`cq_s03_migrator` 执行显式 schema 与 backend 维护。详情接口通过 `metadata_observation` 返回真实来源、观察记录时间、窗口和 observed watermark；当前性与全局最新状态保持明确的 unknown 语义。
