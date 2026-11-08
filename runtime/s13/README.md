# 文档导出工具链

导出服务将研究计划草稿转换为DOCX、HTML和PDF。Web/API提交导出任务，独立worker读取队列并保存产物；Linux渲染器负责文档转换。

## 组件

| 组件 | 职责 |
| --- | --- |
| `citequest.exports_v3` | 导出接口、队列、产物存储和下载 |
| `Dockerfile` | LibreOffice、字体与Python依赖镜像 |
| `release-profile.json` | 镜像、模板、CSL与工具链身份 |
| `font-packages.lock.json`、`fontconfig.xml` | 字体依赖与匹配配置 |

## 准备工具链

1. 准备学校DOCX模板、Harvard CSL文件和产物存储。
2. 从仓库根目录构建渲染器：

```bash
docker build -f runtime/s13/Dockerfile -t citequest-export:local .
```

3. 将镜像ID、模板/CSL摘要、renderer及字体工具链信息配置到导出release profile。
4. 为独立worker准备数据库连接、搜索配置、产物存储和工作目录。

Python依赖由本目录的`requirements.txt`与`uv.lock`管理。镜像包含LibreOffice、Poppler及文档所需字体。

## 运行worker

导出进程配置使用`citequest-export-process/1`：

| 配置 | 用途 |
| --- | --- |
| `workspace`、`release_profile_file`、`release_profile_sha256` | 工作目录与工具链发布配置 |
| `business_dsn_file`、`checkpoint_dsn_file`、`memory_dsn_file` | 数据库连接 |
| `search_config_file`、`material_config_file`、`cursor_key_file` | 来源与分页配置 |
| `artifact_store` | 私有产物存储 |
| `worker_spool` | 导出工作目录 |
| `docker_cli`、`docker_config` | 渲染器的Docker客户端配置 |
| `limits` | 运行额度 |

在安装了对应Python依赖的环境中启动：

```bash
PYTHONPATH=src python -m citequest.exports_v3.cli \
  --config /etc/citequest/export-worker.json --serve
```

使用`--once`处理一次队列轮询。Web配置的`export_config_file`指向`citequest-export-api/1`配置，API与worker使用同一逻辑产物存储和release profile。

## 渲染器接口

PDF渲染容器接收`/inputs/proposal.docx`，将文件写入`/outputs`。入口支持：

```text
--input-root
--output-root
--renderer
--renderer-sha256
```

研究工作站的导出面板负责选择格式、查看任务进度和下载文件。
