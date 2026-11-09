# 文档导出工具链

Web/API 提交导出任务，独立 worker 读取队列并保存 DOCX、HTML 和 PDF。渲染容器使用固定的 LibreOffice、Pandoc、学校模板、CSL 和字体配置。

## CI 发布

在 GitHub Actions 手动运行 **Export release**，选择 `build`。流水线构建 `linux/amd64` 镜像，读取实际 image ID 和容器工具链身份，渲染包含长文、36 行表格和 3 个原生公式的容量夹具，检查文字覆盖、A4 页面和页码，保存 PDF 与每页 PNG。

产物 `export-candidate` 包含：

| 文件 | 内容 |
| --- | --- |
| `candidate.json` | 当前源码、镜像、工具链和各产物的 SHA-256 |
| `toolchain-identity.json` | 实际字体、Python/系统包、模板与源码身份 |
| `layout/` | 真实 DOCX、HTML、PDF、每页 PNG 和覆盖检查 |
| `renderer-image.tar` | 可通过 `docker load` 预载的固定镜像 |

已有真实版面认证结果时，再次手动运行，选择 `certify`，输入候选构建的 `candidate_run_id`、仓库内认证文件路径和文件 SHA-256。认证文件使用 `citequest-capacity-render-certification/2`：

- `status: passed`、`image_id`、`source_sha256`、`toolchain_sha256`、`font_profile` 与候选一致。
- `capacity_artifact_sha256` 对应候选的 `layout/capacity.json`。
- `coverage` 原样绑定实际容量检查，`actual_model_requests` 为 `0`。
- `visual_review` 包含实际认证的 `status`、`pages_reviewed`、`pages_observed_individually`、`png_sha256` 和 `evidence` 引用；页码与所有 PNG 摘要必须完全匹配候选。

流水线消费真实认证，通过后生成 `export-release`，其中包含新的 `release-profile.json`、认证、工具链身份、固定模板及 API/worker 配置。候选和认证版本各自保留。

对应命令：

```bash
python scripts/s13/build_release.py build --output output/export-candidate
python scripts/s13/build_release.py certify \
  --candidate output/export-candidate \
  --certification assets/export/<实际认证文件>.json \
  --certification-sha256 <实际SHA256> \
  --output output/export-release
```

## 部署

API 和 worker 将同一 `export-release` 目录只读挂载至 `/run/export-release`，使用生成的 `export-api.json`、`export-worker.json`。`workspace` 指向应用根 `/opt/citequest`，`release_assets_root` 指向认证发行资产 `/run/export-release`。样例中的 SHA 占位符由流水线生成值替换。Web 的 `export_config_file` 指向 API 配置。

渲染器由独立 worker 所连接的 Docker daemon 启动。先在该 daemon 上执行 `docker load --input renderer-image.tar`，再确认 `docker image inspect <image_id> --format '{{.Id}}'` 与 release 一致。此处使用 Docker image ID，与镜像仓库 manifest digest 分别核对。

Worker 使用 core 的 `export_worker` 构建 target，配置秘密与搜索/材料服务沿用现有角色。为 worker 提供可信 Docker CLI、`docker_config` 和专用 daemon 访问；渲染子容器使用无网络、只读文件系统、非 root 用户和资源限制。

`worker_spool` 在 worker 与 Docker 宿主机上采用同一绝对路径，例如 `/var/lib/citequest/state/export-job-spool`，以便 daemon 正确解析 bind mount。API 和 worker 使用同一个私有产物存储，配置各自的 spool。

```bash
PYTHONPATH=src python -m citequest.exports_v3.cli \
  --config /run/core-config/export-worker.json --serve
```

镜像构建采用 [Docker Build](https://docs.docker.com/reference/cli/docker/buildx/build/)；镜像归档采用 [Docker Save](https://docs.docker.com/reference/cli/docker/image/save/)。
