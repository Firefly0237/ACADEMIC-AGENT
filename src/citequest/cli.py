"""命令行入口：`citequest <subcommand>`（由 pyproject 的 [project.scripts] 注册）。

薄壳原则：这里只做「读配置 → 初始化日志 → 调包内函数」。任何一条子命令的实现
逻辑都不写在本文件里——CLI 是一层协议适配，和 `api/` 的地位相同。

子命令：

- `build-index` —— ingestion → indexing 全流程灌库
- `evaluate`    —— 跑评测并把报告写到 reports/
- `serve`       —— 起 FastAPI
- `mcp`         —— 起 MCP server（stdio）
- `show-config` —— 打印当前生效的参数正本，用于核对环境变量有没有意外覆盖
"""

from __future__ import annotations

import sys
from typing import Annotated, TextIO

import typer

from citequest.config.settings import Settings, get_settings
from citequest.observability.logging import configure_logging, get_logger

__all__ = ["app", "main"]

app = typer.Typer(
    name="citequest",
    help="CiteQuest：学术检索与 RAG 系统",
    no_args_is_help=True,
    add_completion=False,
)
_log = get_logger(__name__)


def _bootstrap(*, log_stream: TextIO | None = None) -> Settings:
    """读配置并初始化日志。每条子命令的第一件事。

    单独抽出来是因为「先配日志再干活」这个顺序容易在新增子命令时被漏掉，
    而漏掉的表现是那条命令的日志格式与别处不同——排查时最容易被误导的一类现象。

    :param log_stream: 只有 `mcp` 子命令需要传（stderr），理由见该命令的 docstring。
    """
    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format, stream=log_stream)
    return settings


@app.command("build-index")
def build_index(
    granularity: Annotated[str, typer.Option(help="只灌某一层：fulltext / abstract / all")] = "all",
    dry_run: Annotated[bool, typer.Option(help="只做 schema 校验与规模估算，不真正写入")] = False,
) -> None:
    """跑 ingestion → indexing 全流程灌库。

    **这是一条昂贵且不可逆的命令**：147 万 chunk 的全量重灌是 6–10 小时量级，
    而且 analyzer 类型、bm25 参数、text_exact 是否存在、占位符策略这四项
    一旦建库就改不动了。所以 `--dry-run` 不是可选的礼貌，是默认应该先跑一遍的步骤。
    """
    settings = _bootstrap()
    if granularity not in ("fulltext", "abstract", "all"):
        raise typer.BadParameter(f"granularity 只能是 fulltext / abstract / all，收到 {granularity}")

    _log.info(
        "build_index_start",
        granularity=granularity,
        dry_run=dry_run,
        collection=settings.milvus.collection_name,
        index_version=settings.milvus.index_version,
        # 这四个参数打进日志是刻意的：它们建库后不可逆，出问题时第一件要查的事
        # 就是「当初这一库到底是用什么参数建的」。
        analyzer=settings.milvus.text_en_analyzer_params,
        bm25_k1=settings.milvus.bm25_k1,
        bm25_b=settings.milvus.bm25_b,
        strip_placeholders=settings.corpus.strip_placeholders,
        estimated_chunks=settings.corpus.estimated_total_chunks,
    )

    # TODO(接真实实现): 调 citequest.indexing.loader.build_index(settings, ...)
    #   —— 由 indexing 模块的负责人填。这里保持薄壳，不在 CLI 里写编排。
    raise NotImplementedError("build-index 待 citequest.indexing.loader 落地后接上")


@app.command("evaluate")
def evaluate(
    benchmark: Annotated[str, typer.Option(help="citrec / litsearch / all")] = "all",
    config: Annotated[
        str, typer.Option(help="消融档位：bm25_only / dense_only / hybrid / hybrid_rerank")
    ] = "hybrid_rerank",
) -> None:
    """跑评测并把报告写到 reports/。"""
    settings = _bootstrap()
    if benchmark not in ("citrec", "litsearch", "all"):
        raise typer.BadParameter(f"未知基准：{benchmark}")
    if config not in settings.eval.ablation_configs:
        raise typer.BadParameter(f"未知消融档位：{config}，可选 {settings.eval.ablation_configs}")

    _log.info(
        "evaluate_start",
        benchmark=benchmark,
        config=config,
        # 评测用的是加大到 100 的 group_limit（线上 60）。这个口径差异必须
        # 出现在日志和报告两处——只写在报告里，跑分时就没人注意得到。
        eval_group_limit=settings.retrieval.eval_group_limit,
        reports_dir=str(settings.eval.reports_dir),
    )

    # 在函数内 import：评测依赖走 PEP 735 的 eval 组（ranx / datasets / scipy），
    # 放模块顶层会让「没装评测依赖就连 `citequest serve` 都起不来」。
    from citequest.evaluation.report import EvalConfigSnapshot
    from citequest.evaluation.retrieval_eval import ABLATION_PRESETS, PipelineRunProducer

    preset = ABLATION_PRESETS[config]
    dataset_id = settings.eval.citrec_dataset_id if benchmark != "litsearch" else settings.eval.litsearch_dataset_id
    snapshot = EvalConfigSnapshot.from_settings(settings, dataset_id=dataset_id, split=settings.eval.citrec_split)
    _log.info("eval_config_snapshot", **snapshot.model_dump(mode="json"))

    # **接真实实现**：装配 `pipelines.factory.build_services(settings)`，把
    # `services.search` 传给 `PipelineRunProducer`，再依次调
    # `datasets.citrec.build_in_corpus_subset` → `headline_survives` →
    # `retrieval_eval.evaluate_dataset` → `retrieval_eval.split_by_layer` →
    # `report.write_report`。这里刻意停在产出器上：**产不出真实 run 就不该落一份报告**，
    # 而报告一旦落盘就会被引用。
    # 构造产出器本身就会把「这一档跑的是什么参数」打进日志（见其 __init__）。
    PipelineRunProducer(preset, settings)
    raise NotImplementedError(
        f"evaluate 待接入真实检索管线（preset={preset.name}，{preset.role}）："
        "把 build_services(settings).search 注入 PipelineRunProducer 即可打通"
    )


@app.command("serve")
def serve(
    reload: Annotated[bool, typer.Option(help="开发模式热重载")] = False,
    workers: Annotated[int, typer.Option(help="worker 数，见下方说明")] = 1,
) -> None:
    """起 FastAPI 服务。

    **默认单 worker，且多数情况下不该改**：并发闸门与 GPU 信号量是进程内状态
    （rerank 4 / 生成 12 / HHEM 8 / 路由 12），多 worker 会让「rerank 并发 4」
    实际变成 4×worker 数，把显存账直接打穿。要扩容就起多个容器，让闸门额度
    随实例数拆分。
    """
    settings = _bootstrap()
    if workers > 1 and reload:
        raise typer.BadParameter("--reload 与多 worker 不能同时用")
    if workers > 1:
        _log.warning(
            "multi_worker_semaphore_warning",
            workers=workers,
            # 不阻止，但必须在日志里留痕——这是会被误用的一个旋钮。
            note="并发闸门是进程内状态，实际并发 = 配置值 × workers",
            sem_generate=settings.concurrency.sem_generate,
        )

    import uvicorn

    _log.info("serve_start", host=settings.api.host, port=settings.api.port, workers=workers)
    # 用字符串路径而不是 import app 对象：热重载要求 uvicorn 自己能重新导入模块。
    uvicorn.run(
        "citequest.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=reload,
        workers=workers if not reload else None,
        log_config=None,  # 日志已由 configure_logging 接管，不让 uvicorn 再配一次
    )


@app.command("mcp")
def mcp() -> None:
    """起 MCP server（stdio only）。

    stdio 传输下 **stdout 是协议通道**，任何写到 stdout 的内容都会破坏 JSON-RPC
    的帧结构，而宿主端报出来的会是一个与日志内容毫无关系的解析错误——所以这是
    本命令唯一一处与其它子命令不同的地方，也是最容易踩的一处。
    """
    import asyncio

    settings = _bootstrap(log_stream=sys.stderr)
    _log.info("mcp_server_start", transport="stdio", tools=5)

    # 在函数内 import：`mcp_server.server` 顶层要 import 官方 `mcp` 包，
    # 放模块顶层会让「没装 mcp 依赖就连 `citequest --help` 都跑不起来」。
    from citequest.mcp_server.server import run_stdio

    asyncio.run(run_stdio(settings))


@app.command("show-config")
def show_config() -> None:
    """打印当前生效的全部参数。

    存在理由很具体：参数正本在代码里，但环境变量可以覆盖。出现「线上表现和设计
    文档对不上」时，第一件事就是确认到底哪几项被环境变量改掉了——靠读代码是查不出来的。
    """
    settings = _bootstrap()
    typer.echo(settings.model_dump_json(indent=2))


def main() -> None:
    """[project.scripts] 的入口函数。"""
    app()


if __name__ == "__main__":
    main()
