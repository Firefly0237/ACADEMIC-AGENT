"""HTTP 薄层：只做协议转换与依赖注入，业务逻辑一律下沉到 `pipelines/`。

- `main.py`       —— FastAPI app 工厂、lifespan、中间件、路由挂载
- `deps.py`       —— Milvus client / reranker / Settings 的依赖注入
- `errors.py`     —— 领域异常 → HTTP 状态码的映射（**只有这一层知道状态码**）
- `middleware.py` —— request_id 注入与访问日志
- `ratelimit.py`  —— 令牌桶：全局请求限流（429）与 AI 回答限流（超限降级成只给搜索结果）
- `sse.py`        —— SSE 事件帧的封装（事件名、心跳、缓冲控制头）
- `routes/`       —— 每个资源一个文件：search / answer / papers（含 facets）/ feedback / health
- `schemas/`      —— 请求与响应模型，与内部 domain 模型解耦

错误码本身定义在 `core.errors` 而不是这里，因为 MCP adapter 与 FastAPI 共用同一个
service 层：错误码定义在 api 层就意味着 MCP 那条路要么复制一份，要么静默地行为
不一致。留在本层的只有「协议细节」——HTTP 状态码、响应头、SSE 帧格式。

一条契约上的硬约定：**部分失败不用 HTTP 状态码表达**。SSE 一旦发出首帧，
状态码已经写进响应行，物理上改不了。所以「检索成功但 AI 失败」仍是 200，
失败以 SSE 事件下发。底线永远是「只返回搜索结果，不返回 AI 失败」。

本包**不在 `__init__` 里 import `main`**：`create_app()` 在模块导入期就会读配置、
建 Milvus 连接，而 `citequest.api.errors` 这类模块被测试单独 import 时不该顺带
起一整个应用。要拿 app 就显式 `from citequest.api.main import create_app`。
"""
