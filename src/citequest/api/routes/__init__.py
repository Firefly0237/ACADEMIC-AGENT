"""端点定义，每个资源一个文件。

- `search.py`   —— `GET /v1/search`，纯检索，分页上限 3 页 × 20
- `answer.py`   —— `GET /v1/answer`，RAG 问答，SSE 流式
- `papers.py`   —— `GET /v1/papers/{paper_id}` 明细、`GET /v1/facets` 全库精确分面
- `feedback.py` —— `POST /v1/feedback`，👍/👎，body 必须带 request_id
- `health.py`   —— `/healthz` 存活、`/readyz` 就绪

路由函数里**只能出现三件事**：参数校验、调 `pipelines` 的一个函数、把结果
渲染成响应模型。出现第四件事就说明业务逻辑漏到 HTTP 层了。

`/healthz` 对匿名调用只返回存活信号，不泄露模型名、版本、endpoint 等配置
——健康检查端点通常不鉴权，把配置摊在上面等于免费送一份内网拓扑。
"""

from citequest.api.routes import answer, feedback, health, papers, search

#: 挂载顺序即注册顺序。集中在这里而不是在 `main.py` 里逐条 import，
#: 是为了让「这套 API 到底有哪些端点」在一个地方看得完。
ALL_ROUTERS = [search.router, answer.router, papers.router, feedback.router, health.router]

__all__ = ["ALL_ROUTERS"]
