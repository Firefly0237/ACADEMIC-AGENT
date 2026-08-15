"""`/v1/answer` 的 SSE 全链路（M6 §2）。

三条不变量在这里被断言，它们比「回答内容对不对」重要得多：

1. **`meta` 永远是第一个事件**，且在任何下游调用之前就发——它同时是「代理没有
   缓冲」的探针；
2. **整条流永远是 HTTP 200**：SSE 一旦发出首帧，状态码已经写进响应行，物理上
   改不了，所以「检索成功但 AI 失败」只能是 200 + `ai_answer_error`；
3. **`search_results` 先于任何 AI 事件**——搜索结果是底线，AI 回答是增强。

骨架阶段生成模型未接（`ChatClient` 抛 NotImplementedError），所以这条流会走到
AI 失败分支——**而这恰恰是最该被测的那条**：它就是线上「AI 挂了」时用户看到的
东西，必须仍然拿得到搜索结果。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.e2e


def _events(client: TestClient, query: str) -> list[tuple[str, dict[str, object]]]:
    """把 SSE 报文解析成 `(event, data)` 列表。

    手工解析而不是引 sse client：这里要断言的正是**报文本身的时序与结构**，
    用一个替我做解析的库会把「事件顺序」这条不变量藏在库的行为里。
    """
    with client.stream("GET", "/v1/answer", params={"q": query}) as response:
        assert response.status_code == 200
        raw = "".join(response.iter_text())

    parsed: list[tuple[str, dict[str, object]]] = []
    # SSE 的帧分隔符是 CRLF CRLF（sse-starlette 按规范发 \r\n）。按 "\n\n" 切会
    # 得到一整块——这类解析错误的表现是「一个事件都没解析出来」，而不是报错。
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        name, payload = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                payload = line.removeprefix("data:").strip()
        if name and payload:
            try:
                parsed.append((name, json.loads(payload)))
            except json.JSONDecodeError:
                parsed.append((name, {"raw": payload}))
    return parsed


class TestSseInvariants:
    def test_meta_is_always_the_first_event(self, client: TestClient) -> None:
        events = _events(client, "what is rotary position embedding used for")
        assert events[0][0] == "meta"
        # meta 里必须带 request_id 与 index_version：这两个是排查一次请求的全部起点。
        assert events[0][1]["request_id"]
        assert events[0][1]["index_version"]

    def test_stream_is_http_200_even_when_ai_fails(self, client: TestClient) -> None:
        # 骨架阶段生成模型未接 → AI 路必然失败。整条流仍然是 200。
        names = [name for name, _ in _events(client, "what is rotary position embedding used for")]
        assert "ai_answer_error" in names or "ai_answer_done" in names

    def test_search_results_arrive_before_any_ai_event(self, client: TestClient) -> None:
        names = [name for name, _ in _events(client, "what is rotary position embedding used for")]
        assert "search_results" in names
        first_ai = next((i for i, n in enumerate(names) if n.startswith("ai_answer")), len(names))
        assert names.index("search_results") < first_ai

    def test_stream_ends_with_done(self, client: TestClient) -> None:
        # `done` 是流的终止标记：缺它的话客户端只能靠连接关闭来判断结束，
        # 而连接关闭与「服务端挂了」在客户端看来是同一件事。
        names = [name for name, _ in _events(client, "what is rotary position embedding used for")]
        assert names[-1] == "done"

    def test_search_results_are_paper_level(self, client: TestClient) -> None:
        events = dict(_events(client, "what is rotary position embedding used for"))
        papers = events["search_results"]["papers"]
        assert isinstance(papers, list)
        assert len(papers) == 2


class TestDegradeIsVisible:
    def test_ai_failure_still_leaves_the_user_with_search_results(self, client: TestClient) -> None:
        # 全局降级底线（K9）：**只返回搜索结果，不返回 AI 失败**。
        events = _events(client, "what is rotary position embedding used for")
        by_name = {name for name, _ in events}
        assert "search_results" in by_name
        errors = [data for name, data in events if name == "ai_answer_error"]
        if errors:
            # 失败必须带一个可读的状态，而不是一个空对象。
            assert errors[0].get("answer_state") or errors[0].get("error")
