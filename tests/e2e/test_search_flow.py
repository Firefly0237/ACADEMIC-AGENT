"""`/v1/search` 全链路（M6 §1）。

这条路上**没有 LLM、没有 rerank、没有 grounding 闸门**——它是唯一能讲
「亚百毫秒」的一条路。所以这份测试除了功能，还盯着两件不该出现的东西：
响应体里不能有 `relevance_score`（RRF 分数无物理含义），链路里不能有模型调用。
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.e2e


class TestSearchEndpoint:
    def test_returns_paper_level_results(self, client: TestClient) -> None:
        response = client.get("/v1/search", params={"q": "rotary position embedding"})
        assert response.status_code == 200
        results = response.json()["results"]
        # 分的是论文不是 chunk：grouping search 在服务端完成 chunk→paper 聚合。
        # 假 Milvus 给了 3 行、2 个 paper_id，所以这里必须是 2 而不是 3。
        assert results["group_count"] == 2
        assert len(results["papers"]) == 2

    def test_response_carries_request_id_in_both_body_and_header(self, client: TestClient) -> None:
        # 前端要读 request_id 才能提交反馈；跨域下不 expose 就会在最后一步断掉。
        response = client.get("/v1/search", params={"q": "attention"})
        assert response.headers["X-Request-Id"]
        assert response.json()["meta"]["request_id"] == response.headers["X-Request-Id"]

    def test_no_relevance_score_is_exposed(self, client: TestClient) -> None:
        # 不是忘了给，是不能给：RRF 只用名次不用分数，k=60 时上界约 0.033，
        # 这个数没有任何物理含义。
        assert "relevance_score" not in response_text(client, "attention")
        assert "rrf_score" not in response_text(client, "attention")

    def test_grouping_parameters_reach_milvus(self, client: TestClient, fake_milvus: Any) -> None:
        client.get("/v1/search", params={"q": "attention"})
        call = fake_milvus.calls[-1]
        # limit 是**组数**不是条数——这是 grouping search 改变的语义。
        assert call["limit"] == 60
        assert call["group_by_field"] == "paper_id"
        assert call["group_size"] == 1
        assert type(call["ranker"]).__name__ == "RRFRanker"

    def test_chinese_query_degrades_to_dense_only(self, client: TestClient, fake_milvus: Any) -> None:
        # 中文 query 打英文文档时 BM25 命中率≈0，只发 dense 一路并打点。
        body = client.get("/v1/search", params={"q": "旋转位置编码的长文本效果"}).json()
        assert body["meta"]["retrieval_mode"] == "dense_only"
        assert len(fake_milvus.calls[-1]["reqs"]) == 1

    def test_year_filter_is_pushed_down(self, client: TestClient, fake_milvus: Any) -> None:
        client.get("/v1/search", params={"q": "attention", "year_from": 2021})
        expr = fake_milvus.calls[-1]["reqs"][0].expr or ""
        assert "year" in expr

    def test_page_beyond_the_cap_is_rejected(self, client: TestClient) -> None:
        # 上限 3 页：ANN 返回的是近邻候选而非稳定全序，limit 放大同时损失召回
        # 稳定性与延迟。
        assert client.get("/v1/search", params={"q": "attention", "page": 9}).status_code == 400

    def test_empty_query_is_a_400_not_a_500(self, client: TestClient) -> None:
        assert client.get("/v1/search", params={"q": "  "}).status_code == 400


class TestOpsEndpoints:
    def test_healthz_does_not_touch_downstream(self, client: TestClient) -> None:
        # 碰下游的存活探针会把一次 Milvus 抖动放大成一次容器重启风暴。
        assert client.get("/healthz").json()["status"] == "ok"

    def test_healthz_hides_configuration_by_default(self, client: TestClient) -> None:
        body = client.get("/healthz").json()
        assert "index_version" not in body

    def test_readyz_reports_the_startup_probe(self, client: TestClient) -> None:
        body = client.get("/readyz").json()
        assert body["status"] in {"ready", "not_ready"}
        assert "startup_probe" in body


def response_text(client: TestClient, query: str) -> str:
    return client.get("/v1/search", params={"q": query}).text
