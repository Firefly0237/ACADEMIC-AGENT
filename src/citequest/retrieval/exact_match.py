"""精确匹配的检索层入口（M3 §4.3、D7）。

底层算子在 `indexing.expressions`（`TEXT_MATCH` / `PHRASE_MATCH`）。本模块只回答
一个检索层的问题：**这条 query 该不该走 sparse 单路，以及短语放宽到第几档。**

三个设计点（M3 §4.3）：

1. **`text_exact` 不用 english analyzer**：english analyzer 带 stemmer，而短语匹配
   是在 analyzer 输出的 token 序列上算位置的——用户输入的确切短语被词干化之后
   就不再「确切」了。所以精确匹配走一个独立的、不做词干化的字段。
2. **不做融合**：`TEXT_MATCH`/`PHRASE_MATCH` 只判命中不给分数，只能当 filter；
   而 RRF 只用名次不用分数。两者叠在一起，带引号的确切短语查询就**没有任何机制
   让「压倒性的精确命中」在排序上体现强度**——一个完全匹配和一个勉强的第一名
   在 RRF 里贡献完全相同，而搜索路又不开 rerank。解法是让这类 query 根本不进融合：
   **单路下量纲问题不存在**。
3. **不引入 Elasticsearch**：Milvus 内部已经是 Tantivy，再装一套倒排引擎意味着
   双写和两套分词器人肉对齐。精确匹配是 BM25 Function 的免费副产品，零新增组件。
"""

from __future__ import annotations

from citequest.config.milvus import MilvusSettings
from citequest.config.retrieval import RetrievalSettings
from citequest.core.query_plan import PhraseMatchSpec
from citequest.indexing.expressions import phrase_match_expr, text_match_expr

__all__ = ["MAX_SLOP", "build_phrase_filter", "build_required_terms_filter", "relax_slop"]

#: slop 上限 2（决策记录 §4.2）。0 = 严格相邻同序；1 = 中间允许插一个 token；
#: 2 = 允许词序反转或插两个。**没有第三档**——再放宽就不是「确切短语」了，
#: 而语义检索那一路本来就在处理「意思相近」。
MAX_SLOP = 2


def build_phrase_filter(spec: PhraseMatchSpec, milvus: MilvusSettings) -> str:
    """短语约束表达式。承载字段恒为 `text_exact`。

    字段名从配置读而不是写字面量：字段名一旦被改，写入端与读取端必须同时变，
    两处各写一份字面量就会出现「写进去了但查不出来」这类静默故障。
    """
    return phrase_match_expr(milvus.text_exact_field, spec.phrase, spec.slop)


def build_required_terms_filter(terms: str, milvus: MilvusSettings, retrieval: RetrievalSettings) -> str:
    """「必须包含某个词」的表达式（M3 §7.1 倒数第二行）。

    多词默认 **OR**（Milvus 语义），要 AND 只能写成
    `TEXT_MATCH(t,'a') and TEXT_MATCH(t,'b')`——**没有第三种写法**，
    也不要试图用单个调用表达 AND，那会静默退化成 OR 并放大召回。
    """
    return text_match_expr(milvus.text_exact_field, terms, logic=retrieval.text_match_default_logic)


def relax_slop(spec: PhraseMatchSpec) -> PhraseMatchSpec | None:
    """短语放宽一档，已在上限则返回 None。

    **放宽必须是显式的一步，不是自动重试链**：原设计里那套「短语三级放宽降级」
    是自建启发式，已被删掉（D7）。留下这个函数是为了让「用户主动点了放宽」
    这种产品动作有一个确定的实现，而不是让检索层自己偷偷放宽——
    偷偷放宽会让用户拿到一批他没要求的近似匹配，且没有任何提示。
    """
    if spec.slop >= MAX_SLOP:
        return None
    return PhraseMatchSpec(phrase=spec.phrase, slop=spec.slop + 1)
