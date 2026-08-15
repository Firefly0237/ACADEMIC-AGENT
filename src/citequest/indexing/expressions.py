"""Milvus 布尔表达式构造：标量过滤 + TEXT_MATCH + PHRASE_MATCH（决策记录 D7，M2 §8）。

放在 indexing 而不是 retrieval，是因为表达式的正确性完全依赖 **schema 里的字段名
与 analyzer 配置**：`PHRASE_MATCH` 作用在 `text_exact` 上而不是 `text_en` 上，正是
因为前者用 whitespace + alphanumonly（stemmer 会让短语匹配行为与用户直觉不一致）。
把它放在检索层就等于让检索层去记 schema 细节。

**这套算子替换掉的是四块自建代码**：布尔 AST 解析器、集合交并差、自写词距离、
短语三级放宽降级。而且它们的语义是**错的**——布尔 NOT 用集合差集实现，只能排除
恰好落进右侧 top-N 的论文，用户明确要求排除的照样返回。改用官方算子顺带修掉了
这个语义 bug。

**不引入 Elasticsearch**：Milvus 内部已经是 Tantivy，再装一套倒排引擎意味着双写、
两套分词器人肉对齐，性价比最差，且与「不新增有状态服务」的判据一致。

一条必须记住的边界：**TEXT_MATCH 只判命中不给分数**，只能当 filter 不能当 ranker。
排序由 D6 的单路 BM25 分数承担。
"""

from __future__ import annotations

from citequest.config.milvus import MilvusSettings
from citequest.core.enums import Granularity
from citequest.core.query_plan import PhraseMatchSpec, QueryFilters

__all__ = [
    "build_filter_expression",
    "combine_expressions",
    "escape_literal",
    "phrase_match_expr",
    "text_match_expr",
]

#: PHRASE_MATCH 需要 Milvus 2.5.17+；本项目底线已是 2.6.x，满足。
_MAX_SLOP = 2


def escape_literal(value: str) -> str:
    """转义字符串字面量里的反斜杠与双引号。

    不转义的后果不只是语法错误——一个带引号的 query 能改写整个过滤表达式的结构，
    这是标准的表达式注入面。用户输入会经由引号短语路由直达 PHRASE_MATCH，所以
    **这个函数是必经之路，不是可选的整洁措施**。
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def text_match_expr(field: str, terms: str, *, logic: str = "OR") -> str:
    """`TEXT_MATCH(field, 'terms')`。

    Milvus 的多词默认是 **OR**；要 AND 只能写成
    `TEXT_MATCH(t,'a') and TEXT_MATCH(t,'b')`——**没有第三种写法**，也不要试图用
    单个调用表达 AND，那会静默退化成 OR 并放大召回。
    """
    tokens = [t for t in terms.split() if t]
    if not tokens:
        raise ValueError("TEXT_MATCH 的词项不能为空")
    if logic.upper() == "AND":
        return " and ".join(f'TEXT_MATCH({field}, "{escape_literal(t)}")' for t in tokens)
    if logic.upper() != "OR":
        raise ValueError(f"logic 只支持 OR / AND，收到 {logic}")
    return f'TEXT_MATCH({field}, "{escape_literal(" ".join(tokens))}")'


def phrase_match_expr(field: str, phrase: str, slop: int = 0) -> str:
    """`PHRASE_MATCH(field, 'phrase', slop)`。

    slop 语义（官方）：0 = 严格相邻同序；1 = 允许中间插一个 token；2 = 允许词序反转
    或插两个。短语命中是在 **analyzer 输出的 token 序列**上算位置的，所以承载字段
    必须是 `text_exact`——用 `text_en` 的话 stemmer 会把 "training" 归一成 "train"，
    用户输入的确切短语反而匹配不上他自己看到的原文。
    """
    if not phrase.strip():
        raise ValueError("PHRASE_MATCH 的短语不能为空")
    if not 0 <= slop <= _MAX_SLOP:
        raise ValueError(f"slop 只允许 0–{_MAX_SLOP}（决策记录 §4.2），收到 {slop}")
    return f'PHRASE_MATCH({field}, "{escape_literal(phrase)}", {slop})'


def combine_expressions(*parts: str | None) -> str:
    """用 `and` 串联非空片段，自动加括号。

    加括号不是风格：`a or b` 与 `c` 直接用 `and` 连起来会变成 `a or (b and c)`，
    这类优先级错误产生的是**看起来合理但范围错了的结果集**，不是报错。
    """
    clauses = [f"({p})" for p in parts if p and p.strip()]
    return " and ".join(clauses)


def build_filter_expression(
    filters: QueryFilters,
    milvus: MilvusSettings,
    *,
    phrase: PhraseMatchSpec | None = None,
    cited_open_alex_id: str | None = None,
) -> str:
    """把 `QueryFilters` 下推成一条 Milvus 过滤表达式。

    **加过滤不是免费的**：过滤选择率一旦很低，HNSW 的图导航会走不通并退化成暴力
    扫描（官方经验边界是 filter ratio 低于 85% 时图索引仍优于 IVF 系列）。所以这里
    只生成**真的需要**的子句——`granularity=ANY` 时不写 granularity 条件，而不是写一个
    恒真的 `granularity in ["fulltext","abstract"]`，后者会让优化器多做一次无用的
    位图运算。

    :param cited_open_alex_id: 「别人在什么语境下引用 X」的过滤（M1 §7.2）。
        它是一次**标量等值过滤 + 一次正常混合检索**，没有任何新机制——被引论文
        不必在库内，25.3% 库内引用边这个约束直接消失。**接真实实现：数组字段
        `cited_open_alex_ids` 的 `ARRAY_CONTAINS` 支持形态须按官方文档核实（M1 §7.3
        选项 A）；核实不通过就走选项 B——先查 cite_edges 边表拿 chunk_id 集合，
        再拼成 `chunk_id in [...]` 下推。**
    """
    parts: list[str] = []

    if filters.granularity is not Granularity.ANY:
        parts.append(f'granularity == "{filters.granularity.value}"')

    # year 同时是 partition key：带年份过滤时命中的是物理分区而不是全库扫。
    if filters.year_from is not None:
        parts.append(f"year >= {filters.year_from}")
    if filters.year_to is not None:
        parts.append(f"year <= {filters.year_to}")

    if filters.primary_category:
        parts.append(f'primary_category == "{escape_literal(filters.primary_category)}"')

    if phrase is not None:
        parts.append(phrase_match_expr(milvus.text_exact_field, phrase.phrase, phrase.slop))

    if cited_open_alex_id:
        parts.append(f'ARRAY_CONTAINS(cited_open_alex_ids, "{escape_literal(cited_open_alex_id)}")')

    # `exclude_retracted` 目前没有对应的标量字段：撤稿是**软删**，标记落在论文级
    # 元数据（单机关系库）而不是 chunk 上，因为撤稿要按 paper_id 精确失效缓存
    # （M1 §8.3，缓存失效纪律里唯一的例外），而 chunk 级字段改一次就是一次全量重写。
    # 过滤在应用侧按 paper_id 黑名单做，不下推——名单是小集合，下推成一个巨大的
    # `not in [...]` 反而会让表达式本身成为瓶颈。
    return combine_expressions(*parts)
