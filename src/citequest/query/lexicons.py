"""路由词表与类目映射的正本（M3 §1.2 / §3.3 / §7.1）。

**为什么词表单独成文件**：M3 §2.5 有一条硬纪律——词表、few-shot 示例、prompt、
阈值共同决定路由行为，**改动其中任何一项都必须换 `ROUTER_CONFIG_VERSION` 并触发
路由回归测试**。散在各判定函数里的字面量做不到「一眼看清改了哪个词」，
也没法用一个版本号把它们一起失效掉（路由缓存 key 就含这个版本号，M3 §5.2）。

**词表宽窄的偏置是有方向的**（M3 §3.6）：`METHOD_WORDS` 宁可宽也不做窄——
多触发一次 fulltext 过滤的代价是少召回（结果变旧变少），少触发一次的代价是
**拿 2024 年的摘要去编方法细节**，而那条路径上所有下游校验全部通过。
代价不对称，判据就必须偏置。

**词表必然漏，这里不声称完备**：漏掉的由 Level 2 的 `needs_fulltext` 兜，
再兜不住由 prompt 里的 granularity 标注兜——三层都是软的。
"""

from __future__ import annotations

__all__ = [
    "CATEGORY_ALIASES",
    "COMPARISON_WORDS",
    "METHOD_WORDS",
    "QUESTION_WORDS",
    "RECENCY_WORDS",
    "ROUTER_CONFIG_VERSION",
    "match_category",
    "scan_lexicon",
]

#: 路由配置版本号。**词表 / few-shot / prompt / 阈值任一改动都必须递增它**
#: （M3 §2.5、§5.2）：它进日志与 span，也是路由缓存 key 的一部分，
#: 不递增就会出现「新配置命中旧配置的缓存结果」——那是正确性 bug 不是命中率问题。
ROUTER_CONFIG_VERSION = "router-v1"

#: 深度信号（M3 §3.3 第 ① 类）：这些词面直接指向 Method / Experiment 段，
#: **摘要在物理上不含这些内容**，所以命中即 `granularity == "fulltext"` 硬过滤。
#: 中英一起收，因为语言判别只影响检索通道与回答语言，不影响这条判据。
METHOD_WORDS: frozenset[str] = frozenset(
    {
        # --- 英文 ---
        "loss function",
        "objective function",
        "hyperparameter",
        "hyperparameters",
        "learning rate",
        "batch size",
        "ablation",
        "ablation study",
        "implementation",
        "implementation detail",
        "implementation details",
        "pseudocode",
        "pseudo-code",
        "training setup",
        "training config",
        "training configuration",
        "optimizer",
        "architecture detail",
        "how does it work",
        "how is it implemented",
        # 表格/图指代：问「表 3 报了多少」必然要正文
        "table",
        "figure",
        "appendix",
        # --- 中文 ---
        "损失函数",
        "目标函数",
        "超参",
        "超参数",
        "学习率",
        "批大小",
        "消融",
        "实现细节",
        "伪代码",
        "训练配置",
        "优化器",
        "怎么实现",
        "如何实现",
        "报了多少",
        "具体做法",
    }
)

#: 时效信号（M3 §3.3 第 ② 类）：命中即落进「只有摘要」的 2023–2025 区间。
#: 它与深度信号**同时命中**时进入唯一无法靠过滤解决的组合（§3.4 第三行）：
#: 用户既要最新又要细节，而这两件事在本库里不同时存在。
RECENCY_WORDS: frozenset[str] = frozenset(
    {
        "latest",
        "recent",
        "recently",
        "state of the art",
        "state-of-the-art",
        "sota",
        "newest",
        "up to date",
        "最新",
        "近期",
        "目前",
        "当前",
        "现在",
    }
)

#: 疑问词（M3 §2.2 规则 5）。命中且 `token_count > 4` → SEARCH_PLUS_QA。
QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "what",
        "why",
        "how",
        "when",
        "which",
        "who",
        "does",
        "do",
        "is",
        "are",
        "can",
        "should",
        "什么",
        "为什么",
        "为何",
        "怎么",
        "如何",
        "哪些",
        "是否",
        "能否",
    }
)

#: 比较词（M3 §2.2 规则 4）。命中且 `token_count > 6` → SEARCH_PLUS_MULTIHOP。
#: 条件比单步 QA 更严是有意的：多跳是全链路最贵的一条，**宁可把该多跳的判成单步**
#: （回答略浅），也不要把单步问题判成多跳（成本翻倍且答案发散）。
COMPARISON_WORDS: frozenset[str] = frozenset(
    {
        "compare",
        "comparison",
        "versus",
        "vs",
        "vs.",
        "difference",
        "differences",
        "trade-off",
        "tradeoff",
        "对比",
        "比较",
        "区别",
        "差异",
        "取舍",
    }
)

#: 词面 → arXiv 类目映射（M3 §7.1）。
#: **映射不命中就不加过滤**，把词交给检索本身处理——静默猜一个类目会把用户
#: 明确没说的约束加上去，而加错的过滤是「结果变少但不报错」。
CATEGORY_ALIASES: dict[str, str] = {
    "nlp": "cs.CL",
    "natural language processing": "cs.CL",
    "computational linguistics": "cs.CL",
    "自然语言处理": "cs.CL",
    "cv": "cs.CV",
    "computer vision": "cs.CV",
    "计算机视觉": "cs.CV",
    "machine learning": "cs.LG",
    "机器学习": "cs.LG",
    "deep learning": "cs.LG",
    "深度学习": "cs.LG",
    "information retrieval": "cs.IR",
    "信息检索": "cs.IR",
    "artificial intelligence": "cs.AI",
    "人工智能": "cs.AI",
    "robotics": "cs.RO",
    "机器人": "cs.RO",
    "cryptography": "cs.CR",
    "security": "cs.CR",
    "安全": "cs.CR",
    "software engineering": "cs.SE",
    "软件工程": "cs.SE",
    "distributed systems": "cs.DC",
    "分布式系统": "cs.DC",
}


def scan_lexicon(lowered_text: str, lexicon: frozenset[str]) -> list[str]:
    """在**已归一化并小写**的文本里扫一遍词表，返回命中的词面。

    直接用子串包含而不是分词后精确匹配，理由是词表里本来就有多词短语
    （"loss function" / "state of the art"）和中文（无空格分词）——
    为它引一个分词器会让这一层从纯字符串操作变成有模型依赖的一层，
    而这一层的存在理由正是「p50 1ms、不付模型的钱」（M3 §1）。

    代价写出来：子串匹配会有假阳性（"table" 命中 "comfortable"）。这类误伤的
    方向是**多加一次 fulltext 过滤**，正好落在 M3 §3.6 判定的安全侧。

    :param lowered_text: 调用方保证已 `normalize_query` 且 `.lower()`，
        本函数不重复做——同一段文本只解析一次是这一层唯一的性能设计（M3 §1.2）。
    """
    return sorted(term for term in lexicon if term in lowered_text)


def match_category(lowered_text: str) -> str | None:
    """词面 → arXiv 类目。命中多个时取**词面最长**的那个。

    取最长是为了让 "natural language processing" 赢过它内部并不存在但同类的短词面，
    也让 "computer vision" 不被更短的通用词覆盖。命不中返回 None，
    调用方据此**不加过滤**，而不是塞一个默认类目（M3 §7.2 第 3 条）。
    """
    hits = [(alias, code) for alias, code in CATEGORY_ALIASES.items() if alias in lowered_text]
    if not hits:
        return None
    hits.sort(key=lambda pair: len(pair[0]), reverse=True)
    return hits[0][1]
