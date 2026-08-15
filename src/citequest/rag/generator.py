"""答案生成：prompt 组装、生成调用、后处理（M4 §3、§4、§7）。

**这一层不生产知识，只组织和约束证据。** 系统对外承诺的是「回答里的每一句都能
追到一段可点击核对的原文」，**不承诺论文本身说的是对的**——本系统保证并度量
faithfulness（回答有没有超出给定证据），**不保证 factuality**（回答说的事在真实
世界里对不对），而且明说不保证。

理由是结构性的：语料就是论文本身，论文是**一手证据不是权威事实**——学术界的常态
就是后来的工作推翻先前的结论。要判 factuality 需要一个「正确答案」的外部权威源，
而在计算机科学这个领域这个源不存在。**一个声称能判学术论文事实性的检索系统，
是在声称自己比这个学科更权威。**

采样参数（M4 §4，全部【设计】值，M7 实测后回填）：

| 参数 | 值 | 理由 |
|---|---|---|
| `temperature` | 0.0 | 事实型问答的强共识；**且首句闸门要求可复现**——温度 >0 会让同一条 query 的首句每次都不一样，闸门通过率不可复现，线上出现「同样的问题有时给答案有时说信息不足」且无法调试 |
| `top_p` | 1.0（显式） | 避免两个采样旋钮同时拧 |
| `max_tokens` | 1500 / 3000（decompose 路） | **防御性上限，不是长度控制手段**；长度由自然语言指令控制 |
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from citequest.config.rag import GenerationSettings
from citequest.core.answer import Evidence
from citequest.core.enums import Granularity, RouteType
from citequest.llm.chat import ChatClient, ChatMessage, Deadline
from citequest.observability.logging import get_logger
from citequest.rag.context import ContextPack, assemble_context
from citequest.rag.prompts.answer import (
    ANSWER_SYSTEM_PROMPT,
    INSUFFICIENT_EVIDENCE_TEMPLATE_EN,
    INSUFFICIENT_EVIDENCE_TEMPLATE_ZH,
    USER_SECTION_TEMPLATE,
)

__all__ = ["AnswerGenerator", "GenerationRequest", "build_insufficient_answer"]

_logger = get_logger(__name__)

#: decompose 路的 max_tokens（M4 §4）：子问题 ≤4，每个子问题的综合陈述需要更多空间。
#: 分阶段设预算是强共识（STORM 对话模拟 500 / 大纲 400 / 正文段落 700 / 润色 4000；
#: OpenScholar 主答 3000 / 反馈重写 2000）。
MULTIHOP_MAX_TOKENS = 3_000


@dataclass(slots=True)
class GenerationRequest:
    """一次生成的全部输入。

    `query` 是**原始 query 原文**，不是 HyDE 产物也不是改写结果（M4 §3.1 第 3 条）：
    把 HyDE 的假设文档塞进生成 prompt 等于让模型看到一段自己编的、没有出处的
    「参考文本」——**那是在幻觉源头上浇油**。
    """

    query: str
    evidences: list[Evidence]
    route: RouteType = RouteType.SEARCH_PLUS_QA
    language: str = "EN"
    coverage_note: str | None = None
    """granularity 路由第三种处置的边界声明（M3 §3.4）。非空时进 system 区——
    它是回答的一部分，不是免责声明。"""


class AnswerGenerator:
    """组装 prompt 并驱动流式生成。

    **它只负责产出 token 流，不做切句也不做校验**——那是 `rag.streaming` 与
    `rag.grounding` 的事。分开的判据是：换一个推理后端不应该顺带换掉切句与闸门。
    """

    def __init__(self, chat: ChatClient, settings: GenerationSettings) -> None:
        self._chat = chat
        self._settings = settings

    def build_messages(self, request: GenerationRequest, context: ContextPack) -> list[ChatMessage]:
        """三区分离的消息列表（M4 §3.1）。

        - System 区 100% 由系统生成，**永不含检索内容**；
        - Evidence 区带显式声明与 `<<< >>>` 定界符；
        - User 区放原始 query，紧邻生成起点。

        Evidence 区放进 system 消息还是 user 消息？这里放 **user 消息的前半段**：
        放进 system 会让「指令区永不含检索内容」这条不变量在实现层被打破，
        而那条不变量正是注入防护第 1 道的定义。
        """
        system = ANSWER_SYSTEM_PROMPT.format(language="Chinese" if request.language == "ZH" else "English")
        if request.coverage_note:
            # 覆盖边界声明进 system 区：它是系统对能力边界的陈述，不是检索内容。
            system = f"{system}\nCoverage note you must state in the answer: {request.coverage_note}\n"
        if context.has_abstract_evidence:
            # 第二道防线：过滤只能拦住已识别的情况，逐块 granularity 标注 + 这一句
            # 才是兜住漏网之鱼的那层（M3 §3.4 实现约束 1）。
            system = f"{system}\nSome evidence blocks are abstract-level. Do not derive method details from them.\n"

        user = f"{context.evidence_text}\n\n{USER_SECTION_TEMPLATE.format(query=request.query)}"
        return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]

    async def stream(
        self,
        request: GenerationRequest,
        *,
        deadline: Deadline | None = None,
    ) -> AsyncIterator[str]:
        """流式生成，逐 token 产出。

        非流式在这里不提供：约 300 字 @30–50 tok/s 要 6–10s，用户等完整回答的体验
        不可接受，而首 token 预算是 p50 ≤1.0s。**没有第二条代码路径就不会有
        「流式与非流式行为不一致」这类 bug。**
        """
        context = assemble_context(request.evidences, self._settings)
        messages = self.build_messages(request, context)
        max_tokens = (
            MULTIHOP_MAX_TOKENS if request.route is RouteType.SEARCH_PLUS_MULTIHOP else self._settings.max_tokens
        )

        _logger.info(
            "generation_started",
            route=request.route.value,
            evidences=len(context.evidences),
            est_tokens=context.estimated_tokens,
            truncated=context.truncated,
            prompt_version=self._settings.prompt_version,
        )
        async for token in self._chat.stream(
            messages,
            temperature=self._settings.temperature,
            max_tokens=max_tokens,
            deadline=deadline,
        ):
            yield token


def build_insufficient_answer(
    query: str,
    evidences: list[Evidence],
    *,
    language: str = "EN",
    reasons: list[str] | None = None,
) -> str:
    """「信息不足」回答（M4 §7.2）。

    三条设计要点：

    1. **说清「为什么不足」而不只是「不足」**——「只有摘要粒度」「本库无 2023 年后
       全文层」是**可行动的信息**，「抱歉我不知道」不是；
    2. **仍然给出检索到的东西 + 引用**，让用户自己判断够不够用。这与「把 factuality
       判断权交还用户」是同一个原则；
    3. `answer_state=insufficient_evidence` 是**正常状态不是错误**，
       前端按正常卡片渲染并标注，不弹错误提示。

    它由调用方在**首句闸门之前**决定是否走这条路——此时尚未推送任何 token，
    「不展示」才是可执行的。
    """
    template = INSUFFICIENT_EVIDENCE_TEMPLATE_ZH if language == "ZH" else INSUFFICIENT_EVIDENCE_TEMPLATE_EN

    lines: list[str] = []
    for evidence in evidences[:3]:
        level = "摘要级" if evidence.granularity is Granularity.ABSTRACT else "全文级"
        if language != "ZH":
            level = "abstract-level" if evidence.granularity is Granularity.ABSTRACT else "full-text"
        snippet = evidence.text.strip().replace("\n", " ")[:140]
        lines.append(f"- [{evidence.citation_id}] {evidence.title}（{evidence.year}，{level}）：{snippet}…")
    if not lines:
        lines.append("- （本次检索没有返回任何符合过滤条件的结果）")

    reason_lines = reasons or _default_reasons(evidences, language)
    return template.format(
        query=query,
        evidence_lines="\n".join(lines),
        reasons="\n".join(f"- {r}" for r in reason_lines),
    )


def _default_reasons(evidences: list[Evidence], language: str) -> list[str]:
    """按证据的实际形态推出原因，**不写死一句通用套话**。

    这里只用**确定性事实**（这批证据是不是全是摘要、有没有结果），不做任何猜测：
    猜错的原因比不给原因更糟——它会把用户引向一个不存在的问题。
    """
    if not evidences:
        return (
            ["过滤条件下没有命中任何论文，可能是年份或领域条件过窄"]
            if language == "ZH"
            else ["No paper matched the filters; the year or category constraint may be too narrow"]
        )
    if all(e.granularity is Granularity.ABSTRACT for e in evidences):
        return (
            ["这个问题需要方法级证据，而符合条件的论文在本库中只有摘要粒度（2023 年后无全文层）"]
            if language == "ZH"
            else [
                "This question needs method-level evidence, but the matching papers are abstract-only "
                "in this corpus (no full text after 2022)"
            ]
        )
    return (
        ["检索到的内容与问题的直接相关度不足"]
        if language == "ZH"
        else ["The retrieved passages are not directly relevant enough to the question"]
    )
