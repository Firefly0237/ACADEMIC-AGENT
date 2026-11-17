"""Complete Markdown projection of an already reviewed experimental proposal."""
from __future__ import annotations

from citequest.agents_v3.contracts import AgentError, digest

from .research_plan import ResearchPlan

EXPERIMENT_FIELDS = (
    ("hypothesis", "待检验假设"),
    ("data_and_access", "数据与获取条件"),
    ("method_and_independent_variables", "方法与自变量"),
    ("baselines_and_controls", "基线与对照"),
    ("metrics_and_units", "指标与单位"),
    ("repetitions_and_uncertainty", "重复实验与不确定性"),
    ("resources_and_upper_bound", "资源需求与上限"),
    ("risks_and_stopping_rules", "风险与停止条件"),
)


def render_experiment(plan: ResearchPlan, intake: dict, literature: list[dict]) -> str:
    """Retain every field and original Unicode; no summary or truncation pass."""
    lines = ["# 实验设计方案", "本方案为待实施的实验设计，尚未执行实验。"]
    if intake.get("explicit_unknowns"):
        lines.append("## 用户明确未知的信息")
        for field, value in intake["explicit_unknowns"].items():
            lines.append(f"**{field}**\n\n{value}")
    lines.append("## 研究问题与预期产出")
    for question in plan.questions:
        lines.extend((f"### {question.id} · {question.question}",
            "关联实验：" + "、".join(question.experiment_ids),
            "动机证据声明：" + "、".join(question.motivation_claim_ids),
            "**预期产出**\n\n" + "\n\n".join(question.expected_outputs),
            "**已知缺口**\n\n" + ("\n\n".join(question.known_gaps) or "未列出额外缺口。")))
    for experiment in plan.experiments:
        values = experiment.model_dump(mode="json")
        lines.extend((f"## {experiment.id} · 对应 {experiment.research_question_id}",
            "状态：proposed_not_executed（拟议，未执行）"))
        for field, title in EXPERIMENT_FIELDS:
            lines.append(f"### {title}\n\n{values[field]}")
        lines.append("### 证据引用\n\n" + ("\n\n".join(experiment.evidence_ref_ids)
            or "未绑定直接文献证据，以上内容按拟议设计理解。"))
    lines.extend(("## 可行性", plan.feasibility, "\n\n".join(plan.feasibility_reasons),
        "## 待验证的创新假设", "\n\n".join(plan.innovation_hypotheses) or "未提出额外创新假设。",
        "## 文献缺口", "\n\n".join(plan.literature_gaps) or "未列出额外文献缺口。"))
    if literature:
        lines.append("## 文献依据与限定条件")
        for item in literature:
            lines.append("### " + item["research_question_id"])
            for claim in item["answer"]["claims"]:
                lines.append(f"**{item['research_question_id']}:{claim['id']} · {claim['status']}** {claim['statement']}")
                for edge in claim["edges"]:
                    lines.extend((f"引用：{edge['ref_id']}（{edge['relation']}）", edge["quote"], edge["rationale"]))
                    lines.extend(edge["conditions"])
                lines.extend(claim["limitations"])
            lines.extend(item["answer"]["gaps"])
    return "\n\n".join(lines)


def experiment_result(upstream: dict) -> dict:
    """Project only the existing publication gate's complete reviewed result."""
    if (upstream.get("status") != "structured_plan_ready"
            or upstream.get("publication_status") != "validated_ai_review"
            or upstream.get("performed_experiments") is not False):
        raise AgentError("experiment_reviewed_research_plan_required")
    plan = ResearchPlan.model_validate(upstream["plan"], strict=True)
    intake = upstream["research_intake"]
    context = upstream["context_manifest"]
    if (plan.brief_sha256 != intake["brief_sha256"]
            or any(context.get(key) != intake[key] for key in ("brief_id", "brief_revision", "brief_sha256"))):
        raise AgentError("experiment_delivery_brief_identity_mismatch")
    return {**upstream, "status": "experiment_design_ready",
        "completion_status": "experiment_design_ready", "school_format_ready": False,
        "text": render_experiment(plan, intake, upstream["literature_ledger"]),
        "context_manifest": {**context, "experiment_delivery_protocol": "citequest-experiment-delivery/1",
            "experiment_plan_sha256": digest(upstream["plan"])}}
