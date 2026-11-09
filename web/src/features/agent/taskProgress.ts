export type PublicProgressEvent = {
  seq?: number; event_type?: string; type?: string; kind?: string;
  payload?: Record<string, unknown>; data?: Record<string, unknown>;
};
export type ProgressBudget = {
  state: string; model_calls?: number; tool_calls?: number; known_ncny?: number; reserved_ncny?: number;
  profile?: { model_calls?: number; tool_calls?: number };
};

const responsibilities: Record<string, string> = {
  qa5_plan: "问题理解与检索规划", qa37_review_goals: "独立需求审查",
  qa5_author: "基于证据回答", qa5_review: "独立回答审查",
  research_outline_write: "研究问题规划", research_outline_review: "独立提纲审查",
  research_literature: "文献研究", research_literature_search: "文献检索",
  research_literature_write: "文献论据撰写", research_literature_review: "独立文献审查",
  research_methods: "实验方法设计", research_review_unit: "独立研究单元审查",
  research_review_global: "跨方案一致性审查", research_review_evidence: "文献依据评审",
  research_review_methods: "实验设计与统计方法评审", research_review_structure: "可行性与结构评审",
  research_review_aggregate: "评审协调", school_write_section: "研究稿件撰写",
  school_review_section: "独立章节审稿", school_review_cross: "跨章节一致性审查",
  school_targeted_reviser: "定向修订", school_resolve_findings: "独立判断采纳问题是否解决",
  decisionrag_navigation: "证据导航", cross_encoder_rerank: "检索结果精排",
};
export const eventType = (event: PublicProgressEvent) => event.event_type || event.type || event.kind || "";
export const eventPayload = (event: PublicProgressEvent) => event.payload || event.data || {};
export function money(value?: number): string {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? `¥${(value / 1e9).toFixed(4)}` : "—";
}
export function progressContext(events: readonly PublicProgressEvent[]) {
  const tools = new Map<string, string>();
  for (const event of events) {
    const payload = eventPayload(event);
    if (eventType(event) === "tool_plan_bound" && typeof payload.tool_execution_id === "string"
        && typeof payload.name === "string" && payload.name.length <= 160)
      tools.set(payload.tool_execution_id, payload.name);
  }
  const latestStage = [...events].reverse().find(event => eventType(event) === "workflow_progress");
  const phase = latestStage && eventPayload(latestStage).phase;
  return {
    latestStage,
    responsibility: typeof phase === "string" ? responsibilities[phase] : undefined,
    rows: events.map(event => {
      const payload = eventPayload(event);
      const toolId = typeof payload.tool_execution_id === "string" ? payload.tool_execution_id : payload.attempt_id;
      const tool = typeof toolId === "string" ? tools.get(toolId) : undefined;
      const usage = eventType(event) === "attempt_settled" && payload.kind === "model"
        && payload.usage && typeof payload.usage === "object" ? payload.usage as Record<string, unknown> : undefined;
      const tokens = usage && Number.isSafeInteger(usage.input) && Number.isSafeInteger(usage.output)
        && (usage.input as number) >= 0 && (usage.output as number) >= 0
        ? { input: usage.input as number, output: usage.output as number } : undefined;
      return { event, tool, tokens, kind: payload.kind === "model" ? "模型" : payload.kind === "tool" || tool ? "工具" : undefined };
    }),
  };
}
