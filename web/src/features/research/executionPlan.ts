export type ExecutionPlanNode = {
  id: string;
  operation: "retrieve_review" | "design_method" | "assemble_review" | "write_review";
  target_id: string;
  goal: string;
  query?: string;
  depends_on: string[];
};

export type ExecutionPlan = {
  protocol: "approval1";
  version: number;
  task_id: string;
  origin_command_id: string;
  brief_sha256: string;
  outline_sha256: string;
  scope_sha256: string;
  parent_sha256: string | null;
  nodes: ExecutionPlanNode[];
  resource_requirement: Record<string, unknown>;
  query_changes: Array<{ target_id: string; before: string; after: string }>;
};

export type PlanApprovalRequest = {
  kind: "execution_plan_approval_required";
  plan_sha256: string;
  plan: ExecutionPlan;
};

export type PlanDecision = {
  plan_sha256: string;
  action: "approve" | "adjust_queries";
  queries?: Record<string, string>;
};

export type ApprovedExecutionPlan = {
  plan: ExecutionPlan;
  plan_sha256: string;
  decision: { command_id: string; operation_id: string; action: "approve" };
};

export function pendingExecutionPlan(result?: { interrupts?: unknown[] } | null): PlanApprovalRequest | undefined {
  return result?.interrupts?.find((value): value is PlanApprovalRequest =>
    !!value && typeof value === "object" && "kind" in value
    && value.kind === "execution_plan_approval_required",
  );
}

/** Only query edits are submitted; goals, dependencies and resource scope stay server-owned. */
export function changedPlanQueries(plan: ExecutionPlan, edits: Record<string, string>): Record<string, string> {
  return Object.fromEntries(plan.nodes
    .filter(node => node.operation === "retrieve_review" && edits[node.target_id] !== undefined
      && edits[node.target_id].trim() !== (node.query || ""))
    .map(node => [node.target_id, edits[node.target_id].trim()]));
}

export const planOperationLabels: Record<ExecutionPlanNode["operation"], string> = {
  retrieve_review: "检索与证据审查",
  design_method: "设计实验方法",
  assemble_review: "汇总与一致性审查",
  write_review: "逐节写作与审查",
};
