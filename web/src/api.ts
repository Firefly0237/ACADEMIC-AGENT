export type Session = {
  authenticated: boolean;
  owner_id: string;
  csrf: string;
  project_id: string | null;
};
export type PaperRef = {
  kind: "metadata_projection" | "native_representation";
  id: string;
};
export type CSLReference = {
  id: string; type: string; title?: string;
  author?: Array<{ literal: string } | { given?: string; family: string }>;
  issued?: { "date-parts": number[][] };
  DOI?: string; URL?: string; "container-title"?: string;
  volume?: string; issue?: string; page?: string; publisher?: string;
};
export type Paper = {
  paper: PaperRef;
  title: string | null;
  abstract: string | null;
  authors: string[] | null;
  year: number | null;
  work_id: string | null;
  work_version_id: string | null;
  coverage: string;
  source_url: string | null;
  pdf_url: string | null;
  pdf_license?: "CC-BY-4.0" | null;
  pdf_is_unmodified?: true | null;
  citation_status?: string;
  csl?: CSLReference;
  citation_missing_fields?: string[];
};
export type SearchPage = {
  snapshot_id: string;
  generation_id: string;
  epoch: number;
  query: string;
  catalog: string;
  mode: string;
  sort: string;
  candidate_count: number;
  candidate_limit: number;
  results: {
    rank: number;
    score: number;
    paper: Paper;
    snippet: string;
    evidence_refs: string[];
  }[];
  next_cursor: string | null;
  facets: Record<string, Record<string, number>>;
  masked_since_search: number;
};
let csrf = "";
export class APIError extends Error {
  constructor(
    public code: string,
    public status: number,
  ) {
    super(code);
  }
}
export async function api<T>(
  path: string,
  body?: unknown,
  method?: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(path, {
    method: method || (body === undefined ? "GET" : "POST"),
    credentials: "same-origin",
    signal,
    headers: {
      ...(body === undefined
        ? {}
        : {
            "Content-Type": "application/json",
            "Idempotency-Key": crypto.randomUUID(),
          }),
      ...(csrf ? { "X-CSRF-Token": csrf } : {}),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const value = await response
    .json()
    .catch(() => ({ code: "invalid_server_response" }));
  if (!response.ok)
    throw new APIError(
      value.code || value.detail?.code || `http_${response.status}`,
      response.status,
    );
  return value;
}
export async function session() {
  const s = await api<Session>("/api/auth/session");
  csrf = s.csrf;
  return s;
}
export const paperPath = (ref: PaperRef) =>
  `/papers/${ref.kind}/${encodeURIComponent(ref.id)}`;
export const errorText = (e: unknown) =>
  e instanceof APIError
    ? {
        search_catalog_requires_published_full_generation:
          "文献索引尚未正式发布，当前无法检索。",
        login_required: "请先登录，再使用私人文献和 AI 任务。",
        session_required: "登录已过期，请重新进入。",
        csrf_required: "会话校验失败，请刷新后重试。",
        request_busy: "服务繁忙，请稍后重试。",
        task_profile_refresh_required: "执行版本已更新。请再次提交，页面会重新读取当前版本；不会自动重发或改变已存在任务。",
        task_kind_not_enabled_for_new_tasks: "此功能暂未开放新任务，历史结果仍可查看。",
        evidence_not_issued_to_task: "此引用不属于当前任务，不能读取原文。",
        evidence_ref_not_issued_to_task: "此引用不属于当前任务，不能读取原文。",
        evidence_unavailable_or_access_revoked: "原文已不可用或访问权限已撤回。",
        source_denied: "当前来源已不可读取。",
        material_access_revoked: "此材料的访问权限已撤回。",
        task_not_found: "任务不存在或当前账户无权读取。",
        agent_task_not_found: "任务不存在或当前账户无权读取。",
        agent_revision_conflict: "任务已在其他操作中更新，请重新打开后继续。",
        agent_attempt_unresolved: "本任务有尚未核对的调用，暂时不能继续。",
        execution_plan_revision_conflict: "计划已更新，请重新读取并核对当前版本，再提交决定。",
        execution_plan_approval_command_changed: "批准记录与当前执行命令不一致，请重新读取任务并核对。",
        execution_plan_decision_required: "请先在执行计划中明确批准当前版本，或提交检索词修改。",
        execution_plan_decision_not_admissible: "当前任务状态无法接收计划决定，请重新读取任务。",
        execution_plan_scope_changed: "计划绑定的档案或任务范围已变化，请重新读取任务并核对。",
        execution_plan_version_limit: "已达到计划版本上限。可以批准当前版本，或取消后重新明确要求。",
        execution_plan_query_invalid: "检索词不能为空，且每项最多 2000 个字符。",
        execution_plan_query_target_invalid: "修改项不属于当前计划，请重新读取并核对检索目标。",
        execution_plan_adjustment_no_change: "检索词没有实际变化，可以直接批准当前版本。",
        execution_plan_structure_changed: "计划步骤与当前任务不一致，暂不能执行，请重新读取任务。",
        task_catalog_pinned_runtime_required: "需要恢复与本任务匹配的执行服务后才能继续。",
      }[e.code] || "这次操作未能完成。请先刷新查看最新状态；若仍无法继续，请将问题提示和刚才的操作告知维护者。"
    : e instanceof Error
      ? e.message
      : "请求未完成";
