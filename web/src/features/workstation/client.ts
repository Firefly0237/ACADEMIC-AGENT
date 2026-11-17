import { APIError } from "../../api";
import type { Session } from "../../api";

export const WORKSPACE = "/api/v3/workstation";

// The feature uses its supplied, current session instead of a second identity
// cache. No retries or model calls are hidden in this transport.
export async function workspaceRequest<T>(
  identity: Session,
  path: string,
  options: { body?: unknown; method?: string; signal?: AbortSignal } = {},
): Promise<T> {
  const response = await fetch(path, {
    credentials: "same-origin",
    method: options.method || (options.body === undefined ? "GET" : "POST"),
    signal: options.signal,
    headers: {
      "X-CSRF-Token": identity.csrf,
      ...(options.body === undefined
        ? {}
        : { "Content-Type": "application/json" }),
    },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  });
  const value = await response
    .json()
    .catch(() => ({ code: "invalid_server_response" }));
  if (!response.ok) {
    throw new APIError(
      value.code || value.detail?.code || `http_${response.status}`,
      response.status,
    );
  }
  return value as T;
}

export function workspaceError(error: unknown): string {
  if (error instanceof APIError) {
    const messages: Record<string, string> = {
      research_intake_migration_required:
        "研究档案服务尚未启用。当前输入仍保留在本页，请勿直接刷新。",
      brief_revision_conflict:
        "这个研究档案已有新版本。你的修改仍保留，请比较后继续。",
      revision_conflict: "条目已在其他地方修改，请重新读取后再保存。",
      idempotency_conflict:
        "这次操作的提交内容有冲突，请先刷新确认已保存内容。",
      login_required: "请登录后使用个人科研工作区。",
      session_required: "会话已过期，请重新登录。",
      csrf_required: "会话校验失败，请重新进入工作区。",
      same_origin_required: "请求未通过同源校验，请从产品页面重新操作。",
      research_brief_needs_input:
        "研究信息或冲突尚未补齐，可以先保存并继续补充。",
      not_found: "内容已不可用或你当前没有访问权限。",
      collection_cycle: "集合不能移入自身或它的子集合，请选择其他父集合。",
      collection_has_children:
        "这个集合仍有子集合，请先移动或处理子集合，再删除它。",
      item_deleted_use_restore:
        "这篇文献已移出收藏，请先恢复原条目，以保留笔记和分组关联。",
    };
    return (
      messages[error.code] ||
      "这项操作暂未完成，请稍后重试；你的未保存输入不会被自动覆盖。"
    );
  }
  return "连接中断，提交结果尚未确认。请先核对已保存内容，未保存输入仍保留在本页。";
}
