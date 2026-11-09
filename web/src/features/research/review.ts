export type ReviewSource = {
  source_task_id: string;
  source_command_id: string;
  command_seq: number;
  event_seq: number;
  task_revision: number;
  document_sha256: string;
  context_sha256: string;
};
export type ReviewReportSource = {
  task_id: string; command_id: string; command_seq: number; event_seq: number; task_revision: number;
  report_sha256: string; context_sha256: string;
};
export type RevisionRequest = { report: ReviewReportSource; finding_keys: string[] };

/** Retain the fixed source context, never inherit a previous operation. */
export function sourceBriefFields(brief: object): Record<string, unknown> {
  const source = brief as Record<string, unknown>;
  const fields = ["schema_version", "kind", "requested_direction", "objective", "catalog", "selected_papers",
    "constraints", "unknowns", "research_brief", "selected_skills", "research_draft_files"];
  return Object.fromEntries(fields.filter(field => Object.hasOwn(source, field)).map(field => [field, source[field]]));
}

export function revisionSelection(findings: ReadonlyArray<{ key: string; section_ids: readonly string[] }>, keys: readonly string[]) {
  if (!keys.length || keys.length > 8 || new Set(keys).size !== keys.length)
    return { valid: false, section_ids: [] as string[], reason: "请选择 1–8 条意见。" };
  const byKey = new Map(findings.map(finding => [finding.key, finding]));
  if (keys.some(key => !byKey.has(key))) return { valid: false, section_ids: [] as string[], reason: "所选意见不属于此评审报告。" };
  const sections = [...new Set(keys.flatMap(key => [...byKey.get(key)!.section_ids]))];
  if (!sections.length || sections.length > 2)
    return { valid: false, section_ids: sections, reason: "本次定向修订最多涉及两个章节，请分次选择。" };
  return { valid: true, section_ids: sections, reason: "" };
}

export function sameReviewSource(left: ReviewSource, right: ReviewSource): boolean {
  return (["source_task_id", "source_command_id", "command_seq", "event_seq", "task_revision", "document_sha256", "context_sha256"] as const)
    .every(key => left[key] === right[key]);
}

export function canRestartResearchReview(task: {
  state: string; reserved_ncny?: number; result_availability?: string;
}): boolean {
  return task.reserved_ncny === 0 && (["failed", "cancelled"].includes(task.state)
    || (task.state === "partial" && task.result_availability === "available_diagnostics"));
}

function ordered(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(ordered);
  if (value && typeof value === "object") return Object.fromEntries(
    Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
      .map(([key, entry]) => [key, ordered(entry)]),
  );
  return value;
}

/** RFC 9562 section 5.5: namespace bytes, UTF-8 name, SHA-1, UUIDv5 bits. */
export async function uuid5(name: string, namespace = "6ba7b811-9dad-11d1-80b4-00c04fd430c8"): Promise<string> {
  const namespaceHex = namespace.replaceAll("-", "");
  if (!/^[a-f0-9]{32}$/i.test(namespaceHex)) throw new Error("Invalid UUID namespace");
  const namespaceBytes = Uint8Array.from(namespaceHex.match(/../g)!, byte => Number.parseInt(byte, 16));
  const nameBytes = new TextEncoder().encode(name);
  const input = new Uint8Array(namespaceBytes.length + nameBytes.length);
  input.set(namespaceBytes); input.set(nameBytes, namespaceBytes.length);
  const bytes = new Uint8Array(await crypto.subtle.digest("SHA-1", input)).slice(0, 16);
  bytes[6] = (bytes[6]! & 0x0f) | 0x50;
  bytes[8] = (bytes[8]! & 0x3f) | 0x80;
  const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

/** This is an idempotency name, never an authorization token. */
export async function reviewOperationId(ownerId: string, body: unknown, retryAfterTaskId?: string): Promise<string> {
  const jsonBody = JSON.parse(JSON.stringify(body));
  return uuid5("citequest/research-review/1\n" + JSON.stringify(ordered({
    owner_id: ownerId, body: jsonBody, retry_after_task_id: retryAfterTaskId || null,
  })));
}
