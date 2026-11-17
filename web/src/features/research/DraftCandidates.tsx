import { useRef, useState, type ReactNode } from "react";
import { api, APIError, type Session } from "../../api";
import { WORKSPACE, workspaceRequest, workspaceError } from "../workstation/client";
import { FIELD_LABELS, type BriefSnapshot, type DraftExtraction } from "./types";

export function DraftCandidates({ identity, taskId, briefId, result, evidence, onSaved }: {
  identity: Session;
  taskId: string;
  briefId: string;
  result: DraftExtraction;
  evidence: (ref: string) => ReactNode;
  onSaved?: (value: BriefSnapshot) => void;
}) {
  const [choices, setChoices] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  const pending = useRef<{ signature: string; body: Record<string, unknown>; revision: number } | null>(null);
  const candidates = result.candidates || [];
  const fields = [...new Set(candidates.map(item => item.field))];
  async function save() {
    setBusy(true); setError("");
    try {
      const choiceSignature = JSON.stringify(choices);
      if (pending.current?.signature === choiceSignature) {
        const latest = await workspaceRequest<BriefSnapshot>(identity, `${WORKSPACE}/briefs/${briefId}`);
        if (latest.brief.revision > pending.current.revision) {
          // A lost response may have committed. Replaying the exact old command
          // either returns its current projection or conflicts without applying
          // an old patch to a new revision; do not manufacture a new operation.
          const recovered = await workspaceRequest<BriefSnapshot>(identity, `${WORKSPACE}/briefs/${briefId}`, {
            method: "PATCH", body: pending.current.body
          });
          setSaved(true); onSaved?.(recovered); return;
        }
      }
      // Re-read the same guarded Task before applying a choice; do not trust a
      // historical browser result after source/Brief/Skill/Memory revocation.
      const current = await api<{ result_availability?: string; result?: DraftExtraction; brief?: {
        research_brief?: { id: string; revision: number; sha256: string }
      } }>(`/api/agent/tasks/${taskId}`);
      if (current.result_availability !== "available" || current.brief?.research_brief?.id !== briefId)
        throw new APIError("research_candidates_no_longer_current", 409);
      const selected = fields.flatMap(field => {
        const candidate = candidates.find(item => item.id === choices[field]);
        if (!candidate) return [];
        const actual = current.result?.candidates?.find(item => item.id === candidate.id);
        if (!actual || JSON.stringify(actual) !== JSON.stringify(candidate))
          throw new APIError("research_candidates_changed", 409);
        return [{ field, value: candidate.value, intent: "suggest" }];
      });
      if (!selected.length) throw new Error("请先选择希望保存的候选信息。");
      const document = await workspaceRequest<BriefSnapshot>(identity, `${WORKSPACE}/briefs/${briefId}`);
      const ref = current.brief!.research_brief!;
      if (document.brief.revision !== ref.revision || document.brief_sha256 !== ref.sha256)
        throw new APIError("brief_revision_conflict", 409);
      // One batch preserves all explicitly selected alternatives before the
      // Brief version changes and invalidates this extraction's old context.
      const body = { expected_revision: document.brief.revision, patches: selected };
      if (pending.current?.signature !== choiceSignature) pending.current = {
        signature: choiceSignature, revision: document.brief.revision,
        body: { ...body, operation_id: crypto.randomUUID(), message_id: crypto.randomUUID() }
      };
      const value = await workspaceRequest<BriefSnapshot>(identity, `${WORKSPACE}/briefs/${briefId}`, {
        method: "PATCH", body: pending.current.body
      });
      setSaved(true); onSaved?.(value);
    } catch (cause) { setError(workspaceError(cause)); }
    finally { setBusy(false); }
  }
  if (saved) return <p role="status">所选内容已存为待确认建议。请在研究档案中采用、修改或拒绝；尚未自动确认任何事实。</p>;
  return <section className="card draft-candidates" aria-label="研究草稿候选信息">
    <h3>从草稿提取的候选信息</h3>
    <p className="muted">每项可选一条候选，也可暂不选择。候选仅代表草稿中的说法，不能证明科研结论或已完成实验。</p>
    {result.completion_status?.startsWith("partial") && <p className="notice">本次提取覆盖不完整；仅展示通过审查的部分候选。</p>}
    {fields.map(field => <fieldset key={field} disabled={busy}>
      <legend>{FIELD_LABELS[field]}</legend>
      <label><input type="radio" name={`${taskId}-${field}`} checked={!choices[field]}
        onChange={() => setChoices(old => ({ ...old, [field]: "" }))} />暂不采用</label>
      {candidates.filter(item => item.field === field).map(item => <div key={item.id}>
        <label><input type="radio" name={`${taskId}-${field}`} checked={choices[field] === item.id}
          onChange={() => setChoices(old => ({ ...old, [field]: item.id }))} />{item.value}</label>
        <details><summary>核对草稿原文</summary>{item.evidence.map((edge, i) => <blockquote key={i}>
          <p className="source-quote">{edge.quote}</p><small>原文字符 {edge.span_start}–{edge.span_end}</small>
          {evidence(edge.ref_id)}
        </blockquote>)}</details>
      </div>)}
    </fieldset>)}
    {!!result.missing_fields?.length && <p>草稿未提供：{result.missing_fields.map(field => FIELD_LABELS[field]).join("、")}。</p>}
    {!!result.incomplete_fields?.length && <p className="notice">仍需补充或拆分：{result.incomplete_fields.map(field => FIELD_LABELS[field]).join("、")}。</p>}
    {!!error && <p className="notice" role="alert">{error}</p>}
    <button disabled={busy || !Object.values(choices).some(Boolean)} onClick={save}>
      {busy ? "核对并保存中…" : "将所选候选保存为待确认建议"}
    </button>
  </section>;
}
