import { useEffect, useRef, useState } from "react";
import { api, errorText } from "../../api";
import { Markdown } from "../../Markdown";
import { sameReviewSource, type ReviewSource, type ReviewReportSource } from "./review";
import type { ResearchVersion } from "./ResearchVersions";
import type { SchoolDraftDocument } from "./SchoolDraft";

export type RevisionOutcomeData = {
  status: "adopted_findings_resolved";
  source_document_sha256: string; review_report_sha256: string;
  source_binding: ReviewSource; report_binding: ReviewReportSource;
  selected_finding_keys: string[]; target_section_ids: string[]; changed_section_ids: string[];
  unchanged_section_sha256s: Record<string, string>;
  unselected_finding_keys: string[];
  unselected_findings_status: "not_selected_not_evaluated_again";
  resolution: { items: Array<{
    finding_key: string; verdict: "resolved" | "unresolved" | "needs_input";
    section_ids: string[]; revised_quote: string; explanation: string;
    citations: Array<{ ref_id: string; quote: string }>;
  }> };
};

export function RevisionOutcome({ outcome, document }: { outcome: RevisionOutcomeData; document: SchoolDraftDocument }) {
  const [before, setBefore] = useState<SchoolDraftDocument>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>();
  const request = useRef<AbortController | null>(null);
  useEffect(() => () => request.current?.abort(), []);
  async function compare() {
    const controller = new AbortController(); request.current?.abort(); request.current = controller;
    setBefore(undefined); setBusy(true); setError(undefined);
    try {
      const source = outcome.source_binding;
      const version = await api<ResearchVersion>(`/api/agent/tasks/${encodeURIComponent(source.source_task_id)}/research-version?command_seq=${source.command_seq}`, undefined, undefined, controller.signal);
      if (controller.signal.aborted) return;
      if (!sameReviewSource(version.source_binding, source) || !version.result.school_document)
        throw new Error("原稿版本无法对应，未展示其他稿件作为对照。");
      setBefore(version.result.school_document);
    } catch (failure) { if (!controller.signal.aborted) setError(failure); }
    finally { if (!controller.signal.aborted) setBusy(false); }
  }
  const verdicts = { resolved: "已处理", unresolved: "仍未解决", needs_input: "需要补充" };
  return <section className="card" aria-label="定向修订结果与版本对照">
    <h3>定向修订结果</h3>
    <p>本次采纳 {outcome.selected_finding_keys.length} 条意见，目标涉及 {outcome.target_section_ids.length} 个章节；实际变更 {outcome.changed_section_ids.length} 个章节。</p>
    <p className="muted">未选章节保持原文的校验记录共 {Object.keys(outcome.unchanged_section_sha256s).length} 项；其余 {outcome.unselected_finding_keys.length} 条评审意见保留，未重新判定。</p>
    {outcome.resolution.items.map(item => <p key={item.finding_key}><strong>{item.finding_key} · {verdicts[item.verdict]}</strong> · {item.explanation}</p>)}
    <button type="button" disabled={busy} onClick={() => void compare()}>{busy ? "正在读取原稿…" : "查看所选章节修订前后"}</button>
    {!!error && <p className="notice" role="alert">{errorText(error)}</p>}
    {before && outcome.target_section_ids.map(sectionId => <details key={sectionId}>
      <summary>{document.section_specs.find(spec => spec.id === sectionId)?.title || sectionId}</summary>
      <h4>修订前</h4>
      {before.sections.find(section => section.section_id === sectionId)?.blocks.map(block => <Markdown key={block.id} text={block.text} />)}
      <h4>修订后</h4>
      {document.sections.find(section => section.section_id === sectionId)?.blocks.map(block => <Markdown key={block.id} text={block.text} />)}
    </details>)}
  </section>;
}
