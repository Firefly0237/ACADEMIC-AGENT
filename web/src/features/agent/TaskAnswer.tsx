import { useState } from "react";
import { Markdown } from "../../Markdown";
import type { DraftExtraction, BriefSuggestion } from "../research/types";
import { SchoolDraft, type SchoolDraftDocument } from "../research/SchoolDraft";
import { ClaimReviewDetails } from "./ClaimReviewDetails";
import { EvidenceLink } from "./EvidenceLink";
import { EvidenceNavigation, type EvidenceNavigationReport } from "./EvidenceNavigation";
import { claimStatusText, relationText } from "./progress";
import { WorksetResult, type WorksetReport } from "./WorksetResult";
import { revisionSelection, type ReviewSource } from "../research/review";
import { RevisionOutcome, type RevisionOutcomeData } from "../research/RevisionOutcome";

export type ManuscriptReviewReport = {
  schema_version: "citequest-research-review/1";
  source_document_sha256: string;
  source_binding_sha256: string;
  role_reports: Array<{
    role: "evidence" | "methods" | "structure";
    assessment: "no_blocking_findings" | "revision_recommended" | "insufficient_information";
    sections: Array<{ section_id: string; status: "no_issue_identified" | "needs_revision" | "insufficient_information"; reason: string }>;
    findings: Array<{
      id: string; section_ids: string[]; severity: "critical" | "major" | "minor";
      kind: "source_support" | "methodology" | "structure" | "requirements" | "missing_information";
      manuscript_quote: string; description: string; recommendation: string;
      citations: Array<{ ref_id: string; quote: string }>;
    }>;
  }>;
  aggregation: {
    decision: "revise" | "needs_input" | "no_blocking_findings";
    reason: string;
    findings: Array<{ finding_key: string; disposition: "retained" | "disputed" | "needs_user_input"; reason: string }>;
    conflicts: Array<{ id: string; finding_keys: string[]; explanation: string }>;
    actions: Array<{ section_ids: string[]; finding_keys: string[]; recommendation: string }>;
  };
};

export type AgentResult = WorksetReport & DraftExtraction & {
  text?: string;
  gaps?: string[];
  citations?: { ref_id: string; quote: string; quote_locator?: { span_start: number; span_end: number; coordinate_system: string } }[];
  claims?: any[];
  presentation?: { format: "prose" | "blocks"; paragraphs: string[][] };
  quality?: Record<string, string>;
  completion_status?: string;
  uncovered_requests?: string[];
  review?: { claims: Array<{ claim_id: string; verdict: string; explanation: string }> };
  review_grid?: { claims: Record<string, { quotes: Record<string, { relation: string; conditions_preserved: boolean; explanation: string }> }> };
  interrupts?: any[];
  suggestions?: BriefSuggestion[];
  school_document?: SchoolDraftDocument;
  review_report?: ManuscriptReviewReport;
  revision_outcome?: RevisionOutcomeData;
  diagnostics?: {
    phase: string; error_codes: string[];
    completed_roles: Array<"evidence" | "methods" | "structure" | "aggregate">;
    remaining_roles: Array<"evidence" | "methods" | "structure" | "aggregate">;
  };
  status?: string;
  requirements_revision?: number;
  requirements_sha256?: string;
  citation_protocol?: string;
  evidence_navigation?: EvidenceNavigationReport;
};

const reviewerNames = { evidence: "文献依据", methods: "实验设计与统计方法", structure: "可行性与结构" };
function ManuscriptReview({ taskId, report, onRevisionRequest, revisionDisabled = false }: {
  taskId: string; report: ManuscriptReviewReport;
  onRevisionRequest?: (findingKeys: string[]) => Promise<void> | void; revisionDisabled?: boolean;
}) {
  const [selected, setSelected] = useState<string[]>([]);
  const [selectionError, setSelectionError] = useState("");
  const findings = report.role_reports.flatMap(role => role.findings.map(finding => ({ key: `${role.role}:${finding.id}`, section_ids: finding.section_ids })));
  const selection = revisionSelection(findings, selected);
  function toggle(key: string, checked: boolean) {
    const next = checked ? [...selected, key] : selected.filter(item => item !== key);
    const proposed = revisionSelection(findings, next);
    if (next.length && !proposed.valid) { setSelectionError(proposed.reason); return; }
    setSelected(next); setSelectionError("");
  }
  const assessments = { no_blocking_findings: "本轮未识别阻断问题", revision_recommended: "建议修订", insufficient_information: "需要补充信息" };
  const severities = { critical: "关键问题", major: "主要问题", minor: "次要问题" };
  const dispositions = { retained: "保留", disputed: "存在分歧", needs_user_input: "需你确认" };
  const sectionStates = { no_issue_identified: "本轮未识别问题", needs_revision: "建议修订", insufficient_information: "信息不足" };
  return <section className="card" aria-label="多智能体评审报告">
    <h3>分角色评审</h3>
    <p className="muted">评审意见对应所选稿件版本，原稿保持保留。</p>
    <p>{report.aggregation.reason}</p>
    {report.role_reports.map(role => <details key={role.role}>
      <summary>{reviewerNames[role.role]} · {assessments[role.assessment]} · {role.findings.length} 项意见</summary>
      {role.findings.map(finding => <article key={finding.id}>
        {onRevisionRequest && <label><input type="checkbox" checked={selected.includes(`${role.role}:${finding.id}`)} disabled={revisionDisabled}
          onChange={event => toggle(`${role.role}:${finding.id}`, event.target.checked)} />采纳此意见用于修订</label>}
        <h4>{finding.id} · {severities[finding.severity]}{finding.kind === "missing_information" ? " · 待补信息" : ""}</h4>
        <p className="muted">对应章节：{finding.section_ids.join("、")}</p>
        <blockquote><p className="source-quote">{finding.manuscript_quote}</p><small>原稿片段</small></blockquote>
        <Markdown text={finding.description} />
        <p><strong>建议：</strong>{finding.recommendation}</p>
        {finding.citations.map((citation, index) => <div key={`${citation.ref_id}:${index}`}>
          <p className="source-quote">{citation.quote}</p>
          <EvidenceLink taskId={taskId} refId={citation.ref_id} focus={citation} />
        </div>)}
      </article>)}
      <details><summary>逐节覆盖 · {role.sections.length} 节</summary>
        {role.sections.map(section => <p key={section.section_id}><strong>{section.section_id} · {sectionStates[section.status]}</strong> · {section.reason}</p>)}
      </details>
    </details>)}
    {!!report.aggregation.findings.length && <details><summary>意见处理情况</summary>
      {report.aggregation.findings.map(finding => <p key={finding.finding_key}><strong>{finding.finding_key} · {dispositions[finding.disposition]}</strong> · {finding.reason}</p>)}
    </details>}
    {!!report.aggregation.conflicts.length && <section aria-label="保留的评审分歧">
      <h4>保留的分歧</h4>
      {report.aggregation.conflicts.map(conflict => <p key={conflict.id}><strong>{conflict.finding_keys.join("、")}</strong> · {conflict.explanation}</p>)}
    </section>}
    {!!report.aggregation.actions.length && <details><summary>建议处理顺序</summary>
      {report.aggregation.actions.map((action, index) => <p key={index}>{action.finding_keys.join("、")} · {action.recommendation}</p>)}
    </details>}
    {onRevisionRequest && <section aria-label="选择定向修订范围">
      <p className="muted">本次最多采纳 8 条意见、涉及两个章节。未选意见完整保留，原稿与评审报告不会被覆盖。</p>
      <p>已选 {selected.length} 条意见 · {selection.section_ids.length} 个章节。</p>
      {selectionError && <p className="notice" role="alert">{selectionError}</p>}
      <button type="button" disabled={revisionDisabled || !selection.valid} onClick={() => void onRevisionRequest(selected)}>修订所选意见</button>
    </section>}
  </section>;
}

export type CoverageReport = {
  claim_coverage?: Array<{ claim_id?: string; state?: string }>;
  coverage_gaps?: string[];
  completion_status?: string;
};

export type ResearchProgressReport = {
  schema_version: "citequest-research-progress/1";
  status: "waiting_input" | "partial" | "rejected" | "structured_plan_ready";
  research_status: "proposed_not_executed";
  school_format_ready: false;
  research_questions_planned: number;
  literature_reviews_completed: number;
  experiments_planned: number;
  method_proposals_completed: number;
  unit_reviews_completed: number;
  cross_plan_review_completed: boolean;
  targeted_revision_round: number;
  waiting_kind: string | null;
  school_writing_status?: "not_started" | "partial";
  school_sections_written?: number;
  school_section_reviews_completed?: number;
};

/** Counts describe completed workflow units; no unpublished plan text is shown. */
export function ResearchProgress({ report }: { report: ResearchProgressReport }) {
  const reasons: Record<string, string> = {
    scope_sync_required: "研究档案已变化，请同步本任务的文献范围。",
    research_intake_required: "仍有必要信息待补充或确认。",
    outline_request_gaps: "部分研究要求尚未纳入方案。",
    research_evidence_insufficient: "当前文献范围内的证据不足，请补充来源或调整问题。",
    research_motivation_insufficient: "文献核验已执行，但尚不足以支撑当前研究问题。",
    research_review_needs_input: "审阅发现需要你确认的信息或角色分歧。",
    research_request_coverage_partial: "方案尚未完整覆盖当前研究要求。",
    research_scope_exceeds_remaining_budget: "原任务剩余资源不足以完成后续步骤，已暂停生成。",
    revision_needs_user_input: "定向修订需要补充信息。",
    research_conflict_resolution_set_exceeds_bound: "当前方案分歧较多，需要先明确研究范围。",
    research_draft_input_required: "请先提供或选择研究草稿。",
    school_writing_incomplete: "正式计划书尚未完成全部章节与审查。",
    other_required_input: "请查看待补充的信息后继续本任务。",
  };
  const heading = { waiting_input: "研究计划等待补充", partial: "研究计划尚未完成",
    rejected: "当前方案未通过审查", structured_plan_ready: "结构化方案已就绪" }[report.status];
  return <section className="notice" aria-label="研究计划进度">
    <strong>{heading}</strong>
    <p>已规划 {report.research_questions_planned} 个研究问题、{report.experiments_planned} 个拟议实验；
      已完成文献审阅 {report.literature_reviews_completed}/{report.research_questions_planned} 项，
      方法设计 {report.method_proposals_completed}/{report.experiments_planned} 项。</p>
    <p className="muted">当前稿逐项审阅已执行 {report.unit_reviews_completed}/{report.research_questions_planned} 项，
      跨方案审阅{report.cross_plan_review_completed ? "已执行" : "尚未完成"}；
      定向修订第 {report.targeted_revision_round} 轮。审阅执行数量不代表全部通过。</p>
    {report.waiting_kind && reasons[report.waiting_kind] && <p>{reasons[report.waiting_kind]}</p>}
    {report.school_writing_status === "partial" && <p>学校草稿已保留 {report.school_sections_written ?? 0}/14 节，
      已执行逐节审阅 {report.school_section_reviews_completed ?? 0}/14 节；未通过的正文不作为完成稿发布。</p>}
    <p className="muted">以上仅为当前保留的工作进度。实验尚未执行，正式学校计划书尚未生成。</p>
  </section>;
}

/** Coverage metadata is not an authorized, evaluated answer body. */
export function TaskCoverage({ report }: { report: CoverageReport }) {
  const gaps = [...new Set((report.coverage_gaps || []).filter(value => typeof value === "string" && value.trim()))];
  const pending = [...new Set((report.claim_coverage || [])
    .filter(row => row.state === "not_evaluated")
    .map(row => row.claim_id).filter((id): id is string => typeof id === "string"))];
  return <section className="notice" aria-label="尚未完成的观点核验">
    <h3>本轮尚无通过核验的观点回答</h3>
    <p>以下仅为覆盖情况与实际缺口，不是已获证据支持的结论。</p>
    {!!pending.length && <p>尚未评估的命题：{pending.join("、")}</p>}
    {!!gaps.length && <ul>{gaps.map(gap => <li key={gap}>{gap}</li>)}</ul>}
    {!gaps.length && <p>当前记录未提供具体缺口说明，请查看执行过程或补充明确的核验范围。</p>}
  </section>;
}

function presentationClaims(result: AgentResult): any[][] | null {
  const format = result.presentation?.format;
  const paragraphs = result.presentation?.paragraphs;
  if (!["prose", "blocks"].includes(format || "") || !Array.isArray(paragraphs)
      || !paragraphs.length || paragraphs.length > 8 || !result.claims?.length) return null;
  const claims = new Map<string, any>();
  for (const claim of result.claims) {
    if (typeof claim.id !== "string" || !claim.id || claims.has(claim.id)
        || typeof claim.statement !== "string" || !claim.statement.trim()) return null;
    if (format === "prose" && (/[\r\n\v\f\x1c-\x1e\x85\u2028\u2029]/.test(claim.statement)
        || claim.statement.includes("$$") || claim.statement.includes("\\[") || claim.statement.includes("\\]"))) return null;
    claims.set(claim.id, claim);
  }
  if (paragraphs.some(group => !Array.isArray(group) || !group.length || group.length > 8
      || group.some(id => typeof id !== "string" || !claims.has(id)))) return null;
  const identifiers = paragraphs.flat();
  if (identifiers.length !== claims.size || new Set(identifiers).size !== claims.size) return null;
  return paragraphs.map(group => group.map(id => claims.get(id)));
}

/** The same read-only renderer is used for the current result and saved turns. */
export function TaskAnswer({ taskId, result, historical = false, sourceBinding, onRevisionRequest, revisionDisabled }: {
  taskId: string; result: AgentResult; historical?: boolean; sourceBinding?: ReviewSource;
  onRevisionRequest?: (findingKeys: string[]) => Promise<void> | void; revisionDisabled?: boolean;
}) {
  const paragraphs = presentationClaims(result);
  const claimCards = result.claims?.map((claim: any) => <article className="claim" key={claim.id}>
    <b>{claim.id} · {claimStatusText(claim.status)}</b>
    <Markdown text={claim.statement} />
    {claim.edges?.map((edge: any, index: number) => <blockquote key={index}>
      <p className="source-quote">{edge.quote}</p>
      {edge.quote_locator && <small>原文字符 {edge.quote_locator.span_start}–{edge.quote_locator.span_end}（按来源文本计数，非 PDF 页码）</small>}
      <small>{relationText(edge.relation)} · {edge.rationale}</small>
      <EvidenceLink taskId={taskId} refId={edge.ref_id} focus={edge} sourceBinding={sourceBinding} />
    </blockquote>)}
    {claim.limitations?.map((text: string) => <p className="muted" key={text}>{text}</p>)}
    <ClaimReviewDetails claimId={claim.id} review={result.review} grid={result.review_grid} />
  </article>);
  return <>
    {result.requirements_revision !== undefined && <p className="muted" aria-label="回答需求版本">本回答依据第 {result.requirements_revision} 版已核对要求；追问中的明确变更由新一轮处理。</p>}
    {result.evidence_navigation && <EvidenceNavigation taskId={taskId} report={result.evidence_navigation} />}
    {result.interrupts?.map((value, index) => <section className="card" key={index} aria-label="待补充的信息">
      <h3>等待补充或确认</h3>
      {value.questions?.map((question: any, ordinal: number) => <p key={ordinal}>{typeof question === "string" ? question : question.question}</p>)}
      {value.input_rule && <p className="muted">请在研究档案中保存或确认信息，再继续本任务；对话文字不会自动改成已确认事实。</p>}
      {value.reason && <p>{value.reason}</p>}
      {!!value.open_conflicts?.length && <p className="notice">有 {value.open_conflicts.length} 项未解决的信息冲突，请先在研究档案中明确选择。</p>}
    </section>)}
    {!!result.documents?.length && <WorksetResult result={result}
      evidence={(ref, focus) => <EvidenceLink taskId={taskId} refId={ref} focus={focus} sourceBinding={sourceBinding} />} />}
    {result.revision_outcome && result.school_document && <RevisionOutcome outcome={result.revision_outcome} document={result.school_document} />}
    {result.school_document && <SchoolDraft taskId={taskId} document={result.school_document} allowExport={!historical || !!sourceBinding} sourceBinding={sourceBinding}
      evidence={ref => <EvidenceLink taskId={taskId} refId={ref} sourceBinding={sourceBinding} />} />}
    {result.review_report && <ManuscriptReview taskId={taskId} report={result.review_report} onRevisionRequest={onRevisionRequest} revisionDisabled={revisionDisabled} />}
    {["research_review_partial", "research_revision_partial"].includes(result.status || "") && result.diagnostics && <section className="notice" aria-label="稿件任务未完成">
      <strong>本轮稿件任务尚未完成</strong>
      {result.status === "research_review_partial" ? <>
        <p>已完成：{result.diagnostics.completed_roles.map(role => role === "aggregate" ? "意见汇总" : reviewerNames[role]).join("、") || "暂无"}。</p>
        <p>尚未完成：{result.diagnostics.remaining_roles.map(role => role === "aggregate" ? "意见汇总" : reviewerNames[role]).join("、") || "请查看执行记录"}。</p>
      </> : <p>本次修订尚未发布通过审查的新稿，原稿与评审报告保留。执行情况见诊断与过程记录。</p>}
    </section>}
    {result.text && result.review_report && <details><summary>查看评审报告全文</summary><Markdown text={result.text} /></details>}
    {result.text && !result.review_report && !result.school_document && !result.claims?.length && !result.documents?.length && <Markdown text={result.text} />}
    {paragraphs ? <>
      <section aria-label="回答正文">
        {paragraphs.map((group, index) => <section key={index}>
          <Markdown text={result.presentation?.format === "blocks"
            ? group.map(claim => `[${claim.id}]\n\n${claim.statement}`).join("\n\n")
            : group.map(claim => `[${claim.id}] ${claim.statement}`).join(" ")} />
          <details>
            <summary>本段来源依据 · {group.map(claim => claim.id).join("、")}</summary>
            {group.map(claim => <div key={claim.id}>
              <strong>{claim.id} · {claimStatusText(claim.status)}</strong>
              {claim.edges?.map((edge: any, ordinal: number) => <EvidenceLink key={ordinal}
                taskId={taskId} refId={edge.ref_id} focus={edge} sourceBinding={sourceBinding} />)}
              {!claim.edges?.length && <p className="muted">本条未提供可定位引用，请查看逐条依据中的不足说明。</p>}
            </div>)}
          </details>
        </section>)}
      </section>
      <details><summary>逐条依据与审查</summary>{claimCards}</details>
    </> : <>
      {result.presentation && <p className="notice" role="status">正文分组与当前论断无法完整对应，以下逐条展示可核查内容。</p>}
      {claimCards}
    </>}
    {result.request_coverage === "partial" && <section className="notice" aria-label="回答尚未覆盖的要求">
      <strong>本次仅完成部分问题。以下要求仍未回答完整：</strong>
      <ul>{result.uncovered_requests?.map((request, index) => <li key={index}>{request}</li>)}</ul>
      <p>上方保留已通过证据核验的内容；这不代表本任务已完整完成。</p>
    </section>}
    {!!result.source_warnings?.length && !result.documents?.length && <p className="notice">部分原文含尚未解释的私用字符。引用按原字符保留，未推断其数学含义；涉及公式时请查看原文核对。</p>}
    {!!result.citations?.length && !result.claims?.length && <details>
      <summary>原文引用 · {result.citations.length} 处</summary>
      {result.citations.map((citation, index) => <blockquote key={index}>
        <p className="source-quote">{citation.quote}</p>
        <EvidenceLink taskId={taskId} refId={citation.ref_id} focus={citation} sourceBinding={sourceBinding} />
      </blockquote>)}
    </details>}
    {result.gaps?.map((gap, index) => <p className="muted" key={index}>覆盖缺口：{gap}</p>)}
    {result.quality && <div className="muted">自动检查已执行；引用字面位置与 AI 语义审查分别核验，关键结论可展开原文复核。
      <details><summary>检查记录</summary>{Object.entries(result.quality).map(([key, value]) => <p key={key}>{key} = {value}</p>)}</details>
    </div>}
  </>;
}
