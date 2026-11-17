import type { ReactNode } from "react";
import { Markdown } from "../../Markdown";
import { claimStatusText } from "./progress";
import type { QuoteFocus } from "./EvidenceLink";

type Edge = {
  ref_id: string;
  quote: string;
  relation: string;
  conditions?: string[];
  quote_locator?: { span_start: number; span_end: number; coordinate_system: string };
};
type Claim = {
  id: string;
  statement: string;
  status: string;
  edges: Edge[];
  limitations?: string[];
};
type SourceCoverage = {
  document_id?: string;
  source_version_sha256?: string;
  source_coverage_kind?: string;
  status: string;
  mapped_windows: number;
  total_windows: number;
  covered_codepoints: number;
  available_codepoints: number;
  reasons: string[];
};
type ComparisonCoverage = {
  claim_id?: string;
  question?: string;
  request_coverage: string;
  uncovered_requests?: string[];
};
export type WorksetReport = {
  kind?: string;
  documents?: Array<{
    document: {
      document_id: string;
      title: string | null;
      coverage_kind: string;
      source_version_sha256?: string;
      missing_ranges?: Record<string, unknown>[];
    };
    coverage: SourceCoverage;
    summary: {
      claims: Claim[];
      gaps: string[];
      evidence_status: string;
    } | null;
    review_status: string;
  }>;
  comparisons?: Claim[];
  completion?: string;
  admission?: { admitted: boolean; reasons: string[] };
  source_warnings?: Array<{ code: string; count: number; ref_id: string; code_points?: string[]; document_id?: string }>;
  source_scope?: SourceCoverage[];
  request_coverage?: "complete" | "partial" | {
    status: "complete" | "partial";
    documents: Array<{ document_id: string; status: string; uncovered_requests: string[] }>;
    comparison: ComparisonCoverage | ComparisonCoverage[] | null;
    global?: { gaps: string[]; uncovered_requirement_ids: string[]; missing_document_ids: string[]; conflicting_claim_ids: string[] } | null;
  };
};

const coverageName = (value: string) =>
  ({
    complete_available_text: "已覆盖本次可用文本",
    partial: "覆盖不完整",
    insufficient: "证据不足",
    not_processed: "未处理",
    failed: "未完成处理",
  })[value] || value;

const sourceKind = (value?: string) => ({metadata_only: "仅题录与摘要", native_literal: "来源原始文本",
  material_full_text: "材料可用全文", material_partial: "材料部分解析文本"} as Record<string,string>)[value || ""] || "按实际可用文本";

function CoverageDetails({ coverage, version, kind }: { coverage: SourceCoverage; version?: string; kind?: string }) {
  return <div className="source-coverage-details">
    <p>来源范围：{sourceKind(coverage.source_coverage_kind || kind)}；已读取 {coverage.covered_codepoints} / {coverage.available_codepoints} 个可用字符。
      {coverage.total_windows > coverage.mapped_windows ? ` 尚有 ${coverage.total_windows-coverage.mapped_windows} 个可用窗口未处理。` : ""}</p>
    <p className="source-version">来源版本（SHA256）：<code>{coverage.source_version_sha256 || version || "未提供版本标识"}</code></p>
    {coverage.source_version_sha256 && version && coverage.source_version_sha256 !== version && <p className="notice">覆盖记录与来源版本不一致，不能据此认定已完成同一版本的全文读取。</p>}
  </div>;
}

export function WorksetResult({
  result,
  evidence,
}: {
  result: WorksetReport;
  evidence: (ref: string, focus?: QuoteFocus) => ReactNode;
}) {
  const requests = typeof result.request_coverage === "object" ? result.request_coverage : null;
  const comparisons = Array.isArray(requests?.comparison) ? requests.comparison : requests?.comparison ? [requests.comparison] : [];
  const renderClaim = (claim: Claim, prefix: string) => (
    <article className="claim" key={`${prefix}.${claim.id}`}>
      <b>
        {prefix}.{claim.id} ·{" "}
        {claimStatusText(claim.status)}
      </b>
      <Markdown text={claim.statement} />
      {claim.edges.map((edge, index) => (
        <blockquote key={`${edge.ref_id}-${index}`}>
          <p className="source-quote">{edge.quote}</p>
          {edge.quote_locator && <small>原文字符 {edge.quote_locator.span_start}–{edge.quote_locator.span_end}（按来源文本计数，非 PDF 页码）</small>}
          {evidence(edge.ref_id, edge)}
          {edge.conditions?.length ? (
            <p className="muted">适用条件：{edge.conditions.join("；")}</p>
          ) : null}
        </blockquote>
      ))}
      {claim.limitations?.map((value, index) => (
        <p className="muted" key={index}>
          {value}
        </p>
      ))}
    </article>
  );
  return (
    <section className="workset-result" aria-label="文献摘要与比较结果">
      <h3>{result.kind === "compare" ? "多文比较" : "文献摘要"}</h3>
      <p className="muted">
        覆盖范围按系统实际取得的文本计算；不代表原 PDF、图片或附件已完整解析。
      </p>
      {!!result.source_warnings?.length && <p className="notice">部分原文含尚未解释的私用字符。引用按原字符保留，未推断其数学含义；涉及公式时请查看原文核对。</p>}
      {result.source_warnings?.map((warning,index) => <p className="muted" key={index}>来源告警：{warning.document_id || "所引来源"} · {warning.count} 处未解释字符
        {warning.code_points?.length ? `（${warning.code_points.join("、")}）` : ""}；原字面保留。</p>)}
      {requests?.status === "partial" && <section className="notice" aria-label="摘要或比较未覆盖的要求">
        <strong>本次仅完成部分摘要或比较要求。</strong>
        {requests.documents.filter(item => item.status !== "complete").map(item => <div key={item.document_id}>
          <p>{item.document_id} · {item.status === "not_assessed" ? "尚未完成覆盖检查" : "仍有待回答内容"}</p>
          <ul>{item.uncovered_requests.map((text,index) => <li key={index}>{text}</li>)}</ul>
        </div>)}
        {comparisons.filter(item => item.request_coverage !== "complete").map((item,index) => <div key={index}>
          <p>比较{item.claim_id ? ` ${item.claim_id}` : ""}{item.question ? ` · ${item.question}` : ""}：仍有未完成要求。</p>
          {!!item.uncovered_requests?.length && <ul>{item.uncovered_requests.map((text,ordinal) => <li key={ordinal}>{text}</li>)}</ul>}
        </div>)}
        {!!requests.global?.gaps.length && <ul>{requests.global.gaps.map((text,index) => <li key={index}>{text}</li>)}</ul>}
        {!!requests.global?.uncovered_requirement_ids.length && <p>未覆盖要求：{requests.global.uncovered_requirement_ids.join("、")}</p>}
        {!!requests.global?.missing_document_ids.length && <p>仍有文献未完整覆盖：{requests.global.missing_document_ids.join("、")}</p>}
        {!!requests.global?.conflicting_claim_ids.length && <p>尚未解决的论断冲突：{requests.global.conflicting_claim_ids.join("、")}</p>}
      </section>}
      {result.admission?.admitted === false && (
        <p className="notice" role="status">
          本次范围超出可完成资源，未生成摘要。请缩小选文范围，已处理范围与缺口如下。
        </p>
      )}
      {result.documents?.map(
        ({ document, coverage, summary, review_status }) => (
          <section className="source-summary" key={document.document_id}>
            <h4>
              {document.document_id} · {document.title || "未提供标题"}
            </h4>
            <p className="coverage-status">
              {coverageName(coverage.status)} · {coverage.mapped_windows}/
              {coverage.total_windows} 个文本窗口
            </p>
            <CoverageDetails coverage={coverage} version={document.source_version_sha256} kind={document.coverage_kind} />
            <progress
              aria-label={`${document.document_id}文本覆盖`}
              max={Math.max(1, coverage.available_codepoints)}
              value={coverage.covered_codepoints}
            />
            {!!document.missing_ranges?.length && <details className="source-missing">
              <summary>解析缺失范围 · {document.missing_ranges.length} 项</summary>
              <p className="muted">以下为解析器记录的缺失位置，不用已读文本补写。</p>
              {document.missing_ranges.map((range,index) => <pre key={index}>{JSON.stringify(range,null,2)}</pre>)}
            </details>}
            {summary ? (
              summary.claims.map((claim) =>
                renderClaim(claim, document.document_id),
              )
            ) : (
              <p className="notice">
                {review_status === "quality_rejected"
                  ? "内容未通过审查，候选未发布。"
                  : "当前没有可发布摘要。"}
              </p>
            )}
            {coverage.reasons.map((reason, index) => (
              <p className="muted" key={index}>
                覆盖说明：{reason}
              </p>
            ))}
            {summary?.gaps.map((gap, index) => (
              <p className="muted" key={index}>
                缺口：{gap}
              </p>
            ))}
          </section>
        ),
      )}
      {!!result.source_scope?.length && <details className="source-scope" aria-label="任务来源覆盖记录">
        <summary>任务记录的来源覆盖 · {result.source_scope.length} 篇</summary>
        {result.source_scope.map((coverage,index) => <div key={coverage.document_id || index}>
          <strong>{coverage.document_id || `来源 ${index+1}`} · {coverageName(coverage.status)} · {coverage.mapped_windows}/{coverage.total_windows} 个文本窗口</strong>
          <CoverageDetails coverage={coverage} />
          {coverage.reasons.map((reason,ordinal) => <p className="muted" key={ordinal}>{reason}</p>)}
        </div>)}
      </details>}
      {!!result.comparisons?.length && (
        <section aria-label="跨文献比较">
          <h4>跨文献比较与条件</h4>
          {result.comparisons.map((claim) => renderClaim(claim, "COMPARE"))}
        </section>
      )}
    </section>
  );
}
