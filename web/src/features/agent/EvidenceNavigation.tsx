import { EvidenceLink } from "./EvidenceLink";

type NavigationStep = {
  stage: "documents" | "passages";
  candidate_ids: string[];
  probabilities: Record<string, number>;
  model_binding?: { model_id?: string; device?: string };
};

export type EvidenceNavigationReport = {
  protocol: "laya1";
  status?: string;
  retrieved_passages: number;
  input_frontier_ref_ids: string[];
  excluded_before_navigation?: number;
  source_text_truncated: boolean;
  full_corpus_or_full_document_coverage: boolean;
  evidence_ref_ids?: string[];
  document_ids_kept?: string[];
  selected_passage_ids?: string[];
  decision_calls: number;
  elapsed_seconds?: number;
  trace: NavigationStep[];
  local_usage?: {
    observed_forward_calls: number;
    input_tokens: number;
    elapsed_seconds: number;
    external_API_cash_ncny: number;
    physical_compute_cost_not_measured: boolean;
  };
};

const count = (value: number | undefined) => Number.isSafeInteger(value) && value! >= 0 ? String(value) : "未记录";
const seconds = (value: number | undefined) => typeof value === "number" && Number.isFinite(value) && value >= 0
  ? `${value.toFixed(2)} 秒` : "未记录";
const probability = (value: number | undefined) => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1
  ? `${(value * 100).toFixed(1)}%` : "未记录";

/** Display only persisted navigation metadata; evidence opens through the existing ACL gate. */
export function EvidenceNavigation({ taskId, report }: { taskId: string; report: EvidenceNavigationReport }) {
  const noCandidates = report.status === "not_run_no_candidates";
  const frontier = Array.isArray(report.input_frontier_ref_ids) ? report.input_frontier_ref_ids : null;
  const selected = Array.isArray(report.evidence_ref_ids) ? report.evidence_ref_ids : null;
  const trace = Array.isArray(report.trace) ? report.trace : [];
  const usage = report.local_usage;
  return <details className="card" aria-label="证据导航过程">
    <summary>{noCandidates ? "证据导航未执行" : "查看证据导航过程"}</summary>
    {noCandidates ? <p>本轮没有可用于导航的候选片段，未运行概率决策。</p> : <>
      <p>检索片段 {count(report.retrieved_passages)} → 本轮导航候选 {count(frontier?.length)} → 入选片段 {count(selected?.length)}。</p>
      {typeof report.excluded_before_navigation === "number" && report.excluded_before_navigation > 0 && <p className="muted">
        有 {report.excluded_before_navigation} 个检索片段未进入本轮有界导航。</p>}
      <p className="muted">相关性概率用于选择阅读片段，不是事实支持度。结论仍需独立来源审查。</p>
      {report.full_corpus_or_full_document_coverage === false && <p className="muted">
        选择范围限于本轮已检索、可授权的片段，不代表全文或全库覆盖。
        {report.source_text_truncated === false && "导航输入片段按原文保留。"}</p>}
      {!!selected?.length && <section aria-label="导航入选证据">
        <strong>入选证据</strong>
        {selected.map((ref, index) => <div key={`${index}-${ref}`}>
          <span>片段 {index + 1}</span><EvidenceLink taskId={taskId} refId={ref} />
        </div>)}
      </section>}
      <details>
        <summary>两级候选与相关性概率 · {count(report.decision_calls)} 次决策</summary>
        {trace.map((step, index) => {
          const kept = step.stage === "documents" ? report.document_ids_kept : report.selected_passage_ids;
          return <section key={`${step.stage}-${index}`}>
            <h4>{step.stage === "documents" ? "文献候选" : "片段候选"}</h4>
            <div style={{ maxWidth: "100%", overflowX: "auto" }}>
              <table>
                <caption>第 {index + 1} 次决策的实际候选记录</caption>
                <thead><tr><th scope="col">候选标识</th><th scope="col">相关性概率</th><th scope="col">本阶段结果</th></tr></thead>
                <tbody>{step.candidate_ids.map(id => <tr key={id}>
                  <td style={{ overflowWrap: "anywhere" }}>{id}</td>
                  <td>{probability(step.probabilities?.[id])}</td>
                  <td>{Array.isArray(kept) ? kept.includes(id) ? "入选" : "未入选" : "未记录"}</td>
                </tr>)}</tbody>
              </table>
            </div>
            {step.model_binding?.model_id && <p className="muted" style={{ overflowWrap: "anywhere" }}>
              决策模型：{step.model_binding.model_id}{step.model_binding.device ? ` · ${step.model_binding.device}` : ""}</p>}
          </section>;
        })}
        {!trace.length && <p>当前结果没有逐步候选记录，未展示推测过程。</p>}
      </details>
      <p className="muted">导航耗时 {seconds(report.elapsed_seconds)}。
        {usage && <>本地概率推理 {count(usage.observed_forward_calls)} 次，输入 {count(usage.input_tokens)} tokens，
          本地执行耗时 {seconds(usage.elapsed_seconds)}。</>}</p>
      {usage?.external_API_cash_ncny === 0 && <p className="muted">本地导航未产生外部API费用。
        {usage.physical_compute_cost_not_measured && "物理计算成本尚未测量。"}生成回答的费用见任务账本。</p>}
    </>}
  </details>;
}
