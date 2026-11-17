import { useState } from "react";
import { SourceNavigation, SourceReferenceList, type ResourceAdmission, type SourceNavigationReport } from "./SourceNavigation";

type NavigationStage = "documents" | "sections" | "passages";
type NavigationLevel = "documents" | "sections" | "leaves";
export type NavigationStep = {
  stage: NavigationStage;
  navigation_level?: NavigationLevel;
  phase?: "screen" | "final";
  batch_index?: number;
  candidate_ids: string[];
  probabilities: Record<string, number>;
  model_binding?: { model_id?: string; device?: string };
};
type NavigationSelection = {
  stage?: NavigationStage;
  level?: NavigationLevel;
  shared_final_candidate_ids: string[];
  kept_ids: string[];
};

export type EvidenceNavigationReport = {
  protocol: "laya1" | "laya2" | "structure1" | "structure2";
  status?: string;
  retrieved_passages?: number;
  input_frontier_ref_ids?: string[];
  excluded_before_navigation?: number;
  source_text_truncated?: boolean;
  full_corpus_or_full_document_coverage?: boolean;
  evidence_ref_ids?: string[];
  document_ids_kept?: string[];
  selected_passage_ids?: string[];
  decision_calls?: number | null;
  elapsed_seconds?: number;
  trace?: NavigationStep[];
  selection_steps?: NavigationSelection[];
  unknown_outcome?: boolean;
  error_code?: string;
  structure?: SourceNavigationReport;
  resource_admission?: ResourceAdmission;
  local_usage?: {
    observed_forward_calls?: number | null;
    input_tokens?: number | null;
    elapsed_seconds?: number;
    external_API_cash_ncny?: number;
    physical_compute_cost_not_measured?: boolean;
  };
};

const count = (value: number | null | undefined) => Number.isSafeInteger(value) && value! >= 0 ? String(value) : "未记录";
const seconds = (value: number | undefined) => typeof value === "number" && Number.isFinite(value) && value >= 0
  ? `${value.toFixed(2)} 秒` : "未记录";
const probability = (value: number | undefined) => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1
  ? `${(value * 100).toFixed(1)}%` : "未记录";
const refs = (value: string[] | undefined) => [...new Set(Array.isArray(value) ? value.filter(id => typeof id === "string" && !!id) : [])];
const stageLabel = (stage: NavigationStage) => ({ documents: "资料", sections: "章节", passages: "片段" })[stage] || "候选";
const stepLabel = (step: NavigationStep) => step.navigation_level
  ? ({ documents: "资料", sections: "章节", leaves: "片段" })[step.navigation_level] : stageLabel(step.stage);
const isStructureProtocol = (protocol: EvidenceNavigationReport["protocol"]) => protocol === "structure1" || protocol === "structure2";

export function navigationStepResult(report: EvidenceNavigationReport, step: NavigationStep, id: string): string {
  const selection = report.selection_steps?.find(value => step.navigation_level
    ? value.level === step.navigation_level : !value.level && value.stage === step.stage);
  if (step.phase === "screen") {
    const next = selection?.shared_final_candidate_ids
      || report.trace?.find(value => value.phase === "final" && (step.navigation_level
        ? value.navigation_level === step.navigation_level : !value.navigation_level && value.stage === step.stage))?.candidate_ids;
    return Array.isArray(next) ? next.includes(id) ? "进入共同复核" : "未进入共同复核" : "复核去向未记录";
  }
  if (step.phase === "final" || report.protocol === "laya1") {
    const kept = selection?.kept_ids || (!isStructureProtocol(report.protocol)
      ? step.stage === "documents" ? report.document_ids_kept : step.stage === "passages" ? report.selected_passage_ids : undefined
      : undefined);
    return Array.isArray(kept) ? kept.includes(id) ? "本轮保留" : "本轮未保留" : "保留结果未记录";
  }
  return "本次筛选结果未记录";
}

function NavigationTechnical({ report }: { report: EvidenceNavigationReport }) {
  const [visibleSteps, setVisibleSteps] = useState(8);
  const trace = Array.isArray(report.trace) ? report.trace : [];
  const usage = report.local_usage;
  return <details className="cq-source-technical">
    <summary>查看筛选与资源记录</summary>
    {report.protocol === "structure2" ? <p>
      相关性概率仅用于本轮预算内的候选排序，不是事实支持度，也不作为固定概率门槛。原文已返回不等于其中的说法已通过核验。
      同一次共同上下文内的评分才可比较，不能把不同批次的分数直接排序。
    </p> : (report.protocol !== "structure1" || trace.some(step => Object.keys(step.probabilities || {}).length > 0)) && <p>
      相关性概率只辅助选择资料，不是事实支持度。同一次共同上下文内的评分才可比较，不能把不同批次的分数直接排序。
    </p>}
    <p>已记录的决策次数：{report.decision_calls === null ? "尚未确认" : count(report.decision_calls)}。查找耗时：{seconds(report.elapsed_seconds)}。</p>
    {report.unknown_outcome && <p>存在结果尚未确认的调用；已记录次数不能当作完整调用总数。</p>}
    {report.error_code && <p>处理记录：{report.error_code}</p>}
    {trace.slice(0, visibleSteps).map((step, index) => <section key={`${step.stage}-${index}`}>
      <h4>{stepLabel(step)} · {step.phase === "screen" ? "分批筛选" : step.phase === "final" ? "共同复核" : "候选记录"}</h4>
      <div className="cq-source-table-wrap">
        <table>
          <caption>第 {index + 1} 次已记录决策</caption>
          <thead><tr><th scope="col">内部候选标识</th><th scope="col">相关性概率</th><th scope="col">本次结果</th></tr></thead>
          <tbody>{(Array.isArray(step.candidate_ids) ? step.candidate_ids : []).map(id => <tr key={id}>
            <td>{id}</td><td>{probability(step.probabilities?.[id])}</td><td>{navigationStepResult(report, step, id)}</td>
          </tr>)}</tbody>
        </table>
      </div>
      {step.model_binding?.model_id && <p>决策模型：{step.model_binding.model_id}{step.model_binding.device ? ` · ${step.model_binding.device}` : ""}</p>}
    </section>)}
    {!trace.length && <p>当前记录没有逐步候选信息。</p>}
    {trace.length > visibleSteps && <button type="button" onClick={() => setVisibleSteps(value => value + 8)}>
      显示更多筛选记录（已显示 {Math.min(visibleSteps, trace.length)} / {trace.length}）
    </button>}
    {usage && <p>已记录本地推理 {count(usage.observed_forward_calls)} 次，输入 {count(usage.input_tokens)} tokens，本地执行耗时 {seconds(usage.elapsed_seconds)}。</p>}
    {usage?.external_API_cash_ncny === 0 && <p>本地查找未产生外部 API 费用。
      {usage.physical_compute_cost_not_measured && "物理计算成本尚未测量。"}生成回答的费用见任务账本。</p>}
    {report.resource_admission && <details><summary>继续查阅的检查详情</summary><pre>{JSON.stringify(report.resource_admission, null, 2)}</pre></details>}
  </details>;
}

/** Use recorded reading states only; selected candidates do not imply full-source reading. */
export function EvidenceNavigation({ taskId, report }: { taskId: string; report: EvidenceNavigationReport }) {
  const selected = refs(report.evidence_ref_ids);
  if (isStructureProtocol(report.protocol) && report.structure?.version === "citequest-source-navigation/1") {
    return <SourceNavigation taskId={taskId} structure={report.structure} selectedRefs={selected}
      resourceAdmission={report.resource_admission} sourceTextTruncated={report.source_text_truncated}
      unknownOutcome={report.unknown_outcome} status={report.status}>
      <NavigationTechnical report={report} />
    </SourceNavigation>;
  }
  const noCandidates = report.status === "not_run_no_candidates";
  const frontier = Array.isArray(report.input_frontier_ref_ids) ? refs(report.input_frontier_ref_ids) : undefined;
  return <section className="cq-source-navigation" aria-label="本次资料范围">
    <h3>本次资料范围</h3>
    {isStructureProtocol(report.protocol) ? <p className="notice">本次资料阅读记录尚不完整，暂未展示结构与阅读状态。</p>
      : <p>检索返回 {count(report.retrieved_passages)} 个片段，本轮筛选范围 {count(frontier?.length)} 个，选用 {selected.length} 个。</p>}
    {noCandidates && <p>这次没有可进一步查找的候选资料，未继续筛选。</p>}
    {report.unknown_outcome && <p className="notice" role="status">上次查阅是否完成仍待核对，请勿重复提交。</p>}
    {report.status === "error" && !report.unknown_outcome && <p className="notice">本次资料查找未完成，请结合已保存的出处与限制查看结果。</p>}
    {!!selected.length && <section aria-label="本轮选用的原文">
      <h4>本轮选用的原文</h4>
      <SourceReferenceList taskId={taskId} refs={selected} label="本轮选用的原文入口" />
    </section>}
    <section aria-label="出处与限制">
      <h4>出处与限制</h4>
      <p>这份记录没有分别保存已查阅和尚未查阅的状态，不能由入选与否推断阅读覆盖。</p>
      <p>范围限于本轮已检索且可访问的片段，不代表全文或整个资料库。未入选不表示不相关，也不表示内容已被否定。</p>
      {typeof report.excluded_before_navigation === "number" && report.excluded_before_navigation > 0 && <p>有 {count(report.excluded_before_navigation)} 个检索片段未进入本次筛选范围。</p>}
      {report.source_text_truncated === true && <p className="notice">本次输入存在截断，请打开出处核对完整上下文。</p>}
      <p>资料相关性不等于事实支持度，结论仍需结合来源核验。</p>
    </section>
    <NavigationTechnical report={report} />
  </section>;
}
