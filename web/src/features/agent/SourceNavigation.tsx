import { useState, type ReactNode } from "react";
import { EvidenceLink } from "./EvidenceLink";
import "./sourceNavigation.css";

export type ReadingState = "source_text_returned" | "not_read" | "partially_read" | "unavailable";
export type ExpansionReason = "not_selected" | "capacity" | "time_limit" | "outside_frontier" |
  "access_unavailable" | "no_children" | "not_recorded" | "budget" | "not_requested";
export type SourceNavigationNode = {
  id: string; parent_id: string | null; kind: "document" | "section" | "passage";
  label: string; read_state: ReadingState; evidence_ref_ids: string[]; why_not_expanded?: ExpansionReason | null;
  metadata_unknown?: string[];
};
export type SourceNavigationReport = {
  version: "citequest-source-navigation/1";
  scope: { documents: number; sections: number; passages: number; coverage: "bounded_current_sources" };
  nodes: SourceNavigationNode[];
  nodes_total?: number;
  nodes_truncated?: boolean;
  scope_ref_id?: string;
  supplemental_evidence_ref_ids: string[];
  limitations: string[];
};
export type ResourceAdmission = {
  status: "admitted" | "rejected" | "not_recorded";
  reason_codes: string[];
  [key: string]: unknown;
};

const readingLabels: Record<ReadingState, string> = {
  source_text_returned: "已返回原文", not_read: "尚未查阅", partially_read: "部分原文已返回", unavailable: "当前无法读取",
};
const kindLabels = { document: "资料", section: "章节", passage: "片段" };
const reasons: Record<ExpansionReason, string> = {
  not_selected: "本轮优先处理了其他部分，这里尚未继续展开。",
  capacity: "本轮可处理的资料量已到上限。",
  time_limit: "本轮查阅时间已到上限。",
  outside_frontier: "这部分超出本次已检索的资料范围。",
  access_unavailable: "当前无法读取这个出处。",
  no_children: "当前记录没有可展开的下级内容。",
  budget: "本轮可用额度不足以继续展开。",
  not_requested: "本轮没有请求继续展开这部分。",
  not_recorded: "没有记录未展开的具体原因。",
};
type NavigationIssue = { kind: "resource" | "authority" | "other"; message: string };
const resourceReasons: Record<string, string> = {
  structure_navigation_active_budget_exhausted: reasons.time_limit,
  structure_navigation_forward_budget_exhausted: reasons.budget,
  structure_navigation_input_budget_exhausted: reasons.budget,
  structure_navigation_request_capacity: reasons.capacity,
  structure_navigation_level_capacity: reasons.capacity,
  laya_complete_context_exceeds_token_budget: reasons.capacity,
  navigation_question_rows_capacity: reasons.capacity,
  navigation_sequence_length_capacity: reasons.capacity,
  navigation_padded_shape_capacity: reasons.capacity,
  navigation_attention_shape_capacity: reasons.capacity,
  navigation_estimated_peak_memory_capacity: "预计内存需求超出本次可用范围，未继续展开。",
  navigation_remaining_deadline_capacity: "本轮剩余查阅时间不足，未继续展开。",
  navigation_calibrated_latency_exceeds_deadline: "预计查阅时间超过本轮剩余时间，未继续展开。",
  navigation_latency_calibration_required: "当前运行条件缺少可用的耗时依据，未继续展开。",
  navigation_concurrency_capacity: "查阅服务正在处理其他请求，本轮未继续展开。",
  navigation_deadline_context_required: "本轮剩余查阅时间未能确认，未继续展开。",
};
const authorityReasons = new Set([
  "navigation_source_authority_changed", "navigation_source_unavailable", "navigation_source_blob_unavailable",
  "navigation_current_native_identity_changed", "navigation_current_native_text_changed", "navigation_current_fragment_changed",
  "navigation_scope_identity_changed", "navigation_scope_selection_changed", "navigation_structure_pointer_changed",
  "structure_navigation_task_scope_changed", "evidence_unavailable_or_access_revoked",
]);
export function navigationIssue(code: string): NavigationIssue {
  if (resourceReasons[code]) return { kind: "resource", message: resourceReasons[code] };
  if (["capacity", "time_limit", "budget"].includes(code)) return { kind: "resource", message: reasons[code as ExpansionReason] };
  if (authorityReasons.has(code) || code === "access_unavailable") return { kind: "authority", message: "来源当前无法读取或版本已变化，未继续展开。" };
  return { kind: "other", message: "本次查阅未完成，具体原因请查看处理记录。" };
}
const totalText = (value: number) => Number.isSafeInteger(value) && value >= 0 ? String(value) : "未记录";
const nodeLabel = (node: SourceNavigationNode) => typeof node.label === "string" && node.label.trim()
  && !/^[a-f\d]{32,}$/i.test(node.label.trim()) ? node.label : kindLabels[node.kind] || "来源位置";
const uniqueRefs = (refs: string[] | undefined) => [...new Set(Array.isArray(refs) ? refs.filter(ref => typeof ref === "string" && !!ref) : [])];
const readingLabel = (node: SourceNavigationNode) => node.kind !== "passage" && node.read_state === "source_text_returned"
  ? readingLabels.partially_read : readingLabels[node.read_state] || "阅读状态未记录";
const expansionReason = (reason: ExpansionReason | null | undefined, fallback: ExpansionReason, capacityOverride?: string) =>
  reason === "capacity" && capacityOverride ? capacityOverride : reasons[reason || fallback] || reasons.not_recorded;

export function SourceReferenceList({ taskId, refs, label }: { taskId: string; refs: string[]; label: string }) {
  const [visible, setVisible] = useState(6);
  const values = uniqueRefs(refs);
  if (!values.length) return null;
  return <div className="cq-source-references" aria-label={label}>
    {values.slice(0, visible).map((ref, index) => <div key={ref} className="cq-source-reference">
      <span>原文 {index + 1}</span><EvidenceLink taskId={taskId} refId={ref} />
    </div>)}
    {values.length > visible && <button type="button" onClick={() => setVisible(value => value + 6)}>
      显示更多原文入口（已显示 {Math.min(visible, values.length)} / {values.length}）
    </button>}
  </div>;
}

export function groupSourceNodes(nodes: SourceNavigationNode[]) {
  const byId = new Map(nodes.map(node => [node.id, node]));
  const groups = nodes.filter(node => node.kind === "document").map(document => ({ document, nodes: [] as SourceNavigationNode[] }));
  const byDocument = new Map(groups.map(group => [group.document.id, group]));
  const ungrouped: SourceNavigationNode[] = [];
  for (const node of nodes) {
    if (node.kind === "document") continue;
    let parent = node.parent_id ? byId.get(node.parent_id) : undefined;
    const seen = new Set([node.id]);
    while (parent && parent.kind !== "document" && !seen.has(parent.id)) {
      seen.add(parent.id);
      parent = parent.parent_id ? byId.get(parent.parent_id) : undefined;
    }
    const group = parent?.kind === "document" ? byDocument.get(parent.id) : undefined;
    if (group) group.nodes.push(node);
    else ungrouped.push(node);
  }
  return { groups, ungrouped };
}

function NodeReading({ taskId, node, capacityOverride }: { taskId: string; node: SourceNavigationNode; capacityOverride?: string }) {
  return <li className="cq-source-reading">
    <div className="cq-source-reading-heading">
      <strong>{kindLabels[node.kind] || "来源位置"} · {nodeLabel(node)}</strong>
      <span>{readingLabel(node)}</span>
    </div>
    {(node.why_not_expanded || node.read_state === "not_read" || node.read_state === "unavailable") && <p>
      未继续展开的原因：{expansionReason(node.why_not_expanded, node.read_state === "unavailable" ? "access_unavailable" : "not_recorded", capacityOverride)}
    </p>}
    <SourceReferenceList taskId={taskId} refs={node.evidence_ref_ids} label="该位置的出处" />
  </li>;
}

function ReadingGroup({ taskId, document, nodes, capacityOverride }: {
  taskId: string; document?: SourceNavigationNode; nodes: SourceNavigationNode[]; capacityOverride?: string;
}) {
  const [visible, setVisible] = useState(8);
  return <details className="cq-source-document">
    <summary>
      <strong>{document ? nodeLabel(document) : "其他来源位置"}</strong>
      <span>{document ? readingLabel(document) : "未关联到资料的记录"} · {nodes.length} 条位置记录</span>
    </summary>
    {document?.why_not_expanded && <p>未继续展开的原因：{expansionReason(document.why_not_expanded, "not_recorded", capacityOverride)}</p>}
    {document && <SourceReferenceList taskId={taskId} refs={document.evidence_ref_ids} label="资料出处" />}
    {!!nodes.length && <ul className="cq-source-reading-list">
      {nodes.slice(0, visible).map(node => <NodeReading key={node.id} taskId={taskId} node={node} capacityOverride={capacityOverride} />)}
    </ul>}
    {!nodes.length && <p>本次没有返回下级位置的阅读记录。</p>}
    {nodes.length > visible && <button type="button" onClick={() => setVisible(value => value + 8)}>
      显示更多位置（已显示 {Math.min(visible, nodes.length)} / {nodes.length}）
    </button>}
  </details>;
}

export function SourceNavigation({ taskId, structure, selectedRefs, resourceAdmission, sourceTextTruncated, unknownOutcome, status, children }: {
  taskId: string; structure: SourceNavigationReport; selectedRefs: string[]; resourceAdmission?: ResourceAdmission;
  sourceTextTruncated?: boolean; unknownOutcome?: boolean; status?: string; children?: ReactNode;
}) {
  const [visibleDocuments, setVisibleDocuments] = useState(6);
  const nodes = Array.isArray(structure.nodes) ? structure.nodes : [];
  const { groups, ungrouped } = groupSourceNodes(nodes);
  const supplemental = uniqueRefs(structure.supplemental_evidence_ref_ids);
  const selected = uniqueRefs(selectedRefs);
  const limitations = Array.isArray(structure.limitations) ? structure.limitations : [];
  const admissionIssues = (Array.isArray(resourceAdmission?.reason_codes) ? resourceAdmission.reason_codes : []).map(navigationIssue);
  const resourceOnly = admissionIssues.length > 0 && admissionIssues.every(issue => issue.kind === "resource");
  const capacityOverride = resourceAdmission?.status === "rejected" && !resourceOnly
    ? "本次查阅未完成，具体原因见上方说明。" : undefined;
  return <section className="cq-source-navigation" aria-label="本次资料范围">
    <h3>本次资料范围</h3>
    <p className="cq-source-scope">{totalText(structure.scope?.documents)} 份资料 · {totalText(structure.scope?.sections)} 个章节 · {totalText(structure.scope?.passages)} 个片段</p>
    <p>本次只在当前获准的资料范围内查找，不代表全文或整个资料库都已查阅。</p>
    {unknownOutcome && <p className="notice" role="status">上次查阅是否完成仍待核对，未把未知结果计为已完成。请勿重复提交。</p>}
    {status === "error" && !unknownOutcome && resourceAdmission?.status !== "rejected" && <p className="notice" role="status">
      本次资料查阅未完成，以下仅显示已经保存的阅读范围与出处。
    </p>}
    {resourceAdmission?.status === "rejected" && <p className="notice" role="status">
      {admissionIssues.some(issue => issue.kind === "authority") ? "本次来源核对未通过。" : "本次未继续展开部分资料。"}
      {[...new Set(admissionIssues.map(issue => issue.message))].join("") || "具体原因没有记录，不能据此判断资料是否相关。"}
    </p>}
    <section aria-label="本轮选用的原文">
      <h4>本轮选用的原文</h4>
      <p>以下是本轮用于回答的资料入口。原文已返回不等于其中的说法已通过核验，请结合回答中的依据与限制阅读。</p>
      {selected.length ? <SourceReferenceList taskId={taskId} refs={selected} label="本轮选用的原文入口" />
        : <p>本轮尚未记录用于回答的原文。</p>}
    </section>
    <section aria-label="出处与限制">
      <h4>出处与限制</h4>
      <p>尚未查阅或未继续展开，不表示这部分资料不相关。</p>
      {sourceTextTruncated === true && <p className="notice">本次输入存在截断，不能据此判断完整内容；请打开出处核对。</p>}
      {!!limitations.length && <ul>{limitations.map((limit, index) => <li key={index}>{limit}</li>)}</ul>}
    </section>
    <details className="cq-source-details">
      <summary>已返回的目录与访问记录 · {nodes.length} 条位置记录</summary>
      <p>“已返回原文”表示系统取得了对应正文，不代表语义核验已完成；资料或章节中的部分正文返回，也不等于整体读完。</p>
      <p>这里只列出本轮返回的目录与访问记录。未返回的位置没有逐项阅读状态，不能把它们全部统计为“尚未查阅”。</p>
      {structure.nodes_total !== undefined && <p>资料结构共 {totalText(structure.nodes_total)} 个位置（含资料、章节和片段），不等于片段总数或未读数量。</p>}
      {structure.nodes_truncated && <p className="notice">本次返回的目录与访问记录已截取，仅展示其中 {nodes.length} 条。其余记录的具体状态未在这里展开。</p>}
      {groups.slice(0, visibleDocuments).map(group => <ReadingGroup key={group.document.id} taskId={taskId} {...group} capacityOverride={capacityOverride} />)}
      {groups.length > visibleDocuments && <button type="button" onClick={() => setVisibleDocuments(value => value + 6)}>
        显示更多资料（已显示 {Math.min(visibleDocuments, groups.length)} / {groups.length}）
      </button>}
      {!!ungrouped.length && <ReadingGroup taskId={taskId} nodes={ungrouped} capacityOverride={capacityOverride} />}
      {!nodes.length && <p>没有保存可展示的阅读位置。</p>}
    </details>
    {!!supplemental.length && <details className="cq-source-details">
      <summary>补充阅读 · {supplemental.length} 处原文</summary>
      <p>本轮在已获准范围内补充取得了以下原文。打开入口可核对出处，不会自动发起后续任务。</p>
      <SourceReferenceList taskId={taskId} refs={supplemental} label="补充阅读原文" />
    </details>}
    {children}
  </section>;
}
