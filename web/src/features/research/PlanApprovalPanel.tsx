import { useState } from "react";
import { changedPlanQueries, planOperationLabels, type ApprovedExecutionPlan, type ExecutionPlan,
  type PlanApprovalRequest, type PlanDecision } from "./executionPlan";
import "./executionPlan.css";

const resourceLabels: Record<string, string> = {
  required_model_calls: "所需模型请求额度", required_tool_calls: "所需资料处理请求额度",
  reserved_output_tokens: "文本生成额度（token）",
};
const questions = (plan: ExecutionPlan) => plan.nodes.filter(node => node.operation === "retrieve_review");
const questionName = (plan: ExecutionPlan, id: string) => `研究问题 ${questions(plan).findIndex(node => node.target_id === id) + 1}`;

function PlanDetails({ plan, sha256 }: { plan: ExecutionPlan; sha256: string }) {
  return <details className="cq-execution-details">
    <summary>查看处理说明与记录</summary>
    <p>确认后，会查找并核对相关资料，结合研究要求拟定方法，再整理方案、检查遗漏和不同意见。
      {plan.nodes.some(node => node.operation === "write_review") && "本次还包括正文整理和逐节检查。"}
      资料不足或结论有分歧时，会说明尚未解决的问题。</p>
    <details className="cq-execution-technical">
      <summary>运行记录与版本信息</summary>
      <p className="muted">以下信息用于核对执行记录，理解研究内容不需要阅读此部分。</p>
      <ul className="cq-execution-nodes">
        {plan.nodes.map(node => <li key={node.id}>
          <div className="cq-execution-node-heading"><strong>{planOperationLabels[node.operation]}</strong><code>{node.id}</code></div>
          <p className="muted">{node.depends_on.length ? `前置记录：${node.depends_on.join("、")}` : "无前置记录"}</p>
        </li>)}
      </ul>
      <dl className="cq-execution-resources">
        {Object.entries(plan.resource_requirement).filter(([key, value]) => resourceLabels[key] && typeof value === "number").map(([key, value]) => <div key={key}>
          <dt>{resourceLabels[key]}</dt><dd>{String(value)}</dd>
        </div>)}
      </dl>
      <p className="muted">这些是容量检查数据，并非费用报价。实际费用和待核费用见任务记录。</p>
      <dl className="cq-execution-bindings">
        <div><dt>查找方案标识</dt><dd><code>{sha256}</code></dd></div>
        <div><dt>研究资料版本标识</dt><dd><code>{plan.brief_sha256}</code></dd></div>
        {plan.parent_sha256 && <div><dt>上一版方案标识</dt><dd><code>{plan.parent_sha256}</code></dd></div>}
      </dl>
    </details>
  </details>;
}

export function PlanApprovalPanel({ request, busy = false, disabled = false, sourceCount, onDecision, onRefresh }: {
  request: PlanApprovalRequest;
  busy?: boolean;
  disabled?: boolean;
  sourceCount?: number;
  onDecision: (decision: PlanDecision) => Promise<void> | void;
  onRefresh: () => Promise<void> | void;
}) {
  const { plan, plan_sha256: sha256 } = request;
  const [edits, setEdits] = useState<Record<string, string>>({});
  const changes = changedPlanQueries(plan, edits);
  const changed = Object.keys(changes).length > 0;
  const invalid = Object.values(changes).some(query => !query || query.length > 2000);
  const retrieval = questions(plan);
  const canAdjust = plan.version < 5;
  return <section className="cq-execution-plan" aria-label="确认资料查找范围">
    <header className="cq-execution-heading">
      <div><p className="cq-execution-eyebrow">开始查找前</p><h3>这些是你想查找的内容吗？</h3></div>
      <span className="cq-execution-status" role="status">第 {plan.version} 版 · 等待你确认</span>
    </header>
    <p className="cq-execution-intro">请核对下面的研究问题。你可以调整用来查找资料的词，确认后再开始处理。</p>
    <p className="cq-execution-scope">查找范围：{sourceCount ? `本次已选的 ${sourceCount} 份资料` : "本次任务可访问的资料"}。实际费用沿用本任务的记录。</p>
    <fieldset disabled={disabled || busy} className="cq-execution-queries">
      <legend>你关心的研究问题</legend>
      {retrieval.map((node, index) => <div className="cq-execution-query" key={node.id}>
        <p className="cq-execution-question-number">研究问题 {index + 1}</p>
        <h4>{node.goal}</h4>
        <label htmlFor={`plan-query-${sha256}-${index}`}>用这些词查找资料（可调整）</label>
        <textarea id={`plan-query-${sha256}-${index}`} rows={2} maxLength={2000}
          aria-label={`研究问题 ${index + 1} 的查找词`}
          value={edits[node.target_id] ?? node.query ?? ""} readOnly={!canAdjust}
          onChange={event => setEdits(current => ({ ...current, [node.target_id]: event.target.value }))} />
        {Object.hasOwn(changes, node.target_id) && <p className="cq-execution-before"><strong>修改前：</strong>{node.query}</p>}
      </div>)}
    </fieldset>
    {!!plan.query_changes.length && <details className="cq-execution-diff" open>
      <summary>已修改 {plan.query_changes.length} 处查找词，请再核对一次</summary>
      {plan.query_changes.map(change => <div key={change.target_id}>
        <strong>{questionName(plan, change.target_id)}</strong><p><span>修改前</span> {change.before}</p><p><span>修改后</span> {change.after}</p>
      </div>)}
    </details>}
    <PlanDetails plan={plan} sha256={sha256} />
    {changed ? <p className="cq-execution-guidance" role="status">先保存修改并核对新版，随后就可以开始查找。保存修改本身不会开始查找资料。</p>
      : <p className="cq-execution-guidance">调整查找词不会改变研究问题、资料权限或任务预算。需要改变这些要求时，请回到研究资料中修改。</p>}
    {!canAdjust && <p className="muted">本轮已达到调整次数上限。可以确认当前内容，或结束本任务、修改研究要求后重新开始。</p>}
    {invalid && <p role="alert" className="notice">查找词不能为空，每项最多 2000 个字符。</p>}
    <div className="cq-execution-actions">
      <button type="button" className="primary" disabled={busy || disabled || changed}
        onClick={() => void onDecision({ plan_sha256: sha256, action: "approve" })}>确认第 {plan.version} 版，开始查找</button>
      {canAdjust && <button type="button" disabled={busy || disabled || !changed || invalid}
        onClick={() => void onDecision({ plan_sha256: sha256, action: "adjust_queries", queries: changes })}>保存修改，重新核对</button>}
      {changed && <button type="button" disabled={busy || disabled} onClick={() => setEdits({})}>撤回未保存的修改</button>}
      <button type="button" disabled={busy} onClick={() => void onRefresh()}>刷新当前内容</button>
    </div>
    {busy && <p className="muted" role="status">正在保存，请稍候…</p>}
    {disabled && !busy && <p className="muted" role="status">现在暂时无法开始。你可以刷新状态；服务恢复后，已保存的内容仍可继续使用。</p>}
  </section>;
}

export function ExecutionPlanRecord({ record }: { record: ApprovedExecutionPlan }) {
  return <section className="cq-execution-record" aria-label="这次确认的查找内容">
    <strong>已确认第 {record.plan.version} 版查找内容</strong>
    <p className="muted">本次处理以你确认的内容为准。结果中的缺口和不同意见仍需结合原文判断。</p>
    <details><summary>回看确认时的研究问题与查找词</summary>
      {questions(record.plan).map((node, index) => <div key={node.id}><h4>研究问题 {index + 1} · {node.goal}</h4><p>{node.query}</p></div>)}
    </details>
    <PlanDetails plan={record.plan} sha256={record.plan_sha256} />
  </section>;
}
