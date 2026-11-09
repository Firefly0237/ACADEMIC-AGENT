import { progressText, taskStateText } from "./progress";
import { money, progressContext, type ProgressBudget, type PublicProgressEvent } from "./taskProgress";

export function TaskProgress({ events, task, hasMore, onMore }: {
  events: PublicProgressEvent[]; task: ProgressBudget; hasMore: boolean; onMore: () => void;
}) {
  const context = progressContext(events);
  return <section className="card" aria-label="Agent 执行过程">
    <h3>执行进展 · {taskStateText(task.state)}</h3>
    {context.latestStage && <p>{progressText(context.latestStage)}</p>}
    {context.responsibility && <p className="muted">最近阶段职责：{context.responsibility}</p>}
    <p className="usage">模型额度已登记 {task.model_calls ?? "—"}{task.profile?.model_calls ? ` / ${task.profile.model_calls}` : ""}
      {" · "}工具额度已登记 {task.tool_calls ?? "—"}{task.profile?.tool_calls ? ` / ${task.profile.tool_calls}` : ""}
      {" · "}已知费用 {money(task.known_ncny)} · 未结算预留 {money(task.reserved_ncny)}</p>
    {task.state === "reconciling" && <p className="notice" role="status">调用结果仍待核对，相关预留保留。</p>}
    <details open={["queued", "running", "reconciling"].includes(task.state)}>
      <summary>步骤与工具记录 · 已读取 {events.length} 条事件</summary>
      <ol className="events" role="log" aria-label="按顺序追加的执行记录" aria-live="polite" aria-relevant="additions text" aria-atomic="false">
        {context.rows.map(({ event, tool, tokens, kind }, index) => <li key={event.seq ?? index}>
          <strong>{progressText(event)}</strong>
          {kind && <span className="tag">{kind}</span>}
          {tool && <p style={{ overflowWrap: "anywhere" }}>工具：{tool}</p>}
          {tokens && <p className="muted">本次已登记输入 {tokens.input} / 输出 {tokens.output} tokens</p>}
          <details><summary>技术详情</summary><pre>{JSON.stringify(event.payload || event.data || event, null, 2)}</pre></details>
        </li>)}
      </ol>
      {!events.length && <p className="muted">等待工作流追加执行记录。</p>}
      {hasMore && <button type="button" onClick={onMore}>读取后续执行记录</button>}
    </details>
  </section>;
}
