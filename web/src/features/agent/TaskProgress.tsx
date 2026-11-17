import { progressText, taskStateText } from "./progress";
import { money, progressContext, type ProgressBudget, type PublicProgressEvent } from "./taskProgressProjection";

export function TaskProgress({ events, task, hasMore, onMore }: {
  events: PublicProgressEvent[]; task: ProgressBudget; hasMore: boolean; onMore: () => void;
}) {
  const context = progressContext(events);
  const guidance: Record<string, string> = {
    queued: "正在排队，暂时不需要你操作。",
    running: "正在处理，你可以继续浏览；进度会自动更新。",
    waiting_input: "需要你确认或补充信息，请查看下方提示后继续。",
    succeeded: "本轮已完成，可在下方查看结果和引用。",
    partial: "本轮已完成部分内容，请查看下方尚未解决的问题。",
    failed: "本轮未完成，请先查看问题说明和已保留的处理记录。",
    cancelled: "任务已停止，已产生的处理记录和费用仍会保留。",
  };
  return <section className="card" aria-label="任务处理进度">
    <h3>处理进度 · {taskStateText(task.state)}</h3>
    {context.latestStage && <p>当前进展：{progressText(context.latestStage)}</p>}
    {guidance[task.state] && <p role="status">{guidance[task.state]}</p>}
    <p className="usage">已记录费用 {money(task.known_ncny)} · 暂占额度（待结算）{money(task.reserved_ncny)}</p>
    <p className="muted">暂占额度不等于实际费用；没有可确认的金额时显示“—”。</p>
    {task.state === "reconciling" && <p className="notice" role="status">上一步是否成功还在核对中。请等待核对完成，不要重复提交；暂占额度会继续保留。</p>}
    <details>
      <summary>查看处理记录 · 已读取 {events.length} 条</summary>
      <p className="usage">模型调用 {task.model_calls ?? "—"}{task.profile?.model_calls ? ` / ${task.profile.model_calls}` : ""}
        {" · "}工具调用 {task.tool_calls ?? "—"}{task.profile?.tool_calls ? ` / ${task.profile.tool_calls}` : ""}</p>
      {context.responsibility && <p className="muted">最近阶段：{context.responsibility}</p>}
      <ol className="events" role="log" aria-label="按顺序追加的执行记录" aria-live="polite" aria-relevant="additions text" aria-atomic="false">
        {context.rows.map(({ event, tool, tokens, kind }, index) => <li key={event.seq ?? index}>
          <strong>{progressText(event)}</strong>
          {kind && <span className="tag">{kind}</span>}
          {tool && <p style={{ overflowWrap: "anywhere" }}>工具：{tool}</p>}
          {tokens && <p className="muted">本次已登记输入 {tokens.input} / 输出 {tokens.output} tokens</p>}
          <details><summary>技术详情</summary><pre>{JSON.stringify(event.payload || event.data || event, null, 2)}</pre></details>
        </li>)}
      </ol>
      {!events.length && <p className="muted">还没有收到处理记录，进度更新后会显示在这里。</p>}
      {hasMore && <button type="button" onClick={onMore}>读取更多处理记录</button>}
    </details>
  </section>;
}
