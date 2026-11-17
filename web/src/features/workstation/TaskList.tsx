import { useEffect, useId, useRef, useState } from "react";
import type { Session } from "../../api";
import { taskStateText } from "../agent/progress";
import { WORKSPACE, workspaceError, workspaceRequest } from "./client";
import type { Page, TaskSummary } from "./types";

type StateGroup = "all" | "needs_input" | "active" | "completed" | "ended";
const groups: Array<[StateGroup, string]> = [
  ["all", "全部任务"],
  ["needs_input", "需要我确认或补充"],
  ["active", "处理中或核对中"],
  ["completed", "已完成"],
  ["ended", "部分完成、未完成或已停止"],
];
const categories: Record<string, string> = {
  qa: "文献问答", summary: "文献摘要", compare: "多文比较", stance: "观点依据",
  research_plan: "研究方案", experiment_design: "实验设计", research_suggestions: "研究建议",
  draft_intake: "草稿整理", research_review: "稿件评审", research_revision: "定向修订",
  research: "研究任务", other: "其他任务",
};
const emptyPage = (): Page<TaskSummary> => ({ items: [], next_cursor: null });
const dateText = (value: string) => new Date(value).toLocaleString("zh-CN", { hour12: false });

export function TaskList({ identity, onOpen, canOpen }: {
  identity: Session; onOpen: (task: TaskSummary) => void; canOpen: (task: TaskSummary) => boolean;
}) {
  const [group, setGroup] = useState<StateGroup>("all");
  const [refresh, setRefresh] = useState(0);
  const [tasks, setTasks] = useState(emptyPage);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const sequence = useRef(0);
  const request = useRef<AbortController | null>(null);
  const groupId = useId();

  async function load(cursor?: string, signal?: AbortSignal) {
    const params = new URLSearchParams({ limit: "20", state_group: group });
    if (cursor) params.set("cursor", cursor);
    return workspaceRequest<Page<TaskSummary>>(identity, `${WORKSPACE}/tasks?${params}`, { signal });
  }

  useEffect(() => {
    const controller = new AbortController();
    request.current?.abort();
    request.current = controller;
    const current = ++sequence.current;
    setTasks(emptyPage());
    setError("");
    if (!identity.authenticated) { setLoading(false); return () => controller.abort(); }
    setLoading(true);
    void load(undefined, controller.signal).then(value => {
      if (current === sequence.current && !controller.signal.aborted) setTasks(value);
    }).catch(cause => {
      if (current === sequence.current && !controller.signal.aborted) setError(workspaceError(cause));
    }).finally(() => {
      if (current === sequence.current && !controller.signal.aborted) setLoading(false);
    });
    return () => { request.current?.abort(); sequence.current += 1; };
  }, [identity.owner_id, identity.authenticated, group, refresh]);

  async function more() {
    if (!tasks.next_cursor || loading) return;
    const controller = new AbortController();
    request.current?.abort();
    request.current = controller;
    const current = ++sequence.current;
    setLoading(true); setError("");
    try {
      const next = await load(tasks.next_cursor, controller.signal);
      if (current !== sequence.current || controller.signal.aborted) return;
      setTasks(previous => ({
        items: [...new Map([...previous.items, ...next.items].map(item => [item.task_id, item])).values()],
        next_cursor: next.next_cursor,
      }));
    } catch (cause) {
      if (current === sequence.current && !controller.signal.aborted) setError(workspaceError(cause));
    } finally {
      if (current === sequence.current && !controller.signal.aborted) setLoading(false);
    }
  }

  return <section aria-labelledby={`${groupId}-heading`}>
    <h2 id={`${groupId}-heading`}>我的任务</h2>
    <p className="muted">查看问答、研究方案、评审与修订的进展。打开后可以继续原任务，接下来的工作由你决定。</p>
    <div className="cq-library-filters">
      <label htmlFor={groupId}>按状态查看
        <select id={groupId} value={group} onChange={event => setGroup(event.target.value as StateGroup)}>
          {groups.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
      </label>
      <button type="button" disabled={loading} onClick={() => setRefresh(value => value + 1)}>刷新任务状态</button>
    </div>
    <p className="muted">打开结果时会重新检查当前可访问范围。检索记录在“浏览与检索历史”中，文件排版与下载在对应任务内。</p>
    {loading && <p role="status">正在读取任务…</p>}
    {error && <p role="alert" className="notice">{error}</p>}
    {tasks.items.map(item => <article className="cq-task-row" key={item.task_id}>
      <div>
        <h3>{categories[item.category || item.kind] || "研究任务"}</h3>
        <p>{taskStateText(item.state)}</p>
        <small>{item.updated_at ? `任务更新于 ${dateText(item.updated_at)}` : `创建于 ${dateText(item.created_at)}`}</small>
        {item.state === "waiting_input" && <p>请打开任务，查看需要确认或补充的内容。</p>}
        {item.state === "reconciling" && <p>上次处理结果还在核对，请勿重复提交。</p>}
      </div>
      <button type="button" disabled={!canOpen(item)} onClick={() => onOpen(item)}>
        {item.state === "waiting_input" ? "查看待确认内容" : "打开任务"}
      </button>
    </article>)}
    {!loading && !error && !tasks.items.length && <p className="empty">
      {group === "all" ? "还没有任务。可以从选中文献提问，或先建立研究档案。" : "当前没有符合此状态的任务。"}
    </p>}
    {tasks.next_cursor && <button type="button" disabled={loading} onClick={() => void more()}>加载更多任务</button>}
  </section>;
}
