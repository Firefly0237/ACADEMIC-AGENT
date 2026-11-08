import { useEffect, useRef, useState } from "react";
import { api, errorText, APIError } from "../../api";
import { Markdown } from "../../Markdown";
import { TaskAnswer, type AgentResult } from "./TaskAnswer";
import { resultAvailabilityText, taskStateText } from "./progress";

type Turn = {
  command_seq: number;
  command_kind: string;
  user_text: string;
  created_at: string;
  state: string;
  brief_revision: number | null;
  answer: AgentResult | null;
  answer_availability: string;
};
type Page = { items: Turn[]; next_after_seq: number | null };

export function TaskConversation({ taskId, revision, state, onUnavailable }: {
  taskId: string; revision: number; state: string; onUnavailable?: (failure: APIError) => void;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [cursor, setCursor] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<unknown>();
  const request = useRef<AbortController | null>(null);
  const identity = useRef(taskId);
  async function load(after = 0) {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setBusy(true); setError(undefined);
    try {
      const value = await api<Page>(`/api/agent/tasks/${encodeURIComponent(taskId)}/conversation?after_seq=${after}&limit=20`,
        undefined, undefined, controller.signal);
      if (controller.signal.aborted || identity.current !== taskId) return;
      if (!Array.isArray(value.items) || value.items.length > 20 || value.items.some((turn, index) => !Number.isSafeInteger(turn.command_seq)
          || turn.command_seq <= after || (index > 0 && turn.command_seq <= value.items[index-1].command_seq)
          || typeof turn.user_text !== "string") || (value.next_after_seq !== null
          && (!value.items.length || value.next_after_seq !== value.items.at(-1)!.command_seq)))
        throw new Error("对话记录格式不正确，未合并不完整历史。请重新读取。");
      setTurns(items => after === 0 ? value.items : Array.from(new Map([...items, ...value.items].map(turn => [turn.command_seq, turn])).values()).sort((a,b) => a.command_seq-b.command_seq));
      setCursor(value.next_after_seq); setLoaded(true);
    } catch (failure) {
      if (!controller.signal.aborted) {
        // Do not retain earlier authorized answers after a denied refresh.
        if (after === 0 || (failure instanceof APIError && [401, 403, 404].includes(failure.status))) {
          setTurns([]); setCursor(null); setLoaded(false);
        }
        if (failure instanceof APIError && [401, 403, 404].includes(failure.status))
          onUnavailable?.(failure);
        setError(failure);
      }
    } finally { if (!controller.signal.aborted) setBusy(false); }
  }
  useEffect(() => {
    identity.current = taskId;
    setTurns([]); setCursor(null); setLoaded(false);
    void load();
    return () => request.current?.abort();
  }, [taskId, revision, state]);
  return <section className="task-conversation" aria-label="多轮对话记录">
    <h3>本任务对话</h3>
    <p className="muted">按提交顺序保存提问与可恢复回答。历史回答每次读取都会重新核对当前权限和上下文；执行事件另列在下方。</p>
    {turns.map(turn => <article className="conversation-turn" key={turn.command_seq}>
      <header><strong>{turn.command_kind === "start" ? "首次提问" : turn.command_kind === "resume" ? "补充信息" : "追问"}</strong>
        <small> · 提交序号 {turn.command_seq} · {turn.state === "done" ? "本轮已结束" : taskStateText(turn.state)}{turn.created_at ? ` · ${new Date(turn.created_at).toLocaleString("zh-CN")}` : ""}</small></header>
      <div className="conversation-question"><Markdown text={turn.user_text} /></div>
      {turn.answer && ["available", "available_for_input"].includes(turn.answer_availability) ? <details className="conversation-answer">
        <summary>查看本轮已保存回答</summary>
        <TaskAnswer taskId={taskId} result={turn.answer} historical />
      </details> : <p className="muted">{turn.answer_availability === "not_retained"
        ? "较早版本未保留该轮可恢复回答。当前任务结果可在下方查看。"
        : resultAvailabilityText(turn.answer_availability)}</p>}
    </article>)}
    {busy && <p className="muted" role="status">正在读取对话记录…</p>}
    {!busy && loaded && turns.length === 0 && <p className="muted">尚无可读取的对话记录；当前任务状态和回答见下方。</p>}
    {!!error && <p className="notice" role="alert">{error instanceof APIError && error.status === 404
      ? "当前执行服务尚未提供完整对话读取；当前任务结果仍可查看。"
      : errorText(error)}</p>}
    <div className="toolbar">
      {cursor !== null && <button disabled={busy} onClick={() => void load(cursor)}>读取后续对话</button>}
      {!!error && <button disabled={busy} onClick={() => void load()}>重新读取对话</button>}
    </div>
  </section>;
}
