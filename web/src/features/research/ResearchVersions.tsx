import { useEffect, useRef, useState, type ReactNode } from "react";
import { api, errorText } from "../../api";
import type { AgentResult } from "../agent/TaskAnswer";
import type { ReviewSource } from "./review";

type Version = {
  command_seq: number; created_at: string; state: string; title: string | null;
  document_sha256: string | null; source_binding: ReviewSource | null;
  availability: "available" | "source_unavailable";
};
type Page = { items: Version[]; next_after_command_seq: number | null };
export type ResearchVersion = { source_binding: ReviewSource; result: AgentResult; source_brief: object };

export function ResearchVersions({ taskId, initialCommandSeq, renderVersion }: {
  taskId: string; initialCommandSeq?: number; renderVersion: (version: ResearchVersion) => ReactNode;
}) {
  const [items, setItems] = useState<Version[]>([]);
  const [cursor, setCursor] = useState<number | null>(null);
  const [selected, setSelected] = useState<number | undefined>(initialCommandSeq);
  const [version, setVersion] = useState<ResearchVersion>();
  const [busy, setBusy] = useState(false);
  const [loadingList, setLoadingList] = useState(false);
  const [error, setError] = useState<unknown>();
  const [listError, setListError] = useState<unknown>();
  const request = useRef<AbortController | null>(null);
  const listRequest = useRef<AbortController | null>(null);
  const scope = useRef(taskId);
  scope.current = taskId;
  async function load(after = 0) {
    listRequest.current?.abort();
    const controller = new AbortController(); listRequest.current = controller;
    setLoadingList(true); setListError(undefined);
    try {
      const page = await api<Page>(`/api/agent/tasks/${encodeURIComponent(taskId)}/research-versions?after_command_seq=${after}&limit=20`, undefined, undefined, controller.signal);
      if (controller.signal.aborted || scope.current !== taskId) return;
      if (!Array.isArray(page.items) || page.items.length > 20 || page.items.some((item, index) => !Number.isSafeInteger(item.command_seq)
          || item.command_seq <= after || (index > 0 && item.command_seq <= page.items[index - 1]!.command_seq))
          || (page.next_after_command_seq !== null && (!Number.isSafeInteger(page.next_after_command_seq) || page.next_after_command_seq <= after)))
        throw new Error("稿件版本列表格式不完整，请重新读取。");
      setItems(old => after === 0 ? page.items : [...new Map([...old, ...page.items].map(item => [item.command_seq, item])).values()]);
      setCursor(page.next_after_command_seq);
    } catch (failure) { if (!controller.signal.aborted) { setListError(failure); if (!after) { setItems([]); setCursor(null); } } }
    finally { if (!controller.signal.aborted) setLoadingList(false); }
  }
  async function choose(commandSeq: number) {
    request.current?.abort();
    const controller = new AbortController(); request.current = controller;
    setSelected(commandSeq); setVersion(undefined); setBusy(true); setError(undefined);
    try {
      const value = await api<ResearchVersion>(`/api/agent/tasks/${encodeURIComponent(taskId)}/research-version?command_seq=${commandSeq}`, undefined, undefined, controller.signal);
      if (controller.signal.aborted || scope.current !== taskId) return;
      if (value.source_binding.source_task_id !== taskId || value.source_binding.command_seq !== commandSeq
          || value.result.status !== "school_document_ready" || !value.result.school_document
          || !value.source_brief || typeof value.source_brief !== "object")
        throw new Error("此稿件版本无法完整读取，请重新选择。");
      setVersion(value);
    } catch (failure) { if (!controller.signal.aborted) setError(failure); }
    finally { if (!controller.signal.aborted) setBusy(false); }
  }
  useEffect(() => {
    setItems([]); setCursor(null); setVersion(undefined); setSelected(initialCommandSeq);
    void load();
    if (initialCommandSeq !== undefined) void choose(initialCommandSeq);
    return () => { listRequest.current?.abort(); request.current?.abort(); };
  }, [taskId, initialCommandSeq]);
  return <section aria-label="研究稿件版本">
    <div className="toolbar">
      <label>稿件版本 <select value={selected ?? ""} disabled={busy}
        onChange={event => { if (event.target.value) void choose(Number(event.target.value)); }}>
        {selected === undefined && <option value="">选择已保存稿件</option>}
        {initialCommandSeq !== undefined && !items.some(item => item.command_seq === initialCommandSeq)
          && <option value={initialCommandSeq}>当前稿 · 提交 {initialCommandSeq}</option>}
        {selected !== undefined && selected !== initialCommandSeq && !items.some(item => item.command_seq === selected)
          && <option value={selected}>已选历史稿 · 提交 {selected}</option>}
        {items.map(item => <option key={item.command_seq} value={item.command_seq}
          disabled={item.availability !== "available"}>
          {item.command_seq === initialCommandSeq ? "当前稿" : "历史稿"} · 提交 {item.command_seq} · {item.title || "研究计划书"}{item.availability !== "available" ? " · 当前不可用" : ""}
        </option>)}
      </select></label>
      <button type="button" disabled={loadingList || busy} onClick={() => { void load(); if (selected !== undefined) void choose(selected); }}>刷新版本列表</button>
      {cursor !== null && <button type="button" disabled={loadingList} onClick={() => void load(cursor)}>读取更多版本</button>}
    </div>
    {busy && <p role="status">正在读取所选稿件版本…</p>}
    {!!listError && <p role="alert" className="notice">{errorText(listError)}</p>}
    {!!error && <p role="alert" className="notice">所选版本当前不可读取：{errorText(error)}
      {selected !== undefined && <button type="button" disabled={busy} onClick={() => void choose(selected)}>重新读取所选版本</button>}
    </p>}
    {!loadingList && !items.length && initialCommandSeq === undefined && <p className="muted">本任务尚无可读取的已完成稿件。</p>}
    {version && <><p className="muted">正在查看提交 {version.source_binding.command_seq} 的稿件；下方评审与导出使用此版本。</p>{renderVersion(version)}</>}
  </section>;
}
