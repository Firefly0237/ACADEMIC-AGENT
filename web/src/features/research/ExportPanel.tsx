import { useEffect, useRef, useState } from "react";
import { api, errorText } from "../../api";

type ExportJob = {
  id: string; state: string; availability: string; error_code?: string | null;
  created_at: string; artifacts: Array<{ format: string; size_bytes: number; content_url: string }>;
};
type Page = { items: ExportJob[]; next_cursor: string | null };
const states: Record<string, string> = { queued: "等待排版", running: "正在排版与检查", succeeded: "排版完成", failed: "未完成", cancelled: "已取消" };

export function ExportPanel({ taskId }: { taskId: string }) {
  const [jobs, setJobs] = useState<ExportJob[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [error, setError] = useState<unknown>();
  const [busy, setBusy] = useState(false);
  const operation = useRef<string | null>(null);
  const active = jobs.some(job => ["queued", "running"].includes(job.state));
  const mounted = useRef(true);
  async function load(next?: string | null, signal?: AbortSignal) {
    const page = await api<Page>(`/api/agent/tasks/${taskId}/exports?limit=10${next ? `&cursor=${encodeURIComponent(next)}` : ""}`,
      undefined, undefined, signal);
    if (!mounted.current) return;
    setJobs(old => next ? [...new Map([...old, ...page.items].map(item => [item.id, item])).values()] : page.items);
    setCursor(page.next_cursor);
  }
  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    void load(null, controller.signal).catch(cause => { if (cause.name !== "AbortError") setError(cause); });
    return () => { mounted.current = false; controller.abort(); };
  }, [taskId]);
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    const timer = setTimeout(() => void load(null, controller.signal).catch(cause => {
      if (cause.name !== "AbortError") { setJobs([]); setError(cause); }
    }), 1500);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [active, jobs]);
  async function create() {
    setBusy(true); setError(undefined);
    operation.current ||= crypto.randomUUID();
    try {
      const job = await api<ExportJob>(`/api/agent/tasks/${taskId}/exports`, {
        operation_id: operation.current, formats: ["docx", "html", "pdf"]
      });
      operation.current = null;
      if (mounted.current) { setJobs(old => [job, ...old.filter(item => item.id !== job.id)]); await load(); }
    } catch (cause) { if (mounted.current) setError(cause); }
    finally { if (mounted.current) setBusy(false); }
  }
  return <section className="card" aria-label="研究计划书导出">
    <h3>导出可编辑草案</h3>
    <p className="muted">基于本任务已审查的同一版本排版，不重新调用模型。下载前仍核对当前权限与档案；公开模板和开放字体需按院系当期要求复核，签名与日期由你填写。</p>
    <button disabled={busy || active} onClick={create}>{busy ? "正在提交…" : active ? "排版任务执行中" : "生成 DOCX / HTML / PDF"}</button>
    <button disabled={busy} onClick={() => load().catch(setError)}>刷新导出状态</button>
    {!!error && <p role="alert" className="notice">{errorText(error)} 请检查档案或来源是否变化；导出失败不会自动重试。</p>}
    <ul className="export-jobs">{jobs.map(job => <li key={job.id}>
      <strong>{states[job.state] || job.state}</strong> · {new Date(job.created_at).toLocaleString("zh-CN")}
      {job.availability === "source_unavailable" && <p className="notice">当前来源或档案已变化，旧导出不可下载。</p>}
      {job.error_code && <p className="notice">本次排版未通过（{job.error_code}）。保留该记录后可明确另起排版。</p>}
      {job.availability === "available" && job.artifacts.map(artifact =>
        artifact.content_url.startsWith(`/api/agent/exports/${job.id}/artifacts/`) && <a className="button" key={artifact.format}
          href={artifact.content_url} download>{artifact.format.toUpperCase()} · {(artifact.size_bytes/1024).toFixed(1)} KiB</a>)}
      {["queued", "running"].includes(job.state) && <button onClick={async () => {
        try { await api(`/api/agent/exports/${job.id}/cancel`, {operation_id: crypto.randomUUID()}); await load(); }
        catch(cause) { setError(cause); }
      }}>取消排版</button>}
    </li>)}</ul>
    {cursor && <button disabled={busy} onClick={() => load(cursor).catch(setError)}>查看更早的导出记录</button>}
  </section>;
}
