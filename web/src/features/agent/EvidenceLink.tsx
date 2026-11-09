import { useEffect, useId, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, errorText, paperPath, type PaperRef } from "../../api";
import type { ReviewSource } from "../research/review";

export type QuoteLocator = { span_start: number; span_end: number; coordinate_system: string };
export type QuoteFocus = { quote: string; quote_locator?: QuoteLocator };
type Evidence = QuoteLocator & {
  text: string;
  source_ref?: { kind: string; id: string };
  selection?: { kind: string; id: string };
  file_id?: string;
  source_url?: string;
  coverage?: string;
  coverage_kind?: string;
  locator?: { kind?: string; page_1based?: number };
};

// Python source offsets count Unicode codepoints, not JavaScript UTF-16 units.
// Only the exact server-issued coordinates and literal quote can select text.
function quoteParts(value: Evidence, focus?: QuoteFocus) {
  if (!focus?.quote_locator || typeof focus.quote !== "string") return null;
  const locator = focus.quote_locator, points = Array.from(value.text);
  if (!Number.isSafeInteger(value.span_start) || !Number.isSafeInteger(value.span_end)
      || !Number.isSafeInteger(locator.span_start) || !Number.isSafeInteger(locator.span_end)
      || typeof value.coordinate_system !== "string" || value.coordinate_system !== locator.coordinate_system
      || !value.coordinate_system.startsWith("unicode_codepoints_in_")
      || value.span_end - value.span_start !== points.length
      || locator.span_start < value.span_start || locator.span_end > value.span_end
      || locator.span_end <= locator.span_start) return null;
  const start = locator.span_start - value.span_start, end = locator.span_end - value.span_start;
  if (points.slice(start, end).join("") !== focus.quote) return null;
  return [points.slice(0, start).join(""), points.slice(start, end).join(""), points.slice(end).join("")];
}

function sourcePath(value: Evidence): string | null {
  const source = value.selection || value.source_ref;
  if (source && ["metadata_projection", "native_representation"].includes(source.kind))
    return paperPath(source as PaperRef);
  const fileId = value.file_id || (source?.kind === "material_file" ? source.id : null);
  return fileId ? `/materials/${encodeURIComponent(fileId)}` : null;
}

function allowedSourceURL(value?: string): string | undefined {
  if (!value) return undefined;
  try {
    const url = new URL(value, window.location.origin);
    return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password ? url.href : undefined;
  } catch { return undefined; }
}

export function EvidenceLink({ taskId, refId, focus, sourceBinding }: { taskId: string; refId: string; focus?: QuoteFocus; sourceBinding?: ReviewSource }) {
  const [value, setValue] = useState<Evidence>();
  const [error, setError] = useState<unknown>();
  const [busy, setBusy] = useState(false);
  const request = useRef<AbortController | null>(null);
  const highlight = useRef<HTMLElement | null>(null);
  const regionId = useId();
  useEffect(() => {
    setValue(undefined); setError(undefined); setBusy(false);
    return () => request.current?.abort();
  }, [taskId, refId, sourceBinding?.command_seq]);
  const parts = value ? quoteParts(value, focus) : null;
  useEffect(() => {
    if (value && parts) highlight.current?.scrollIntoView({ block: "nearest" });
  }, [value]);
  async function open() {
    if (value) { setValue(undefined); return; }
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setValue(undefined); setError(undefined); setBusy(true);
    try {
      const result = await api<Evidence>(`/api/agent/tasks/${encodeURIComponent(taskId)}/evidence?ref_id=${encodeURIComponent(refId)}${sourceBinding ? `&source_command_seq=${sourceBinding.command_seq}` : ""}`,
        undefined, undefined, controller.signal);
      if (typeof result.text !== "string") throw new Error("原文窗口格式不正确，请重新读取。");
      if (!controller.signal.aborted) setValue(result);
    } catch (failure) { if (!controller.signal.aborted) setError(failure); }
    finally { if (!controller.signal.aborted) setBusy(false); }
  }
  const coverage = value?.coverage || value?.coverage_kind;
  const path = value ? sourcePath(value) : null;
  const original = allowedSourceURL(value?.source_url);
  return <div className="evidence-link">
    <button className="link-button" disabled={busy} aria-expanded={!!value} aria-controls={regionId} onClick={open}>
      {busy ? "正在核对原文权限…" : value ? "收起原文证据" : focus?.quote_locator ? "定位引用原文 ↗" : "查看当前可授权原文 ↗"}
    </button>
    {!!error && <p className="notice" role="alert">{errorText(error)} 原文未展示；可重新读取。</p>}
    {value && <section id={regionId} className="evidence" aria-label="原文证据窗口">
      <p><strong>当前可授权原文</strong> · {coverage === "metadata" || coverage === "metadata_only" ? "仅题录与摘要" : coverage === "native_literal" ? "来源原始文本" : "材料已解析文本"}</p>
      <p className="muted">原文字符 {value.span_start}–{value.span_end}，结束位置不含在内；字符位置不等同于 PDF 页码。</p>
      {parts ? <p className="evidence-location">已定位引用：字符 {focus!.quote_locator!.span_start}–{focus!.quote_locator!.span_end}</p>
        : focus?.quote_locator ? <p className="notice" role="status">引用坐标与当前窗口无法精确对应，未自动高亮。请核对完整原文窗口。</p> : null}
      <pre className="evidence-text">{parts ? <>{parts[0]}<mark ref={highlight} tabIndex={-1}>{parts[1]}</mark>{parts[2]}</> : value.text}</pre>
      {value.locator?.kind === "pdf" && Number.isSafeInteger(value.locator.page_1based) && value.locator.page_1based! > 0
        && <p className="muted">解析来源定位：原 PDF 第 {value.locator.page_1based} 页。请打开材料核对该页。</p>}
      {path && <p><Link to={path}>打开来源详情与原文入口 ↗</Link></p>}
      {original && <p><a href={original} target="_blank" rel="noreferrer">打开原始材料 ↗</a></p>}
    </section>}
  </div>;
}
