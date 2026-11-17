import { useEffect, useRef, useState } from "react";
import * as pdfjs from "pdfjs-dist";
import worker from "pdfjs-dist/build/pdf.worker.min.mjs?url";
pdfjs.GlobalWorkerOptions.workerSrc = worker;
export function PDF({ url }: { url: string }) {
  return <PDFReader key={url} url={url} />;
}
function PDFReader({ url }: { url: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const rendering = useRef<pdfjs.RenderTask | null>(null);
  const [document, setDocument] = useState<pdfjs.PDFDocumentProxy | null>(null);
  const [page, setPage] = useState(1);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(true);
  const count = document?.numPages || 0;
  useEffect(() => {
    let disposed = false;
    // Canvas rendering only: no annotation action layer, scripting manager or XFA forms.
    const task = pdfjs.getDocument({
      url,
      enableXfa: false,
      cMapUrl: "/pdfjs/6.5.438/cmaps/",
      cMapPacked: true,
      standardFontDataUrl: "/pdfjs/6.5.438/standard_fonts/",
      wasmUrl: "/pdfjs/6.5.438/wasm/",
    });
    task.promise
      .then((doc) => {
        if (disposed) return;
        setDocument(doc);
      })
      .catch((e) => {
        if (!disposed) { setError(e.message || "PDF 无法加载，请使用原文链接。"); setBusy(false); }
      });
    return () => {
      disposed = true;
      rendering.current?.cancel();
      void task.destroy();
    };
  }, [url]);
  useEffect(() => {
    if (!document) return;
    let disposed = false;
    let ownRender: pdfjs.RenderTask | null = null;
    setBusy(true); setError("");
    const previous = rendering.current;
    void (async () => {
      // PDF.js forbids concurrent render tasks on the same canvas. Cancellation
      // is asynchronous: await the previous task before the next draw.
      await previous?.promise.catch(() => undefined);
      const source = await document.getPage(page);
      if (disposed || !canvas.current) return;
      const viewport = source.getViewport({ scale: 1.2 });
      if (![viewport.width, viewport.height].every(value => Number.isFinite(value) && value > 0 && value < 100000))
        throw new Error("该页尺寸超出在线查看范围，请使用原文入口。");
      const outputScale = Math.min(window.devicePixelRatio || 1, 2,
        Math.sqrt(8000000 / (viewport.width * viewport.height)));
      canvas.current.height = Math.ceil(viewport.height * outputScale);
      canvas.current.width = Math.ceil(viewport.width * outputScale);
      canvas.current.style.width = `${viewport.width}px`;
      canvas.current.style.height = `${viewport.height}px`;
      ownRender = source.render({ canvas: canvas.current, viewport,
        transform: outputScale === 1 ? undefined : [outputScale, 0, 0, outputScale, 0, 0] });
      rendering.current = ownRender;
      await ownRender.promise;
    })().catch(cause => {
      if (!disposed) setError(cause.message || "该页无法显示，请使用原文入口。");
    }).finally(() => { if (!disposed) setBusy(false); });
    return () => { disposed = true; ownRender?.cancel(); };
  }, [document, page]);
  return (
    <section className="pdf">
      <div className="toolbar">
        <button disabled={!count || page <= 1} onClick={() => setPage(page - 1)}>
          上一页
        </button>
        <label>
          第{" "}
          <input
            aria-label="PDF页码"
            type="number"
            min={1}
            max={count || 1}
            value={page}
            disabled={!count}
            onChange={(e) =>
              setPage(Math.max(1, Math.min(count, Number(e.target.value) || 1)))
            }
          />{" "}
          / {count || "…"} 页
        </label>
        <button
          disabled={!count || page >= count}
          onClick={() => setPage(page + 1)}
        >
          下一页
        </button>
        <a href={url} target="_blank" rel="noreferrer">
          另窗阅读
        </a>
      </div>
      {busy && <p role="status">正在显示第 {page} 页…</p>}
      {error && <p role="alert">{error}</p>}
        <div className="pdf-scroll" hidden={!!error}>
          <canvas ref={canvas} aria-label={`PDF 第 ${page} 页`} aria-busy={busy} />
        </div>
    </section>
  );
}
