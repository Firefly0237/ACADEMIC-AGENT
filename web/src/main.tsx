import {
  StrictMode,
  Suspense,
  lazy,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";
import { createRoot } from "react-dom/client";
import {
  BrowserRouter,
  Link,
  Route,
  Routes,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";
import {
  api,
  APIError,
  errorText,
  paperPath,
  session,
  type Session,
  type Paper,
  type SearchPage,
  type PaperRef,
} from "./api";
const Markdown = lazy(() =>
  import("./Markdown").then((module) => ({ default: module.Markdown })),
);
const PDF = lazy(() =>
  import("./PDF").then((module) => ({ default: module.PDF })),
);
import "./style.css";
import type { SkillPin } from "./features/agent/ContextSettings";
import type {
  BriefField,
  BriefSnapshot,
  BriefSuggestion,
  BriefHelpRequest,
  DraftExtraction,
  PlanningInput,
} from "./features/research/types";
import { progressText, resultAvailabilityText, taskFailureText, taskKindText, taskStateText } from "./features/agent/progress";
import { canContinueTask } from "./features/agent/continuation";
import { newTaskProfile } from "./features/agent/profiles";
import type { AgentResult, CoverageReport, ResearchProgressReport } from "./features/agent/TaskAnswer";
import { CapabilityStatus } from "./features/agent/CapabilityStatus";
import { EvidenceLink } from "./features/agent/EvidenceLink";
const TaskAnswer = lazy(() => import("./features/agent/TaskAnswer").then(module => ({ default: module.TaskAnswer })));
const TaskCoverage = lazy(() => import("./features/agent/TaskAnswer").then(module => ({ default: module.TaskCoverage })));
const ResearchProgress = lazy(() => import("./features/agent/TaskAnswer").then(module => ({ default: module.ResearchProgress })));
const TaskConversation = lazy(() => import("./features/agent/TaskConversation").then(module => ({ default: module.TaskConversation })));

const WorkspacePanel = lazy(() =>
  import("./features/workstation/WorkspacePanel").then((module) => ({
    default: module.WorkspacePanel,
  })),
);

const ContextSettings = lazy(() =>
  import("./features/agent/ContextSettings").then((module) => ({
    default: module.ContextSettings,
  })),
);
const DraftCandidates = lazy(() => import("./features/research/DraftCandidates").then(module => ({ default: module.DraftCandidates })));

let current: Session | null = null;
function Notice({ error }: { error: unknown }) {
  return (
    <p className="notice" role="alert">
      {errorText(error)}
    </p>
  );
}
function App() {
  const [identity, setIdentity] = useState<Session | null>(null);
  const [error, setError] = useState<unknown>();
  useEffect(() => {
    session()
      .then((s) => {
        current = s;
        setIdentity(s);
      })
      .catch(setError);
  }, []);
  return (
    <div className="shell">
      <aside className="rail">
        <Link to="/" aria-label="CiteQuest首页" className="brand">
          CQ<span>CiteQuest</span>
        </Link>
        <nav>
          <Link to="/">
            ⌕ <span>学术搜索</span>
          </Link>
          <Link to="/workspace">
            ▤ <span>科研工作区</span>
          </Link>
          <Link to="/upload">
            ＋ <span>导入文献</span>
          </Link>
        </nav>
        <div className="account">
          {identity?.authenticated ? (
            <>
              <span className="avatar">研</span>
              <span>我的工作区</span>
              <button
                onClick={() =>
                  api("/api/auth/logout", {}).then(() => location.assign("/"))
                }
              >
                退出
              </button>
            </>
          ) : (
            <a href="/auth/login">登录个人中心 →</a>
          )}
          <small>计算机科学 · 有界公开语料</small>
        </div>
      </aside>
      <main>
        {error ? (
          <Notice error={error} />
        ) : !identity ? (
          <p className="loading">正在连接工作区…</p>
        ) : (
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/search" element={<Results />} />
            <Route path="/papers/:kind/:id" element={<DetailRoute />} />
            <Route path="/workspace" element={<Workspace />} />
            <Route path="/materials/:id" element={<MaterialDetailRoute />} />
            <Route path="/upload" element={<Upload />} />
            <Route
              path="*"
              element={
                <section className="content">
                  <h1>页面不存在</h1>
                  <Link to="/">返回首页</Link>
                </section>
              }
            />
          </Routes>
        )}
      </main>
    </div>
  );
}
function SearchBox({
  initial = "",
  compact = false,
  initialIntent = "search",
}: {
  initial?: string;
  compact?: boolean;
  initialIntent?: string;
}) {
  const [q, setQ] = useState(initial);
  const [intent, setIntent] = useState(initialIntent);
  const navigate = useNavigate();
  useEffect(() => setQ(initial), [initial]);
  return (
    <form
      className={`searchbox ${compact ? "compact" : ""}`}
      onSubmit={(e) => {
        e.preventDefault();
        if (q.trim())
          navigate(
            `/search?q=${encodeURIComponent(q.trim())}&intent=${intent}`,
          );
      }}
    >
      <label
        className="sr-only"
        htmlFor={compact ? "query-compact" : "query-home"}
      >
        输入论文标题、主题或观点
      </label>
      <input
        id={compact ? "query-compact" : "query-home"}
        maxLength={2000}
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="搜索论文、研究主题，或输入一个待考究的观点"
        required
      />
      <div className="search-actions">
        <select
          aria-label="任务类型"
          value={intent}
          onChange={(e) => setIntent(e.target.value)}
        >
          <option value="search">文献搜索</option>
          <option value="support">寻找支持证据</option>
          <option value="refute">寻找反驳证据</option>
        </select>
        <button className="primary" type="submit" aria-label="开始搜索">
          搜索 ↗
        </button>
      </div>
    </form>
  );
}
function Home() {
  return (
    <section className="home">
      <div className="eyebrow">SEARCH · EVIDENCE · RESEARCH</div>
      <h1>从一个问题，走向可靠证据。</h1>
      <p className="subtitle">搜索文献，考究观点，在原文之上开展研究。</p>
      <SearchBox />
      <div className="entries">
        <Link to="/upload">
          <b>＋ 导入一篇文献</b>
          <span>PDF、Word、文本与论文链接</span>
        </Link>
        <Link to="/workspace">
          <b>▤ 我的科研工作区</b>
          <span>材料、选文问答与执行记录</span>
        </Link>
        <button onClick={() => document.getElementById("query-home")?.focus()}>
          <b>⌕ 考究一个观点</b>
          <span>选择支持或反驳，保留矛盾与不足</span>
        </button>
      </div>
      <p className="scope-note">
        结果来自实际发布的固定语料版本；不代表全网、全学科或最新文献。
      </p>
    </section>
  );
}
function SearchSnippet({ text }: { text: string }) {
  const id = useId();
  const [expanded, setExpanded] = useState(false);
  const long = text.length > 300;
  return <div className="search-excerpt">
    <p className="muted">命中的原文片段</p>
    <p id={id} className={`snippet${long && !expanded ? " snippet-collapsed" : ""}`}>{text}</p>
    {long && <button type="button" className="excerpt-toggle" aria-expanded={expanded} aria-controls={id}
      onClick={() => setExpanded(value => !value)}>{expanded ? "收起检索片段" : "展开完整检索片段"}</button>}
  </div>;
}
function Results() {
  const [params, setParams] = useSearchParams();
  const q = params.get("q") || "";
  const intent = params.get("intent") || "search";
  const [page, setPage] = useState<SearchPage | null>(null);
  const [error, setError] = useState<unknown>();
  const [busy, setBusy] = useState(false);
  const [stack, setStack] = useState<SearchPage[]>([]);
  const [selected, setSelected] = useState<PaperRef[]>([]);
  const [historyError, setHistoryError] = useState<unknown>();
  const [historyUnsupported, setHistoryUnsupported] = useState(false);
  const historyOperations = useRef(new Map<string, string>());
  const historySnapshot = useRef<string | null>(null);
  const identity = current;
  const sort = params.get("sort") || "relevance",
    catalog = params.get("catalog") || "application_papers",
    mode = params.get("mode") || "hybrid",
    field = params.get("field") || "text";
  const saveSearchHistory = async (value: SearchPage, operationId: string) => {
    try {
      await api("/api/v3/workstation/search-activities", { operation_id: operationId, snapshot_id: value.snapshot_id });
      if (historySnapshot.current === value.snapshot_id) setHistoryError(undefined);
    } catch (failure) {
      if (failure instanceof APIError && failure.status === 404 && ["not_found", "http_404"].includes(failure.code)) {
        setHistoryUnsupported(true);
        if (historySnapshot.current === value.snapshot_id) setHistoryError(undefined);
        return;
      }
      if (historySnapshot.current === value.snapshot_id) setHistoryError(failure);
    }
  };
  useEffect(() => {
    historySnapshot.current = page?.snapshot_id || null;
    setHistoryError(undefined);
    if (!page || !identity?.authenticated || historyUnsupported) return;
    const key = identity.owner_id + ":" + page.snapshot_id;
    if (historyOperations.current.has(key)) return;
    const operation = crypto.randomUUID();
    historyOperations.current.set(key, operation);
    void saveSearchHistory(page, operation);
  }, [page?.snapshot_id, identity?.authenticated, identity?.owner_id, historyUnsupported]);
  useEffect(() => {
    const controller = new AbortController();
    setBusy(true);
    setError(undefined);
    setPage(null);
    setStack([]);
    setSelected([]);
    api<SearchPage>(
      "/api/v3/search",
      {
        q,
        sort,
        catalog,
        mode,
        field,
        page_size: 10,
        candidate_limit: 100,
        ttl_seconds: 900,
        filters: {
          year_from: params.get("from") ? Number(params.get("from")) : null,
          year_to: params.get("to") ? Number(params.get("to")) : null,
        },
      },
      undefined,
      controller.signal,
    )
      .then(value => { if (!controller.signal.aborted) setPage(value); })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e);
      })
      .finally(() => {
        if (!controller.signal.aborted) setBusy(false);
      });
    return () => controller.abort();
  }, [q, sort, catalog, mode, field, params.get("from"), params.get("to")]);
  const update = (k: string, v: string) => {
    const p = new URLSearchParams(params);
    p.set(k, v);
    setParams(p);
  };
  const next = async () => {
    if (!page?.next_cursor) return;
    setBusy(true);
    try {
      const value = await api<SearchPage>(
        `/api/v3/search/snapshots/${page.snapshot_id}?cursor=${encodeURIComponent(page.next_cursor)}`,
      );
      setStack([...stack, page]);
      setPage(value);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="results">
      <SearchBox compact initial={q} initialIntent={intent} />
      <div className="toolbar search-filters">
        <label>
          语料{" "}
          <select
            aria-label="语料"
            value={catalog}
            onChange={(e) => update("catalog", e.target.value)}
          >
            <option value="application_papers">已核全文应用论文</option>
            <option value="standard_metadata">标准题名与摘要</option>
          </select>
        </label>
        <label>搜索字段 <select aria-label="搜索字段" value={field} onChange={e => update("field", e.target.value)}>
          <option value="text">题名与正文</option><option value="author">作者</option><option value="doi">DOI</option>
        </select></label>
        <label>
          排序{" "}
          <select aria-label="排序" value={sort} onChange={(e) => update("sort", e.target.value)}>
            <option value="relevance">相关性</option>
            <option value="year_desc">年份降序</option>
            <option value="title">标题</option>
          </select>
        </label>
        <label>
          检索{" "}
          <select aria-label="检索方式" value={mode} onChange={(e) => update("mode", e.target.value)}>
            <option value="hybrid">混合检索</option>
            <option value="bm25">关键词 BM25</option>
            <option value="dense">语义检索</option>
          </select>
        </label>
        <label>
          起始年份{" "}
          <input
            type="number"
            min="1000"
            max="2200"
            placeholder="不限"
            value={params.get("from") || ""}
            onChange={(e) => update("from", e.target.value)}
          />
        </label>
        <label>截止年份 <input type="number" min="1000" max="2200" placeholder="不限"
          value={params.get("to") || ""} onChange={e => update("to", e.target.value)} /></label>
      </div>
      {intent !== "search" && (
        <section className="ai-block">
          <div className="eyebrow">
            观点证据 · {intent === "support" ? "寻找支持" : "寻找反驳"}
          </div>
          <TaskComposer
            initial={q}
            kind="stance"
            direction={intent === "support" ? "support" : "refute"}
            catalog={catalog}
          />
        </section>
      )}
      {!!error && <Notice error={error} />}{" "}
      {historyUnsupported && <p className="muted" role="status" aria-label="检索历史服务状态">当前服务尚未开放检索条件保存；浏览记录仍可使用。本次检索未保存为可重新检索的条件。</p>}
      {!!historyError && page && <section className="notice" role="alert">
        检索结果已返回，但历史保存未完成：{errorText(historyError)}{" "}
        <button onClick={() => {
          const operation = identity && historyOperations.current.get(identity.owner_id + ":" + page.snapshot_id);
          if (operation) void saveSearchHistory(page, operation);
        }}>重试保存历史</button>
      </section>}
      {busy && (
        <p className="loading" role="status">
          正在检索实际文献…
        </p>
      )}
      {page && (
        <>
          <div className="result-meta">
            本次有界候选 {page.candidate_count} / 上限 {page.candidate_limit} ·
            版本 {page.generation_id.slice(0, 8)} · 普通搜索 0 次生成调用
            {page.masked_since_search > 0 &&
              ` · ${page.masked_since_search}条已因权限或状态变化隐藏`}
          </div>
          {page.results.length === 0 && (
            <p className="empty">
              该范围内没有匹配结果。可调整关键词或语料；这不表示相关研究不存在。
            </p>
          )}
          {page.results.map((hit) => (
            <article
              className="result"
              key={hit.paper.paper.kind + hit.paper.paper.id}
            >
              <div className="result-top">
                <label>
                  <input
                    aria-label={`选择 ${hit.paper.title || "未提供标题"}`}
                    type="checkbox"
                    checked={selected.some(
                      (p) =>
                        p.id === hit.paper.paper.id &&
                        p.kind === hit.paper.paper.kind,
                    )}
                    onChange={(e) =>
                      setSelected(
                        e.target.checked
                          ? [...selected, hit.paper.paper]
                          : selected.filter((p) => p.id !== hit.paper.paper.id),
                      )
                    }
                  />{" "}
                  选文
                </label>
                <span className="tag">
                  {hit.paper.coverage === "metadata"
                    ? "题名 / 摘要"
                    : "已核原生正文"}
                </span>
              </div>
              <h2>
                <Link to={paperPath(hit.paper.paper)}>
                  {hit.paper.title || "源记录未提供标题"}
                </Link>
              </h2>
              <p className="byline">
                {hit.paper.authors?.join(", ") || "作者字段未提供"} ·{" "}
                {hit.paper.year || "年份未知"}
              </p>
              <SearchSnippet key={`${page.snapshot_id}:${hit.paper.paper.id}`} text={hit.snippet} />
              <div className="result-links">
                <Link to={paperPath(hit.paper.paper)}>详情与问答 →</Link>
                {hit.paper.source_url && (
                  <a
                    href={hit.paper.source_url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    原文来源 ↗
                  </a>
                )}
              </div>
            </article>
          ))}
          <div className="toolbar pagination">
            <button
              disabled={busy || stack.length === 0}
              onClick={() => {
                setPage(stack[stack.length - 1]);
                setStack(stack.slice(0, -1));
              }}
            >
              上一页
            </button>
            <span>第 {stack.length + 1} 页 · 当前快照内分页</span>
            <button disabled={busy || !page.next_cursor} onClick={next}>
              下一页
            </button>
          </div>
          {selected.length > 0 && (
            <section className="card">
              <h2>选中文献 · {selected.length} 篇</h2>
              <TaskComposer selected={selected} catalog={catalog} />
            </section>
          )}
        </>
      )}
    </section>
  );
}
function DetailRoute() {
  const { kind, id } = useParams();
  return <Detail key={`${kind}:${id}`} />;
}
function MaterialDetailRoute() {
  const { id } = useParams();
  return <MaterialDetail key={id} />;
}
function Detail() {
  const { kind = "", id = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const [paper, setPaper] = useState<Paper | null>(null);
  const [error, setError] = useState<unknown>();
  const [citation, setCitation] = useState("");
  const [showPDF, setShowPDF] = useState(false);
  const [similar, setSimilar] = useState<SearchPage | null>(null);
  const [saved, setSaved] = useState(false);
  const saveOperation = useRef(crypto.randomUUID());
  const viewOperation = useRef(crypto.randomUUID());
  useEffect(() => {
    let cancelled = false;
    setPaper(null);
    setCitation("");
    setSimilar(null);
    setShowPDF(false);
    setSaved(false);
    setError(undefined);
    saveOperation.current = crypto.randomUUID();
    viewOperation.current = crypto.randomUUID();
    api<Paper>(`/api/v3/papers/${kind}/${id}`)
      .then((value) => {
        if (!cancelled) setPaper(value);
      })
      .catch((e) => {
        if (!cancelled) setError(e);
      });
    return () => {
      cancelled = true;
    };
  }, [kind, id]);
  useEffect(() => {
    if (paper && current?.authenticated)
      api("/api/v3/workstation/paper-activity", {
        operation_id: viewOperation.current,
        paper: paper.paper,
      }).catch(setError);
  }, [paper]);
  const cite = async () => {
    if (!paper) return;
    try {
      const { Cite } = await import("@citation-js/core");
      await Promise.all([
        import("@citation-js/plugin-csl"),
        import("@citation-js/plugin-bibtex"),
      ]);
      const data: any = {
        id: paper.paper.id,
        type: "article-journal",
        ...(paper.title ? { title: paper.title } : {}),
        ...(paper.authors
          ? { author: paper.authors.map((a) => ({ literal: a })) }
          : {}),
        ...(paper.year ? { issued: { "date-parts": [[paper.year]] } } : {}),
        ...(paper.source_url ? { URL: paper.source_url } : {}),
      };
      setCitation(
        new Cite([data]).format("bibliography", {
          format: "text",
          template: "harvard1",
          lang: "en-US",
        }),
      );
    } catch (e) {
      setError(e);
    }
  };
  return (
    <section className="content paper">
      <Link className="back" to="/">
        ← 学术搜索
      </Link>
      {!!error && <Notice error={error} />}{" "}
      {!paper ? (
        <p>正在读取文献…</p>
      ) : (
        <>
          <span className="tag">
            {paper.coverage === "metadata"
              ? "元数据记录 · 全文未入库"
              : "已核原生正文 · 不等于已解析PDF页码"}
          </span>
          <h1>{paper.title || "标题未提供"}</h1>
          <p className="byline">
            {paper.authors?.join(", ") || "作者字段未提供"} ·{" "}
            {paper.year || "年份未知"}
          </p>
          <div className="toolbar">
            {paper.source_url && (
              <a
                className="button"
                href={paper.source_url}
                target="_blank"
                rel="noreferrer"
              >
                原文来源 ↗
              </a>
            )}
            {paper.pdf_url && (
              <>
                <a
                  className="button"
                  href={paper.pdf_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  PDF来源 / 下载 ↗
                </a>
                <button onClick={() => setShowPDF(!showPDF)}>在线 PDF</button>
              </>
            )}
            <button onClick={cite}>Harvard 引用</button>
            <button
              disabled={saved}
              onClick={async () => {
                try {
                  if (!current?.authenticated)
                    throw new APIError("login_required", 401);
                  await api("/api/v3/workstation/saved-papers", {
                    operation_id: saveOperation.current,
                    paper: paper.paper,
                  });
                  setSaved(true);
                } catch (e) {
                  setError(e);
                }
              }}
            >
              {saved ? "已收藏" : "收藏到文献库"}
            </button>
            <button
              onClick={() =>
                api<SearchPage>(`/api/v3/papers/${kind}/${id}/similar`, {
                  mode: "hybrid",
                  page_size: 10,
                  candidate_limit: 50,
                })
                  .then(setSimilar)
                  .catch(setError)
              }
            >
              相似主题
            </button>
          </div>
          {paper.pdf_license && paper.pdf_is_unmodified && <p className="muted">
            原文 PDF 保留来源署名，字节未经修改。许可：<a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">CC BY 4.0</a>。
          </p>}
          {citation && (
            <section className="citation">
              <p>{citation}</p>
              {paper.citation_status === "incomplete_metadata" && (
                <small>
                  源字段不全：引用仅包含已有字段，请按学校要求核对。
                </small>
              )}
              <button
                onClick={() =>
                  navigator.clipboard.writeText(citation).catch(setError)
                }
              >
                复制引用
              </button>
            </section>
          )}
          <h2>摘要</h2>
          <p className="abstract">
            {paper.abstract || "该源记录未提供摘要。未使用模型补写。"}
          </p>
          {showPDF && paper.pdf_url && <PDF url={paper.pdf_url} />}
          <section className="card">
            <h2>与这篇文献对话</h2>
            <p className="muted">
              默认仅使用这篇文献；正文缺失或证据不足会明确提示。原文引用可核查。
            </p>
            <TaskComposer
              selected={[paper.paper]}
              resumeTaskId={params.get("task") || undefined}
              onTaskChange={taskId => {
                const next = new URLSearchParams(params);
                if (taskId) next.set("task", taskId);
                else next.delete("task");
                setParams(next, { replace: true });
              }}
              catalog={
                paper.coverage === "metadata"
                  ? "standard_metadata"
                  : "application_papers"
              }
            />
          </section>
          {similar && (
            <section>
              <h2>相关文献</h2>
              {similar.results.map((h) => (
                <p key={h.paper.paper.id}>
                  <Link to={paperPath(h.paper.paper)}>
                    {h.paper.title || "源标题缺失"}
                  </Link>
                </p>
              ))}
            </section>
          )}
        </>
      )}
    </section>
  );
}
type Task = {
  kind?: string;
  brief?: {
    question?: string;
    requested_direction?: string;
    selected_skills?: SkillPin[];
    selected_papers?: Array<PaperRef | { kind: "material_file"; id: string }>;
    research_brief?: { id: string; revision: number; sha256: string };
    research_action?: "plan" | "suggest" | "extract";
  };
  known_ncny?: number;
  reserved_ncny?: number;
  active_seconds?: number;
  dispatch_timing?: {
    queue_seconds: number;
    persisted_queue_seconds: number;
    current_queue_seconds: number;
    execution_seconds: number;
    target_total_seconds: number;
    human_waiting_input_excluded: boolean;
  };
  id: string;
  revision: number;
  state: string;
  result?: AgentResult | null;
  result_availability?: string;
  coverage_report?: CoverageReport;
  research_progress?: ResearchProgressReport;
  error_code?: string;
  runtime_version?: string;
  runtime_availability?: string;
  dispatch_runtime_availability?: string;
  cancellation_requested?: boolean;
  profile?: { model_calls?: number; tool_calls?: number; active_seconds?: number; cash_ncny?: number };
  input_tokens?: number;
  output_tokens?: number;
  model_calls?: number;
  tool_calls?: number;
};
function TaskComposer({
  initial = "",
  selected = [],
  kind: initialKind = "qa",
  direction = "neutral",
  catalog = "application_papers",
  resumeTaskId,
  onTaskChange,
  onBriefSaved,
}: {
  initial?: string;
  resumeTaskId?: string;
  onTaskChange?: (taskId: string | undefined) => void;
  onBriefSaved?: (value: BriefSnapshot) => void;
  selected?: Array<PaperRef | { kind: "material_file"; id: string }>;
  kind?: string;
  direction?: string;
  catalog?: string;
}) {
  const [question, setQuestion] = useState(initial);
  const [selectedSkills, setSelectedSkills] = useState<SkillPin[]>([]);
  const [contextOpen, setContextOpen] = useState(false);
  const [selectedDirection, setSelectedDirection] = useState(direction);
  const [kind, setKind] = useState(initialKind);
  const [task, setTask] = useState<Task | null>(null);
  const [events, setEvents] = useState<any[]>([]);
  const [error, setError] = useState<unknown>();
  const [sending, setSending] = useState(false);
  const seq = useRef(0);
  const eventTaskId = useRef<string | undefined>(resumeTaskId);
  const historyCursor = useRef(0);
  const pendingOperation = useRef<{ signature: string; id: string } | null>(
    null,
  );
  const [eventHasMore, setEventHasMore] = useState(false);
  function reportTaskError(failure: unknown, taskId?: string) {
    if (taskId && eventTaskId.current !== taskId) return;
    setError(failure);
    if (failure instanceof APIError && [401, 403, 404].includes(failure.status)) {
      setTask(null);
      setEvents([]);
      setQuestion("");
      eventTaskId.current = undefined;
      historyCursor.current = 0;
      seq.current = 0;
      setEventHasMore(false);
    }
  }
  async function loadEventPage(id: string, after = 0) {
    const value = await api<{ events: any[] }>(
      `/api/agent/tasks/${id}/events?after_seq=${after}`,
    );
    if (eventTaskId.current !== id) return;
    setEvents((items) =>
      Array.from(
        new Map(
          [...items, ...value.events].map((item) => [item.seq, item]),
        ).values(),
      ).sort((a, b) => a.seq - b.seq),
    );
    setEventHasMore(value.events.length === 200);
    if (value.events.length) historyCursor.current = Math.max(historyCursor.current, value.events.at(-1).seq);
    if (value.events.length)
      seq.current = Math.max(seq.current, value.events.at(-1).seq);
  }
  const questionId = useId();
  const selectedCount =
    task?.brief && "selected_papers" in task.brief
      ? (task.brief.selected_papers as unknown[])?.length || selected.length
      : selected.length;
  useEffect(() => {
    let cancelled = false;
    if (resumeTaskId && task?.id === resumeTaskId) return;
    if (resumeTaskId) {
      setError(undefined);
      eventTaskId.current = resumeTaskId;
      historyCursor.current = 0;
      seq.current = 0;
      setEvents([]);
      setTask(null);
      setQuestion("");
      api<Task>(`/api/agent/tasks/${resumeTaskId}`)
        .then((value) => {
          if (cancelled) return;
          const sources = value.brief?.selected_papers;
          if (selected.length && (!sources || sources.length !== selected.length
              || selected.some(source => !sources.some(item => item.kind === source.kind && item.id === source.id))))
            throw new Error("此任务绑定的来源与当前阅读页不同，请从科研工作区的任务记录打开原会话。");
          setTask(value);
          setSelectedDirection(value.brief?.requested_direction || "neutral");
          setSelectedSkills(value.brief?.selected_skills || []);
          void loadEventPage(value.id).catch(failure => reportTaskError(failure, value.id));
        })
        .catch(error => { if (!cancelled) reportTaskError(error); });
    }
    return () => { cancelled = true; };
  }, [resumeTaskId]);
  useEffect(() => {
    if (!task || !["queued", "running"].includes(task.state)) return;
    let cancelled = false;
    let timeout: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const next = await api<Task>(`/api/agent/tasks/${task.id}`);
        if (cancelled) return;
        setTask(next);
        if (["queued", "running"].includes(next.state))
          timeout = setTimeout(poll, 1200);
      } catch (e) {
        if (!cancelled) reportTaskError(e);
      }
    };
    timeout = setTimeout(poll, 300);
    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  }, [task?.id, task?.state]);
  useEffect(() => {
    if (!task || !["queued", "running"].includes(task.state)) return;
    const stream = new EventSource(
      `/api/agent/tasks/${task.id}/events/stream?after_seq=${seq.current}`,
    );
    stream.addEventListener("progress", (event) => {
      if (eventTaskId.current !== task.id) return;
      try {
        const item = JSON.parse((event as MessageEvent).data);
        if (item.seq > seq.current) {
          seq.current = item.seq;
          setEvents((items) => [...items, item]);
        }
      } catch {
        setError(new Error("执行事件格式不正确，请重新读取任务。"));
      }
    });
    stream.addEventListener("settled", () => stream.close());
    stream.addEventListener("unavailable", () => {
      stream.close();
      setError(new Error("执行事件当前不可读取，请刷新并检查登录状态。"));
    });
    return () => stream.close();
  }, [task?.id, task?.state]);
  const submit = async () => {
    setError(undefined);
    setSending(true);
    try {
      if (!current?.authenticated || !current.project_id)
        throw new APIError("login_required", 401);
      const operation = (intent: unknown) => {
        const signature = JSON.stringify(intent);
        if (pendingOperation.current?.signature !== signature)
          pendingOperation.current = { signature, id: crypto.randomUUID() };
        return pendingOperation.current.id;
      };
      if (task) {
        if (!canContinueTask(task))
          throw new Error("该任务当前无法继续执行。可查看已授权的历史结果和执行记录；需要恢复可用的执行服务后再继续。");
        let synchronizedBrief:
          { id: string; revision: number; sha256: string } | undefined;
        if (task.kind === "research" && task.brief?.research_brief) {
          const saved = await api<{
            brief: { id: string; revision: number };
            brief_sha256: string;
          }>(`/api/v3/workstation/briefs/${task.brief.research_brief.id}`);
          synchronizedBrief = {
            id: saved.brief.id,
            revision: saved.brief.revision,
            sha256: saved.brief_sha256,
          };
        }
        const command = {
          ...(synchronizedBrief ? { research_brief: synchronizedBrief } : {}),
          expected_revision: task.revision,
          kind: task.state === "waiting_input" ? "resume" : "followup",
          message: question,
          selected_skills: selectedSkills,
          ...(task.kind === "stance"
            ? { requested_direction: selectedDirection }
            : {}),
        };
        setTask(
          await api<Task>(`/api/agent/tasks/${task.id}/commands`, {
            operation_id: operation({ task_id: task.id, ...command }),
            ...command,
          }),
        );
      } else {
        const profile = await newTaskProfile(kind);
        const created = await api<Task>("/api/agent/tasks", {
            project_id: current.project_id,
            operation_id: operation({
              project_id: current.project_id,
              kind,
              question,
              selected,
              catalog,
              selectedDirection,
              selectedSkills,
            }),
            profile_id: profile,
            kind,
            brief: {
              question,
              kind,
              objective:
                kind === "summary"
                  ? "Summarize the selected sources with grounded atomic claims."
                  : kind === "compare"
                    ? "Compare selected sources by evidence-backed methods, data and limitations."
                    : "Answer the research question using permitted evidence.",
              catalog,
              selected_papers: selected,
              selected_skills: selectedSkills,
              constraints: [],
              unknowns: [],
              requested_direction:
                kind === "stance" ? selectedDirection : "neutral",
            },
          });
        eventTaskId.current = created.id;
        historyCursor.current = 0;
        seq.current = 0;
        setEvents([]);
        setTask(created);
        onTaskChange?.(created.id);
        void loadEventPage(created.id).catch(failure => reportTaskError(failure, created.id));
      }
      setQuestion("");
    } catch (e) {
      reportTaskError(e);
    } finally {
      setSending(false);
    }
  };
  return (
    <div className="task">
      {!current?.authenticated ? (
        <p className="muted">
          <a href="/auth/login">登录</a>后发起 AI 任务，普通文献搜索无需登录。
        </p>
      ) : (
        <>
          <details
            onToggle={(event) => setContextOpen(event.currentTarget.open)}
          >
            <summary>技能与项目记忆</summary>
            {contextOpen && (
              <Suspense fallback={<p>正在读取上下文设置…</p>}>
                <ContextSettings
                  identity={current}
                  selected={selectedSkills}
                  onSelect={setSelectedSkills}
                  selectionDisabled={
                    sending ||
                    (!!task &&
                      ["queued", "running", "reconciling"].includes(task.state))
                  }
                />
              </Suspense>
            )}
          </details>
          {task?.kind === "research" && (
            <p className="muted">
              继续任务会重新读取并同步当前已保存的研究档案；模型调用与费用沿用本任务累计记录。
            </p>
          )}
          {!task && <CapabilityStatus kind={kind} />}
          <label className="sr-only" htmlFor={questionId}>
            研究问题或追问
          </label>
          <textarea
            id={questionId}
            maxLength={8000}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={
              task
                ? "继续追问，或补充需要澄清的信息…"
                : "输入问题；也可切换为单 / 多文摘要或比较…"
            }
          />
          <div className="toolbar">
            {(task?.kind || kind) === "stance" && (
              <label>
                取向
                <select
                  aria-label="观点取向"
                  value={selectedDirection}
                  onChange={(e) => {
                    setSelectedDirection(e.target.value);
                    if (task && !question.trim())
                      setQuestion(task.brief?.question || initial);
                  }}
                >
                  <option value="support">寻找支持</option>
                  <option value="refute">寻找反驳</option>
                </select>
              </label>
            )}
            {!task && initialKind !== "stance" && (
              <select
                aria-label="AI任务类型"
                value={kind}
                onChange={(e) => setKind(e.target.value)}
              >
                <option value="qa">文献问答</option>
                <option value="summary">AI 摘要</option>
                <option value="compare">多文比较</option>
              </select>
            )}
            <button
              className="primary"
              disabled={
                sending ||
                !question.trim() ||
                (!!task && !canContinueTask(task)) ||
                (!task &&
                  ["summary", "compare"].includes(kind) &&
                  (selected.length < (kind === "compare" ? 2 : 1) ||
                    selected.length > 10))
              }
              onClick={submit}
            >
              {sending ? "提交中…" : task ? "继续本任务" : "发起任务"} ↗
            </button>
            {task && (
              <button
                disabled={sending || ["queued", "running"].includes(task.state)}
                onClick={() => {
                  setTask(null);
                  setEvents([]);
                  seq.current = 0;
                  historyCursor.current = 0;
                  eventTaskId.current = undefined;
                  setEventHasMore(false);
                  pendingOperation.current = null;
                  onTaskChange?.(undefined);
                }}
              >
                新建独立任务
              </button>
            )}
            {task &&
              ["queued", "running", "waiting_input"].includes(task.state) && (
                <button
                  onClick={() =>
                    api<Task>(`/api/agent/tasks/${task.id}/commands`, {
                      operation_id: crypto.randomUUID(),
                      expected_revision: task.revision,
                      kind: "cancel",
                      message: "",
                    })
                      .then(setTask)
                      .catch(reportTaskError)
                  }
                >
                  取消任务
                </button>
              )}
          </div>
          {["summary", "compare"].includes(task?.kind || kind) && (
            <p className="muted">
              当前选择 {selectedCount}{" "}
              篇；摘要需至少1篇，比较需至少2篇，每次最多10篇。实际全文覆盖与预算在执行前检查，材料不足会明确报告。
            </p>
          )}
        </>
      )}
      {!!error && <Notice error={error} />}{" "}
      {task && (
        <section className="task-output" aria-live="polite">
          <div className="task-state">
            <span className={task.state === "running" ? "pulse" : ""}>●</span>{" "}
            {taskStateText(task.state)} · 任务 {task.id.slice(0, 8)}
          </div>
          <TaskConversation taskId={task.id} revision={task.revision} state={task.state}
            onUnavailable={failure => reportTaskError(failure, task.id)} />
          <h3>当前轮次</h3>
          {task.brief?.question && <p className="conversation-question">{task.brief.question}</p>}
          {task.state === "partial" && <p className="notice" role="status">本轮仅完成部分内容。已确认的回答与尚未覆盖的问题会分别列出，可在本任务内继续补充。</p>}
          {task.state === "reconciling" && <p className="notice" role="status">存在尚未确认结果的调用，正在核对；费用预留与执行记录保留，当前不能继续追问。</p>}
          <details open={["queued", "running"].includes(task.state)}>
            <summary>执行过程 · {events.length} 条事件</summary>
            <ol className="events">
              {events.map((event, i) => (
                <li key={event.seq || i}>
                  <strong>{progressText(event)}</strong>
                  <details>
                    <summary>执行记录</summary>
                    <pre>
                      {JSON.stringify(
                        event.payload || event.data || event,
                        null,
                        2,
                      )}
                    </pre>
                  </details>
                </li>
              ))}
            </ol>
            {eventHasMore && (
              <button
                onClick={() =>
                  loadEventPage(task.id, historyCursor.current).catch(
                    failure => reportTaskError(failure, task.id),
                  )
                }
              >
                读取后续执行记录
              </button>
            )}
          </details>
          {task.runtime_availability === "pinned_legacy_runtime_required" && (
            <p className="notice">
              该历史任务绑定旧版执行环境，当前运行器不会自动接管。原费用与执行记录仍保留。
            </p>
          )}
          {task.dispatch_runtime_availability && task.dispatch_runtime_availability !== "current" && (
            <p className="notice" role="status">
              本任务当前只能查看已授权的结果和记录；恢复匹配的执行服务后才能继续，原有费用和任务范围保持保留。
            </p>
          )}
          {task.error_code && (
            <section className="notice" role="status"><p>{taskFailureText(task.error_code)}</p><details><summary>查看错误标识</summary><code>{task.error_code}</code></details></section>
          )}
          {task.result_availability &&
            !["available", "available_for_input"].includes(
              task.result_availability,
            ) && (
              <p className="notice">
                {resultAvailabilityText(task.result_availability)}
              </p>
            )}
          {current?.authenticated && task.brief?.research_brief && task.result?.candidates && (
            <DraftCandidates key={`${task.id}-${task.revision}`} identity={current} taskId={task.id}
              briefId={task.brief.research_brief.id} result={task.result}
              evidence={ref => <EvidenceLink taskId={task.id} refId={ref} />}
              onSaved={value => { onBriefSaved?.(value); api<Task>(`/api/agent/tasks/${task.id}`).then(setTask).catch(reportTaskError); }} />
          )}
          {task.result && <TaskAnswer taskId={task.id} result={task.result} />}
          {task.kind === "stance" && !task.result && task.coverage_report && <TaskCoverage report={task.coverage_report} />}
          {task.kind === "research" && !task.result?.school_document && task.research_progress && <ResearchProgress report={task.research_progress} />}
          <div className="usage">
            模型 {task.model_calls ?? "—"}{task.profile?.model_calls ? ` / ${task.profile.model_calls}` : ""} 次 · 工具 {task.tool_calls ?? "—"}{task.profile?.tool_calls ? ` / ${task.profile.tool_calls}` : ""} 次
            · 已知输入 {task.input_tokens ?? "—"} / 输出{" "}
            {task.output_tokens ?? "—"} tokens · 活跃{" "}
            {task.active_seconds === undefined
              ? "—"
              : task.active_seconds.toFixed(2)}{" "}
            {task.profile?.active_seconds ? ` / ${task.profile.active_seconds}` : ""} 秒 · 已知费用{" "}
            {task.known_ncny === undefined
              ? "—"
              : `¥${(task.known_ncny / 1e9).toFixed(4)}`}{" "}
            {task.reserved_ncny
              ? `· 未决预留 ¥${(task.reserved_ncny / 1e9).toFixed(4)}`
              : ""}{" "}
            {task.profile?.cash_ncny ? ` / 上限 ¥${(task.profile.cash_ncny / 1e9).toFixed(2)}` : ""} · 多轮共用本任务预算，等待补充不计活跃时间
          </div>
          {task.dispatch_timing && (
            <div className="usage" aria-label="任务排队与执行时长">
              累计排队 {task.dispatch_timing.queue_seconds.toFixed(2)} 秒
              {task.dispatch_timing.current_queue_seconds > 0 ? `（本轮仍在等待 ${task.dispatch_timing.current_queue_seconds.toFixed(2)} 秒）` : ""}
              {" · "}累计执行 {task.dispatch_timing.execution_seconds.toFixed(2)} 秒
              {" · "}排队与执行合计 {task.dispatch_timing.target_total_seconds.toFixed(2)} 秒
              {task.dispatch_timing.human_waiting_input_excluded ? "，等待用户补充的停留不计入。" : ""}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
function TaskHistory() {
  const [tasks, setTasks] = useState<any[]>([]);
  const [error, setError] = useState<unknown>();
  useEffect(() => {
    if (current?.authenticated)
      api<{ items: any[] }>("/api/agent/tasks")
        .then((value) => setTasks(value.items))
        .catch(setError);
  }, []);
  return (
    <details className="history">
      <summary>最近任务与恢复入口</summary>
      {!!error && <Notice error={error} />}
      <p className="muted">最近50个本人任务；恢复原任务继续使用原预算。</p>
      {tasks.map((task) => (
        <p key={task.id}>
          <Link to={`/workspace?task=${task.id}`}>
            {taskKindText(task.kind)} · {taskStateText(task.state)} · {task.id.slice(0, 8)}
          </Link>
        </p>
      ))}
    </details>
  );
}
function MaterialDetail() {
  const { id = "" } = useParams();
  const [file, setFile] = useState<any>();
  const [error, setError] = useState<unknown>();
  const [readiness, setReadiness] = useState<any>();
  const [parts, setParts] = useState<any[]>([]);
  const [after, setAfter] = useState<number | null>(0);
  const [params, setParams] = useSearchParams();
  const consentOperation = useRef<{ signature: string; id: string } | null>(null);
  const [consentBusy, setConsentBusy] = useState(false);
  const load = async () => {
    try {
      const value = await api<any>(`/api/v3/materials/${id}`);
      setFile(value);
      setReadiness(await api(`/api/v3/materials/${id}/ai-readiness`));
    } catch (e) {
      setError(e);
    }
  };
  const setProcessingConsent = async (allowed: boolean) => {
    setConsentBusy(true);
    setError(undefined);
    try {
      const signature = JSON.stringify({ id, revision: file.revision, allowed });
      if (consentOperation.current?.signature !== signature)
        consentOperation.current = { signature, id: crypto.randomUUID() };
      await api(`/api/v3/materials/${id}/processing-consent`, {
        operation_id: consentOperation.current.id,
        expected_revision: file.revision,
        allowed,
      });
      consentOperation.current = null;
      await load();
    } catch (error) { setError(error); }
    finally { setConsentBusy(false); }
  };
  useEffect(() => {
    void load();
    const timer = setInterval(() => void load(), 3000);
    return () => clearInterval(timer);
  }, [id]);
  const more = async () => {
    try {
      const value = await api<any>(
        `/api/v2/files/${id}/fragments?after=${after || 0}&page_size=20`,
      );
      setParts((items) => [...items, ...value.items]);
      setAfter(value.next_after);
    } catch (e) {
      setError(e);
    }
  };
  return (
    <section className="content material">
      <Link className="back" to="/workspace">
        ← 我的材料
      </Link>
      <h1>材料阅读与问答</h1>
      {!!error && <Notice error={error} />}{" "}
      {!file ? (
        <p>正在读取材料…</p>
      ) : (
        <>
          <p className="byline">
            {file.media_type} · {file.status} · {id.slice(0, 8)}
          </p>
          {file.coverage && (
            <pre className="muted">
              实际解析覆盖：{JSON.stringify(file.coverage)}
            </pre>
          )}
          <p className="muted">
            材料保持私有；原文阅读范围与 AI 已解析证据范围分别显示。
          </p>
          {file.can_download && (
            <a
              className="button"
              href={`/api/v2/files/${id}/content`}
              target="_blank"
              rel="noreferrer"
            >
              原始文件 / 下载 ↗
            </a>
          )}
          {file.can_preview && file.media_type === "application/pdf" && (
            <PDF url={`/api/v2/files/${id}/content`} />
          )}{" "}
          {["ready", "partial"].includes(file.status) &&
            file.media_type !== "application/pdf" && (
              <section>
                {parts.map((part) => (
                  <article className="source-part" key={part.fragment_id}>
                    <pre>{part.source_text}</pre>
                    <small>原始片段 {part.ordinal}</small>
                  </article>
                ))}
                <button disabled={after === null} onClick={more}>
                  {parts.length ? "继续阅读后续片段" : "读取材料原文"}
                </button>
              </section>
            )}
          {!["ready", "partial"].includes(file.status) && (
            <p className="notice">
              材料尚未解析完成。状态会自动更新；解析失败不会作为可问答成功。
            </p>
          )}
          <section className="card">
            <h2>与材料对话</h2>
            <p className="muted">
              AI 就绪：
              {readiness?.ready ? "可使用" : readiness?.code || "检查中"}
              。模型使用当前已解析片段，缺页不会由模型补写。修改选文范围须显式新建任务。
            </p>
            {file.outbound_policy !== "selected_cloud_processing_allowed" && (
              <button
                disabled={consentBusy || !["ready", "partial"].includes(file.status)}
                onClick={() => void setProcessingConsent(true)}
              >
                允许该材料发送至已配置模型 API
              </button>
            )}
            {file.outbound_policy === "selected_cloud_processing_allowed" && (
              <button disabled={consentBusy} onClick={() => void setProcessingConsent(false)}>
                撤回模型 API 处理授权
              </button>
            )}
            {!readiness?.ready &&
              file.outbound_policy === "selected_cloud_processing_allowed" && (
                <button
                  onClick={() =>
                    api(`/api/v3/materials/${id}/index`, {})
                      .then(load)
                      .catch(setError)
                  }
                >
                  构建本材料的语义索引
                </button>
              )}
            {readiness?.ready ? (
              <TaskComposer
                selected={[{ kind: "material_file", id }]}
                resumeTaskId={params.get("task") || undefined}
                onTaskChange={taskId => {
                  const next = new URLSearchParams(params);
                  if (taskId) next.set("task", taskId);
                  else next.delete("task");
                  setParams(next, { replace: true });
                }}
              />
            ) : (
              <p className="muted">就绪前不发起付费模型调用。</p>
            )}
          </section>
        </>
      )}
    </section>
  );
}
function Workspace() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const [researchTaskId, setResearchTaskId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [researchSkills, setResearchSkills] = useState<SkillPin[]>([]);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [briefRefresh, setBriefRefresh] = useState<{ id: string; revision: number }>();
  const briefSaved = (value: BriefSnapshot) => setBriefRefresh({ id: value.brief.id, revision: value.brief.revision });
  const tasksByBriefAction = useRef(new Map<string, string>());
  const operationRef = useRef<{ signature: string; id: string } | null>(null);
  const operation = (value: unknown) => {
    const signature = JSON.stringify(value);
    if (operationRef.current?.signature !== signature)
      operationRef.current = { signature, id: crypto.randomUUID() };
    return operationRef.current.id;
  };
  async function launchResearch(
    input: PlanningInput,
    action: "plan" | "suggest" | "extract",
    fields: BriefField[] = [],
  ) {
    if (!current?.authenticated || !current.project_id)
      throw new APIError("login_required", 401);
    const reference = {
      id: input.brief_id,
      revision: input.brief_revision,
      sha256: input.brief_sha256,
    };
    const question =
      action === "plan"
        ? "根据当前已确认研究档案与文献证据生成上交研究计划书，未执行的实验必须写为拟议，未知信息如实标明。"
        : action === "extract"
          ? "逐窗读取关联草稿，提取指定研究字段的候选信息、不同说法和缺项，不得自动确认事实。"
          : "针对指定研究字段提出有明确待确认条件的建议，不得当作已确认事实。";
    let task: Task;
    const taskKey = `${input.brief_id}:${action}`;
    let existing = tasksByBriefAction.current.get(taskKey);
    // Recover an explicitly opened Task after reload without replacing its
    // pending subgraph or resetting its lifetime counters with a new Task.
    const opened = researchTaskId || params.get("task");
    if (!existing && opened) {
      const candidate = await api<Task>(`/api/agent/tasks/${opened}`);
      if (candidate.kind === "research" && candidate.brief?.research_brief?.id === input.brief_id) {
        const candidateKey = `${input.brief_id}:${candidate.brief.research_action || "plan"}`;
        tasksByBriefAction.current.set(candidateKey, candidate.id);
        if (candidateKey === taskKey) existing = candidate.id;
      }
    }
    if (existing) {
      const previous = await api<Task>(`/api/agent/tasks/${existing}`);
      if (previous.kind !== "research" || previous.brief?.research_brief?.id !== input.brief_id ||
          (previous.brief.research_action || "plan") !== action)
        throw new Error("研究任务的操作类型或档案已变化，请核对原任务后继续。");
      if (["queued", "running", "reconciling"].includes(previous.state)) {
        setResearchTaskId(existing);
        return previous;
      }
      if (!["succeeded", "partial", "waiting_input"].includes(previous.state))
        throw new Error(
          "该研究任务已终止。原失败记录会保留，请明确另起研究任务。",
        );
      if (!canContinueTask(previous))
        throw new Error("该研究任务当前缺少可用的执行服务。已授权结果和记录仍可查看，请恢复服务后继续原任务。");
      const body = {
        expected_revision: previous.revision,
        kind: previous.state === "waiting_input" ? "resume" : "followup",
        message: question,
        research_brief: reference,
        research_action: action,
        ...(action === "suggest" ? { research_help_fields: fields } : {}),
        ...(action === "extract" ? { research_extract_fields: fields } : {}),
        selected_skills: researchSkills,
      };
      task = await api<Task>(`/api/agent/tasks/${existing}/commands`, {
        ...body,
        operation_id: operation({ task_id: existing, ...body }),
      });
    } else {
      const body = {
        project_id: current.project_id,
        profile_id: await newTaskProfile("research"),
        kind: "research",
        brief: {
          kind: "research",
          question,
          catalog: "application_papers",
          selected_papers: input.selected_sources,
          research_draft_files: input.draft_source_ids || [],
          research_brief: reference,
          selected_skills: researchSkills,
          ...(action === "suggest" ? { research_action: action, research_help_fields: fields }
            : action === "extract" ? { research_action: action, research_extract_fields: fields } : {}),
        },
      };
      task = await api<Task>("/api/agent/tasks", {
        ...body,
        operation_id: operation(body),
      });
    }
    tasksByBriefAction.current.set(taskKey, task.id);
    setResearchTaskId(task.id);
    return task;
  }
  async function help(
    request: BriefHelpRequest,
  ): Promise<BriefSuggestion[] | void> {
    const saved = await api<BriefSnapshot>(
      `/api/v3/workstation/briefs/${request.brief.id}`,
    );
    if (saved.brief.revision !== request.brief.revision)
      throw new APIError("brief_revision_conflict", 409);
    const input: PlanningInput = {
      schema_version: "citequest-research-planning-input/1",
      brief_id: saved.brief.id,
      brief_revision: saved.brief.revision,
      brief_sha256: saved.brief_sha256,
      user_confirmed: {},
      explicit_unknowns: {},
      selected_sources: saved.brief.selected_sources,
      draft_source_ids: saved.brief.draft_source_ids,
      research_status: "proposed_not_executed",
      school_format_ready: false,
    };
    let task = await launchResearch(
      input,
      "suggest",
      request.fields.length
        ? request.fields
        : ["methods", "baselines", "metrics"],
    );
    const deadline = Date.now() + 180000;
    while (
      ["queued", "running"].includes(task.state) &&
      Date.now() < deadline
    ) {
      await new Promise((resolve) => setTimeout(resolve, 1200));
      task = await api<Task>(`/api/agent/tasks/${task.id}`);
    }
    if (
      task.state === "succeeded" &&
      task.result_availability === "available" &&
      task.result?.status === "research_suggestions_ready"
    )
      return task.result.suggestions;
    if (["failed", "cancelled"].includes(task.state))
      throw new Error("本次建议未完成，请查看任务记录；研究档案保持原状态。");
  }
  if (!current?.authenticated)
    return (
      <section className="content">
        <h1>我的科研工作区</h1>
        <p>
          <a href="/auth/login">登录个人中心</a>，管理文献、材料与研究任务。
        </p>
      </section>
    );
  return (
    <section className="content">
      {!!error && <Notice error={error} />}
      {(researchTaskId || params.get("task")) && (
        <section className="card">
          <h2>研究任务</h2>
          <TaskComposer
            key={researchTaskId || params.get("task") || "research"}
            resumeTaskId={researchTaskId || params.get("task") || undefined}
            onBriefSaved={briefSaved}
          />
        </section>
      )}
      <details onToggle={(event) => setSettingsOpen(event.currentTarget.open)}>
        <summary>研究任务的技能与项目记忆</summary>
        {settingsOpen && (
          <Suspense fallback={<p>正在载入设置…</p>}>
            <ContextSettings
              identity={current}
              selected={researchSkills}
              onSelect={setResearchSkills}
            />
          </Suspense>
        )}
      </details>
      <Suspense fallback={<p role="status">正在载入科研工作区…</p>}>
        <WorkspacePanel
          identity={current}
          briefRefresh={briefRefresh}
          renderTask={(request) => <TaskComposer {...request} onBriefSaved={briefSaved} />}
          onOpenTask={setResearchTaskId}
          onOpenWork={async (id) => {
            try {
              const result = await api<{
                items: Array<{ paper: PaperRef | null }>;
              }>("/api/v3/workstation/resolve-work-selections", {
                work_ids: [id],
                require_full_text: false,
              });
              if (!result.items[0]?.paper)
                throw new Error(
                  "该文献暂无可访问的论文详情，请从材料列表查看文件。",
                );
              navigate(paperPath(result.items[0].paper));
            } catch (e) {
              setError(e);
            }
          }}
          resolveSelections={async (items) => {
            const result = await api<{
              items: Array<{ status: string; paper: PaperRef | null }>;
            }>("/api/v3/workstation/resolve-work-selections", {
              work_ids: items.map((item) => item.work_id),
              require_full_text: true,
            });
            if (
              result.items.some(
                (item) => item.status !== "resolved" || !item.paper,
              )
            )
              throw new Error(
                "部分文献暂无已准入正文，无法开始全文任务。请仅选正文已就绪的文献，或从材料页提问。",
              );
            return result.items.map((item) => item.paper!);
          }}
          onResearchHelp={help}
          onResearchExtract={async request => {
            const saved = await api<BriefSnapshot>(`/api/v3/workstation/briefs/${request.brief.id}`);
            if (saved.brief.revision !== request.brief.revision) throw new APIError("brief_revision_conflict", 409);
            await launchResearch({ schema_version: "citequest-research-planning-input/1", brief_id: saved.brief.id,
              brief_revision: saved.brief.revision, brief_sha256: saved.brief_sha256,
              user_confirmed: {}, explicit_unknowns: {}, selected_sources: saved.brief.selected_sources,
              draft_source_ids: saved.brief.draft_source_ids, research_status: "proposed_not_executed", school_format_ready: false
            }, "extract", request.fields.length ? request.fields.slice(0,4) : ["topic", "purpose", "methods", "resources"]);
          }}
          onResearchStart={async (input) => {
            setError(undefined);
            try {
              await launchResearch(input, "plan");
            } catch (e) {
              setError(e);
              throw e;
            }
          }}
        />
      </Suspense>
    </section>
  );
}
function Upload() {
  const [tab, setTab] = useState("file");
  const [text, setText] = useState("");
  const [title, setTitle] = useState("");
  const [link, setLink] = useState("");
  const [result, setResult] = useState<any>();
  const [error, setError] = useState<unknown>();
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const run = async () => {
    setError(undefined);
    setBusy(true);
    try {
      if (!current?.authenticated) throw new APIError("login_required", 401);
      if (tab === "text") {
        setResult(
          await api(
            "/api/v2/materials/text",
            { title, text, format: "paste" },
            "POST",
          ),
        );
      } else if (tab === "link") {
        setResult(
          await api(
            link.startsWith("10.")
              ? "/api/v2/imports/doi"
              : "/api/v2/imports/url",
            link.startsWith("10.") ? { doi: link } : { url: link },
          ),
        );
      } else {
        const file = input.current?.files?.[0];
        if (!file) throw new Error("请先选择文件。");
        const maxBytes = file.name.toLowerCase().endsWith(".pdf")
          ? 50 * 1024 * 1024
          : file.name.toLowerCase().endsWith(".docx")
            ? 20 * 1024 * 1024
            : 2 * 1024 * 1024;
        if (file.size > maxBytes)
          throw new Error(
            "文件超过该类型上限：PDF 50MiB / Word 20MiB / 文本 2MiB。",
          );
        const bytes = await file.arrayBuffer();
        const hash = Array.from(
          new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
        )
          .map((x) => x.toString(16).padStart(2, "0"))
          .join("");
        const media = file.name.toLowerCase().endsWith(".pdf")
          ? "application/pdf"
          : file.name.toLowerCase().endsWith(".docx")
            ? "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            : file.name.toLowerCase().endsWith(".md")
              ? "text/markdown"
              : "text/plain";
        const ticket = await fetch("/api/v2/uploads", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Idempotency-Key": crypto.randomUUID(),
            "X-CSRF-Token": current.csrf,
          },
          body: JSON.stringify({
            file_name: file.name,
            media_type: media,
            size_bytes: file.size,
            expected_sha256: hash,
            project_id: current.project_id,
          }),
        });
        const body = await ticket.json();
        if (!ticket.ok)
          throw new APIError(body.code || "upload_rejected", ticket.status);
        const response = await fetch(body.upload_url, {
          method: "PUT",
          headers: {
            "Content-Type": "application/octet-stream",
            "X-CSRF-Token": current.csrf,
          },
          body: bytes,
        });
        const receipt = await response.json();
        if (!response.ok)
          throw new APIError(receipt.code || "upload_failed", response.status);
        setResult(receipt);
      }
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="content">
      <div className="eyebrow">IMPORT · KEEP THE SOURCE</div>
      <h1>导入研究材料</h1>
      <p className="muted">
        保留原始文件与来源位置。解析在后台执行，缺页与失败会保留记录。上传后默认禁止向外部模型发送正文，需在个人工作区单独允许。
      </p>
      <div className="tabs" role="tablist">
        {[
          ["file", "上传文件"],
          ["text", "粘贴文本"],
          ["link", "DOI / 论文链接"],
        ].map(([id, label]) => (
          <button
            role="tab"
            aria-selected={tab === id}
            key={id}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="card">
        {tab === "file" ? (
          <label>
            PDF / DOCX / TXT / Markdown
            <input type="file" ref={input} accept=".pdf,.docx,.txt,.md" />
          </label>
        ) : tab === "text" ? (
          <>
            <label>
              材料标题
              <input
                value={title}
                maxLength={500}
                onChange={(e) => setTitle(e.target.value)}
              />
            </label>
            <label>
              原始文本
              <textarea
                value={text}
                maxLength={200000}
                onChange={(e) => setText(e.target.value)}
                rows={12}
              />
            </label>
          </>
        ) : (
          <label>
            DOI 或受支持的 HTTPS 论文链接
            <input
              value={link}
              onChange={(e) => setLink(e.target.value)}
              placeholder="10.… 或 https://arxiv.org/abs/…"
            />
          </label>
        )}
        <button className="primary" disabled={busy} onClick={run}>
          {busy ? "正在核验和保存…" : "保存材料"}
        </button>
        {!!error && <Notice error={error} />}{" "}
        {result && (
          <>
            <p className="success">服务端已接收，解析状态请到工作区查看。</p>
            <pre className="receipt">{JSON.stringify(result, null, 2)}</pre>
            <Link to="/workspace">进入工作区 →</Link>
          </>
        )}
      </div>
    </section>
  );
}
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Suspense
        fallback={
          <p className="loading" role="status">
            正在加载阅读组件…
          </p>
        }
      >
        <App />
      </Suspense>
    </BrowserRouter>
  </StrictMode>,
);
