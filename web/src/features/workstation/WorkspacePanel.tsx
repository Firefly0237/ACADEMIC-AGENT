import {
  lazy,
  Suspense,
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from "react";
import { Link } from "react-router-dom";
import { paperPath } from "../../api";
import { BriefEditor } from "../research/BriefEditor";
import type { BriefSnapshot, BriefSummary } from "../research/types";
import { WORKSPACE, workspaceError, workspaceRequest } from "./client";
import type {
  Collection,
  EvidenceSelection,
  HistoryItem,
  LibraryItem,
  Note,
  Page,
  SavedPaperItem,
  SourceAvailability,
  TaskLaunch,
  TaskSummary,
  WorkspacePanelProps,
} from "./types";
import { CollectionManager } from "./CollectionManager";
import "./workspace.css";

const Markdown = lazy(() =>
  import("../../Markdown").then((module) => ({ default: module.Markdown })),
);
type Entry =
  { kind: "paper"; item: SavedPaperItem } | { kind: "work"; item: LibraryItem };
type Material = {
  file_id: string;
  display_title?: string;
  media_type: string;
  status: string;
  revision: number;
  outbound_policy: string;
};
const keyOf = (entry: Entry) =>
  entry.kind === "paper"
    ? `paper:${entry.item.saved_paper_id}`
    : `work:${entry.item.library_item_id}`;
const titleOf = (entry: Entry) =>
  entry.kind === "paper"
    ? entry.item.metadata?.title || "来源当前不可用"
    : entry.item.title || "来源未提供题名";
const page = <T,>(): Page<T> => ({ items: [], next_cursor: null });
const day = (value: string) =>
  new Date(value).toLocaleString("zh-CN", { hour12: false });
const kindName = (value: string) =>
  ({
    qa: "文献问答",
    summary: "文献摘要",
    compare: "多文比较",
    stance: "观点证据",
    research: "研究计划",
    research_plan: "研究计划",
  })[value] || "研究任务";
const stateName = (value: string) =>
  ({
    queued: "排队中",
    running: "正在执行",
    waiting_input: "等待补充",
    succeeded: "已完成",
    partial: "部分完成",
    failed: "未完成",
    cancelled: "已取消",
    reconciling: "正在核对执行结果",
  })[value] || "状态待核实";

export function WorkspacePanel(props: WorkspacePanelProps) {
  return <WorkspaceScope key={props.identity.owner_id} {...props} />;
}

function WorkspaceScope({
  identity,
  renderTask,
  resolveSelections,
  onOpenWork,
  onOpenTask,
  onResearchHelp,
  onResearchStart,
  onResearchExtract,
  briefRefresh,
}: WorkspacePanelProps) {
  const [tab, setTab] = useState("library");
  const [saved, setSaved] = useState(page<SavedPaperItem>);
  const [legacy, setLegacy] = useState(page<LibraryItem>);
  const [collections, setCollections] = useState(page<Collection>);
  const [history, setHistory] = useState(page<HistoryItem>);
  const [tasks, setTasks] = useState(page<TaskSummary>);
  const [briefs, setBriefs] = useState(page<BriefSummary>);
  const [materials, setMaterials] = useState(page<Material>);
  const [query, setQuery] = useState("");
  const [tag, setTag] = useState("");
  const [collection, setCollection] = useState("");
  const [filters, setFilters] = useState({
    query: "",
    tag: "",
    collection: "",
  });
  const [selected, setSelected] = useState<Record<string, Entry>>({});
  const [launch, setLaunch] = useState<TaskLaunch | null>(null);
  const [edit, setEdit] = useState<Entry | null>(null);
  const [editTags, setEditTags] = useState("");
  const [editCollections, setEditCollections] = useState<string[]>([]);
  const [notes, setNotes] = useState(page<Note>);
  const [noteText, setNoteText] = useState("");
  const [editNote, setEditNote] = useState<Note | null>(null);
  const [noteDelete, setNoteDelete] = useState<Note | null>(null);
  const [remove, setRemove] = useState<Entry | null>(null);
  const [undo, setUndo] = useState<Entry | null>(null);
  const [clearHistory, setClearHistory] = useState<{
    through: string;
    operation_id: string;
  } | null>(null);
  const [brief, setBrief] = useState<BriefSnapshot | null>(null);
  const [briefKey, setBriefKey] = useState("new");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const epoch = useRef(0);
  const operations = useRef(
    new Map<string, { signature: string; operation: string }>(),
  );
  const activeFilters = useRef(filters);
  activeFilters.current = filters;
  const libraryRequest = useRef(0);
  const selectedEntries = Object.values(selected);
  useEffect(() => {
    if (!briefRefresh || briefRefresh.id !== brief?.brief.id || briefRefresh.revision <= brief.brief.revision) return;
    const controller = new AbortController();
    get<BriefSnapshot>(`${WORKSPACE}/briefs/${briefRefresh.id}`, controller.signal).then(setBrief)
      .catch(cause => { if (cause.name !== "AbortError") setError(workspaceError(cause)); });
    return () => controller.abort();
  }, [briefRefresh?.id, briefRefresh?.revision, brief?.brief.id]);

  function operation(key: string, payload: unknown) {
    const signature = JSON.stringify(payload);
    const previous = operations.current.get(key);
    if (previous?.signature === signature) return previous.operation;
    const value = crypto.randomUUID();
    operations.current.set(key, { signature, operation: value });
    return value;
  }
  async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
    const current = epoch.current;
    const result = await workspaceRequest<T>(identity, path, { signal });
    if (current !== epoch.current)
      throw new DOMException("identity changed", "AbortError");
    return result;
  }
  async function mutate<T>(
    path: string,
    body?: unknown,
    method = "POST",
  ): Promise<T> {
    const current = epoch.current;
    const result = await workspaceRequest<T>(identity, path, { body, method });
    if (current !== epoch.current)
      throw new DOMException("identity changed", "AbortError");
    return result;
  }
  function params(cursor?: string | null) {
    const value = new URLSearchParams({ limit: "20" });
    if (activeFilters.current.query)
      value.set("query", activeFilters.current.query);
    if (activeFilters.current.tag) value.set("tag", activeFilters.current.tag);
    if (activeFilters.current.collection)
      value.set("collection_id", activeFilters.current.collection);
    if (cursor) value.set("cursor", cursor);
    return value;
  }
  async function loadLibrary(signal?: AbortSignal) {
    const request = ++libraryRequest.current;
    const results = await Promise.allSettled([
      get<Page<SavedPaperItem>>(
        `${WORKSPACE}/saved-papers?${params()}`,
        signal,
      ),
      get<Page<LibraryItem>>(`${WORKSPACE}/library?${params()}`, signal),
    ]);
    if (signal?.aborted || request !== libraryRequest.current) return;
    if (results[0].status === "fulfilled") setSaved(results[0].value);
    if (results[1].status === "fulfilled") setLegacy(results[1].value);
    const failures = results.filter((result) => result.status === "rejected");
    if (failures.length)
      setError(
        "部分文献列表暂不可用。" +
          workspaceError((failures[0] as PromiseRejectedResult).reason),
      );
  }
  useEffect(() => {
    epoch.current += 1;
    operations.current.clear();
    setSelected({});
    setEdit(null);
    setBrief(null);
    setBriefKey("new");
    setSaved(page());
    setLegacy(page());
    setHistory(page());
    setTasks(page());
    setMaterials(page());
    setBriefs(page());
    setNotice("");
    setError("");
    setLaunch(null);
  }, [identity.owner_id]);
  useEffect(() => {
    if (!identity.authenticated) return;
    const controller = new AbortController();
    get<Page<Collection>>(
      `${WORKSPACE}/collections?limit=50`,
      controller.signal,
    )
      .then(setCollections)
      .catch((cause) => {
        if (!controller.signal.aborted) setError(workspaceError(cause));
      });
    return () => controller.abort();
  }, [identity.owner_id, identity.authenticated]);
  useEffect(() => {
    if (!identity.authenticated) return;
    const controller = new AbortController();
    setLoading(true);
    setError("");
    const work =
      tab === "library"
        ? loadLibrary(controller.signal)
        : tab === "history"
          ? get<Page<HistoryItem>>(
              `${WORKSPACE}/history?limit=20`,
              controller.signal,
            ).then(setHistory)
          : tab === "tasks"
            ? get<Page<TaskSummary>>(
                `${WORKSPACE}/tasks?limit=20`,
                controller.signal,
              ).then(setTasks)
            : tab === "materials"
              ? get<Page<Material>>(
                  "/api/v3/materials?page_size=20",
                  controller.signal,
                ).then(setMaterials)
              : identity.project_id
                ? get<Page<BriefSummary>>(
                    `${WORKSPACE}/projects/${identity.project_id}/briefs?limit=20`,
                    controller.signal,
                  ).then(setBriefs)
                : Promise.resolve();
    work
      .catch((cause) => {
        if (!controller.signal.aborted) setError(workspaceError(cause));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [
    tab,
    filters,
    identity.owner_id,
    identity.authenticated,
    identity.project_id,
  ]);

  async function action(work: () => Promise<void>) {
    const current = epoch.current;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await work();
    } catch (cause) {
      if (current === epoch.current) setError(workspaceError(cause));
    } finally {
      if (current === epoch.current) setBusy(false);
    }
  }
  function toggle(entry: Entry, checked: boolean) {
    setSelected((current) => {
      const next = { ...current };
      if (checked) next[keyOf(entry)] = entry;
      else delete next[keyOf(entry)];
      return next;
    });
  }
  async function launchSelected(kind: "qa" | "summary" | "compare") {
    const papers = selectedEntries
      .filter((entry) => entry.kind === "paper")
      .map((entry) => entry.item as SavedPaperItem);
    const works = selectedEntries
      .filter((entry) => entry.kind === "work")
      .map((entry) => entry.item as LibraryItem);
    if (papers.some((item) => item.source_availability !== "available")) {
      setError("所选文献来源当前不可用，请先重新核对来源状态。");
      return;
    }
    const resolved = works.length ? await resolveSelections!(works) : [];
    const selections: EvidenceSelection[] = [
      ...papers.map((item) => item.paper),
      ...resolved,
    ];
    const unique = selections.filter(
      (value, index) =>
        selections.findIndex(
          (other) => other.kind === value.kind && other.id === value.id,
        ) === index,
    );
    if (
      unique.length < (kind === "compare" ? 2 : 1) ||
      unique.length > (kind === "qa" ? 20 : 10)
    ) {
      setError("问答最多20篇，摘要或比较最多10篇；比较需要至少2篇。");
      return;
    }
    if (
      kind !== "qa" &&
      unique.some((value) => value.kind === "metadata_projection")
    ) {
      setError("选择中有仅题录文献，缺少全文时不能生成全文摘要或比较。");
      return;
    }
    setLaunch({
      selected: unique,
      kind,
      catalog: unique.every((value) => value.kind === "metadata_projection")
        ? "standard_metadata"
        : "application_papers",
    });
  }
  async function useForResearch() {
    if (!identity.project_id) {
      setError("当前账户没有可用研究项目，请重新进入个人中心。");
      return;
    }
    const papers = selectedEntries
      .filter((entry) => entry.kind === "paper")
      .map((entry) => entry.item as SavedPaperItem);
    const works = selectedEntries
      .filter((entry) => entry.kind === "work")
      .map((entry) => entry.item as LibraryItem);
    const resolved = works.length ? await resolveSelections!(works) : [];
    const selected_sources = [
      ...(brief?.brief.selected_sources || []),
      ...papers.map((item) => item.paper),
      ...resolved,
    ].filter(
      (value, index, values) =>
        values.findIndex(
          (other) => other.kind === value.kind && other.id === value.id,
        ) === index,
    );
    if (selected_sources.length > 20) {
      setError("当前研究档案最多关联20份资料，请先调整选文范围。");
      return;
    }
    const payload = brief
      ? {
          expected_revision: brief.brief.revision,
          patches: [],
          selected_sources,
        }
      : { project_id: identity.project_id, patches: [], selected_sources };
    const operationId = operation("research-sources", payload);
    const value = await mutate<BriefSnapshot>(
      brief ? `${WORKSPACE}/briefs/${brief.brief.id}` : `${WORKSPACE}/briefs`,
      {
        ...payload,
        operation_id: operationId,
        message_id: operationId,
      },
      brief ? "PATCH" : "POST",
    );
    setBrief(value);
    setBriefKey(value.brief.id);
    setTab("research");
    setNotice("已把所选文献关联到研究档案，资料范围会随档案版本保存。");
  }
  async function attachDraft(fileId: string) {
    if (!identity.project_id) throw new Error("请先选择可用研究项目。");
    const selected_sources = [...(brief?.brief.selected_sources || [])];
    if (!selected_sources.some(item=>item.kind==="material_file"&&item.id===fileId))
      selected_sources.push({kind:"material_file",id:fileId});
    const draft_source_ids=Array.from(new Set([...(brief?.brief.draft_source_ids||[]),fileId]));
    if(selected_sources.length>20||draft_source_ids.length>4)throw new Error("一个档案最多关联20份资料，其中草稿最多4份。");
    const payload=brief?{expected_revision:brief.brief.revision,patches:[],selected_sources,draft_source_ids}
      :{project_id:identity.project_id,patches:[],selected_sources,draft_source_ids};
    const id=operation("research-drafts",payload);
    const value=await mutate<BriefSnapshot>(brief?`${WORKSPACE}/briefs/${brief.brief.id}`:`${WORKSPACE}/briefs`,
      {...payload,operation_id:id,message_id:id},brief?"PATCH":"POST");
    setBrief(value);setBriefKey(value.brief.id);setTab("research");
    setNotice("草稿已关联。AI读取仍需解析就绪和你的处理许可；它不会被当作学术参考文献。");
  }
  function applySourceAvailability(
    entry: Entry,
    availability?: SourceAvailability,
  ) {
    if (
      entry.kind === "paper" &&
      availability &&
      availability !== "available"
    ) {
      const unavailable: Entry = {
        kind: "paper",
        item: {
          ...entry.item,
          metadata: null,
          source_availability: availability,
          qa_scope: "unavailable",
          fulltext_summary_available: false,
        },
      };
      setEdit((current) =>
        current && keyOf(current) === keyOf(entry) ? unavailable : current,
      );
      setSelected((current) =>
        current[keyOf(entry)]
          ? { ...current, [keyOf(entry)]: unavailable }
          : current,
      );
      setSaved((current) => ({
        ...current,
        items: current.items.map((item) =>
          item.saved_paper_id === entry.item.saved_paper_id
            ? unavailable.item
            : item,
        ),
      }));
    }
  }
  async function openEdit(entry: Entry) {
    setEdit(entry);
    setEditTags(entry.item.tags.join("、"));
    setEditCollections(entry.item.collection_ids);
    setNoteText("");
    setEditNote(null);
    setNotes(page());
    const path =
      entry.kind === "paper"
        ? `${WORKSPACE}/saved-papers/${entry.item.saved_paper_id}/notes?limit=20`
        : `${WORKSPACE}/library/${entry.item.library_item_id}/notes?limit=20`;
    const notePage = await get<Page<Note>>(path);
    setNotes(notePage);
    applySourceAvailability(entry, notePage.source_availability);
  }
  async function saveOrganization() {
    if (!edit) return;
    const tags = editTags
      .split(/[、,，\n]/u)
      .map((value) => value.trim())
      .filter(Boolean);
    const body = {
      expected_revision: edit.item.revision,
      tags: [...new Set(tags)],
      collection_ids: editCollections,
    };
    const value =
      edit.kind === "paper"
        ? await mutate<SavedPaperItem>(
            `${WORKSPACE}/saved-papers/${edit.item.saved_paper_id}`,
            { ...body, operation_id: operation("organization", body) },
            "PATCH",
          )
        : await mutate<LibraryItem>(
            `/api/v2/library/items/${edit.item.library_item_id}`,
            body,
            "PATCH",
          );
    setEdit({ kind: edit.kind, item: value } as Entry);
    await loadLibrary();
    setNotice("分组与标签已保存。移出分组不会删除文献。");
  }
  async function saveNote() {
    if (!edit || !noteText.trim()) return;
    const body = { body: noteText };
    if (edit.kind === "paper") {
      const receipt = await mutate<{
        source_availability?: SourceAvailability;
      }>(
        editNote
          ? `${WORKSPACE}/saved-paper-notes/${editNote.note_id}`
          : `${WORKSPACE}/saved-papers/${edit.item.saved_paper_id}/notes`,
        {
          ...body,
          operation_id: operation("note", {
            ...body,
            note_id: editNote?.note_id,
            item: keyOf(edit),
          }),
          ...(editNote ? { expected_revision: editNote.revision } : {}),
        },
        editNote ? "PATCH" : "POST",
      );
      applySourceAvailability(edit, receipt.source_availability);
    } else {
      await mutate(
        editNote ? `/api/v2/notes/${editNote.note_id}` : "/api/v2/notes",
        editNote
          ? { ...body, expected_revision: editNote.revision }
          : {
              ...body,
              library_item_id: edit.item.library_item_id,
              evidence_ids: [],
              expected_revision: null,
              client_operation_id: operation("note", {
                ...body,
                item: keyOf(edit),
              }),
            },
        editNote ? "PATCH" : "POST",
      );
    }
    setNoteText("");
    setEditNote(null);
    const path =
      edit.kind === "paper"
        ? `${WORKSPACE}/saved-papers/${edit.item.saved_paper_id}/notes?limit=20`
        : `${WORKSPACE}/library/${edit.item.library_item_id}/notes?limit=20`;
    const notePage = await get<Page<Note>>(path);
    setNotes(notePage);
    applySourceAvailability(edit, notePage.source_availability);
    setNotice("笔记已保存。");
  }
  async function deleteEntry(entry: Entry, restore = false) {
    const value =
      entry.kind === "paper"
        ? await mutate<SavedPaperItem>(
            `${WORKSPACE}/saved-papers/${entry.item.saved_paper_id}/lifecycle`,
            {
              operation_id: operation("lifecycle", {
                id: keyOf(entry),
                revision: entry.item.revision,
                restore,
              }),
              expected_revision: entry.item.revision,
              deleted: !restore,
            },
          )
        : await mutate<LibraryItem>(
            `/api/v2/library/items/${entry.item.library_item_id}${restore ? "/restore" : ""}`,
            restore ? {} : undefined,
            restore ? "POST" : "DELETE",
          );
    setRemove(null);
    setUndo(restore ? null : ({ kind: entry.kind, item: value } as Entry));
    setEdit(null);
    setSelected((current) => {
      const next = { ...current };
      delete next[keyOf(entry)];
      return next;
    });
    await loadLibrary();
    setNotice(
      restore
        ? "收藏已恢复，原有分组和笔记关系保留。"
        : "已移出收藏；原文和上传材料没有被删除。",
    );
  }
  function historyLink(item: HistoryItem) {
    if (item.kind === "search" && item.search) {
      const search = item.search;
      const params = new URLSearchParams({ q: search.query, catalog: search.catalog, mode: search.mode,
        field: search.field, sort: search.sort });
      if (search.filters.year_from !== null) params.set("from", String(search.filters.year_from));
      if (search.filters.year_to !== null) params.set("to", String(search.filters.year_to));
      return <div>
        <p>{search.query}</p>
        <small className="muted">原检索版本 {search.original_generation_id.slice(0, 8)} · 再检索将使用当前语料，结果可能变化。</small>
        {search.requires_source_reselection
          ? <p className="muted">原查询限定了所选文献；请回到文献库重新选择范围。</p>
          : <Link to={`/search?${params.toString()}`}>按原条件重新检索</Link>}
      </div>;
    }
    const parts = item.target_id.split(":");
    if (
      parts[0] === "paper" &&
      ["metadata_projection", "native_representation"].includes(parts[1]) &&
      parts[2]
    ) {
      if (item.source_availability && item.source_availability !== "available")
        return <span className="muted">来源当前不可用，已隐藏题录</span>;
      return (
        <Link
          to={paperPath({
            kind: parts[1] as "metadata_projection" | "native_representation",
            id: parts[2],
          })}
        >
          {item.display_title || "重新查看文献（标题未提供）"}
        </Link>
      );
    }
    if (parts[0] === "file" && parts[1])
      return <Link to={`/materials/${parts[1]}`}>查看材料</Link>;
    if (parts[0] === "task" && parts[1])
      return (
        <button onClick={() => onOpenTask?.(parts[1])} disabled={!onOpenTask}>
          查看任务
        </button>
      );
    if (parts[0] === "work" && parts[1])
      return (
        <button onClick={() => onOpenWork?.(parts[1])} disabled={!onOpenWork}>
          查看文献
        </button>
      );
    return null;
  }
  function entryCard(entry: Entry) {
    const meta = entry.kind === "paper" ? entry.item.metadata : entry.item;
    const unavailable =
      entry.kind === "paper" && entry.item.source_availability !== "available";
    return (
      <article className="cq-library-card" key={keyOf(entry)}>
        <input
          aria-label={`选择文献：${titleOf(entry)}`}
          type="checkbox"
          checked={!!selected[keyOf(entry)]}
          onChange={(event) => toggle(entry, event.target.checked)}
          disabled={unavailable}
        />
        <div className="cq-entry-body">
          <h3>
            {entry.kind === "paper" ? (
              <Link to={paperPath(entry.item.paper)}>{titleOf(entry)}</Link>
            ) : (
              <button
                className="cq-title-button"
                disabled={!onOpenWork}
                onClick={() => onOpenWork?.(entry.item.work_id)}
              >
                {titleOf(entry)}
              </button>
            )}
          </h3>
          <p className="muted">
            {meta?.authors?.join("、") ||
              (unavailable ? "来源当前不可访问" : "源未提供作者")}{" "}
            {meta?.year ? `· ${meta.year}` : ""}
          </p>
          {entry.kind === "paper" && (
            <p className="cq-coverage">
              {unavailable
                ? "保留收藏关系，来源内容不可用。"
                : entry.item.qa_scope === "metadata_only"
                  ? "题录与已有摘要可阅读、引用；全文摘要需要可用正文。"
                  : "有可用原生正文；AI任务仍需当前授权与证据检查。"}
            </p>
          )}
          <div className="cq-tag-list">
            {entry.item.tags.map((value) => (
              <span className="tag" key={value}>
                {value}
              </span>
            ))}
          </div>
          <div className="cq-inline-actions">
            <button
              disabled={busy}
              onClick={() => action(() => openEdit(entry))}
            >
              分组、标签与笔记
            </button>
            <button disabled={busy} onClick={() => setRemove(entry)}>
              移出收藏
            </button>
          </div>
        </div>
      </article>
    );
  }

  if (!identity.authenticated)
    return (
      <section className="cq-workspace">
        <h1>我的科研工作区</h1>
        <p>
          <a href="/auth/login">登录个人中心</a>后管理文献、研究材料与任务。
        </p>
      </section>
    );
  const canResolve =
    !selectedEntries.some((entry) => entry.kind === "work") ||
    !!resolveSelections;
  const selectedSourcesAvailable = selectedEntries.every(
    (entry) =>
      entry.kind === "work" || entry.item.source_availability === "available",
  );
  const canSummarize = selectedEntries.every((entry) =>
    entry.kind === "paper"
      ? entry.item.fulltext_summary_available
      : !!resolveSelections,
  );
  return (
    <section className="cq-workspace" data-owner-scope="current-session">
      <header className="cq-workspace-heading">
        <div>
          <p className="eyebrow">PERSONAL RESEARCH WORKSPACE</p>
          <h1>我的科研工作区</h1>
          <p className="muted">
            文献、笔记、研究方向与实际任务，在同一个私人项目中继续。
          </p>
        </div>
        <Link className="button" to="/upload">
          ＋ 导入材料
        </Link>
      </header>
      <nav className="cq-workspace-tabs" aria-label="工作区内容">
        {[
          ["library", "文献库"],
          ["materials", "我的材料"],
          ["research", "研究项目"],
          ["history", "浏览与检索历史"],
          ["tasks", "任务与产物"],
        ].map(([value, label]) => (
          <button
            key={value}
            aria-current={tab === value ? "page" : undefined}
            onClick={() => {
              setTab(value);
              setEdit(null);
            }}
          >
            {label}
          </button>
        ))}
      </nav>
      {error && (
        <p className="notice" role="alert">
          {error}
        </p>
      )}
      {notice && (
        <p className="cq-success" role="status">
          {notice}
        </p>
      )}
      {loading && (
        <p className="loading" role="status">
          正在读取当前账户的内容…
        </p>
      )}
      {undo && (
        <div className="cq-undo" role="status">
          刚才移出的收藏可以恢复。
          <button
            disabled={busy}
            onClick={() => action(() => deleteEntry(undo, true))}
          >
            恢复收藏
          </button>
        </div>
      )}
      {tab === "library" && (
        <>
          <form
            className="cq-library-filters"
            onSubmit={(event: FormEvent) => {
              event.preventDefault();
              setFilters({ query, tag, collection });
              setEdit(null);
            }}
          >
            <label>
              在文献库中查找
              <input
                value={query}
                maxLength={200}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="文献题名关键词"
              />
            </label>
            <label>
              集合
              <select
                value={collection}
                onChange={(event) => setCollection(event.target.value)}
              >
                <option value="">全部集合</option>
                {collections.items.map((item) => (
                  <option value={item.collection_id} key={item.collection_id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              标签
              <input
                value={tag}
                maxLength={100}
                onChange={(event) => setTag(event.target.value)}
                placeholder="精确标签"
              />
            </label>
            <button className="primary">应用筛选</button>
          </form>
          <CollectionManager
            identity={identity}
            data={collections}
            onChange={setCollections}
            onNotice={setNotice}
          />
          <div className="cq-selection-bar">
            <span>
              已选择 {selectedEntries.length} 篇
              {selectedEntries.length ? "（可跨页保留）" : ""}
            </span>
            <button
              disabled={!selectedEntries.length}
              onClick={() => {
                setSelected({});
                setLaunch(null);
              }}
            >
              清除选择
            </button>
            <button
              disabled={
                busy ||
                !renderTask ||
                !selectedEntries.length ||
                selectedEntries.length > 20 ||
                !selectedSourcesAvailable ||
                !canResolve
              }
              onClick={() => action(() => launchSelected("qa"))}
            >
              对选中文献提问
            </button>
            <button
              disabled={
                busy ||
                !renderTask ||
                !selectedEntries.length ||
                selectedEntries.length > 10 ||
                !canResolve ||
                !canSummarize
              }
              onClick={() => action(() => launchSelected("summary"))}
            >
              单篇或多篇摘要
            </button>
            <button
              disabled={
                busy ||
                !renderTask ||
                selectedEntries.length < 2 ||
                selectedEntries.length > 10 ||
                !canResolve ||
                !canSummarize
              }
              onClick={() => action(() => launchSelected("compare"))}
            >
              比较选中文献
            </button>
            <button
              disabled={
                busy ||
                !identity.project_id ||
                !selectedEntries.length ||
                !selectedSourcesAvailable ||
                !canResolve
              }
              onClick={() => action(useForResearch)}
            >
              {brief ? "添加选文到当前研究档案" : "用选文建立研究档案"}
            </button>
          </div>
          {selectedEntries.length > 0 && (
            <details className="cq-selected-review">
              <summary>查看已选文献范围</summary>
              <ul>
                {selectedEntries.map((entry) => (
                  <li key={keyOf(entry)}>
                    <span>{titleOf(entry)}</span>
                    <button onClick={() => toggle(entry, false)}>
                      从本次选择移除
                    </button>
                  </li>
                ))}
              </ul>
            </details>
          )}
          {!canResolve && (
            <p className="muted">
              部分条目尚未定位到可核验的文献版本，请先阅读其材料或等待正文定位功能接线。
            </p>
          )}
          {selectedEntries.length > 0 && !selectedSourcesAvailable && (
            <p className="notice">
              部分已选文献来源当前不可用，请移除这些选择或重新核对来源状态；不会用旧题录代替当前证据。
            </p>
          )}
          {selectedEntries.length > 0 &&
            selectedSourcesAvailable &&
            !canSummarize && (
              <p className="muted">
                当前选择包含仅题录文献，暂不能做全文摘要或比较；可以就现有题录信息提问。
              </p>
            )}
          {!renderTask && (
            <p className="muted">
              AI任务入口尚未接线。收藏、分组与笔记仍可正常使用。
            </p>
          )}
          <div className="cq-library-list">
            {saved.items.map((item) => entryCard({ kind: "paper", item }))}
          </div>
          {saved.partial_scan && (
            <p className="muted" role="status">
              本次仅核对了部分收藏，当前结果还不完整。可以继续查找或缩小关键词、集合与标签范围。
            </p>
          )}
          {(saved.source_check_incomplete ||
            saved.library_changed_during_read) && (
            <p className="notice" role="status">
              {saved.source_check_incomplete
                ? "部分来源暂时未能核对，当前结果不完整。可稍后刷新，已保存的收藏与笔记仍会保留。"
                : "读取期间收藏已变化，本次隐藏了过时条目。刷新后可查看当前列表。"}
            </p>
          )}
          {!loading &&
            !saved.partial_scan &&
            !saved.source_check_incomplete &&
            !saved.library_changed_during_read &&
            !saved.items.length &&
            !legacy.items.length &&
            !error && (
              <p className="empty">
                还没有收藏文献。在搜索结果或详情页收藏后，可在这里分组、记笔记或开始研究。
              </p>
            )}
          {saved.next_cursor && (
            <button
              disabled={busy}
              onClick={() =>
                action(async () => {
                  const next = await get<Page<SavedPaperItem>>(
                    `${WORKSPACE}/saved-papers?${params(saved.next_cursor)}`,
                  );
                  setSaved((current) => ({
                    ...next,
                    items: [...current.items, ...next.items],
                    next_cursor: next.next_cursor,
                  }));
                })
              }
            >
              加载更多收藏文献
            </button>
          )}
          {legacy.items.length > 0 && (
            <details open className="cq-legacy-items">
              <summary>其他已保存文献条目 · {legacy.items.length}</summary>
              {legacy.items.map((item) => entryCard({ kind: "work", item }))}
              {legacy.next_cursor && (
                <button
                  disabled={busy}
                  onClick={() =>
                    action(async () => {
                      const next = await get<Page<LibraryItem>>(
                        `${WORKSPACE}/library?${params(legacy.next_cursor)}`,
                      );
                      setLegacy((current) => ({
                        items: [...current.items, ...next.items],
                        next_cursor: next.next_cursor,
                      }));
                    })
                  }
                >
                  加载更多文献条目
                </button>
              )}
            </details>
          )}
        </>
      )}
      {edit && tab === "library" && (
        <section className="cq-entry-editor" aria-label="文献分组与笔记">
          <div className="cq-section-heading">
            <h2>{titleOf(edit)}</h2>
            <button onClick={() => setEdit(null)}>关闭编辑</button>
          </div>
          <fieldset disabled={busy}>
            <legend>文献组织</legend>
            <label>
              标签（用顿号或逗号分隔）
              <input
                value={editTags}
                maxLength={3000}
                onChange={(event) => setEditTags(event.target.value)}
              />
            </label>
            <div className="cq-check-grid">
              {collections.items.map((item) => (
                <label key={item.collection_id}>
                  <input
                    type="checkbox"
                    checked={editCollections.includes(item.collection_id)}
                    onChange={(event) =>
                      setEditCollections((current) =>
                        event.target.checked
                          ? [...current, item.collection_id]
                          : current.filter(
                              (value) => value !== item.collection_id,
                            ),
                      )
                    }
                  />
                  {item.name}
                </label>
              ))}
            </div>
            <button
              className="primary"
              onClick={() => action(saveOrganization)}
            >
              保存分组与标签
            </button>
          </fieldset>
          <h3>我的笔记</h3>
          {notes.source_availability &&
            notes.source_availability !== "available" && (
              <p className="notice">
                来源当前不可访问。下面仅为你的个人笔记，不是可交给研究助手的原文证据。
              </p>
            )}
          <p className="muted">
            个人笔记与原文证据分开保存，不会自动变成AI已经核实的研究结论。
          </p>
          {notes.items.map((note) => (
            <article className="cq-note" key={note.note_id}>
              <Suspense fallback={<p className="cq-literal">{note.body}</p>}>
                <Markdown text={note.body} />
              </Suspense>
              <div className="cq-inline-actions">
                <button
                  onClick={() => {
                    setEditNote(note);
                    setNoteText(note.body);
                  }}
                >
                  编辑笔记
                </button>
                <button onClick={() => setNoteDelete(note)}>删除笔记</button>
              </div>
            </article>
          ))}
          {notes.next_cursor && (
            <button
              disabled={busy}
              onClick={() =>
                action(async () => {
                  const path =
                    edit.kind === "paper"
                      ? `${WORKSPACE}/saved-papers/${edit.item.saved_paper_id}/notes`
                      : `${WORKSPACE}/library/${edit.item.library_item_id}/notes`;
                  const next = await get<Page<Note>>(
                    `${path}?limit=20&cursor=${encodeURIComponent(notes.next_cursor!)}`,
                  );
                  applySourceAvailability(edit, next.source_availability);
                  setNotes((current) => ({
                    ...current,
                    items: [...current.items, ...next.items],
                    next_cursor: next.next_cursor,
                    source_availability:
                      next.source_availability ?? current.source_availability,
                  }));
                })
              }
            >
              继续读取笔记
            </button>
          )}
          <label>
            {editNote ? "修改这条笔记" : "添加笔记"}
            <textarea
              value={noteText}
              maxLength={15000}
              disabled={busy}
              onChange={(event) => setNoteText(event.target.value)}
              placeholder="记录阅读理解、待验证的问题；支持Markdown和公式。"
            />
          </label>
          <button
            disabled={busy || !noteText.trim()}
            onClick={() => action(saveNote)}
          >
            {editNote ? "保存笔记修改" : "保存新笔记"}
          </button>
          {editNote && (
            <button
              onClick={() => {
                setEditNote(null);
                setNoteText("");
              }}
            >
              取消编辑
            </button>
          )}
        </section>
      )}
      {remove && (
        <section
          className="cq-confirm"
          role="alertdialog"
          aria-label="确认移出收藏"
        >
          <h3>将“{titleOf(remove)}”移出收藏？</h3>
          <p>这不会删除原文或上传文件。分组和笔记关联会保留，之后可以恢复。</p>
          <button
            disabled={busy}
            onClick={() => action(() => deleteEntry(remove))}
          >
            确认移出收藏
          </button>
          <button onClick={() => setRemove(null)}>保留收藏</button>
        </section>
      )}
      {noteDelete && (
        <section
          className="cq-confirm"
          role="alertdialog"
          aria-label="确认删除笔记"
        >
          <p>删除这条个人笔记？这不会修改论文原文。</p>
          <button
            disabled={busy}
            onClick={() =>
              action(async () => {
                const receipt = await mutate<{
                  source_availability?: SourceAvailability;
                }>(
                  edit?.kind === "paper"
                    ? `${WORKSPACE}/saved-paper-notes/${noteDelete.note_id}`
                    : `/api/v2/notes/${noteDelete.note_id}`,
                  edit?.kind === "paper"
                    ? {
                        operation_id: operation("delete-note", noteDelete),
                        expected_revision: noteDelete.revision,
                      }
                    : undefined,
                  "DELETE",
                );
                if (edit)
                  applySourceAvailability(edit, receipt.source_availability);
                setNotes((current) => ({
                  ...current,
                  source_availability:
                    receipt.source_availability ?? current.source_availability,
                  items: current.items.filter(
                    (item) => item.note_id !== noteDelete.note_id,
                  ),
                }));
                setNoteDelete(null);
                setNotice("笔记已删除。");
              })
            }
          >
            确认删除笔记
          </button>
          <button onClick={() => setNoteDelete(null)}>保留笔记</button>
        </section>
      )}
      {tab === "materials" && (
        <section>
          <h2>我的研究材料</h2>
          <p className="muted">
            原始文件、可用解析正文和AI授权分别判断；导入成功不代表全文已经完整解析。
          </p>
          {materials.items.map((item) => (
            <article className="cq-library-card" key={item.file_id}>
              <div className="cq-entry-body">
                <h3>
                  <Link to={`/materials/${item.file_id}`}>
                    {item.display_title || "研究材料"}
                  </Link>
                </h3>
                <p>
                  {item.media_type} ·{" "}
                  {{
                    queued: "等待解析",
                    processing: "正在解析",
                    ready: "解析就绪",
                    partial: "部分可用",
                    failed: "解析未完成",
                  }[item.status] || "状态待核实"}
                </p>
                <p className="muted">
                  {item.outbound_policy === "selected_cloud_processing_allowed"
                    ? "已允许选定内容发送到配置的模型，执行前仍核对范围。"
                    : "当前仅在本地保存；可在详情页管理AI处理许可。"}
                </p>
                <Link to={`/materials/${item.file_id}`}>
                  阅读、下载与查看授权 →
                </Link>
                <button disabled={busy} onClick={()=>action(()=>attachDraft(item.file_id))}>作为研究草稿关联到档案</button>
                {renderTask && (
                  <button
                    disabled={
                      item.outbound_policy !==
                      "selected_cloud_processing_allowed"
                    }
                    onClick={() =>
                      setLaunch({
                        selected: [{ kind: "material_file", id: item.file_id }],
                        kind: "qa",
                        catalog: "application_papers",
                      })
                    }
                  >
                    就这篇材料提问
                  </button>
                )}
              </div>
            </article>
          ))}
          {!loading && !error && !materials.items.length && (
            <p className="empty">
              还没有上传材料，可以导入PDF、Word文档、链接或纯文字。
            </p>
          )}
          {materials.next_cursor && (
            <button
              disabled={busy}
              onClick={() =>
                action(async () => {
                  const next = await get<Page<Material>>(
                    `/api/v3/materials?page_size=20&cursor=${encodeURIComponent(materials.next_cursor!)}`,
                  );
                  setMaterials((current) => ({
                    items: [...current.items, ...next.items],
                    next_cursor: next.next_cursor,
                  }));
                })
              }
            >
              加载更多材料
            </button>
          )}
        </section>
      )}
      {tab === "research" && (
        <section>
          {!identity.project_id ? (
            <p className="notice">
              当前账户尚未建立研究项目，请重新进入已登录的个人中心。
            </p>
          ) : (
            <>
              <div className="cq-inline-actions">
                <button
                  onClick={() => {
                    setBrief(null);
                    setBriefKey(crypto.randomUUID());
                  }}
                >
                  新建研究档案
                </button>
                {briefs.items.map((item) => (
                  <button
                    key={item.brief_id}
                    onClick={() =>
                      action(async () => {
                        const value = await get<BriefSnapshot>(
                          `${WORKSPACE}/briefs/${item.brief_id}`,
                        );
                        setBrief(value);
                        setBriefKey(item.brief_id);
                      })
                    }
                  >
                    {item.topic || "未命名研究档案"} · 第{item.revision}版
                  </button>
                ))}
              </div>
              {briefs.next_cursor && (
                <button
                  disabled={busy}
                  onClick={() =>
                    action(async () => {
                      const next = await get<Page<BriefSummary>>(
                        `${WORKSPACE}/projects/${identity.project_id}/briefs?limit=20&cursor=${encodeURIComponent(briefs.next_cursor!)}`,
                      );
                      setBriefs((current) => ({
                        items: [...current.items, ...next.items],
                        next_cursor: next.next_cursor,
                      }));
                    })
                  }
                >
                  加载更多研究档案
                </button>
              )}
              <BriefEditor
                key={`${identity.owner_id}:${briefKey}`}
                identity={identity}
                projectId={identity.project_id}
                initial={brief}
                onSaved={(value) => {
                  setBrief(value);
                  setBriefs((current) => ({
                    ...current,
                    items: [
                      {
                        brief_id: value.brief.id,
                        project_id: value.brief.project_id,
                        revision: value.brief.revision,
                        topic: value.brief.facts.topic?.value || null,
                        created_at:
                          current.items.find(
                            (item) => item.brief_id === value.brief.id,
                          )?.created_at || "",
                        readiness: value.readiness,
                      },
                      ...current.items.filter(
                        (item) => item.brief_id !== value.brief.id,
                      ),
                    ],
                  }));
                }}
                onResearchHelp={onResearchHelp}
                onResearchStart={onResearchStart}
                onResearchExtract={onResearchExtract}
              />
            </>
          )}
        </section>
      )}
      {tab === "history" && (
        <section>
          <div className="cq-section-heading">
            <h2>浏览、检索与文件活动</h2>
            <button
              disabled={busy || !history.items.length}
              onClick={() =>
                setClearHistory({
                  through: new Date().toISOString(),
                  operation_id: crypto.randomUUID(),
                })
              }
            >
              清理已有历史
            </button>
          </div>
          <p className="muted">
            下载请求和实际下载完成分别记录；历史清理不会删除文献、笔记或文件。
          </p>
          {history.items.map((item) => (
            <article className="cq-history-row" key={item.history_id}>
              <div>
                <strong>
                  {{
                    view: "浏览记录",
                    search: "文献检索",
                    download_requested: "发起下载",
                    download_completed: "下载完成记录",
                    export: "导出活动",
                  }[item.kind] || "研究活动"}
                </strong>
                <small>{day(item.occurred_at)}</small>
              </div>
              {historyLink(item)}
            </article>
          ))}
          {!loading && !error && !history.items.length && (
            <p className="empty">尚无可显示的历史活动。</p>
          )}
          {history.next_cursor && (
            <button
              disabled={busy}
              onClick={() =>
                action(async () => {
                  const next = await get<Page<HistoryItem>>(
                    `${WORKSPACE}/history?limit=20&cursor=${encodeURIComponent(history.next_cursor!)}`,
                  );
                  setHistory((current) => ({
                    items: [...current.items, ...next.items],
                    next_cursor: next.next_cursor,
                  }));
                })
              }
            >
              加载更早的历史
            </button>
          )}
          {clearHistory && (
            <section
              className="cq-confirm"
              role="alertdialog"
              aria-label="确认清理历史"
            >
              <p>
                清理截至 {day(clearHistory.through)}{" "}
                的现有历史？之后的新活动会保留，文献与文件不受影响。
              </p>
              <button
                disabled={busy}
                onClick={() =>
                  action(async () => {
                    await mutate(`${WORKSPACE}/history/clear`, {
                      ...clearHistory,
                      kinds: [
                        "view",
                        "search",
                        "download_requested",
                        "download_completed",
                        "export",
                      ],
                    });
                    setHistory(
                      await get<Page<HistoryItem>>(
                        `${WORKSPACE}/history?limit=20`,
                      ),
                    );
                    setClearHistory(null);
                    setNotice("已有历史已清理，新的活动会继续记录。");
                  })
                }
              >
                确认清理历史
              </button>
              <button onClick={() => setClearHistory(null)}>取消清理</button>
            </section>
          )}
        </section>
      )}
      {tab === "tasks" && (
        <section>
          <h2>任务与已保存产物</h2>
          <p className="muted">
            这里显示真实任务状态。打开产物时会重新核对来源权限、记忆版本和内容审查结果。
          </p>
          {tasks.items.map((item) => (
            <article className="cq-task-row" key={item.task_id}>
              <div>
                <h3>{kindName(item.kind)}</h3>
                <p>
                  {stateName(item.state)} · 第{item.revision}版
                </p>
                <small>{day(item.created_at)}</small>
              </div>
              <button
                disabled={
                  !onOpenTask &&
                  (!renderTask ||
                    !["qa", "summary", "compare", "stance"].includes(item.kind))
                }
                onClick={() => {
                  if (onOpenTask) onOpenTask(item.task_id);
                  else
                    setLaunch({
                      selected: [],
                      kind: item.kind as TaskLaunch["kind"],
                      catalog: "application_papers",
                      resumeTaskId: item.task_id,
                    });
                }}
              >
                打开任务与产物
              </button>
            </article>
          ))}
          {!loading && !error && !tasks.items.length && (
            <p className="empty">
              还没有研究任务。可以从选中文献提问，或先建立研究档案。
            </p>
          )}
          {tasks.next_cursor && (
            <button
              disabled={busy}
              onClick={() =>
                action(async () => {
                  const next = await get<Page<TaskSummary>>(
                    `${WORKSPACE}/tasks?limit=20&cursor=${encodeURIComponent(tasks.next_cursor!)}`,
                  );
                  setTasks((current) => ({
                    items: [...current.items, ...next.items],
                    next_cursor: next.next_cursor,
                  }));
                })
              }
            >
              加载更多任务
            </button>
          )}
        </section>
      )}
      {launch && renderTask && (
        <section className="cq-task-launch" aria-label="文献AI任务">
          <div className="cq-section-heading">
            <h2>
              {launch.resumeTaskId ? "继续已有任务" : kindName(launch.kind)}
            </h2>
            <button onClick={() => setLaunch(null)}>收起任务区域</button>
          </div>
          <div
            key={`${launch.resumeTaskId || "new"}:${launch.kind}:${launch.selected.map((item) => `${item.kind}:${item.id}`).join("|")}`}
          >
            {renderTask(launch)}
          </div>
        </section>
      )}
    </section>
  );
}
