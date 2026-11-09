import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { APIError, paperPath } from "../../api";
import {
  WORKSPACE,
  workspaceError,
  workspaceRequest,
} from "../workstation/client";
import {
  BRIEF_FIELDS,
  FIELD_LABELS,
  type BriefEditorProps,
  type BriefField,
  type BriefSnapshot,
  type PlanningInput,
  type MaterialBibliography,
  type SchoolProfiles,
} from "./types";
import "./research.css";

const Markdown = lazy(() =>
  import("../../Markdown").then((module) => ({ default: module.Markdown })),
);
type Edit = { value: string; intent: "set" | "revise" | "unknown" | "clear" };
type BibliographyEdit = {
  file_id: string; file_sha256: string; representation_id: string; title: string;
  authors: string; year: string; doi: string; source_url: string; confirmed: boolean;
};
type BibliographySource = {
  source: { file_id: string; file_sha256: string; representation_id: string; material_name: string };
  bibliography: MaterialBibliography | null;
  source_matches_confirmation: boolean;
};
const id = () => crypto.randomUUID();
const LEGACY_SCHOOL_PROFILE = "sjtu-scs-2024-master-cn-capacity-demo/3";

export function BriefEditor({
  identity,
  projectId,
  initial,
  onSaved,
  onResearchHelp,
  onResearchStart,
  onResearchExtract,
}: BriefEditorProps) {
  const [snapshot, setSnapshot] = useState(initial);
  const [edits, setEdits] = useState<Partial<Record<BriefField, Edit>>>({});
  const [bibliographyEdits, setBibliographyEdits] = useState<Record<string, BibliographyEdit>>({});
  const [schoolProfiles, setSchoolProfiles] = useState<SchoolProfiles | null>(null);
  const [schoolProfileEdit, setSchoolProfileEdit] = useState<string>();
  const [schoolProfileError, setSchoolProfileError] = useState("");
  const [schoolProfileReload, setSchoolProfileReload] = useState(0);
  const [latest, setLatest] = useState<BriefSnapshot | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [custom, setCustom] = useState<Record<string, string>>({});
  const pending = useRef<{
    signature: string;
    body: Record<string, unknown>;
  } | null>(null);
  const owner = useRef(identity.owner_id);
  owner.current = identity.owner_id;
  const bibliographyScope = useRef("");
  bibliographyScope.current = `${identity.owner_id}:${projectId}:${initial?.brief.id || ""}`;
  useEffect(() => {
    setSnapshot(initial);
    setEdits({});
    setBibliographyEdits({});
    setSchoolProfileEdit(undefined);
    setLatest(null);
    setError("");
    setNotice("");
    pending.current = null;
  }, [initial?.brief.id, projectId, identity.owner_id]);

  useEffect(() => {
    const controller = new AbortController();
    setSchoolProfiles(null);
    setSchoolProfileError("");
    void workspaceRequest<SchoolProfiles>(identity, `${WORKSPACE}/school-profiles`, { signal: controller.signal })
      .then(value => {
        if (!controller.signal.aborted) setSchoolProfiles(value);
      })
      .catch(cause => {
        if (!controller.signal.aborted) setSchoolProfileError(workspaceError(cause));
      });
    return () => controller.abort();
  }, [identity.owner_id, identity.csrf, schoolProfileReload]);

  const dirty = Object.keys(edits).length > 0 || Object.keys(bibliographyEdits).length > 0 || schoolProfileEdit !== undefined;
  useEffect(() => {
    if (
      initial &&
      snapshot &&
      initial.brief.id === snapshot.brief.id &&
      initial.brief.revision > snapshot.brief.revision
    ) {
      if (dirty) setLatest(initial);
      else setSnapshot(initial);
    }
  }, [initial, snapshot, dirty]);
  const document = snapshot?.brief;
  const savedSchoolProfileId = document?.school_profile_id || (snapshot ? LEGACY_SCHOOL_PROFILE : schoolProfiles?.default_id);
  const selectedSchoolProfileId = schoolProfileEdit ?? savedSchoolProfileId;
  const selectedSchoolProfile = schoolProfiles?.profiles.find(profile => profile.id === selectedSchoolProfileId);
  const schoolProfileLabel = (profileId: string | null | undefined) =>
    schoolProfiles?.profiles.find(profile => profile.id === (profileId || LEGACY_SCHOOL_PROFILE))?.label || profileId || "原有学校格式";
  const missingBibliography = snapshot?.readiness.missing_material_bibliography || [];
  const experimentReadiness = snapshot?.readiness.experiment_readiness;
  const materialReferences = document?.selected_sources.filter(source => source.kind === "material_file"
    && !document.draft_source_ids?.includes(source.id)) || [];
  const questions = snapshot?.readiness.questions || [
    {
      field: "topic" as BriefField,
      question: "希望研究的具体方向或拟定题目是什么？",
    },
    {
      field: "purpose" as BriefField,
      question: "希望解决什么问题，适用范围是什么？",
    },
    {
      field: "research_questions" as BriefField,
      question: "最希望回答哪些可以检验的问题？",
    },
  ];
  const priority = questions.slice(0, 3).map((question) => question.field);
  const fields = priority.length
    ? priority
    : (["topic", "resources", "timeline"] as BriefField[]);
  const proposals =
    document?.proposals.filter((proposal) => proposal.state === "open") || [];
  const preview = (text: string) => (
    <Suspense fallback={<p className="cq-literal">{text}</p>}>
      <Markdown text={text} />
    </Suspense>
  );

  function operation(payload: Record<string, unknown>) {
    const signature = JSON.stringify(payload);
    if (pending.current?.signature === signature) return pending.current.body;
    const body = { ...payload, operation_id: id(), message_id: id() };
    pending.current = { signature, body };
    return body;
  }

  async function showError(cause: unknown) {
    const bibliographyMessages: Record<string, string> = {
      research_material_bibliography_required: "请先在上传参考文献区确认所列材料的题名、作者与年份，再开始研究计划。",
      material_bibliography_confirmation_required: "请完整填写题名、作者与有效年份，并明确勾选书目确认。",
      material_bibliography_source_changed: "材料解析版本已变化。当前填写仍保留，请重新读取这份材料并确认题录。",
      material_not_readable: "这份材料尚未解析就绪。请在材料详情完成解析后再确认题录。",
      material_not_in_research_reference_scope: "这份材料不在当前参考文献范围内；研究草稿不能自动作为文献引用。",
    };
    setError(cause instanceof APIError ? bibliographyMessages[cause.code] || workspaceError(cause) : workspaceError(cause));
    if (cause instanceof APIError && cause.status === 409 && document) {
      try {
        const server = await workspaceRequest<BriefSnapshot>(
          identity,
          `${WORKSPACE}/briefs/${document.id}`,
        );
        if (owner.current === identity.owner_id) setLatest(server);
      } catch {
        /* Original error and local edits remain visible. */
      }
    }
  }

  function accept(value: BriefSnapshot, clear = true) {
    if (owner.current !== identity.owner_id) return;
    setSnapshot(value);
    setLatest(null);
    setError("");
    pending.current = null;
    if (clear) { setEdits({}); setBibliographyEdits({}); setSchoolProfileEdit(undefined); }
    onSaved?.(value);
  }

  async function save(): Promise<BriefSnapshot> {
    const patches = Object.entries(edits).map(([field, edit]) => ({
      field,
      intent: edit!.intent,
      ...(edit!.intent === "clear" ? {} : { value: edit!.value }),
    }));
    const material_bibliography: Record<string, MaterialBibliography> = {};
    for (const [key, entry] of Object.entries(bibliographyEdits)) {
      const authors = entry.authors.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
      const year = Number(entry.year);
      if (!entry.title.trim() || !authors.length || authors.length > 100 || authors.some(author => author.length > 500) || !Number.isSafeInteger(year)
        || year < 1000 || year > 2200 || !entry.confirmed)
        throw new APIError("material_bibliography_confirmation_required", 400);
      material_bibliography[key] = { file_id: entry.file_id, file_sha256: entry.file_sha256,
        representation_id: entry.representation_id, title: entry.title.trim(), authors, year, confirmed: true,
        provenance: "user_confirmed", ...(entry.doi.trim() ? { doi: entry.doi.trim() } : {}),
        ...(entry.source_url.trim() ? { source_url: entry.source_url.trim() } : {}) };
    }
    if (snapshot && !patches.length && !Object.keys(material_bibliography).length && schoolProfileEdit === undefined) return snapshot;
    const payload = snapshot
      ? { expected_revision: document!.revision, patches,
          ...(Object.keys(material_bibliography).length ? { material_bibliography } : {}),
          ...(schoolProfileEdit !== undefined ? { school_profile_id: schoolProfileEdit } : {}) }
      : { project_id: projectId, patches, selected_sources: [],
          ...(schoolProfileEdit !== undefined ? { school_profile_id: schoolProfileEdit } : {}) };
    const value = await workspaceRequest<BriefSnapshot>(
      identity,
      snapshot ? `${WORKSPACE}/briefs/${document!.id}` : `${WORKSPACE}/briefs`,
      { method: snapshot ? "PATCH" : "POST", body: operation(payload) },
    );
    accept(value);
    return value;
  }

  async function editBibliography(fileId: string) {
    if (!document) return;
    const scope = bibliographyScope.current;
    const value = await workspaceRequest<BibliographySource>(identity,
      `${WORKSPACE}/briefs/${document.id}/material-bibliography/${fileId}`);
    if (scope !== bibliographyScope.current) return;
    const metadata = value.bibliography;
    setBibliographyEdits(current => ({ ...current, [fileId]: { file_id: value.source.file_id,
      file_sha256: value.source.file_sha256, representation_id: value.source.representation_id,
      title: metadata?.title || "", authors: metadata?.authors.join("\n") || "", year: metadata ? String(metadata.year) : "",
      doi: metadata?.doi || "", source_url: metadata?.source_url || "", confirmed: false } }));
    setNotice(`正在确认${value.source.material_name}的题录。文件名和内嵌作者信息不会自动当作论文题录。`);
  }

  function editBibliographyField(fileId: string, field: keyof BibliographyEdit, value: string | boolean) {
    setBibliographyEdits(current => ({ ...current, [fileId]: { ...current[fileId], [field]: value,
      ...(field === "confirmed" ? {} : { confirmed: false }) } }));
  }

  async function action(work: () => Promise<void>) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await work();
    } catch (cause) {
      await showError(cause);
    } finally {
      setBusy(false);
    }
  }

  async function help() {
    if (!onResearchHelp) return;
    const current = await save();
    const result = await onResearchHelp({
      brief: current.brief,
      fields: current.readiness.missing_fields.slice(0, 3),
    });
    if (!result?.length) {
      setNotice(
        "请在助手入口继续补充或查看实际任务状态；当前字段不会自动填入未经确认的内容。",
      );
      return;
    }
    const patches = result.map((item) => ({
      field: item.field,
      value: item.value,
      intent: "suggest",
    }));
    const saved = await workspaceRequest<BriefSnapshot>(
      identity,
      `${WORKSPACE}/briefs/${current.brief.id}`,
      {
        method: "PATCH",
        body: operation({ expected_revision: current.brief.revision, patches }),
      },
    );
    accept(saved);
    setNotice("助手建议已单独列出。请检查后采用、修改或保留原说明。");
  }

  async function resolve(
    proposalId: string,
    decision: "keep_current" | "use_candidate" | "custom",
  ) {
    const value = await workspaceRequest<BriefSnapshot>(
      identity,
      `${WORKSPACE}/briefs/${document!.id}/resolutions`,
      {
        body: operation({
          expected_revision: document!.revision,
          proposal_id: proposalId,
          decision,
          ...(decision === "custom" ? { value: custom[proposalId] } : {}),
        }),
      },
    );
    accept(value);
    setNotice("这项说明已按你的选择保存，相关检查将使用新版本。");
  }

  function editField(field: BriefField, value: string) {
    setEdits((current) => ({
      ...current,
      [field]: {
        value,
        intent: !value.trim()
          ? "clear"
          : document?.facts[field]
            ? "revise"
            : "set",
      },
    }));
  }

  function fieldEditor(field: BriefField) {
    const fact = document?.facts[field];
    const edit = edits[field];
    const value = edit?.value ?? fact?.value ?? "";
    return (
      <div className="cq-brief-field" key={field}>
        <label htmlFor={`brief-${field}`}>
          {FIELD_LABELS[field]}{" "}
          <span className="cq-field-state">
            {edit
              ? "未保存"
              : fact?.status === "confirmed"
                ? "你已确认"
                : fact
                  ? "明确未确定"
                  : "待补充"}
          </span>
        </label>
        <textarea
          id={`brief-${field}`}
          value={value}
          maxLength={8000}
          disabled={busy}
          onChange={(event) => editField(field, event.target.value)}
          placeholder={
            questions.find((question) => question.field === field)?.question ||
            `补充${FIELD_LABELS[field]}，也可以先明确未知。`
          }
        />
        <div className="cq-inline-actions">
          <button
            type="button"
            disabled={busy}
            onClick={() =>
              setEdits((current) => ({
                ...current,
                [field]: {
                  intent: "unknown",
                  value: "尚未确定，希望研究助手提供建议。",
                },
              }))
            }
          >
            暂未确定
          </button>
          {value && (
            <details>
              <summary>查看正文与公式预览</summary>
              {preview(value)}
            </details>
          )}
        </div>
      </div>
    );
  }

  return (
    <section className="cq-brief" aria-labelledby="cq-brief-title">
      <div className="cq-section-heading">
        <div>
          <p className="eyebrow">RESEARCH BRIEF</p>
          <h2 id="cq-brief-title">把研究方向逐步补成可执行计划</h2>
        </div>
        <span className="tag">
          {snapshot ? `已保存 · 第 ${document!.revision} 版` : "新研究档案"}
        </span>
      </div>
      <p className="muted">
        先回答当前最关键的三项即可。方法、基线等尚未确定时可以先保存或请助手提出建议，建议经你确认后才成为研究要求。
      </p>
      <section className="cq-brief-sources" aria-label="学校与年级格式">
        <label htmlFor="brief-school-profile">学校与年级格式</label>
        <select id="brief-school-profile" value={selectedSchoolProfileId || ""}
          disabled={busy || !!latest || !schoolProfiles}
          onChange={event => setSchoolProfileEdit(event.target.value === savedSchoolProfileId ? undefined : event.target.value)}>
          {!schoolProfiles && <option value={selectedSchoolProfileId || ""}>{schoolProfileError ? "格式选项未能读取" : "正在读取格式选项…"}</option>}
          {schoolProfiles?.profiles.map(profile => <option key={profile.id} value={profile.id}>{profile.label}</option>)}
        </select>
        {schoolProfileError && <p role="alert">格式选项暂未读取成功。<button type="button" onClick={() => setSchoolProfileReload(value => value + 1)}>重新读取</button></p>}
        {schoolProfileEdit !== undefined && <p className="muted">格式选择尚未保存，将与本次研究信息一起保存为新版本。</p>}
        {snapshot && !document?.school_profile_id && schoolProfileEdit === undefined && <p className="muted">此档案沿用原有的 2024 级格式。</p>}
        {selectedSchoolProfile && <>
          <p><a href={selectedSchoolProfile.source_url} target="_blank" rel="noreferrer">官方开题要求 ↗</a>
            {selectedSchoolProfile.template_source_url && <>{" · "}<a href={selectedSchoolProfile.template_source_url} target="_blank" rel="noreferrer">原始模板 ↗</a></>}</p>
          {!!selectedSchoolProfile.section_titles.length && <p className="muted">正文内容：{selectedSchoolProfile.section_titles.join("；")}。</p>}
          <p className="muted">文献综述不少于 {selectedSchoolProfile.literature_minimum_han} 汉字（<a href={selectedSchoolProfile.literature_requirement_source_url || selectedSchoolProfile.source_url} target="_blank" rel="noreferrer">要求出处</a>）。
            系统写作篇幅为 {selectedSchoolProfile.demonstration_minimum_han}–{selectedSchoolProfile.demonstration_maximum_han} 汉字。</p>
        </>}
      </section>
      {!!document?.selected_sources.length && (
        <section className="cq-brief-sources">
          <h3>已关联的研究资料 · {document.selected_sources.length} 份</h3>
          <p className="muted">
            保存来源引用不等于正文已就绪；执行前仍核对解析、授权和证据范围。可在文献库选文后添加到当前档案。
          </p>
          <ul>
            {document.selected_sources.map((source, index) => (
              <li key={`${source.kind}:${source.id}`}>
                <Link
                  to={
                    source.kind === "material_file"
                      ? `/materials/${source.id}`
                      : paperPath({ kind: source.kind, id: source.id })
                  }
                >
                  查看第 {index + 1} 份来源 ·{" "}
                  {source.kind === "material_file"
                    ? document.draft_source_ids?.includes(source.id) ? "研究草稿（不作学术引文）" : "研究材料"
                    : source.kind === "metadata_projection"
                      ? "题录文献"
                      : "正文文献"}
                </Link>
                <button
                  disabled={busy || dirty || !!latest}
                  onClick={() =>
                    action(async () => {
                      const selected_sources = document.selected_sources.filter(
                        (item) =>
                          item.kind !== source.kind || item.id !== source.id,
                      );
                      const value = await workspaceRequest<BriefSnapshot>(
                        identity,
                        `${WORKSPACE}/briefs/${document.id}`,
                        {
                          method: "PATCH",
                          body: operation({
                            expected_revision: document.revision,
                            selected_sources,
                            draft_source_ids: (document.draft_source_ids || []).filter(id => id !== source.id),
                            patches: [],
                          }),
                        },
                      );
                      accept(value);
                      setNotice(
                        "来源范围已更新，后续产物将使用新版本重新核对。",
                      );
                    })
                  }
                >
                  移除这份来源
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
      {!!materialReferences.length && <section className="cq-brief-sources" aria-label="上传参考文献题录确认">
        <h3>上传参考文献题录</h3>
        <p className="muted">核对题名、作者与年份后集中保存。这里保存用户确认的信息；论文身份、版权与原文证据分别核验。</p>
        {!!missingBibliography.length && <p className="notice" role="status">
          还有 {missingBibliography.length} 份上传参考文献待确认题录，完成后可开始研究计划。
        </p>}
        {materialReferences.map((source, index) => {
          const entry = bibliographyEdits[source.id];
          const saved = document?.material_bibliography?.[source.id];
          return <details key={source.id} open={!!entry}>
            <summary>第 {index + 1} 份上传参考文献 · {saved?.title || source.id.slice(0, 8)} · {saved ? "用户已确认" : "待确认"}</summary>
            <p><Link to={`/materials/${source.id}`}>打开材料核对原文 ↗</Link></p>
            {!entry ? <button disabled={busy || !!latest} onClick={() => action(() => editBibliography(source.id))}>
              {saved ? "修改并重新确认题录" : "填写并确认题录"}
            </button> : <div className="cq-brief-grid">
              <label>论文题名<input maxLength={500} value={entry.title} disabled={busy}
                onChange={event => editBibliographyField(source.id, "title", event.target.value)} /></label>
              <label>作者（每行一位，按原文顺序）<textarea rows={3} maxLength={20000} value={entry.authors} disabled={busy}
                onChange={event => editBibliographyField(source.id, "authors", event.target.value)} /></label>
              <label>发表年份<input type="number" min={1000} max={2200} value={entry.year} disabled={busy}
                onChange={event => editBibliographyField(source.id, "year", event.target.value)} /></label>
              <label>DOI（可选）<input maxLength={512} value={entry.doi} disabled={busy}
                onChange={event => editBibliographyField(source.id, "doi", event.target.value)} /></label>
              <label>原文链接（可选）<input type="url" maxLength={2000} value={entry.source_url} disabled={busy}
                onChange={event => editBibliographyField(source.id, "source_url", event.target.value)} /></label>
              <label><input type="checkbox" checked={entry.confirmed} disabled={busy}
                onChange={event => editBibliographyField(source.id, "confirmed", event.target.checked)} />
                我已核对这份材料的题名、作者与年份</label>
              <button disabled={busy} onClick={() => setBibliographyEdits(current => {
                const next = { ...current }; delete next[source.id]; return next;
              })}>取消这份题录编辑</button>
            </div>}
          </details>;
        })}
      </section>}
      {error && (
        <div className="notice" role="alert">
          <p>{error}</p>
          {dirty && <p>本页草稿仍保留。离开或刷新前可复制备份。</p>}
        </div>
      )}
      {notice && (
        <p className="cq-success" role="status">
          {notice}
        </p>
      )}
      {latest && (
        <section className="cq-version-conflict" aria-label="版本冲突处理">
          <h3>其他位置已有新版本，你的草稿没有丢失</h3>
          <p>请比较下列修改。继续编辑不会直接覆盖服务器，你仍需再点保存。</p>
          {!!Object.keys(bibliographyEdits).length && <p>本页尚有上传文献题录未保存；确认最新版本后，仍保留这些填写供你核对。</p>}
          {schoolProfileEdit !== undefined && <p>服务器格式：{schoolProfileLabel(latest.brief.school_profile_id)}；本页选择：{schoolProfileLabel(schoolProfileEdit)}。格式选择仍保留，请核对后保存。</p>}
          <div className="cq-table-scroll">
            <table>
              <thead>
                <tr>
                  <th>字段</th>
                  <th>服务器当前说明</th>
                  <th>本页未保存内容</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(edits).map(([field, edit]) => (
                  <tr key={field}>
                    <th>{FIELD_LABELS[field as BriefField]}</th>
                    <td className="cq-literal">
                      {latest.brief.facts[field as BriefField]?.value ||
                        "未填写"}
                    </td>
                    <td className="cq-literal">{edit!.value || "准备清除"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <button
            type="button"
            onClick={() => {
              setSnapshot(latest);
              setLatest(null);
              setError("");
              pending.current = null;
            }}
          >
            以最新版本继续编辑，保留我的草稿
          </button>
        </section>
      )}
      <div className="cq-brief-grid">{fields.map(fieldEditor)}</div>
      <details className="cq-more-fields">
        <summary>查看和编辑全部研究信息</summary>
        <div className="cq-brief-grid">
          {BRIEF_FIELDS.filter((field) => !fields.includes(field)).map(
            fieldEditor,
          )}
        </div>
      </details>
      <div className="cq-inline-actions cq-save-row">
        <button
          className="primary"
          type="button"
          disabled={busy || !!latest || (!!snapshot && !dirty)}
          onClick={() =>
            action(async () => {
              await save();
              setNotice("研究信息已保存。可以补充下一组问题或稍后继续。");
            })
          }
        >
          {busy ? "正在处理…" : "保存当前信息"}
        </button>
        <button
          type="button"
          disabled={
            busy ||
            !onResearchHelp ||
            !!latest ||
            !(edits.topic?.value || document?.facts.topic?.value)
          }
          onClick={() => action(help)}
          title={
            !onResearchHelp
              ? "研究建议助手尚未接线"
              : "建议不会自动覆盖你已确认的内容"
          }
        >
          请研究助手提出建议
        </button>
        <button type="button" disabled={busy || !onResearchExtract || !!latest || !document?.draft_source_ids?.length}
          onClick={() => action(async () => {
            const current = await save();
            await onResearchExtract?.({ brief: current.brief, fields: current.readiness.missing_fields.slice(0, 4) });
            setNotice("已发起草稿提取。候选内容与覆盖情况会显示在任务区，选择保存后还需明确确认。");
          })}>
          从关联草稿提取信息
        </button>
        {dirty && (
          <button
            type="button"
            onClick={() =>
              action(async () => {
                await navigator.clipboard.writeText(
                  Object.entries(edits)
                    .map(
                      ([field, edit]) =>
                        `${FIELD_LABELS[field as BriefField]}\n${edit!.value}`,
                    )
                    .concat(Object.values(bibliographyEdits).map(entry =>
                      `上传文献题录\n${entry.title}\n${entry.authors}\n${entry.year}\n${entry.doi}\n${entry.source_url}`))
                    .concat(schoolProfileEdit !== undefined ? [`学校与年级格式\n${schoolProfileLabel(schoolProfileEdit)}\n${schoolProfileEdit}`] : [])
                    .join("\n\n"),
                );
                setNotice("未保存内容已复制。请保管好私人研究信息。");
              })
            }
          >
            复制未保存内容
          </button>
        )}
      </div>
      {!onResearchHelp && (
        <p className="muted">
          研究建议助手尚未启用；当前支持保存、补全与冲突管理。
        </p>
      )}
      {proposals.length > 0 && (
        <section className="cq-proposals" aria-label="待确认建议与冲突">
          <h3>待确认的建议与不同说明</h3>
          {dirty && (
            <p className="muted">请先保存字段修改，再处理建议或冲突。</p>
          )}
          {proposals.map((proposal) => (
            <article className="cq-proposal" key={proposal.id}>
              <h4>
                {FIELD_LABELS[proposal.field]} ·{" "}
                {proposal.kind === "conflict"
                  ? "信息冲突"
                  : "助手建议，尚未确认"}
              </h4>
              {document?.facts[proposal.field] && (
                <div>
                  <strong>当前已保存说明</strong>
                  {preview(document.facts[proposal.field]!.value)}
                </div>
              )}
              <div>
                <strong>
                  {proposal.kind === "conflict"
                    ? "新提交的不同说明"
                    : "待核查的候选建议"}
                </strong>
                {preview(proposal.candidate.value)}
              </div>
              <div className="cq-inline-actions">
                <button
                  disabled={busy || dirty || !!latest}
                  onClick={() =>
                    action(() => resolve(proposal.id, "keep_current"))
                  }
                >
                  保留原说明
                </button>
                <button
                  disabled={busy || dirty || !!latest}
                  onClick={() =>
                    action(() => resolve(proposal.id, "use_candidate"))
                  }
                >
                  采用这条说明
                </button>
              </div>
              <details>
                <summary>写下自己的决定</summary>
                <label>
                  最终说明
                  <textarea
                    value={custom[proposal.id] || ""}
                    maxLength={8000}
                    onChange={(event) =>
                      setCustom((current) => ({
                        ...current,
                        [proposal.id]: event.target.value,
                      }))
                    }
                  />
                </label>
                <button
                  disabled={
                    busy || dirty || !!latest || !custom[proposal.id]?.trim()
                  }
                  onClick={() => action(() => resolve(proposal.id, "custom"))}
                >
                  保存我的决定
                </button>
              </details>
            </article>
          ))}
        </section>
      )}
      <section className="cq-planning-gate">
        <h3>下一步：研究与实验设计</h3>
        <p>
          {snapshot?.readiness.state === "ready_for_planning"
            ? "主要研究信息已补齐；执行前仍会核对当前选源、权限、资源和证据。"
            : `仍有 ${snapshot?.readiness.missing_fields.length ?? 12} 项研究信息待补充${missingBibliography.length ? `，另有 ${missingBibliography.length} 份上传参考文献待确认题录` : ""}${proposals.some((proposal) => proposal.kind === "conflict") ? "，并有信息冲突待解决" : ""}。可以保存后分次继续。`}
        </p>
        <p className="muted">
          研究助手只会使用已保存要求和获准的资料；拟议实验不会被写成已经完成的结果。
        </p>
        <button
          type="button"
          className="primary"
          disabled={
            busy ||
            dirty ||
            !!latest ||
            !onResearchStart ||
            missingBibliography.length > 0 ||
            snapshot?.readiness.state !== "ready_for_planning"
          }
          onClick={() =>
            action(async () => {
              const input = await workspaceRequest<PlanningInput>(
                identity,
                `${WORKSPACE}/briefs/${document!.id}/planning-input?expected_revision=${document!.revision}`,
              );
              await onResearchStart?.(input, "plan");
              setNotice(
                "当前版本已送至任务入口，排队或执行情况请查看实际任务状态。",
              );
            })
          }
        >
          生成研究计划书
        </button>
        <h4>辅助实验设计</h4>
        <p>
          {experimentReadiness?.state === "ready_for_experiment"
            ? "实验设计所需的基本信息已保存，可以开始规划方法、对照、指标和评估步骤。"
            : experimentReadiness
              ? `还有 ${experimentReadiness.missing_fields.length} 项实验背景待补充${experimentReadiness.open_conflicts.length ? "，并有信息冲突待解决" : ""}。可在上方查看和编辑全部研究信息后保存。`
              : "保存研究档案后，将核对实验设计需要的信息。"}
        </p>
        {!!experimentReadiness?.questions.length && <ul>
          {experimentReadiness.questions.map(({ field, question }) => <li key={field}>
            {FIELD_LABELS[field]}：{question}
          </li>)}
        </ul>}
        <p className="muted">尚未确定的条件可以明确标为“暂未确定”；助手会保留这些未知并提出待验证的设计，不会写成已完成的实验结果。</p>
        <button
          type="button"
          disabled={busy || dirty || !!latest || !onResearchStart ||
            experimentReadiness?.state !== "ready_for_experiment"}
          onClick={() => action(async () => {
            const input = await workspaceRequest<PlanningInput>(
              identity,
              `${WORKSPACE}/briefs/${document!.id}/planning-input?expected_revision=${document!.revision}&action=experiment`,
            );
            await onResearchStart?.(input, "experiment");
            setNotice("当前版本已送至实验设计任务，执行过程和补充问题请查看实际任务状态。");
          })}
        >
          辅助实验设计
        </button>
        {!onResearchStart && (
          <p className="muted">多智能体研究任务尚未启用。</p>
        )}
      </section>
    </section>
  );
}
