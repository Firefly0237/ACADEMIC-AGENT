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
} from "./types";
import "./research.css";

const Markdown = lazy(() =>
  import("../../Markdown").then((module) => ({ default: module.Markdown })),
);
type Edit = { value: string; intent: "set" | "revise" | "unknown" | "clear" };
const id = () => crypto.randomUUID();

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
  useEffect(() => {
    setSnapshot(initial);
    setEdits({});
    setLatest(null);
    setError("");
    setNotice("");
    pending.current = null;
  }, [initial?.brief.id, projectId, identity.owner_id]);

  const dirty = Object.keys(edits).length > 0;
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
    setError(workspaceError(cause));
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
    if (clear) setEdits({});
    onSaved?.(value);
  }

  async function save(): Promise<BriefSnapshot> {
    const patches = Object.entries(edits).map(([field, edit]) => ({
      field,
      intent: edit!.intent,
      ...(edit!.intent === "clear" ? {} : { value: edit!.value }),
    }));
    if (snapshot && !patches.length) return snapshot;
    const payload = snapshot
      ? { expected_revision: document!.revision, patches }
      : { project_id: projectId, patches, selected_sources: [] };
    const value = await workspaceRequest<BriefSnapshot>(
      identity,
      snapshot ? `${WORKSPACE}/briefs/${document!.id}` : `${WORKSPACE}/briefs`,
      { method: snapshot ? "PATCH" : "POST", body: operation(payload) },
    );
    accept(value);
    return value;
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
            : `仍有 ${snapshot?.readiness.missing_fields.length ?? 12} 项待补充${proposals.some((proposal) => proposal.kind === "conflict") ? "，并有信息冲突待解决" : ""}。可以保存后分次继续。`}
        </p>
        <p className="muted">
          研究助手只会使用已保存要求和获准的资料；拟议实验不会被写成已经完成的结果。
        </p>
        <p className="muted">当前学校格式档按公开的上交计算机系 2024 硕士模板进行容量验证：正文 8000–12000 汉字、综述至少 5000 汉字。这是本项目的长文验证档，不代表学校对所有研究计划的统一字数要求；信息或证据不足会停止并报告。</p>
        <button
          type="button"
          className="primary"
          disabled={
            busy ||
            dirty ||
            !!latest ||
            !onResearchStart ||
            snapshot?.readiness.state !== "ready_for_planning"
          }
          onClick={() =>
            action(async () => {
              const input = await workspaceRequest<PlanningInput>(
                identity,
                `${WORKSPACE}/briefs/${document!.id}/planning-input?expected_revision=${document!.revision}`,
              );
              await onResearchStart?.(input);
              setNotice(
                "当前版本已送至任务入口，排队或执行情况请查看实际任务状态。",
              );
            })
          }
        >
          开始研究计划任务
        </button>
        {!onResearchStart && (
          <p className="muted">多智能体正式研究与学校计划书生成尚未启用。</p>
        )}
      </section>
    </section>
  );
}
