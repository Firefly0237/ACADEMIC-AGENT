import { useEffect, useRef, useState } from "react";
import type { Session } from "../../api";
import { workspaceError, workspaceRequest } from "../workstation/client";

export type SkillPin = {
  skill_id: string;
  version_id: string;
  content_sha256: string;
  head_revision: number;
};
type Skill = {
  pin: SkillPin;
  name: string;
  description: string;
  deleted: boolean;
};
type Memory = {
  memory_id: string;
  revision: number;
  kind: "preference" | "decision" | "constraint";
  value: string;
};
const SAMPLE =
  "---\nname: concise-research\ndescription: 简洁呈现研究问题、证据与待确认条件\n---\n使用清晰的学术语言。区分已知事实、拟议方案与证据不足，并保留可核验引用。\n";

export function ContextSettings({
  identity,
  selected,
  onSelect,
  selectionDisabled = false,
}: {
  identity: Session;
  selected: SkillPin[];
  onSelect: (pins: SkillPin[]) => void;
  selectionDisabled?: boolean;
}) {
  const [skills, setSkills] = useState<Skill[]>([]),
    [memories, setMemories] = useState<Memory[]>([]);
  const [next, setNext] = useState<string | null>(null),
    [error, setError] = useState(""),
    [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false),
    [markdown, setMarkdown] = useState(SAMPLE),
    [editing, setEditing] = useState<Skill | null>(null);
  const [memory, setMemory] = useState(""),
    [memoryKind, setMemoryKind] = useState<Memory["kind"]>("preference"),
    [editMemory, setEditMemory] = useState<Memory | null>(null);
  const newSkillId = useRef(crypto.randomUUID()),
    newMemoryId = useRef(crypto.randomUUID());
  const pending = useRef<{ signature: string; id: string } | null>(null);
  const base = `/api/agent/projects/${identity.project_id}`;
  const request = <T,>(path: string, body?: unknown, method?: string) =>
    workspaceRequest<T>(identity, path, { body, method });
  function operation(value: unknown) {
    const signature = JSON.stringify(value);
    if (pending.current?.signature !== signature)
      pending.current = { signature, id: crypto.randomUUID() };
    return pending.current.id;
  }
  async function load(after?: string) {
    const page = await request<{ skills: Skill[]; next_name: string | null }>(
      `${base}/skills?limit=20&include_deleted=true${after ? `&after_name=${encodeURIComponent(after)}` : ""}`,
    );
    setSkills((items) => (after ? [...items, ...page.skills] : page.skills));
    setNext(page.next_name);
    if (!after)
      setMemories(
        (await request<{ memories: Memory[] }>(`${base}/memories`)).memories,
      );
  }
  useEffect(() => {
    let active = true;
    load().catch((e) => {
      if (active) setError(workspaceError(e));
    });
    return () => {
      active = false;
    };
  }, [identity.owner_id, identity.project_id]);
  async function action(work: () => Promise<void>) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await work();
    } catch (e) {
      setError(workspaceError(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="context-settings">
      <p className="muted">
        技能最多选3项，仅提供指令；不会增加工具权限。独立审稿不使用生成者的技能正文。项目记忆会在后续调用前复核。
      </p>
      {!!error && (
        <p className="notice" role="alert">
          {error}
        </p>
      )}
      {!!notice && <p role="status">{notice}</p>}
      <h4>本次任务的技能</h4>
      {skills.map((skill) => (
        <div className="card" key={skill.pin.skill_id}>
          <label>
            <input
              type="checkbox"
              checked={selected.some(
                (pin) =>
                  pin.skill_id === skill.pin.skill_id &&
                  pin.version_id === skill.pin.version_id &&
                  pin.head_revision === skill.pin.head_revision,
              )}
              disabled={
                busy ||
                selectionDisabled ||
                skill.deleted ||
                (!selected.some((pin) => pin.skill_id === skill.pin.skill_id) &&
                  selected.length >= 3)
              }
              onChange={(event) =>
                onSelect(
                  event.target.checked
                    ? [
                        ...selected.filter(
                          (pin) => pin.skill_id !== skill.pin.skill_id,
                        ),
                        skill.pin,
                      ]
                    : selected.filter(
                        (pin) => pin.skill_id !== skill.pin.skill_id,
                      ),
                )
              }
            />
            {skill.name} · v{skill.pin.head_revision}
            {skill.deleted ? " · 已停用" : ""}
          </label>
          <p>{skill.description}</p>
          <div className="toolbar">
            <button
              disabled={busy || skill.deleted}
              onClick={() =>
                action(async () => {
                  const value = await request<{ markdown: string }>(
                    `${base}/skills/${skill.pin.skill_id}/versions/${skill.pin.version_id}`,
                  );
                  setEditing(skill);
                  setMarkdown(value.markdown);
                })
              }
            >
              编辑当前版本
            </button>
            <button
              disabled={busy}
              onClick={() =>
                action(async () => {
                  const body = {
                    expected_revision: skill.pin.head_revision,
                    deleted: !skill.deleted,
                  };
                  await request(
                    `${base}/skills/${skill.pin.skill_id}/lifecycle`,
                    {
                      ...body,
                      operation_id: operation({
                        skill: skill.pin.skill_id,
                        ...body,
                      }),
                    },
                  );
                  await load();
                  setNotice(
                    skill.deleted
                      ? "技能已恢复，请显式选择当前版本。"
                      : "技能已停用；引用旧版本的结果会标为不可用。",
                  );
                })
              }
            >
              {skill.deleted ? "恢复" : "停用"}
            </button>
          </div>
        </div>
      ))}
      {selected.some(
        (pin) =>
          !skills.some(
            (skill) =>
              !skill.deleted &&
              skill.pin.skill_id === pin.skill_id &&
              skill.pin.version_id === pin.version_id &&
              skill.pin.head_revision === pin.head_revision &&
              skill.pin.content_sha256 === pin.content_sha256,
          ),
      ) && (
        <p className="notice">
          所选技能版本未在当前列表中，可能已更新或停用。请加载后续页或清除选择，再选当前版本。
        </p>
      )}
      <div className="toolbar">
        <button
          disabled={busy || selectionDisabled}
          onClick={() => onSelect([])}
        >
          清除技能选择
        </button>
        {next && (
          <button disabled={busy} onClick={() => action(() => load(next))}>
            更多技能
          </button>
        )}
        <button disabled={busy} onClick={() => action(() => load())}>
          刷新技能与记忆
        </button>
      </div>
      <details>
        <summary>{editing ? `修改 ${editing.name}` : "新增 SKILL.md"}</summary>
        <p className="muted">
          支持 UTF-8 的指令型
          SKILL.md，合计最多16KiB；不运行脚本、路径或外部资源。
        </p>
        <label>
          导入技能文件
          <input
            type="file"
            accept=".md,text/markdown,text/plain"
            disabled={busy}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (!file) return;
              action(async () => {
                if (file.size > 16384) throw new Error("技能文件超过16KiB");
                setEditing(null);
                setMarkdown(await file.text());
              });
            }}
          />
        </label>
        <label>
          技能内容
          <textarea
            value={markdown}
            maxLength={16384}
            onChange={(event) => setMarkdown(event.target.value)}
            rows={9}
          />
        </label>
        <button
          disabled={busy || !markdown.trim()}
          onClick={() =>
            action(async () => {
              if (new TextEncoder().encode(markdown).length > 16384)
                throw new Error("技能UTF-8内容超过16KiB");
              const body = {
                expected_revision: editing?.pin.head_revision || 0,
                markdown,
              };
              const id = editing?.pin.skill_id || newSkillId.current;
              await request(
                `${base}/skills/${id}`,
                { ...body, operation_id: operation({ id, ...body }) },
                "PUT",
              );
              newSkillId.current = crypto.randomUUID();
              setEditing(null);
              setMarkdown(SAMPLE);
              await load();
              setNotice("技能已保存为新版本。任务需要显式选择它。");
            })
          }
        >
          保存新版本
        </button>
        {editing && (
          <button
            onClick={() => {
              setEditing(null);
              setMarkdown(SAMPLE);
            }}
          >
            取消编辑
          </button>
        )}
      </details>
      <h4>项目记忆</h4>
      <p className="muted">
        偏好是可选上下文；已确认的决定和约束必须保留。修改或删除后，使用旧版本的结果会失效。
      </p>
      {memories.map((item) => (
        <div className="card" key={item.memory_id}>
          <p>
            {item.kind === "preference"
              ? "偏好"
              : item.kind === "decision"
                ? "已确认决定"
                : "已确认约束"}{" "}
            · v{item.revision}
          </p>
          <p className="preserve">{item.value}</p>
          <div className="toolbar">
            <button
              disabled={busy}
              onClick={() => {
                setEditMemory(item);
                setMemoryKind(item.kind);
                setMemory(item.value);
              }}
            >
              修改
            </button>
            <button
              disabled={busy}
              onClick={() =>
                action(async () => {
                  const body = { expected_revision: item.revision };
                  await request(
                    `${base}/memories/${item.memory_id}`,
                    {
                      ...body,
                      operation_id: operation({
                        id: item.memory_id,
                        ...body,
                        deleted: true,
                      }),
                    },
                    "DELETE",
                  );
                  await load();
                  setNotice("记忆已删除，后续上下文不再装入旧版本。");
                })
              }
            >
              删除
            </button>
          </div>
        </div>
      ))}
      <label>
        记忆类型
        <select
          value={memoryKind}
          onChange={(event) =>
            setMemoryKind(event.target.value as Memory["kind"])
          }
        >
          <option value="preference">偏好</option>
          <option value="decision">已确认决定</option>
          <option value="constraint">已确认约束</option>
        </select>
      </label>
      <label>
        我希望保存的说明
        <textarea
          value={memory}
          maxLength={8000}
          onChange={(event) => setMemory(event.target.value)}
          rows={3}
        />
      </label>
      <button
        disabled={busy || !memory.trim()}
        onClick={() =>
          action(async () => {
            const id = editMemory?.memory_id || newMemoryId.current,
              body = {
                expected_revision: editMemory?.revision || 0,
                kind: memoryKind,
                value: memory,
                source_refs: [],
              };
            await request(
              `${base}/memories/${id}`,
              { ...body, operation_id: operation({ id, ...body }) },
              "PUT",
            );
            newMemoryId.current = crypto.randomUUID();
            setMemory("");
            setEditMemory(null);
            await load();
            setNotice(
              "记忆已提交。持久存储同步完成前，新模型调用会等待；可刷新查看当前版本。",
            );
          })
        }
      >
        {editMemory ? "保存修改" : "保存记忆"}
      </button>
    </div>
  );
}
