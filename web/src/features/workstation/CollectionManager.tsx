import { useRef, useState } from "react";
import type { Session } from "../../api";
import { WORKSPACE, workspaceError, workspaceRequest } from "./client";
import type { Collection, Page } from "./types";

export function CollectionManager({
  identity,
  data,
  onChange,
  onNotice,
}: {
  identity: Session;
  data: Page<Collection>;
  onChange: (value: Page<Collection>) => void;
  onNotice: (value: string) => void;
}) {
  const [name, setName] = useState("");
  const [parent, setParent] = useState("");
  const [editing, setEditing] = useState<Collection | null>(null);
  const [removing, setRemoving] = useState<Collection | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const intent = useRef<{ signature: string; id: string } | null>(null);
  function operation(body: unknown) {
    const signature = JSON.stringify(body);
    if (intent.current?.signature === signature) return intent.current.id;
    const id = crypto.randomUUID();
    intent.current = { signature, id };
    return id;
  }
  async function action(work: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await work();
    } catch (cause) {
      setError(workspaceError(cause));
    } finally {
      setBusy(false);
    }
  }
  function choose(item: Collection | null) {
    setEditing(item);
    setName(item?.name || "");
    setParent(item?.parent_id || "");
    setError("");
  }
  return (
    <details className="cq-collection-create">
      <summary>管理文献集合</summary>
      {error && (
        <p className="notice" role="alert">
          {error}当前名称输入仍保留。
        </p>
      )}
      <form
        onSubmit={(event) => {
          event.preventDefault();
          action(async () => {
            const body = { name: name.trim(), parent_id: parent || null };
            const item = await workspaceRequest<Collection>(
              identity,
              editing
                ? `/api/v2/collections/${editing.collection_id}`
                : "/api/v2/collections",
              {
                method: editing ? "PATCH" : "POST",
                body: editing
                  ? { ...body, expected_revision: editing.revision }
                  : { ...body, client_operation_id: operation(body) },
              },
            );
            onChange({
              ...data,
              items: editing
                ? data.items.map((value) =>
                    value.collection_id === item.collection_id ? item : value,
                  )
                : [...data.items, item],
            });
            choose(null);
            intent.current = null;
            onNotice(
              editing
                ? "集合名称与层级已保存。"
                : "集合已创建，文献可以同时加入多个集合。",
            );
          });
        }}
      >
        <label>
          {editing ? "修改集合名称" : "新集合名称"}
          <input
            value={name}
            maxLength={500}
            disabled={busy}
            onChange={(event) => setName(event.target.value)}
          />
        </label>
        <label>
          父集合
          <select
            aria-label="父集合"
            value={parent}
            disabled={busy}
            onChange={(event) => setParent(event.target.value)}
          >
            <option value="">无，作为顶层集合</option>
            {data.items
              .filter((item) => item.collection_id !== editing?.collection_id)
              .map((item) => (
                <option value={item.collection_id} key={item.collection_id}>
                  {item.name}
                </option>
              ))}
          </select>
        </label>
        <button disabled={busy || !name.trim()}>
          {editing ? "保存集合修改" : "创建集合"}
        </button>
        {editing && (
          <button type="button" onClick={() => choose(null)}>
            取消集合编辑
          </button>
        )}
      </form>
      <ul className="cq-collection-list">
        {data.items.map((item) => (
          <li key={item.collection_id}>
            <span>
              {item.name}
              {item.parent_id
                ? ` · 属于${data.items.find((parent) => parent.collection_id === item.parent_id)?.name || "一个父集合"}`
                : ""}
            </span>
            <button disabled={busy} onClick={() => choose(item)}>
              编辑集合
            </button>
            <button disabled={busy} onClick={() => setRemoving(item)}>
              删除集合
            </button>
          </li>
        ))}
      </ul>
      {data.next_cursor && (
        <button
          disabled={busy}
          onClick={() =>
            action(async () => {
              const next = await workspaceRequest<Page<Collection>>(
                identity,
                `${WORKSPACE}/collections?limit=50&cursor=${encodeURIComponent(data.next_cursor!)}`,
              );
              onChange({
                items: [...data.items, ...next.items],
                next_cursor: next.next_cursor,
              });
            })
          }
        >
          读取更多集合
        </button>
      )}
      {removing && (
        <section
          className="cq-confirm"
          role="alertdialog"
          aria-label="确认删除集合"
        >
          <p>
            删除“{removing.name}
            ”集合？文献和笔记会保留，仅移除分组关系；有子集合时需先调整子集合。
          </p>
          <button
            disabled={busy}
            onClick={() =>
              action(async () => {
                await workspaceRequest(
                  identity,
                  `/api/v2/collections/${removing.collection_id}`,
                  { method: "DELETE" },
                );
                onChange({
                  ...data,
                  items: data.items.filter(
                    (item) => item.collection_id !== removing.collection_id,
                  ),
                });
                if (editing?.collection_id === removing.collection_id)
                  choose(null);
                setRemoving(null);
                onNotice("集合已删除，文献及笔记保留。");
              })
            }
          >
            确认删除集合
          </button>
          <button disabled={busy} onClick={() => setRemoving(null)}>
            保留集合
          </button>
        </section>
      )}
    </details>
  );
}
