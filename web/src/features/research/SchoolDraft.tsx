import { useId, type ReactNode } from "react";
import { Markdown } from "../../Markdown";
import { ExportPanel } from "./ExportPanel";
import "./research.css";

export type SchoolDraftDocument = {
  title: string;
  sections: Array<{ section_id: string; blocks: Array<{ id: string; kind: string; text: string; source_aliases: string[] }> }>;
  section_specs: Array<{ id: string; title: string }>;
  source_aliases: Record<string, string[]>;
  reference_admission: { references: Array<{ id: string; title: string; author?: Array<{ literal?: string; family?: string; given?: string }>;
    issued?: { "date-parts": number[][] }; URL?: string }> };
};
const blockKinds: Record<string,string> = { evidence: "文献依据", proposal: "拟议设计", user_fact: "已确认要求", limitation: "限制与未知" };
export function SchoolDraft({ taskId, document, evidence, allowExport = true }: { taskId: string; document: SchoolDraftDocument; evidence: (ref: string) => ReactNode; allowExport?: boolean }) {
  const prefix = useId().replaceAll(":", "");
  return <section className="school-draft" aria-label="学校研究计划书草案">
    <h2>{document.title}</h2>
    <p className="notice">研究计划草稿 · 实验尚未执行 · 非已获学校批准的提交稿。格式依据上交计算机硕士 2024 级公开模板。</p>
    <details><summary>格式依据与适用范围</summary>
      <p className="muted">当前稿件参考上述特定年级公开模板，不代表任意学院或专业的通用格式要求。具体提交要求以所属学院适用规定为准。</p>
    </details>
    <nav className="school-contents" aria-label="计划书章节目录"><strong>章节</strong><ol>{document.section_specs.map(spec =>
      <li key={spec.id}><a href={`#${prefix}-${spec.id}`}>{spec.title}</a></li>)}</ol></nav>
    {document.section_specs.map(spec => {
      const section = document.sections.find(item => item.section_id === spec.id);
      return <section id={`${prefix}-${spec.id}`} key={spec.id} className="school-section">
        <h3>{spec.title}</h3>
        {section?.blocks.map(block => <article key={block.id}>
          <small className="tag">{blockKinds[block.kind] || "研究内容"}</small>
          <Markdown text={block.text.replace(/\[@(S[1-9][0-9]*)\]/g, (token, alias) =>
            document.source_aliases[alias] ? `[${alias}](#${prefix}-reference-${alias})` : token)} />
          {!!block.source_aliases.length && <details><summary>本段引用的原文依据</summary>{block.source_aliases.map(alias =>
            <div key={alias}><strong>{alias}</strong>{(document.source_aliases[alias] || []).map(ref => <div key={ref}>{evidence(ref)}</div>)}</div>)}</details>}
        </article>)}
      </section>;
    })}
    <section aria-label="核准参考文献"><h3>核准参考文献</h3>
      <p className="muted">以下为当前已准入题录。正式参考文献排版见导出文件，不补写来源中缺失的作者、年份或出版信息。</p>
      <ol>{document.reference_admission.references.map(item => <li id={`${prefix}-reference-${item.id}`} key={item.id}>
        <strong>{item.id}</strong> · {item.author?.map(author => author.literal || [author.given,author.family].filter(Boolean).join(" ")).join(", ")}
        {item.issued?.["date-parts"]?.[0]?.[0] ? ` (${item.issued["date-parts"][0][0]})` : ""} · {item.title}
        {(document.source_aliases[item.id] || []).map(ref => <div key={ref}>{evidence(ref)}</div>)}
      </li>)}</ol>
    </section>
    {allowExport && <ExportPanel taskId={taskId} />}
  </section>;
}
