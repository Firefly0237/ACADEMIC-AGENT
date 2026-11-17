// rehype-katex records parse failures as VFile messages and may still render a
// colored partial fallback. Surface that official diagnostic to the reader;
// a visible formula alone must not imply that parsing succeeded.
type Node = { type?: string; tagName?: string; properties?: Record<string, unknown>; children?: unknown[]; value?: string };
const text = (value: string) => ({ type: "text", value });
function mathSources(root: { children: unknown[] }): string[] {
  const expressions: string[] = [];
  const visit = (value: unknown) => {
    if (!value || typeof value !== "object") return;
    const node = value as Node;
    if (node.tagName === "annotation" && node.properties?.encoding === "application/x-tex")
      expressions.push((node.children || []).map(child => (child as Node).value || "").join(""));
    else if (Array.isArray(node.properties?.className) && node.properties.className.includes("katex-error"))
      expressions.push((node.children || []).map(child => (child as Node).value || "").join(""));
    else (node.children || []).forEach(visit);
  };
  root.children.forEach(visit);
  return expressions;
}
export function mathDiagnostics() {
  return (tree: { children: unknown[] }, file: { messages: Array<{ source?: string }> }) => {
    const failures = file.messages.filter(message => message.source === "rehype-katex");
    const expressions = mathSources(tree);
    if (failures.length) tree.children.push({ type: "element", tagName: "p", properties: { className: ["notice", "math-warning"], role: "status" },
      children: [{ type: "text", value: `有 ${failures.length} 处公式未能按预期排版，已保留可显示的表达式。请查看原文核对。` }] });
    if (expressions.length) tree.children.push({ type: "element", tagName: "details", properties: { className: ["math-source"] }, children: [
      { type: "element", tagName: "summary", properties: {}, children: [text("公式原始表达式与显示边界")] },
      { type: "element", tagName: "p", properties: {}, children: [text("为避免极端输入影响页面，自定义尺寸最多显示为 32em，宏展开最多 1000 次。原始表达式保留如下；排版或符号含义请对照论文核对。结构正确不代表数学结论已获验证。")] },
      ...expressions.map(expression => ({ type: "element", tagName: "pre", properties: {}, children: [
        { type: "element", tagName: "code", properties: {}, children: [text(expression)] },
      ] })),
    ] });
  };
}
