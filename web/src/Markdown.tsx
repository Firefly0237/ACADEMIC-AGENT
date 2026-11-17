import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import { mathDiagnostics } from "./mathDiagnostics";
import "katex/dist/katex.min.css";
// Sanitize untrusted Markdown before KaTeX builds its own trusted output.
const schema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    code: [
      ...(defaultSchema.attributes?.code || []),
      ["className", "language-math", "math-inline", "math-display"],
    ],
  },
};
export function Markdown({ text }: { text: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[
          [rehypeSanitize, schema],
          [rehypeKatex, { throwOnError: false, trust: false, strict: "warn", maxSize: 32, maxExpand: 1000 }],
          mathDiagnostics,
        ]}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
