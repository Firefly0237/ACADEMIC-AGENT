import { cp, mkdir, readFile } from "node:fs/promises";
const metadata = JSON.parse(
  await readFile(
    new URL("../node_modules/pdfjs-dist/package.json", import.meta.url),
    "utf8",
  ),
);
if (metadata.version !== "6.5.438")
  throw new Error("Pinned PDF.js version required");
for (const directory of ["cmaps", "standard_fonts", "wasm"]) {
  const target = new URL(
    `../dist/pdfjs/${metadata.version}/${directory}/`,
    import.meta.url,
  );
  await mkdir(target, { recursive: true });
  await cp(
    new URL(`../node_modules/pdfjs-dist/${directory}/`, import.meta.url),
    target,
    { recursive: true },
  );
}
