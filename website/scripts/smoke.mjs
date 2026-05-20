import { createServer } from "node:http";
import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "dist");

const mimeTypes = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".txt", "text/plain; charset=utf-8"],
  [".svg", "image/svg+xml"],
  [".png", "image/png"],
  [".jpg", "image/jpeg"],
  [".jpeg", "image/jpeg"],
  [".webp", "image/webp"],
  [".woff2", "font/woff2"],
]);

function isInsideDist(filePath) {
  const relative = path.relative(dist, filePath);
  return relative && !relative.startsWith("..") && !path.isAbsolute(relative);
}

async function existingFile(filePath) {
  if (!isInsideDist(filePath)) return null;
  try {
    const info = await stat(filePath);
    return info.isFile() ? filePath : null;
  } catch {
    return null;
  }
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", "http://127.0.0.1");
  const pathname = decodeURIComponent(url.pathname);
  const requested = pathname === "/" ? "/index.html" : pathname;
  const filePath = await existingFile(path.join(dist, requested));
  const fallbackPath = path.join(dist, "index.html");
  const resolvedPath = filePath ?? (path.extname(pathname) ? null : fallbackPath);

  if (!resolvedPath) {
    res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    res.end("Not found");
    return;
  }

  const body = await readFile(resolvedPath);
  const contentType = mimeTypes.get(path.extname(resolvedPath)) ?? "application/octet-stream";
  res.writeHead(200, { "content-type": contentType });
  res.end(body);
});

await new Promise((resolve) => {
  server.listen(0, "127.0.0.1", resolve);
});

const { port } = server.address();
const base = `http://127.0.0.1:${port}`;

const checks = [
  { path: "/", includes: ["id=\"root\""] },
  { path: "/docs", includes: ["id=\"root\""] },
  { path: "/docs#openclaw", includes: ["id=\"root\""] },
  { path: "/science", includes: ["id=\"root\""] },
  { path: "/benchmarks", includes: ["id=\"root\""] },
  { path: "/roadmap", includes: ["id=\"root\""] },
  { path: "/vision", includes: ["id=\"root\""] },
  {
    path: "/llms.txt",
    includes: ["Operator startup", "Native Helix", "17-phase", "27 tools", "OpenClaw", "uninstall --purge-data"],
    excludes: ["15-phase", "19 tools", "Dual mode"],
  },
  {
    path: "/llms-full.txt",
    includes: ["operator surface", "Native Helix", "17-Phase Consolidation", "27 MCP tools", "engram-brain", "uninstall --purge-data"],
    excludes: ["15-Phase", "19 tools exposed", "Dual Mode"],
  },
];

async function readBuiltAssetsText() {
  const assetsDir = path.join(dist, "assets");
  const names = await readdir(assetsDir);
  const chunks = await Promise.all(
    names
      .filter((name) => name.endsWith(".js") || name.endsWith(".css"))
      .map((name) => readFile(path.join(assetsDir, name), "utf8")),
  );
  return chunks.join("\n");
}

try {
  for (const check of checks) {
    const response = await fetch(`${base}${check.path}`);
    if (!response.ok) {
      throw new Error(`${check.path} returned HTTP ${response.status}`);
    }

    const body = await response.text();
    for (const expected of check.includes ?? []) {
      if (!body.includes(expected)) {
        throw new Error(`${check.path} did not include ${JSON.stringify(expected)}`);
      }
    }
    for (const rejected of check.excludes ?? []) {
      if (body.includes(rejected)) {
        throw new Error(`${check.path} still included stale text ${JSON.stringify(rejected)}`);
      }
    }
  }

  const builtAssets = await readBuiltAssetsText();
  const assetIncludes = [
    "Install a local brain",
    "engramctl storage",
    "engramctl doctor",
    "engramctl bootstrap /path/to/project",
    "engram-brain",
    "OPERATOR TASK FLOW",
    "How To Read This",
    "uninstall --purge-data",
  ];
  for (const expected of assetIncludes) {
    if (!builtAssets.includes(expected)) {
      throw new Error(`built assets did not include ${JSON.stringify(expected)}`);
    }
  }

  console.log(`Website smoke passed for ${checks.length} routes.`);
} finally {
  server.close();
}
