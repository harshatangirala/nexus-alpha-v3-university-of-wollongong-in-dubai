// Assembles the NEXUS Alpha v3 notebook from tools/notebook_src/*.pct (authoring aid only: it never executes Python).
// Usage (repository root): node tools/build_notebook.mjs tools/notebook_src NEXUS_Alpha_v3.ipynb [concatenated_code.py]
// Cell markers: a line "# %% [markdown]" starts a markdown cell, a line "# %%" starts a code cell.
import fs from 'fs';
import path from 'path';

const [srcDir, outPath, pyOut] = process.argv.slice(2);
if (!srcDir || !outPath) {
  console.error('usage: node tools/build_notebook.mjs <src_dir> <out.ipynb> [concatenated_code.py]');
  process.exit(2);
}
const files = fs.readdirSync(srcDir).filter(f => f.endsWith('.pct')).sort();
const cells = [];
for (const f of files) {
  const lines = fs.readFileSync(path.join(srcDir, f), 'utf8').replace(/\r\n/g, '\n').split('\n');
  let cur = null;
  const flush = () => {
    if (!cur) return;
    while (cur.src.length && cur.src[cur.src.length - 1].trim() === '') cur.src.pop();
    while (cur.src.length && cur.src[0].trim() === '') cur.src.shift();
    if (cur.src.length) cells.push(cur);
  };
  for (const ln of lines) {
    const t = ln.trim();
    if (t === '# %% [markdown]') { flush(); cur = { type: 'markdown', src: [], file: f }; continue; }
    if (t === '# %%') { flush(); cur = { type: 'code', src: [], file: f }; continue; }
    if (cur) cur.src.push(ln);
  }
  flush();
}
const toSource = src => src.map((l, j) => (j < src.length - 1 ? l + '\n' : l));
const nb = {
  cells: cells.map((c, i) => c.type === 'markdown'
    ? { cell_type: 'markdown', id: `md-${String(i).padStart(3, '0')}`, metadata: {}, source: toSource(c.src) }
    : { cell_type: 'code', id: `code-${String(i).padStart(3, '0')}`, execution_count: null, metadata: {}, outputs: [], source: toSource(c.src) }),
  metadata: {
    accelerator: 'GPU',
    colab: { provenance: [], gpuType: 'A100', machine_shape: 'hm', toc_visible: true },
    kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' },
    language_info: { name: 'python' },
    nexus_alpha_v3: { built_from: 'tools/notebook_src', executed_before_delivery: false },
  },
  nbformat: 4,
  nbformat_minor: 5,
};
fs.writeFileSync(outPath, JSON.stringify(nb, null, 1) + '\n');
const nCode = cells.filter(c => c.type === 'code').length;
const nLines = cells.filter(c => c.type === 'code').reduce((a, c) => a + c.src.length, 0);
console.log(`wrote ${outPath}: ${cells.length} cells (${nCode} code, ${cells.length - nCode} markdown), ${nLines} code lines from ${files.length} source files`);
if (pyOut) {
  const parts = cells.filter(c => c.type === 'code').map((c, i) => `# ---- code cell ${i + 1} (${c.file})\n` + c.src.join('\n'));
  fs.writeFileSync(pyOut, parts.join('\n\n') + '\n');
  console.log(`wrote concatenated code for static checks: ${pyOut}`);
}
