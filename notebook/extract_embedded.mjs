// Extracts the Python sources embedded as r''' ... ''' strings (CORE_SRC, ENGINE_SRC, SCREEN_SRC, APP_SRC) from the
// concatenated notebook code, writes them as modules for static checking, and writes all.py = core + notebook code.
// Usage: node tools/extract_embedded.mjs <concatenated.py> <out_dir>
import fs from 'fs';
import path from 'path';
const [src, outDir] = process.argv.slice(2);
const code = fs.readFileSync(src, 'utf8');
const names = { CORE_SRC: 'nav3_core.py', ENGINE_SRC: 'nav3_engine.py', SCREEN_SRC: 'screen_reader.py', APP_SRC: 'app.py' };
fs.mkdirSync(outDir, { recursive: true });
const got = {};
for (const [v, file] of Object.entries(names)) {
  const start = code.indexOf(`${v} = r'''`);
  if (start < 0) { console.error(`missing ${v}`); process.exit(1); }
  const body0 = start + `${v} = r'''`.length;
  const end = code.indexOf(`'''`, body0);
  got[v] = code.slice(body0, end);
  fs.writeFileSync(path.join(outDir, file), got[v]);
  console.log(`${file}: ${got[v].split('\n').length} lines`);
}
fs.writeFileSync(path.join(outDir, 'all.py'), got.CORE_SRC + '\n\n# ======== notebook code\n' + code);
console.log('all.py written');
