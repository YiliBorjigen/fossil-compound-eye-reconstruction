import {readFile,writeFile,mkdir} from 'node:fs/promises';
const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8');
const html=await read('src/index.html'),css=await read('src/style.css'),core=await read('src/engine.mjs'),app=await read('src/app.mjs');
const js=core.replace(/^export /gm,'')+'\n'+app.replace(/^import .*?;\s*$/gm,'');
const output=html.replace('/* INLINE_STYLE */',css).replace('/* INLINE_SCRIPT */',js);
await mkdir(new URL('../dist/',import.meta.url),{recursive:true});
await writeFile(new URL('../dist/index.html',import.meta.url),output);
await writeFile(new URL('../dist/splitscope.html',import.meta.url),output);
console.log(`Built self-contained SplitScope (${Buffer.byteLength(output)} bytes).`);
