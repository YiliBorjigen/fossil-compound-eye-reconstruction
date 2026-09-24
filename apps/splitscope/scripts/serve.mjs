import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
const port=Number(process.env.PORT||4173);
createServer(async(req,res)=>{
  const path=new URL(req.url,'http://localhost').pathname;
  if(!['/','/index.html','/splitscope.html'].includes(path)){res.writeHead(404);res.end('Not found');return;}
  try{const html=await readFile(new URL('../dist/index.html',import.meta.url));res.writeHead(200,{'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store'});res.end(html);}
  catch{res.writeHead(503);res.end('Run npm run build first.');}
}).listen(port,'0.0.0.0',()=>console.log(`SplitScope listening on ${port}`));
