import {VERSION,DEFAULTS,generateData,runExperiment,parseCSV,dataCSV,resultCSV} from './engine.mjs';
const $=id=>document.getElementById(id);
const COLOURS={random:'#87f0ce',spatial:'#ffb478',buffered:'#c4acff'};
const LABELS={random:'Random',spatial:'Spatial',buffered:'Buffered'};
const REGIONS={center:[0.5,0.5],left:[0,0.5],right:[1,0.5],top:[0.5,1]};
const descriptions={field:'Nearby observations share a pattern. Synthetic data, not measurements.',facets:'A curved surface with local texture. Synthetic, not a biological lens model.',noise:'Independent Gaussian values: location contains no signal. A useful negative control.',custom:'Your local CSV. Distances use the supplied coordinate units, without axis rescaling.'};
let config={...DEFAULTS},points=[],experiment,customData=null,view='split',timer,toastTimer;
function num(v,digits=3){if(v===null||v===undefined||!Number.isFinite(v))return '—';if(v!==0&&(Math.abs(v)>=1e5||Math.abs(v)<.0001))return v.toExponential(2);return v.toFixed(digits);}
function showToast(text){$('toast').textContent=text;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,4500);}
function readHash(){
  const p=new URLSearchParams(location.hash.slice(1));
  if(['field','facets','noise'].includes(p.get('preset')))config.preset=p.get('preset');
  const parse=(key,min,max,integer=false)=>{const raw=p.get(key);if(raw===null)return;const v=Number(raw);if(Number.isFinite(v)&&v>=min&&v<=max&&(!integer||Number.isInteger(v)))config[key]=v;};
  parse('seed',0,999999,true);parse('noise',0,.8);parse('k',1,20,true);parse('bufferFraction',0,.4);
  const region=REGIONS[p.get('region')]||REGIONS.center;[config.anchorX,config.anchorY]=region;
}
function syncControls(){
  $('preset').value=config.preset;$('noise').value=config.noise;$('neighbors').value=config.k;$('buffer').value=config.bufferFraction*100;$('seed').value=config.seed;
  $('region').value=Object.keys(REGIONS).find(k=>REGIONS[k][0]===config.anchorX&&REGIONS[k][1]===config.anchorY)||'center';
}
function settings(){
  config.preset=$('preset').value;config.noise=Number($('noise').value);config.k=Number($('neighbors').value);config.bufferFraction=Number($('buffer').value)/100;
  const seedValue=$('seed').valueAsNumber;config.seed=Number.isFinite(seedValue)?Math.max(0,Math.min(999999,Math.round(seedValue))):42;
  [config.anchorX,config.anchorY]=REGIONS[$('region').value];
  $('noise-output').value=num(config.noise,2);$('neighbors-output').value=config.k;$('buffer-output').value=Math.round(config.bufferFraction*100)+'%';
  $('noise').disabled=config.preset==='custom'||config.preset==='noise';
  $('dataset-description').textContent=descriptions[config.preset];
}
function colour(t,error=false){
  t=Math.max(0,Math.min(1,t));
  const stops=error?[[35,58,74],[193,124,83],[255,212,135]]:[[34,62,104],[72,161,169],[185,245,175]];
  const s=t<.5?0:1,u=t<.5?t*2:(t-.5)*2;
  return `rgb(${stops[s].map((v,i)=>Math.round(v+(stops[s+1][i]-v)*u)).join(',')})`;
}
function projection(b,size=300,padding=24){
  const usable=size-2*padding;
  const radius=v=>(v/b.span)*usable;
  const extraX=(usable-radius(b.xMax-b.xMin))/2,extraY=(usable-radius(b.yMax-b.yMin))/2;
  return {x:v=>padding+extraX+radius(v-b.xMin),y:v=>size-padding-extraY-radius(v-b.yMin),radius};
}
function mapSVG(result,mode=view){
  const b=experiment.bounds,p=projection(b),test=new Set(result.test),excluded=new Set(result.excluded),pred=new Map(result.predictions.map(v=>[v.index,v]));
  const allErrors=experiment.results.flatMap(r=>r.predictions.map(v=>Math.abs(v.residual))),maxError=Math.max(1e-12,...allErrors);
  let svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300" role="img" aria-label="${LABELS[result.id]} holdout ${mode} map"><title>${LABELS[result.id]} holdout: ${result.train.length} training, ${result.test.length} test, ${result.excluded.length} excluded observations</title><rect width="300" height="300" fill="#0b131e"/>`;
  for(let i=0;i<=4;i++){const v=24+i*63;svg+=`<path d="M24 ${v}H276M${v} 24V276" stroke="#243243" stroke-width=".5" opacity=".7"/>`;}
  if(result.id!=='random'){
    svg+=`<circle cx="${p.x(experiment.anchor.x)}" cy="${p.y(experiment.anchor.y)}" r="${p.radius(experiment.regionRadius)}" fill="none" stroke="${COLOURS[result.id]}" stroke-dasharray="3 4" stroke-width=".8" opacity=".6"/>`;
  }
  const radius=Math.max(1.7,Math.min(4.5,65/Math.sqrt(points.length)));
  points.forEach((point,i)=>{
    const isTest=test.has(i),isExcluded=excluded.has(i),q=pred.get(i);
    let fill=isExcluded?'#192533':isTest?'#87f0ce':'#4e6c88',opacity=1,stroke=isExcluded?'#354352':'none';
    if(mode==='values')fill=colour((point.value-b.valueMin)/(b.valueMax-b.valueMin||1));
    if(mode==='error'){fill=q?colour(Math.abs(q.residual)/maxError,true):'#26384a';opacity=q?1:.38;}
    const status=isTest?'test':isExcluded?'excluded':'train';
    const title=`Row ${i+1} · ${status}\nx ${num(point.x,2)}, y ${num(point.y,2)}\nValue ${num(point.value)}${q?'\nPrediction '+num(q.predicted)+' · error '+num(q.residual)+'\nNearest training '+num(q.nearestDistance,2):''}`;
    svg+=`<circle cx="${num(p.x(point.x),3)}" cy="${num(p.y(point.y),3)}" r="${radius}" fill="${fill}" stroke="${stroke}" stroke-width=".6" opacity="${opacity}"><title>${title}</title></circle>`;
  });
  svg+=`<g fill="#8397af" font-family="monospace" font-size="8"><text x="24" y="291">${num(b.xMin,1)}</text><text x="276" y="291" text-anchor="end">${num(b.xMax,1)} · x</text><text x="6" y="14">y · ${num(b.yMax,1)}</text></g></svg>`;
  return svg;
}
function chartSVG(){
  const w=540,h=216,pad={l:46,r:18,t:14,b:35};
  const all=experiment.results.flatMap(r=>r.predictions),maxD=Math.max(1e-9,...all.map(p=>p.nearestDistance))*1.05,maxE=Math.max(1e-9,...all.map(p=>Math.abs(p.residual)))*1.05;
  const x=v=>pad.l+v/maxD*(w-pad.l-pad.r),y=v=>h-pad.b-v/maxE*(h-pad.t-pad.b);
  let svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="Scatter plot of absolute prediction error against nearest training distance"><title>Held-out error versus nearest training distance, one dot per prediction</title>`;
  for(let i=0;i<=4;i++){const vx=maxD*i/4,vy=maxE*i/4;svg+=`<path d="M${pad.l} ${y(vy)}H${w-pad.r}" stroke="#293748" stroke-width=".7"/><g fill="#93a6be" font-family="monospace" font-size="9"><text x="${pad.l-8}" y="${y(vy)+3}" text-anchor="end">${num(vy,1)}</text><text x="${x(vx)}" y="${h-pad.b+15}" text-anchor="middle">${num(vx,1)}</text></g>`;}
  for(const result of experiment.results)for(const q of result.predictions)svg+=`<circle cx="${num(x(q.nearestDistance),2)}" cy="${num(y(Math.abs(q.residual)),2)}" r="2.4" fill="${COLOURS[result.id]}" opacity=".65"><title>${LABELS[result.id]} · row ${q.index+1} · distance ${num(q.nearestDistance,2)} · absolute error ${num(Math.abs(q.residual))}</title></circle>`;
  svg+=`<g fill="#93a6be" font-family="sans-serif" font-size="9"><text x="290" y="212" text-anchor="middle">Distance to nearest training point</text><text x="12" y="111" text-anchor="middle" transform="rotate(-90 12 111)">Absolute error</text></g></svg>`;
  return svg;
}
function render(){
  settings();
  points=config.preset==='custom'?customData:generateData(config);
  experiment=runExperiment(points,config);
  $('sample-count').textContent=`${points.length.toLocaleString()} observations · ${experiment.results[0].test.length} held out`;
  $('buffer-note').textContent=`${num(experiment.buffer,2)} coordinate units · ${Math.round(config.bufferFraction*100)}% of the longest span.`;
  for(const r of experiment.results){
    $('map-'+r.id).innerHTML=mapSVG(r);
    $('rmse-'+r.id).textContent=num(r.metrics?.rmse);
    $('counts-'+r.id).textContent=`${r.train.length} train · ${r.test.length} test${r.excluded.length?' · '+r.excluded.length+' out':''}`;
    $('distance-'+r.id).textContent=num(r.metrics?.meanDistance,2);
  }
  if(view==='split')$('legend').innerHTML='<span class="legend-item"><i class="legend-dot" style="background:#4e6c88"></i>Train</span><span class="legend-item"><i class="legend-dot" style="background:#87f0ce"></i>Test</span><span class="legend-item"><i class="legend-dot" style="background:#192533;border:1px solid #536377"></i>Excluded</span>';
  else{
    const b=experiment.bounds,max=view==='values'?b.valueMax:Math.max(0,...experiment.results.flatMap(r=>r.predictions.map(p=>Math.abs(p.residual))));
    $('legend').innerHTML=`<span>${view==='values'?'Observed value':'Absolute test error'}</span><span>${num(view==='values'?b.valueMin:0,2)}</span><i class="legend-gradient" style="background:linear-gradient(90deg,${colour(0,view==='error')},${colour(.5,view==='error')},${colour(1,view==='error')})"></i><span>${num(max,2)}</span><span>Shared scale</span>`;
  }
  const random=experiment.results[0],buffered=experiment.results[2],ratio=random.metrics?.rmse>1e-12&&buffered.metrics?buffered.metrics.rmse/random.metrics.rmse:null;
  $('gap-number').textContent=ratio===null?'—':num(ratio,1)+'×';
  $('gap-title').textContent=ratio===null?'No valid RMSE ratio':'Buffered / random RMSE';
  $('gap-copy').textContent=ratio===null?'A ratio is unavailable when random error is zero or the buffer leaves no training observations.':`Same model, different prediction conditions. The buffer changes training distance and sample size. This gap alone does not establish leakage.`;
  const warnings=experiment.results.filter(r=>r.reason).map(r=>`${LABELS[r.id]}: ${r.reason}`);
  if(config.preset==='custom')warnings.push('CSV coordinates are used as supplied. Both axes must share a linear unit; do not use unprojected latitude/longitude.');
  $('warning').hidden=!warnings.length;$('warning').textContent=warnings.join(' ');
  $('distance-chart').innerHTML=chartSVG();
  $('metrics-table').innerHTML=experiment.results.map(r=>`<tr><td><span style="background:${COLOURS[r.id]}"></span>${LABELS[r.id]}</td><td>${num(r.metrics?.rmse)}</td><td>${num(r.metrics?.meanRmse)}</td><td>${num(r.metrics?.r2,2)}</td></tr>`).join('');
  $('share').disabled=config.preset==='custom';$('share').title=config.preset==='custom'?'Custom data is not included in share links. Export the full report instead.':'Copy a reproducible link without uploading data';
}
function schedule(){clearTimeout(timer);timer=setTimeout(render,100);}
function download(filename,content,type){const blob=new Blob([content],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=filename;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function exportFigure(){
  let svg='<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="575" viewBox="0 0 1200 575"><rect width="1200" height="575" fill="#0b1018"/><g fill="#edf4fb" font-family="sans-serif"><text x="36" y="47" font-size="26">SplitScope · Same data. Different question.</text><text x="36" y="77" font-size="14" fill="#99a9bd">Coordinate-only kNN · k='+config.k+' · seed '+config.seed+' · '+points.length+' observations · buffer '+num(experiment.buffer,2)+' coordinate units</text></g>';
  experiment.results.forEach((r,i)=>{const x=36+i*392;const inner=mapSVG(r).replace('<svg xmlns="http://www.w3.org/2000/svg"','<svg x="'+x+'" y="126" width="340" height="340"');svg+=`<text x="${x}" y="111" fill="${COLOURS[r.id]}" font-size="18" font-family="sans-serif">${i+1}. ${LABELS[r.id]}</text>`+inner+`<g fill="#edf4fb" font-family="monospace" font-size="16"><text x="${x}" y="493">RMSE ${num(r.metrics?.rmse)}</text><text x="${x}" y="518" fill="#99a9bd" font-size="12">${r.train.length} train / ${r.test.length} test / ${r.excluded.length} excluded</text></g>`;});
  svg+='<text x="36" y="552" fill="#99a9bd" font-size="12" font-family="sans-serif">One split per strategy. A validation gap is not proof of leakage. Synthetic data unless CSV imported. SplitScope v'+VERSION+'</text></svg>';
  download('splitscope-comparison.svg',svg,'image/svg+xml');showToast('Figure saved as editable SVG.');
}
for(const id of ['noise','neighbors','buffer'])$(id).addEventListener('input',()=>{settings();schedule();});
for(const id of ['preset','region'])$(id).addEventListener('change',render);
$('seed').addEventListener('change',()=>{settings();$('seed').value=config.seed;render();});
$('reseed').addEventListener('click',()=>{$('seed').value=(config.seed+1)%1000000;render();});
$('reset').addEventListener('click',()=>{config={...DEFAULTS};customData=null;$('preset').querySelector('[value=custom]').disabled=true;syncControls();setView('split');try{history.replaceState(null,'',location.pathname+location.search);}catch{}render();showToast('Experiment reset to seed 42.');});
function setView(next){view=next;for(const v of ['split','values','error']){const b=$('view-'+v);b.classList.toggle('selected',v===view);b.setAttribute('aria-pressed',String(v===view));}if(experiment)render();}
for(const v of ['split','values','error'])$('view-'+v).addEventListener('click',()=>setView(v));
$('import-open').addEventListener('click',()=>{$('import-error').textContent='';$('import-dialog').showModal();});
$('csv-file').addEventListener('change',async()=>{const file=$('csv-file').files[0];if(!file)return;if(file.size>1000000){$('import-error').textContent='Choose a CSV smaller than 1 MB.';return;}try{$('csv-text').value=await file.text();$('import-error').textContent='';}catch{$('import-error').textContent='This file could not be read. Try pasting the CSV instead.';}});
$('sample-csv').addEventListener('click',()=>download('splitscope-example.csv',dataCSV(generateData({size:10})),'text/csv'));
$('import-apply').addEventListener('click',()=>{try{const parsed=parseCSV($('csv-text').value);customData=parsed;$('preset').querySelector('[value=custom]').disabled=false;$('preset').value='custom';render();$('import-dialog').close();showToast(`Loaded ${parsed.length} observations locally.`);}catch(error){$('import-error').textContent=error.message;}});
$('export-csv').addEventListener('click',()=>{render();download('splitscope-predictions.csv',resultCSV(points,experiment),'text/csv');showToast('Assignments and predictions exported for all three splits.');});
$('export-json').addEventListener('click',()=>{render();download('splitscope-report.json',JSON.stringify({tool:'SplitScope',...experiment,data:points,method:'Single holdout per strategy; inverse-distance kNN using x,y only.',limitations:['Random and spatial tests use different locations.','Buffering also changes training-set size.','No uncertainty estimate; no evidence of cross-specimen generalisation.']},null,2),'application/json');showToast('Full report saved, including data and settings.');});
$('export-svg').addEventListener('click',()=>{render();exportFigure();});
$('share').addEventListener('click',async()=>{
  render();if(config.preset==='custom')return;
  if(location.protocol==='file:'){showToast('Share links need the hosted app. Save the full report to share an offline experiment.');return;}
  const p=new URLSearchParams({preset:config.preset,seed:config.seed,noise:config.noise,k:config.k,bufferFraction:config.bufferFraction,region:$('region').value});
  const url=new URL(location.href);url.hash=p.toString();history.replaceState(null,'',url);
  try{await navigator.clipboard.writeText(url.href);showToast('Experiment link copied. Data is generated from its settings.');}catch{showToast('Settings saved in the address bar. Copy that URL to share.');}
});
window.addEventListener('hashchange',()=>{if(!location.hash.includes('='))return;config={...DEFAULTS};readHash();syncControls();render();});
readHash();syncControls();render();
// Progressive WebMCP support; browsers without it use the standard controls.
const modelContext=document.modelContext;
if(modelContext?.registerTool){
  const lifecycle=new AbortController();
  window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
  try{Promise.resolve(modelContext.registerTool({name:'get_spatial_validation_summary',title:'Read validation summary',description:'Read the visible settings and numerical metrics. Does not return imported rows.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true,untrustedContentHint:true},execute:async(input)=>{
    if(!input||typeof input!=='object'||Array.isArray(input)||Object.keys(input).length)throw new Error('Expected an empty object.');
    return {version:VERSION,config:experiment.config,sampleCount:points.length,results:experiment.results.map(r=>({strategy:r.id,train:r.train.length,test:r.test.length,excluded:r.excluded.length,metrics:r.metrics}))};
  }},{signal:lifecycle.signal})).catch(()=>{});}catch{}
}
