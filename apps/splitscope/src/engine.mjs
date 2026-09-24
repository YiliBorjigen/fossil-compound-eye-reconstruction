/** SplitScope numerical core. No DOM, network, or external dependencies. MIT. */
export const VERSION = '1.0.0';
export const DEFAULTS = Object.freeze({preset:'field', seed:42, noise:0.12, k:6, bufferFraction:0.12, testFraction:0.25, anchorX:0.5, anchorY:0.5});
export function randomGenerator(seed) {
  let state = Number(seed) >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let t = Math.imul(state ^ state >>> 15, 1 | state);
    t ^= t + Math.imul(t ^ t >>> 7, 61 | t);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
const mean = values => values.reduce((s,v)=>s+v,0)/values.length;
const distance = (a,b) => Math.hypot(a.x-b.x,a.y-b.y);
export function bounds(points) {
  const xs=points.map(p=>p.x), ys=points.map(p=>p.y), vs=points.map(p=>p.value);
  const xMin=Math.min(...xs), xMax=Math.max(...xs), yMin=Math.min(...ys), yMax=Math.max(...ys);
  return {xMin,xMax,yMin,yMax,valueMin:Math.min(...vs),valueMax:Math.max(...vs),span:Math.max(xMax-xMin,yMax-yMin)};
}
export function validatePoints(points) {
  if(!Array.isArray(points)||points.length<20||points.length>1600) throw new Error('Use between 20 and 1,600 observations.');
  const seen=new Set();
  points.forEach((p,i)=>{
    if(!p||!['x','y','value'].every(key=>Number.isFinite(p[key])&&Math.abs(p[key])<=1e12)) throw new Error(`Row ${i+2}: x, y and value must be finite numbers with magnitude ≤ 1e12.`);
    const key=`${p.x},${p.y}`;
    if(seen.has(key)) throw new Error(`Row ${i+2}: duplicate coordinates. Aggregate repeated measurements or use a grouped validation tool.`);
    seen.add(key);
  });
  if(bounds(points).span===0) throw new Error('Coordinates need a non-zero spatial extent.');
  return points;
}
export function generateData({preset='field',seed=42,noise=0.12,size=28}={}) {
  if(!['field','facets','noise'].includes(preset)) throw new Error('Unknown synthetic preset.');
  if(!Number.isInteger(size)||size<5||size>40) throw new Error('Grid size must be an integer from 5 to 40.');
  if(!Number.isFinite(noise)||noise<0||noise>2) throw new Error('Noise must be between 0 and 2.');
  const rand=randomGenerator(seed);
  const normal=()=>Math.sqrt(-2*Math.log(Math.max(1e-12,rand())))*Math.cos(2*Math.PI*rand());
  const phases=Array.from({length:5},()=>rand()*Math.PI*2);
  const points=[];
  for(let j=0;j<size;j++) for(let i=0;i<size;i++) {
    const x=(i+0.18*(rand()-0.5))/(size-1),y=(j+0.18*(rand()-0.5))/(size-1);
    const field=0.9*Math.sin(7*x+phases[0])*Math.cos(6*y+phases[1])+0.45*Math.sin(12*x+8*y+phases[2])+0.3*Math.cos(4*x-9*y+phases[3]);
    const surface=1.4*Math.exp(-((x-0.48)**2+(y-0.53)**2)/0.13)+0.3*Math.sin(10*x+phases[0])*Math.cos(8*y+phases[1])+0.15*Math.cos(19*x+6*y);
    const signal=preset==='field'?field:preset==='facets'?surface:0;
    points.push({x:x*100,y:y*100,value:preset==='noise'?normal():signal+noise*normal()});
  }
  return points;
}
export function makeSplits(points,options={}) {
  validatePoints(points);
  const o={...DEFAULTS,...options};
  if(!Number.isFinite(o.testFraction)||o.testFraction<=0||o.testFraction>=1) throw new Error('Test fraction must be between 0 and 1.');
  if(!Number.isFinite(o.bufferFraction)||o.bufferFraction<0||o.bufferFraction>2) throw new Error('Buffer fraction must be between 0 and 2.');
  if(![o.anchorX,o.anchorY].every(v=>Number.isFinite(v)&&v>=0&&v<=1)) throw new Error('Region position must be between 0 and 1.');
  const b=bounds(points),n=points.length, count=Math.min(n-1,Math.max(1,Math.round(n*o.testFraction)));
  const anchor={x:b.xMin+o.anchorX*(b.xMax-b.xMin),y:b.yMin+o.anchorY*(b.yMax-b.yMin)};
  const indices=points.map((_,i)=>i), shuffled=[...indices],rand=randomGenerator(o.seed);
  for(let i=n-1;i>0;i--) {const j=Math.floor(rand()*(i+1));[shuffled[i],shuffled[j]]=[shuffled[j],shuffled[i]];}
  const randomTest=new Set(shuffled.slice(0,count));
  const regionOrder=[...indices].sort((i,j)=>distance(points[i],anchor)-distance(points[j],anchor)||i-j);
  const spatialTest=new Set(regionOrder.slice(0,count));
  const buffer=o.bufferFraction*b.span;
  const spatialTrain=indices.filter(i=>!spatialTest.has(i));
  const testIndices=[...spatialTest].sort((a,b)=>a-b);
  const excluded=spatialTrain.filter(i=>testIndices.some(j=>distance(points[i],points[j])<buffer));
  const excludedSet=new Set(excluded);
  const split=(id,test,train,excluded=[])=>({id,test,train,excluded});
  return {
    bounds:b,buffer,anchor,regionRadius:distance(points[regionOrder[count-1]],anchor),
    splits:[
      split('random',indices.filter(i=>randomTest.has(i)),indices.filter(i=>!randomTest.has(i))),
      split('spatial',testIndices,spatialTrain),
      split('buffered',testIndices,spatialTrain.filter(i=>!excludedSet.has(i)),excluded)
    ]
  };
}
export function evaluateSplit(points,split,k=6) {
  if(!Number.isInteger(k)||k<1) throw new Error('Neighbour count must be a positive integer.');
  const assigned=[...split.train,...split.test,...split.excluded];
  if(assigned.some(i=>!Number.isInteger(i)||i<0||i>=points.length)||new Set(assigned).size!==assigned.length||assigned.length!==points.length) throw new Error('Split must assign every point exactly once.');
  const effectiveK=Math.min(k,split.train.length);
  if(!effectiveK||!split.test.length) return {...split,effectiveK,predictions:[],metrics:null,reason:'No training observations remain. Reduce the buffer or move the region.'};
  const trainMean=mean(split.train.map(i=>points[i].value));
  const predictions=split.test.map(index=>{
    const neighbours=split.train.map(i=>({index:i,d:distance(points[index],points[i])})).sort((a,b)=>a.d-b.d||a.index-b.index).slice(0,effectiveK);
    const exact=neighbours.filter(p=>p.d===0);
    // Rescale inverse-distance weights by the nearest distance to avoid overflow.
    const denom=exact.length?1:neighbours.reduce((s,p)=>s+neighbours[0].d/p.d,0);
    const predicted=exact.length?mean(exact.map(p=>points[p.index].value)):neighbours.reduce((s,p)=>s+points[p.index].value*(neighbours[0].d/p.d),0)/denom;
    return {index,predicted,actual:points[index].value,residual:predicted-points[index].value,nearestDistance:neighbours[0].d};
  });
  const mse=mean(predictions.map(p=>p.residual**2)),actualMean=mean(predictions.map(p=>p.actual));
  const variance=mean(predictions.map(p=>(p.actual-actualMean)**2));
  return {...split,effectiveK,predictions,metrics:{rmse:Math.sqrt(mse),mae:mean(predictions.map(p=>Math.abs(p.residual))),r2:variance>0?1-mse/variance:null,meanRmse:Math.sqrt(mean(predictions.map(p=>(p.actual-trainMean)**2))),meanDistance:mean(predictions.map(p=>p.nearestDistance))},reason:null};
}
export function runExperiment(points,options={}) {
  const config={...DEFAULTS,...options},layout=makeSplits(points,config);
  return {version:VERSION,config,...layout,results:layout.splits.map(s=>evaluateSplit(points,s,config.k))};
}
/** Small RFC-style CSV reader: quoted fields, doubled quotes, CRLF and BOM. */
export function parseCSV(text) {
  if(typeof text!=='string'||text.length>1000000) throw new Error('CSV must be text under 1 MB.');
  text=text.replace(/^\uFEFF/,'');
  const rows=[];let row=[],field='',quoted=false,closed=false;
  for(let i=0;i<text.length;i++) {
    const c=text[i];
    if(quoted){if(c==='"'){if(text[i+1]==='"'){field+='"';i++;}else{quoted=false;closed=true;}}else field+=c;continue;}
    if(c==='"'){if(field.length||closed) throw new Error('Malformed CSV quotation.');quoted=true;}
    else if(c===','){row.push(field);field='';closed=false;}
    else if(c==='\n'||c==='\r'){if(c==='\r'&&text[i+1]==='\n')i++;row.push(field);if(row.some(v=>v.trim()!==''))rows.push(row);row=[];field='';closed=false;}
    else {if(closed&&c.trim())throw new Error('Unexpected text after a quoted field.');if(!closed)field+=c;}
  }
  if(quoted)throw new Error('Unclosed quoted field in CSV.');
  row.push(field);if(row.some(v=>v.trim()!==''))rows.push(row);
  if(!rows.length)throw new Error('CSV is empty. Include the header x,y,value.');
  const headers=rows.shift().map(h=>h.trim().toLowerCase());
  if(new Set(headers).size!==headers.length)throw new Error('CSV contains duplicate column names.');
  const columns=['x','y','value'].map(h=>headers.indexOf(h));
  if(columns.includes(-1))throw new Error('CSV needs columns named x, y and value.');
  const points=rows.map((r,i)=>{
    if(r.length!==headers.length)throw new Error(`Row ${i+2}: expected ${headers.length} columns.`);
    const values=columns.map(c=>r[c].trim());
    if(values.some(v=>!v||!/^[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?$/i.test(v)))throw new Error(`Row ${i+2}: x, y and value must be decimal numbers.`);
    return {x:Number(values[0]),y:Number(values[1]),value:Number(values[2])};
  });
  return validatePoints(points);
}
export function dataCSV(points){return 'x,y,value\n'+points.map(p=>`${p.x},${p.y},${p.value}`).join('\n')+'\n';}
export function resultCSV(points,experiment){
  const rows=['strategy,row,x,y,value,assignment,prediction,residual,nearest_training_distance'];
  for(const r of experiment.results){const test=new Set(r.test),excluded=new Set(r.excluded),pred=new Map(r.predictions.map(p=>[p.index,p]));
    points.forEach((p,i)=>{const q=pred.get(i);rows.push([r.id,i+1,p.x,p.y,p.value,test.has(i)?'test':excluded.has(i)?'excluded':'train',q?.predicted??'',q?.residual??'',q?.nearestDistance??''].join(','));});}
  return rows.join('\n')+'\n';
}
