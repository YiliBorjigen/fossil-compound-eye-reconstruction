import test from 'node:test';
import assert from 'node:assert/strict';
import {generateData,makeSplits,evaluateSplit,runExperiment,parseCSV,dataCSV,resultCSV} from '../src/engine.mjs';
const points=generateData({size:12}), options={seed:123,k:6,bufferFraction:.12};
test('generation and full experiments are reproducible',()=>{
  assert.deepEqual(generateData({size:12}),points);
  assert.deepEqual(runExperiment(points,options),runExperiment(points,options));
  assert.notDeepEqual(generateData({size:12,seed:43}),points);
});
test('every strategy assigns each observation exactly once and matches test sizes',()=>{
  const {splits}=makeSplits(points,options);
  for(const s of splits){const all=[...s.train,...s.test,...s.excluded];assert.equal(all.length,points.length);assert.equal(new Set(all).size,points.length);assert.equal(s.test.length,36);}
  assert.deepEqual(splits[1].test,splits[2].test);
});
test('buffer removes exactly the training observations too close to any test point',()=>{
  const {splits,buffer}=makeSplits(points,options),s=splits[2];
  const minDist=i=>Math.min(...s.test.map(j=>Math.hypot(points[i].x-points[j].x,points[i].y-points[j].y)));
  assert(s.excluded.length>0);
  for(const i of s.train)assert(minDist(i)>=buffer);
  for(const i of s.excluded)assert(minDist(i)<buffer);
});
test('zero buffer equals the spatial holdout; a maximal buffer leaves explicit unavailable metrics',()=>{
  const r=runExperiment(points,{...options,bufferFraction:0});
  assert.deepEqual(r.results[1].metrics,r.results[2].metrics);
  const empty=runExperiment(points,{...options,bufferFraction:2}).results[2];
  assert.equal(empty.train.length,0);assert.equal(empty.metrics,null);assert.equal(empty.predictions.length,0);assert.match(empty.reason,/No training/);
});
test('changing only test outcomes cannot change predictions or split assignments',()=>{
  const split=makeSplits(points,options).splits[2];
  const testIndices=new Set(split.test),changed=points.map((p,i)=>({...p,value:testIndices.has(i)?10000+i:p.value}));
  const original=evaluateSplit(points,split,6),perturbed=evaluateSplit(changed,split,6);
  assert.deepEqual(original.predictions.map(p=>p.predicted),perturbed.predictions.map(p=>p.predicted));
  assert.deepEqual(makeSplits(points,options).splits,makeSplits(changed,options).splits);
});
test('hand-computable inverse-distance prediction and metrics',()=>{
  const p=[{x:0,y:0,value:0},{x:3,y:0,value:12},{x:1,y:0,value:5}];
  const r=evaluateSplit(p,{id:'manual',train:[0,1],test:[2],excluded:[]},2);
  assert.equal(r.predictions[0].predicted,4);assert.equal(r.metrics.rmse,1);assert.equal(r.metrics.mae,1);assert.equal(r.metrics.meanRmse,1);assert.equal(r.metrics.meanDistance,1);assert.equal(r.metrics.r2,null);
});
test('uniform coordinate scaling preserves predictions and rescales distances',()=>{
  const a=runExperiment(points,options),b=runExperiment(points.map(p=>({...p,x:p.x*1000,y:p.y*1000})),options);
  a.results.forEach((r,i)=>{assert.deepEqual(r.train,b.results[i].train);assert.deepEqual(r.test,b.results[i].test);assert(Math.abs(r.metrics.rmse-b.results[i].metrics.rmse)<1e-12);assert(Math.abs(r.metrics.meanDistance*1000-b.results[i].metrics.meanDistance)<1e-9);});
});
test('constant targets give zero RMSE and undefined R²',()=>{
  for(const r of runExperiment(points.map(p=>({...p,value:0})),options).results){assert.equal(r.metrics.rmse,0);assert.equal(r.metrics.r2,null);}
});
test('k is capped by available training observations and malformed splits rejected',()=>{
  const r=evaluateSplit(points,{id:'small',train:[0],test:[1],excluded:Array.from({length:142},(_,i)=>i+2)},20);
  assert.equal(r.effectiveK,1);assert.equal(r.predictions[0].predicted,points[0].value);
  assert.throws(()=>evaluateSplit(points,{train:[0],test:[0],excluded:[]}),/exactly once/);
  assert.throws(()=>evaluateSplit(points,{train:[0],test:[1],excluded:[]}),/exactly once/);
});
test('CSV round trip handles decimal exponents, quoting, CRLF, BOM and extra columns',()=>{
  assert.deepEqual(parseCSV(dataCSV(points)),points);
  const csv='\uFEFF"x","y","value","note"\r\n'+points.map(p=>`${p.x},${p.y},${p.value},"a,b"`).join('\r\n');
  assert.deepEqual(parseCSV(csv),points);
});
test('invalid and ambiguous CSV inputs fail with actionable messages',()=>{
  assert.throws(()=>parseCSV(''),/empty/);
  assert.throws(()=>parseCSV('a,b,c\n1,2,3'),/columns named/);
  assert.throws(()=>parseCSV('x,y,value\n,1,2'),/decimal/);
  assert.throws(()=>parseCSV('x,y,value\n0,NaN,2'),/decimal/);
  assert.throws(()=>parseCSV('x,y,value\n0,1,Infinity'),/decimal/);
  assert.throws(()=>parseCSV('x,y,value\n0,1,"2'),/Unclosed/);
  assert.throws(()=>parseCSV('x,y,value\n0,1,2,3'),/expected 3/);
  assert.throws(()=>parseCSV('x,y,value\n0,1,2'),/20 and/);
  assert.throws(()=>parseCSV(dataCSV([...points,points[0]])),/duplicate coordinates/);
  assert.throws(()=>parseCSV(dataCSV(points.map(p=>({...p,x:1e13+p.x})))),/finite numbers/);
});
test('results CSV includes all rows for all strategies, including excluded assignments',()=>{
  const r=runExperiment(points,options),csv=resultCSV(points,r),rows=csv.trim().split('\n');
  assert.equal(rows.length,1+3*points.length);assert(rows.some(r=>r.includes(',excluded,')));
  assert(!csv.includes('NaN'));assert(!csv.includes('Infinity'));assert(!csv.includes('undefined'));
});
