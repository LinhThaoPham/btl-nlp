import test from 'node:test';
import assert from 'node:assert/strict';
import {analyze,buildRepresentation,embedToken} from '../src/engine.js';
import {rougeDetails,selectionMetrics,randomSelection,textRank,compareMethods,tuneAndEvaluate,validateDataset} from '../src/research.js';
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-8,`${a} != ${b}`);
test('embedding averages repeated occurrences, partial OOV parts and leaves unknown sentence zero',()=>{
  const vectors={a:[1,0],b:[0,1]},model={dim:2,name:'test-fixture'};
  assert.deepEqual(embedToken('a_b',vectors,2).vector,[.5,.5]);
  assert.deepEqual(embedToken('a_unknown',vectors,2).vector,[1,0]);
  const r=analyze('a a b\na_b\nunknown',[],{representation:'embedding',weight:'tf',normalize:false,stopwords:false,fixedK:2,kMode:'fixed'},{vectors,model});
  close(r.rawA[0][0],2/3);close(r.rawA[1][0],1/3);assert.deepEqual(r.columns[2],[0,0]);
  assert.equal(r.embedding.sentences[1].fallback,1);assert.equal(r.embedding.used,4);assert.equal(r.embedding.total,5);assert.ok(r.reconstructionError<1e-8);
});
test('POS, top-m and external IDF build the expected matrix',()=>{
  const rows=[[{word:'học_sinh',pos:'N'},{word:'đang',pos:'R'}],[{word:'học',pos:'V'},{word:'rất',pos:'R'}]];
  const r=buildRepresentation('Học sinh đang\nHọc rất',{tokenizer:'underthesea',vocab:'pos',weight:'tf',stopwords:false},{rows});assert.deepEqual(new Set(r.vocab),new Set(['học_sinh','học']));
  assert.deepEqual(buildRepresentation('a a b\nc',{vocab:'top-tf',topM:1,stopwords:false}).vocab,['a']);
  const ext={documentCount:10,df:{a:2,b:5},tokenizer:'basic',name:'test'};
  const x=buildRepresentation('a a\nb',{vocab:'external',normalize:false,stopwords:false},{external:ext});close(x.rawA[0][0],2*Math.log(5));
});
test('ROUGE counts multiplicity and traces a valid ordered LCS',()=>{
  const d=rougeDetails('a a b c','a b b c');assert.equal(d.metrics.rouge1.match,3);assert.equal(d.alignment.length,3);assert.equal(d.unigrams.find(x=>x.token==='a').match,1);
  for(let i=1;i<d.alignment.length;i++){assert.ok(d.alignment[i].candidate>d.alignment[i-1].candidate);assert.ok(d.alignment[i].reference>d.alignment[i-1].reference);}
  close(selectionMetrics([0,2],[1,2],3).f1,.5);
});
test('baselines use fixed budget; seeded random repeatable; dangling PageRank sums to one',()=>{
  assert.deepEqual(randomSelection(20,4,9),randomSelection(20,4,9));assert.equal(new Set(randomSelection(20,4,9)).size,4);
  const rank=textRank([[0,0],[0,0],[0,0]]);rank.scores.forEach(s=>close(s,1/3));
  const cmp=compareMethods('alpha beta\nbeta gamma\ndelta epsilon\nalpha delta',{count:2,stopwords:false,weight:'tf',tokenizer:'basic'}, {},'alpha beta delta epsilon');
  cmp.rows.forEach(row=>assert.equal(row.selected.length,2));assert.equal(cmp.randomAggregate.seeds.length,5);assert.ok(cmp.rows.every(row=>Number.isFinite(row.rouge.rougeL.f1)));
});
test('tuning is independent of test references and rejects dev/test duplicates',async()=>{
  const rows=[{id:'d',split:'dev',text:'alpha beta\nbeta gamma\ndelta',reference:'alpha beta'},{id:'t',split:'test',text:'red blue\nblue green\nyellow',reference:'red blue'}];
  const prepare=async row=>({resources:{},evalRows:row.text.split('\n').map(s=>s.split(' ')),referenceTokens:row.reference.split(' ')});
  const a=await tuneAndEvaluate(rows,{count:1,weight:'tf',tokenizer:'basic'},prepare);
  const b=await tuneAndEvaluate([rows[0],{...rows[1],reference:'yellow'}],{count:1,weight:'tf',tokenizer:'basic'},prepare);
  assert.deepEqual(a.frozen,b.frozen);assert.equal(a.trials.length,7);assert.equal(a.results.length,1);
  assert.throws(()=>validateDataset([rows[0],{...rows[0],id:'x',split:'test'}]),/trùng/);
});
