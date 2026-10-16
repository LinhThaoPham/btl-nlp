import {analyze,buildRepresentation,tokenize,evaluate,cosine,clamp,splitSentences} from './engine.js';

export function selectionMetrics(selected,gold,n){
  for(const list of [selected,gold])if(!Array.isArray(list)||list.some(i=>!Number.isInteger(i)||i<0||i>=n))throw new Error('Chỉ số câu tham chiếu nằm ngoài văn bản.');
  const a=new Set(selected),b=new Set(gold),matches=[...a].filter(i=>b.has(i));
  const precision=a.size?matches.length/a.size:0,recall=b.size?matches.length/b.size:0;
  return {match:matches.length,matches,precision,recall,f1:precision+recall?2*precision*recall/(precision+recall):0};
}
export function rougeDetails(candidate,reference){
  const a=tokenize(candidate),b=tokenize(reference);
  if(a.length*b.length>4_000_000)throw new Error('Giải thích LCS tối đa 4 triệu cặp token. Hãy dùng bản tóm tắt ngắn hơn.');
  const metrics=evaluate(candidate,reference);
  const grams=(tokens,n)=>{const map=new Map();for(let i=0;i<=tokens.length-n;i++){const g=tokens.slice(i,i+n).join(' ');map.set(g,(map.get(g)||0)+1);}return map;};
  const rows=n=>{const x=grams(a,n),y=grams(b,n);return [...new Set([...x.keys(),...y.keys()])].map(token=>({token,candidate:x.get(token)||0,reference:y.get(token)||0,match:Math.min(x.get(token)||0,y.get(token)||0)}));};
  const width=b.length+1,dp=new Uint16Array((a.length+1)*width);
  for(let i=1;i<=a.length;i++)for(let j=1;j<=b.length;j++)dp[i*width+j]=a[i-1]===b[j-1]?dp[(i-1)*width+j-1]+1:Math.max(dp[(i-1)*width+j],dp[i*width+j-1]);
  const alignment=[];let i=a.length,j=b.length;
  while(i&&j){if(a[i-1]===b[j-1]){alignment.push({token:a[i-1],candidate:i-1,reference:j-1});i--;j--;}else if(dp[(i-1)*width+j]>=dp[i*width+j-1])i--;else j--;}
  alignment.reverse();return {metrics,candidate:a,reference:b,unigrams:rows(1),bigrams:rows(2),alignment};
}
export function randomSelection(n,c,seed){
  let state=seed>>>0;const next=()=>{state+=0x6D2B79F5;let t=state;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296;};
  const indices=Array.from({length:n},(_,i)=>i);for(let i=n-1;i>0;i--){const j=Math.floor(next()*(i+1));[indices[i],indices[j]]=[indices[j],indices[i]];}return indices.slice(0,c).sort((a,b)=>a-b);
}
export function textRank(columns,{damping=.85,tolerance=1e-9,maxIterations=200}={}){
  const n=columns.length,edges=columns.map((a,i)=>columns.map((b,j)=>i===j?0:Math.max(0,cosine(a,b)))),sums=edges.map(row=>row.reduce((s,x)=>s+x,0));
  let scores=Array(n).fill(1/n),iterations=0;
  for(;iterations<maxIterations;iterations++){
    const dangling=scores.reduce((s,x,j)=>s+(sums[j]===0?x:0),0);
    const next=scores.map((_,i)=>(1-damping)/n+damping*(dangling/n+scores.reduce((s,x,j)=>s+(sums[j]?x*edges[j][i]/sums[j]:0),0)));
    const delta=next.reduce((s,x,i)=>s+Math.abs(x-scores[i]),0);scores=next;if(delta<tolerance){iterations++;break;}
  }
  return {scores,iterations,edges};
}
export const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:0;
export const std=a=>a.length?Math.sqrt(mean(a.map(x=>(x-mean(a))**2))):0;
const top=(scores,c)=>scores.map((_,i)=>i).sort((a,b)=>Math.abs(scores[b]-scores[a])<=1e-10?a-b:scores[b]-scores[a]).slice(0,c).sort((a,b)=>a-b);
export function compareMethods(text,options,resources={},reference='',gold=null,referenceTokens=null,seeds=[11,29,47,71,101]){
  const n=splitSentences(text).length,c=clamp(Math.round(Number(options.count)||1),1,n),rows=[];
  const cfg={...options};
  const score=(selected)=>{
    const summary=selected.map(j=>splitSentences(text)[j]).join(' ');
    const candidate=options.tokenizer==='underthesea'?selected.flatMap(j=>resources.rows[j].map(t=>t.word)).join(' '):summary;
    return {summary,rouge:reference?evaluate(candidate,referenceTokens?.join(' ')||reference):null,selection:gold?selectionMetrics(selected,gold,n):null};
  };
  const add=(method,fn)=>{const start=performance.now(),output=fn(),elapsedMs=performance.now()-start;const selected=output.selected;rows.push({method,selected,...score(selected),elapsedMs,...output});};
  add('LEAD',()=>({selected:Array.from({length:c},(_,i)=>i)}));
  for(const seed of seeds)add(`Random (${seed})`,()=>({selected:randomSelection(n,c,seed),seed}));
  add('TF-IDF trực tiếp',()=>{const r=buildRepresentation(text,{...cfg,representation:'terms',weight:'tfidf'},resources);return {selected:top(r.columns.map(v=>v.reduce((s,x)=>s+x,0)),c),numericBytes:numericBytes(r)};});
  add('TextRank',()=>{const r=buildRepresentation(text,{...cfg,representation:'terms',weight:'tfidf'},resources),rank=textRank(r.columns);return {selected:top(rank.scores,c),iterations:rank.iterations,numericBytes:numericBytes(r)+n*n*8};});
  for(const method of ['sj','gl'])add(method==='sj'?'SVD Steinberger–Ježek':'SVD Gong–Liu',()=>{const r=analyze(text,[],{...cfg,method},resources);return {selected:r.selected,k:r.k,numericBytes:numericBytes(r)};});
  const random=rows.filter(r=>r.seed!==undefined),randomAggregate={seeds,rougeL:reference?{mean:mean(random.map(r=>r.rouge.rougeL.f1)),std:std(random.map(r=>r.rouge.rougeL.f1))}:null};
  return {rows,randomAggregate,count:c,options:cfg,tokenizer:options.tokenizer,reference,scope:'SVD; baseline dùng TF-IDF, cùng tokenizer, lọc từ vựng, chuẩn hóa L2 và số câu.'};
}
export function comparePrimaryMethods(text,options,embeddingResources={},reference='',embeddingReferenceTokens=null,gold=null){
  const n=splitSentences(text).length,c=clamp(Math.round(Number(options.count)||1),1,n),lines=splitSentences(text),rows=[];
  const sharedResources=embeddingResources;
  const score=selected=>{const summary=selected.map(j=>lines[j]).join(' '),candidate=selected.flatMap(j=>sharedResources.rows[j].map(t=>t.word)).join(' ');return {summary,rouge:reference?evaluate(candidate,embeddingReferenceTokens?.join(' ')||reference):null,selection:gold?selectionMetrics(selected,gold,n):null};};
  const add=(method,fn)=>{const start=performance.now(),output=fn(),elapsedMs=performance.now()-start,selected=output.selected;rows.push({method,selected,...score(selected),elapsedMs,...output});};
  add('LEAD',()=>({selected:Array.from({length:c},(_,i)=>i)}));
  add('TF-IDF → SVD',()=>{const r=analyze(text,[],{...options,method:options.method||'sj',representation:'terms',tokenizer:'underthesea',weight:'tfidf'},sharedResources);return {selected:r.selected,k:r.k,numericBytes:numericBytes(r)};});
  add('Word2Vec / PhoW2V → SVD',()=>{const r=analyze(text,[],{...options,method:options.method||'sj',representation:'embedding',tokenizer:'underthesea',vocab:'all',weight:'tfidf'},sharedResources);return {selected:r.selected,model:r.embedding?.model?.name,k:r.k,numericBytes:numericBytes(r)};});
  return {rows,count:c,options,reference,scope:'LEAD, TF-IDF → SVD và Word2Vec/PhoW2V → SVD; cùng văn bản, tokenizer Underthesea, số câu và cấu hình SVD.'};
}
export function numericBytes(r){
  const keys=['tf','rawA','A','columns','U','V','VT','Sigma'];
  return keys.reduce((s,key)=>s+(r[key]||[]).reduce((n,row)=>n+row.length*8,0),0);
}
export function validateDataset(input){
  if(!Array.isArray(input)||input.length<2||input.length>100)throw new Error('Dataset cần 2–100 bản ghi.');
  const ids=new Set(),texts=new Set();
  const rows=input.map((row,index)=>{
    if(!row||typeof row.text!=='string'||typeof row.reference!=='string'||!row.reference.trim())throw new Error(`Bản ghi ${index+1} thiếu text/reference.`);
    if(!['dev','test'].includes(row.split))throw new Error(`Bản ghi ${index+1} cần split là dev hoặc test.`);
    const id=String(row.id||index+1),fingerprint=row.text.normalize('NFC').toLocaleLowerCase('vi').replace(/\s+/g,' ').trim();
    if(ids.has(id)||texts.has(fingerprint))throw new Error('Dataset có ID hoặc văn bản trùng. Loại trùng để tránh rò rỉ dev/test.');ids.add(id);texts.add(fingerprint);
    const n=splitSentences(row.text).length;if(n<2||n>120||row.text.length>60000)throw new Error(`Bản ghi ${id} cần 2–120 câu, tối đa 60.000 ký tự.`);
    if(row.gold!==undefined){if(!Array.isArray(row.gold)||row.gold.some(i=>!Number.isInteger(i)))throw new Error(`gold của ${id} phải là mảng số câu nguyên.`);selectionMetrics([],row.gold.map(i=>i-1),n);}
    return {...row,id};
  });
  if(!rows.some(r=>r.split==='dev')||!rows.some(r=>r.split==='test'))throw new Error('Cần ít nhất một văn bản dev và một văn bản test.');
  return rows;
}
export async function tuneAndEvaluate(dataset,base,prepare,progress=()=>{}){
  const rows=validateDataset(dataset),prepared=[];
  for(let i=0;i<rows.length;i++){progress(`Tiền xử lý ${i+1}/${rows.length}`);prepared.push({...rows[i],...await prepare(rows[i])});}
  const grid=[.7,.8,.9,.95].map(energy=>({kMode:'energy',energy})).concat([.3,.5,.7].map(gamma=>({kMode:'threshold',gamma})));
  const dev=prepared.filter(r=>r.split==='dev'),test=prepared.filter(r=>r.split==='test');
  const trials=[];
  for(let i=0;i<grid.length;i++){
    progress(`Thử cấu hình ${i+1}/${grid.length} trên tập dev`);const perDocument=[];
    for(const row of dev){const r=analyze(row.text,[],{...base,...grid[i],method:'sj'},row.resources);const candidate=r.selected.flatMap(j=>row.evalRows[j]).join(' ');perDocument.push({id:row.id,f1:evaluate(candidate,row.referenceTokens.join(' ')).rougeL.f1});}
    trials.push({config:grid[i],meanF1:mean(perDocument.map(x=>x.f1)),perDocument});
  }
  trials.sort((a,b)=>b.meanF1-a.meanF1);const frozen={...base,...trials[0].config,method:'sj'};
  const results=[];
  for(let i=0;i<test.length;i++){progress(`Đánh giá test ${i+1}/${test.length}`);const row=test[i];results.push({id:row.id,...compareMethods(row.text,frozen,row.resources,row.reference,row.gold?.map(j=>j-1)||null,row.referenceTokens)});}
  const names=results[0].rows.map(r=>r.method),aggregate=names.map(method=>{const samples=results.map(r=>r.rows.find(x=>x.method===method));return {method,documents:samples.length,rouge1:mean(samples.map(r=>r.rouge.rouge1.f1)),rouge2:mean(samples.map(r=>r.rouge.rouge2.f1)),rougeL:mean(samples.map(r=>r.rouge.rougeL.f1)),elapsedMs:mean(samples.map(r=>r.elapsedMs)),numericBytes:mean(samples.map(r=>r.numericBytes||0))};});
  const randomMeans=aggregate.filter(r=>r.method.startsWith('Random')).map(r=>r.rougeL);
  return {schemaVersion:1,createdAt:new Date().toISOString(),devIds:dev.map(r=>r.id),testIds:test.map(r=>r.id),trials,frozen,aggregate,results,randomSummary:{rougeLMean:mean(randomMeans),rougeLStdAcrossSeeds:std(randomMeans)},external:prepared[0].resources.external||null,model:prepared[0].resources.model||null,metric:'Macro-average ROUGE F1; tokenizer thống nhất; không bỏ stopword khi chấm.',dataset:rows};
}
