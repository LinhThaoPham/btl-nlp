import {analyze,DEFAULTS,splitSentences,evaluate,tokenize} from './engine.js';
import {prepareResources} from './nlp-client.js';
import {mean,validateDataset,numericBytes} from './research.js';

export function parseVietNews(raw,meta={}) {
  if(typeof raw!=='string')throw new Error('Tệp VietNews phải là văn bản UTF-8.');
  const blocks=raw.replace(/^\uFEFF/,'').replace(/\r\n?/g,'\n').trim().split(/\n\s*\n/).map(x=>x.trim()).filter(Boolean);
  if(blocks.length<3)throw new Error('Cần ít nhất 3 khối: tiêu đề, tham chiếu và nội dung, ngăn bởi dòng trống.');
  const [title,reference,...body]=blocks;
  // Exclude the short final caption block from every VietNews experiment.
  const last=body.at(-1),candidate=body.length>1&&last.split('\n').length<=4&&last.length<900;
  const cleaned=candidate?body.slice(0,-1).join('\n\n'):body.join('\n\n');
  return {...meta,title,reference,text:cleaned,bodyWithoutTrailing:cleaned,trailingCandidate:candidate?last:'',excludeTrailing:candidate,raw};
}
export function vietnewsOptions(count=3,extra={}) {
  return {...DEFAULTS,tokenizer:'basic',representation:'terms',weight:'tfidf',vocab:'all',count,method:'sj',kMode:'energy',energy:.8,...extra};
}
export function scoreVietNews(record,resources,options) {
  const start=performance.now(),r=analyze(record.text,[],options,resources),elapsedMs=performance.now()-start;
  const summary=r.selected.map(i=>r.lines[i]).join(' '),metrics=evaluate(summary,record.reference);
  return {representation:options.representation,summary,selected:r.selected.map(i=>i+1),metrics,k:r.k,rank:r.rank,sentences:r.lines.length,summaryTokens:tokenize(summary).length,referenceTokens:tokenize(record.reference).length,elapsedMs,numericBytes:numericBytes(r),coverage:r.embedding?{used:r.embedding.used,total:r.embedding.total}:null,model:r.embedding?.model||null,options};
}
async function prepare(record,options){
  const n=splitSentences(record.text).length;
  if(n<2||n>120||record.text.length>60000)throw new Error(`Bài ${record.id||''} cần 2–120 câu và tối đa 60.000 ký tự.`);
  return prepareResources(record.text,[],{...options,representation:'embedding'});
}
export async function compareVietNews(record,options,progress=()=>{}){
  const cfg=vietnewsOptions(options.count,options);progress('Tra vector PhoW2V cho bài đang chọn…');
  const resources=await prepare(record,cfg);
  const rows=['terms','embedding'].map(representation=>scoreVietNews(record,resources,{...cfg,representation}));
  return {schemaVersion:1,kind:'single',createdAt:new Date().toISOString(),record,rows,settings:cfg,model:resources.model,preprocessMs:resources.elapsedMs||0,tokenizer:'basic (giữ token có dấu _ của VietNews)',scope:'Cùng văn bản, tokenizer, số câu, L2 và Steinberger–Ježek; không dùng bản tham chiếu để chọn câu.'};
}
export async function benchmarkVietNews(records,options,progress=()=>{}){
  const dataset=validateDataset(records),cfg=vietnewsOptions(options.count,options),prepared=[];
  for(let i=0;i<dataset.length;i++){progress(`Tra vector ${i+1}/${dataset.length}: ${dataset[i].id}`);prepared.push({record:dataset[i],resources:await prepare(dataset[i],cfg)});}
  const dev=prepared.filter(x=>x.record.split==='dev'),test=prepared.filter(x=>x.record.split==='test');
  const grid=[.7,.8,.9,.95].map(energy=>({kMode:'energy',energy})).concat([.3,.5,.7].map(gamma=>({kMode:'threshold',gamma})));
  const frozen={},trials={};
  for(const representation of ['terms','embedding']){
    trials[representation]=[];
    for(let i=0;i<grid.length;i++){
      progress(`${representation==='terms'?'TF-IDF':'PhoW2V'}: thử cấu hình ${i+1}/7 trên ${dev.length} bài dev`);
      const perDocument=dev.map(({record,resources})=>({id:record.id,f1:scoreVietNews(record,resources,{...cfg,...grid[i],representation}).metrics.rougeL.f1}));
      trials[representation].push({config:grid[i],meanF1:mean(perDocument.map(x=>x.f1)),perDocument});
    }
    trials[representation].sort((a,b)=>b.meanF1-a.meanF1);
    frozen[representation]={...cfg,...trials[representation][0].config,representation};
  }
  const results=[];
  for(let i=0;i<test.length;i++){
    progress(`Chấm test ${i+1}/${test.length}: ${test[i].record.id}`);
    const {record,resources}=test[i];results.push({id:record.id,title:record.title,source:record.source,rows:['terms','embedding'].map(key=>scoreVietNews(record,resources,frozen[key]))});
  }
  const aggregate=['terms','embedding'].map(representation=>{const samples=results.map(r=>r.rows.find(x=>x.representation===representation));return {representation,documents:samples.length,rouge1:mean(samples.map(x=>x.metrics.rouge1.f1)),rouge2:mean(samples.map(x=>x.metrics.rouge2.f1)),rougeL:mean(samples.map(x=>x.metrics.rougeL.f1)),elapsedMs:mean(samples.map(x=>x.elapsedMs))};});
  return {schemaVersion:1,kind:'batch',createdAt:new Date().toISOString(),devIds:dev.map(x=>x.record.id),testIds:test.map(x=>x.record.id),frozen,trials,results,aggregate,dataset,model:prepared[0].resources.model,metric:'Macro ROUGE F1 trên tập test; cùng ngân sách 7 cấu hình/biểu diễn; giữ token gốc; không bỏ stopword khi chấm.'};
}
