import {splitSentences,tokenize} from './engine.js';
export async function nlp(data){
  const response=await fetch('/api/nlp',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  let result;try{result=await response.json();}catch{throw new Error('Dịch vụ NLP không khả dụng. Chạy npm run dev hoặc npm run preview trên máy có .venv.');}
  if(!response.ok||result.error)throw new Error(result.error||'Không kết nối được dịch vụ NLP.');return result;
}
export async function prepareResources(text,topics,options,external){
  const lines=splitSentences(text);
  if(options.tokenizer!=='underthesea'&&options.representation!=='embedding')return {external};
  const data=await nlp({action:'prepare',lines:[...lines,...topics],tokenizer:options.tokenizer,embedding:options.representation==='embedding'});
  return {...data,rows:data.rows.slice(0,lines.length),topicRows:data.rows.slice(lines.length),external};
}
export async function evaluationRows(lines,tokenizer){
  return tokenizer==='underthesea'?(await nlp({action:'prepare',lines,tokenizer})).rows.map(row=>row.map(t=>t.word)):lines.map(s=>tokenize(s));
}
