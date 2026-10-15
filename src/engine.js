import { Matrix, SingularValueDecomposition } from 'ml-matrix';

export const DEFAULTS = { representation:'terms', tokenizer:'basic', topM:200, weight: 'tfidf', vocab: 'all', stopwords: true, normalize: true, kMode: 'energy', energy: 0.8, gamma: 0.5, fixedK: 2, count: 3, method: 'sj', alpha: 0, mmr: false, lambda: 0.7, topicWeight: 0.4 };
const STOP = new Set('và là của các những một được đã đang sẽ để với cho trong trên từ thì mà có này đó rất cũng khi như về ở đến bởi hoặc nhưng vì nên do lại hơn vào ra bị chúng tôi bạn họ nó mình phải vẫn chỉ còn theo mỗi cùng nào'.split(' '));
export const norm = v => Math.hypot(...v);
export const dot = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);
export const clamp = (x,a,b) => Math.min(b,Math.max(a,x));
export function tokenize(text, removeStop = false) {
  const tokens = text.toLocaleLowerCase('vi').normalize('NFC').match(/[\p{L}\p{N}]+(?:_[\p{L}\p{N}]+)*/gu) || [];
  return removeStop ? tokens.filter(t=>!STOP.has(t)) : tokens;
}
export function splitSentences(text) {
  // VietNews may insert a newline after an anonymized name (e.g. Tr.).
  // Keep its continuation with the name, including tokenized “Tr .”.
  const source=text.trim();
  const repaired=source.replace(/\s+/gu,(space,offset)=>{
    const before=source.slice(0,offset),after=source.slice(offset+space.length);
    const abbreviation=before.match(/(?:^|[\s_])([\p{L}]{1,4})\s*\.$/u)?.[1];
    if(abbreviation&&(/^(TP|TX|TS|ThS|PGS|GS)$/u.test(abbreviation)||(/^[A-ZĐ][\p{L}]{0,2}$/u.test(abbreviation)&&/^[\p{Ll}(]/u.test(after))))return '\u0000';
    return space;
  });
  return repaired.split(/\n+|(?<=[.!?…。])\s+(?=[^\s])/u).map(s=>s.replaceAll('\u0000',' ').trim()).filter(Boolean);
}
export function cosine(a,b) { const n=norm(a)*norm(b); return n ? clamp(dot(a,b)/n,-1,1) : 0; }
const transpose = a => a[0]?.map((_,j)=>a.map(r=>r[j])) || [];
const compare = (scores,a,b) => Math.abs(scores[b]-scores[a])<=1e-10 ? a-b : scores[b]-scores[a];
export function weightValue(f, df, n, weight, entropy = 1) {
  if(weight==='binary') return f?1:0;
  if(weight==='logtf') return Math.log1p(f);
  if(weight==='tfidf') return f*Math.log(n/df);
  if(weight==='entropy') return Math.log1p(f)*entropy;
  return f;
}
export function buildRepresentation(text, options={}, resources={}) {
  const cfg={...DEFAULTS,...options}, lines=splitSentences(text),n=lines.length;
  if(n<2) throw new Error('Nhập ít nhất 2 câu, phân cách bằng xuống dòng hoặc dấu kết thúc câu.');
  if(n>120 || text.length>60000) throw new Error('Mỗi lần phân tích tối đa 120 câu và 60.000 ký tự. Hãy chia văn bản thành các đoạn nhỏ hơn.');
  if(cfg.tokenizer==='underthesea' && !resources.rows)throw new Error('Cần dịch vụ Underthesea để tách từ tiếng Việt.');
  const tagged=resources.rows || lines.map(s=>tokenize(s).map(word=>({word,pos:''})));
  if(tagged.length!==n)throw new Error('Số câu tiền xử lý không khớp văn bản.');
  const tokens=tagged.map(row=>row.map(t=>t.word).filter(w=>!cfg.stopwords||!STOP.has(w))), counts=tokens.map(t=>{const c=new Map();for(const w of t)c.set(w,(c.get(w)||0)+1);return c;});
  const removedStopwords=tagged.flat().length-tokens.flat().length;
  let vocab=[...new Set(tokens.flat())].sort((a,b)=>a.localeCompare(b,'vi'));
  if(cfg.vocab==='df2') vocab=vocab.filter(w=>counts.filter(c=>c.has(w)).length>=2);
  if(cfg.vocab==='pos'){
    if(cfg.tokenizer!=='underthesea')throw new Error('Lọc POS cần chọn tách từ Underthesea.');
    const contentWords=new Set(tagged.flat().filter(t=>/^(N|V|A)/.test(t.pos)).map(t=>t.word));
    vocab=vocab.filter(w=>contentWords.has(w));
  }
  if(['top-tf','top-tfidf'].includes(cfg.vocab)){
    const score=w=>{const f=counts.reduce((s,c)=>s+(c.get(w)||0),0),df=counts.filter(c=>c.has(w)).length;return cfg.vocab==='top-tf'?f:f*Math.log(n/df);};
    vocab.sort((a,b)=>score(b)-score(a)||a.localeCompare(b,'vi'));
    vocab=vocab.slice(0,clamp(Math.round(Number(cfg.topM)||200),1,1200)).sort((a,b)=>a.localeCompare(b,'vi'));
  }
  const external=cfg.vocab==='external'?resources.external:null;
  if(cfg.vocab==='external'){
    if(!external?.documentCount||!external.df)throw new Error('Hãy nạp kho ngữ liệu ngoài trước khi phân tích.');
    if(external.tokenizer!==cfg.tokenizer)throw new Error('Kho ngoài dùng tokenizer khác. Hãy dựng lại kho với tokenizer hiện tại.');
    vocab=vocab.filter(w=>Object.hasOwn(external.df,w));
  }
  if(vocab.length>1200)throw new Error('Tối đa 1.200 token phân biệt. Chọn top-m hoặc chia văn bản nhỏ hơn.');
  const termVocab=[...vocab],tf=vocab.map(w=>counts.map(c=>c.get(w)||0)),df=tf.map(row=>row.filter(x=>x>0).length);
  const idf=external?vocab.map(w=>Math.log(external.documentCount/external.df[w])):df.map(d=>Math.log(n/d));
  const entropy=tf.map(row=>{const total=row.reduce((s,x)=>s+x,0);return n>1?clamp(1+row.reduce((s,f)=>{const p=f/total;return s+(p>0?p*Math.log(p):0);},0)/Math.log(n),0,1):0;});
  let rawA=tf.map((row,i)=>row.map(f=>cfg.weight==='tfidf'?f*idf[i]:weightValue(f,df[i],n,cfg.weight,entropy[i]))),embedding=null;
  if(cfg.representation==='embedding'){
    if(!resources.model?.dim||!resources.vectors)throw new Error('Chưa có Word2Vec/PhoW2V. Nạp mô hình ở mục Mô hình ngôn ngữ.');
    const dim=resources.model.dim;
    const detail=Object.fromEntries(termVocab.map(w=>[w,embedToken(w,resources.vectors,dim)]));
    const sentences=tokens.map(row=>{
      const kept=row.filter(w=>Object.hasOwn(detail,w)),used=kept.filter(w=>detail[w].vector),sum=Array(dim).fill(0);
      for(const w of used)detail[w].vector.forEach((v,i)=>sum[i]+=v);
      return {total:kept.length,used:used.length,direct:used.filter(w=>detail[w].route==='direct').length,fallback:used.filter(w=>detail[w].route==='parts').length,missing:kept.filter(w=>!detail[w].vector),vector:sum.map(v=>used.length?v/used.length:0),words:kept};
    });
    rawA=Array.from({length:dim},(_,i)=>sentences.map(s=>s.vector[i]));
    vocab=Array.from({length:dim},(_,i)=>`E${i+1}`);
    embedding={model:resources.model,dim,detail,sentences,total:sentences.reduce((s,x)=>s+x.total,0),used:sentences.reduce((s,x)=>s+x.used,0)};
  }
  const m=vocab.length;
  const rawNorms=Array.from({length:n},(_,j)=>norm(rawA.map(r=>r[j])));
  const A=rawA.map(row=>row.map((x,j)=>cfg.normalize&&rawNorms[j]?x/rawNorms[j]:x));
  const columns=Array.from({length:n},(_,j)=>A.map(row=>row[j])), norms=columns.map(norm);
  return {cfg,lines,n,m,tagged,tokens,removedStopwords,termVocab,vocab,tf,df,idf,entropy,rawA,rawNorms,A,columns,norms,embedding,external:external?{name:external.name,documentCount:external.documentCount}:null};
}
export function embedToken(word,vectors,dim){
  const valid=v=>Array.isArray(v)&&v.length===dim&&v.every(Number.isFinite);
  if(valid(vectors[word]))return {route:'direct',parts:[word],vector:vectors[word]};
  const parts=word.split('_').filter(p=>valid(vectors[p]));
  if(word.includes('_')&&parts.length)return {route:'parts',parts,vector:Array.from({length:dim},(_,i)=>parts.reduce((s,w)=>s+vectors[w][i],0)/parts.length)};
  return {route:'missing',parts:[],vector:null};
}
export function analyze(text, topics=[], options={}, resources={}) {
  const input=buildRepresentation(text,options,resources);
  const {cfg,lines,n,m,tokens,vocab,tf,df,idf,entropy,rawA,rawNorms,A,columns,norms}=input;
  let U=Array.from({length:m},()=>[]),V=Array.from({length:n},()=>[]),sigma=[],rank=0,reconstructionError=0,orthogonalError=0;
  const notices=[];
  if(m && norms.some(x=>x>1e-14)) {
    const svd=new SingularValueDecomposition(new Matrix(A), {autoTranspose:true});
    const diagonal=svd.diagonal, tolerance=diagonal[0]*Math.max(m,n)*Number.EPSILON*10;
    rank=diagonal.filter(x=>x>tolerance).length;sigma=diagonal.slice(0,rank);
    U=svd.leftSingularVectors.to2DArray().map(row=>row.slice(0,rank));
    V=svd.rightSingularVectors.to2DArray().map(row=>row.slice(0,rank));
    // Fix a display convention without changing A: largest |u| entry is positive.
    for(let i=0;i<rank;i++){let pivot=0;for(let t=1;t<m;t++)if(Math.abs(U[t][i])>Math.abs(U[pivot][i]))pivot=t;if(U[pivot][i]<0){U.forEach(r=>r[i]*=-1);V.forEach(r=>r[i]*=-1);}}
    let residual=0,total=0;
    for(let t=0;t<m;t++)for(let j=0;j<n;j++){const reconstructed=sigma.reduce((s,x,i)=>s+U[t][i]*x*V[j][i],0);residual+=(reconstructed-A[t][j])**2;total+=A[t][j]**2;}
    reconstructionError=Math.sqrt(residual/total);
    for(let i=0;i<rank;i++)for(let j=0;j<rank;j++){orthogonalError=Math.max(orthogonalError,Math.abs(U.reduce((s,r)=>s+r[i]*r[j],0)-(i===j?1:0)),Math.abs(V.reduce((s,r)=>s+r[i]*r[j],0)-(i===j?1:0)));}
  } else notices.push(m?'Ma trận A toàn 0. Chọn câu theo thứ tự xuất hiện; thử TF để giữ các từ có mặt ở mọi câu.':'Từ vựng rỗng sau tiền xử lý. Chọn câu theo vị trí; thử tắt stopword hoặc bộ lọc df.');
  const totalEnergy=sigma.reduce((s,x)=>s+x*x,0);let accumulated=0;
  const energy=sigma.map(x=>{accumulated+=x*x;return accumulated/totalEnergy;});
  const count=clamp(Math.round(Number(cfg.count)||1),1,n);
  let k=0;if(rank){if(cfg.method==='gl')k=Math.min(count,rank);else if(cfg.kMode==='fixed')k=clamp(Math.round(Number(cfg.fixedK)||1),1,rank);else if(cfg.kMode==='threshold')k=Math.max(1,sigma.filter(s=>s>=clamp(Number(cfg.gamma),0.01,1)*sigma[0]).length);else{k=energy.findIndex(e=>e>=clamp(Number(cfg.energy),0.01,1)-1e-12)+1;if(!k)k=rank;}}
  const contributions=V.map(row=>row.slice(0,k).map((v,i)=>(sigma[i]*v)**2));
  const baseScores=contributions.map(row=>Math.sqrt(row.reduce((s,x)=>s+x,0)));
  const adjusted=baseScores.map((s,j)=>norms[j]?s/Math.pow(norms[j],clamp(Number(cfg.alpha)||0,0,1)):0);
  const maximum=Math.max(...adjusted,0),normalized=adjusted.map(s=>maximum?s/maximum:0);
  const projected=V.map(row=>row.slice(0,k).map((v,i)=>sigma[i]*v));
  const topicResults=topics.map(label=>{
    const topicWords=(resources.topicRows?.[topics.indexOf(label)]?.map(t=>t.word)||tokenize(label)).filter(w=>!cfg.stopwords||!STOP.has(w));
    const tc=new Map();for(const w of topicWords)tc.set(w,(tc.get(w)||0)+1);
    let q,matched;
    if(input.embedding){const embedded=topicWords.map(w=>({w,...embedToken(w,resources.vectors,input.embedding.dim)})).filter(x=>x.vector);q=Array.from({length:m},(_,i)=>embedded.length?embedded.reduce((s,x)=>s+x.vector[i],0)/embedded.length:0);matched=embedded.map(x=>x.w);}
    else{q=vocab.map((w,i)=>cfg.weight==='tfidf'?(tc.get(w)||0)*idf[i]:weightValue(tc.get(w)||0,df[i],n,cfg.weight,entropy[i]));matched=vocab.filter(w=>tc.has(w));}
    const qnorm=norm(q);if(cfg.normalize&&qnorm)q.forEach((_,i)=>q[i]/=qnorm);
    const latent=Array.from({length:k},(_,i)=>U.reduce((s,row,t)=>s+row[i]*q[t],0));
    const relevance=projected.map(z=>Math.max(0,cosine(z,latent)));
    return {label,q,latent,relevance,matched,active:norm(latent)>1e-12};
  });
  const activeTopics=topicResults.filter(t=>t.active), relevance=Array.from({length:n},(_,j)=>activeTopics.length?Math.max(...activeTopics.map(t=>t.relevance[j])):0);
  const beta=activeTopics.length&&cfg.method!=='gl'?clamp(Number(cfg.topicWeight)||0,0,1):0;
  const finalScores=normalized.map((s,j)=>(1-beta)*s+beta*relevance[j]);
  if(topics.length&&!activeTopics.length)notices.push('Chủ đề không có trọng số trong từ vựng hoặc không chiếu được lên các chiều giữ lại. Kết quả dùng điểm SVD gốc.');
  if(cfg.method==='gl'&&(cfg.mmr||topics.length||cfg.alpha>0))notices.push('Gong–Liu dùng đúng quy tắc mỗi chiều chọn một câu. Chủ đề, α và MMR chỉ áp dụng khi chọn Steinberger–Ježek.');
  if(rank&&cfg.method==='sj'&&k===rank&&cfg.normalize)notices.push('k = rank và các cột khác 0 đã chuẩn hóa L2: điểm SVD đều bằng 1. Việc chọn câu dùng quy tắc hòa điểm, chủ đề hoặc MMR tùy cấu hình.');
  const chosen=[],trace=[];
  if(!rank){for(let j=0;j<count;j++){chosen.push(j);trace.push({iteration:j+1,index:j,reason:'Dự phòng theo vị trí',score:0,rows:[]});}}
  else if(cfg.method==='gl'){
    for(let i=0;i<k;i++){const values=V.map(row=>Math.abs(row[i]));const candidates=Array.from({length:n},(_,j)=>j).filter(j=>!chosen.includes(j)).sort((a,b)=>compare(values,a,b));const best=candidates[0];chosen.push(best);trace.push({iteration:i+1,index:best,reason:`Chiều ${i+1}: |v${i+1},j| lớn nhất`,score:values[best],rows:candidates.map(j=>({index:j,importance:values[j],redundancy:0,value:values[j]}))});}
    if(count>rank){notices.push(`c = ${count} lớn hơn rank = ${rank}. Gong–Liu chọn ${rank} câu; bổ sung ${count-rank} câu theo điểm SVD tổng hợp (mở rộng dự phòng).`);for(const j of Array.from({length:n},(_,i)=>i).sort((a,b)=>compare(baseScores,a,b))){if(chosen.length>=count)break;if(!chosen.includes(j)){chosen.push(j);trace.push({iteration:chosen.length,index:j,reason:'Bổ sung theo điểm SVD tổng hợp',score:baseScores[j],rows:[]});}}}
  } else {
    const lambda=cfg.mmr?clamp(Number(cfg.lambda),0,1):1;
    for(let step=0;step<count;step++){
      const candidates=Array.from({length:n},(_,j)=>j).filter(j=>!chosen.includes(j)).map(j=>{const redundancy=chosen.length?Math.max(0,...chosen.map(l=>cosine(columns[j],columns[l]))):0;return {index:j,importance:finalScores[j],redundancy,value:lambda*finalScores[j]-(1-lambda)*redundancy};});
      candidates.sort((a,b)=>Math.abs(b.value-a.value)<=1e-10?a.index-b.index:b.value-a.value);const best=candidates[0];chosen.push(best.index);trace.push({iteration:step+1,index:best.index,score:best.value,reason:cfg.mmr?'Tối đa MMR':'Điểm cuối cao nhất còn lại',rows:candidates});
    }
  }
  const selected=[...chosen].sort((a,b)=>a-b),VT=transpose(V),Sigma=sigma.map((s,i)=>sigma.map((_,j)=>i===j?s:0));
  return {...input,U,V,VT,Sigma,sigma,rank,k,count,energy,totalEnergy,contributions,baseScores,adjusted,normalized,projected,topicResults,relevance,beta,finalScores,selected,selectionOrder:chosen,trace,notices,reconstructionError,orthogonalError};
}
export function inspectCell(r,t,j) {
  const f=r.tf[t][j],weighted=r.rawA[t][j],colNorm=r.rawNorms[j];
  return {word:r.vocab[t],sentence:j+1,f,df:r.df[t],n:r.lines.length,idf:r.idf[t],entropy:r.entropy[t],weighted,colNorm,normalized:r.A[t][j]};
}
export function gram(r){return r.columns.map(a=>r.columns.map(b=>dot(a,b)));}
export function evaluate(candidate,reference){
 const a=tokenize(candidate),b=tokenize(reference);if(a.length>5000||b.length>5000)throw new Error('Đánh giá tối đa 5.000 token ở mỗi bản tóm tắt.');
 const metric=(match,ca,cb)=>{const p=ca?match/ca:0,r=cb?match/cb:0;return {match,precision:p,recall:r,f1:p+r?2*p*r/(p+r):0};};
 const ngram=(tokens,n)=>{const counts=new Map();for(let i=0;i<=tokens.length-n;i++){const key=tokens.slice(i,i+n).join('\u0000');counts.set(key,(counts.get(key)||0)+1);}return counts;};
 const rouge=n=>{const x=ngram(a,n),y=ngram(b,n);return metric([...x].reduce((s,[g,c])=>s+Math.min(c,y.get(g)||0),0),Math.max(0,a.length-n+1),Math.max(0,b.length-n+1));};
 let prev=new Uint32Array(b.length+1);for(const x of a){const next=new Uint32Array(b.length+1);for(let j=1;j<=b.length;j++)next[j]=x===b[j-1]?prev[j-1]+1:Math.max(prev[j],next[j-1]);prev=next;}
 return {rouge1:rouge(1),rouge2:rouge(2),rougeL:metric(prev[b.length],a.length,b.length)};
}
