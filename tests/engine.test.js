import test from 'node:test';
import assert from 'node:assert/strict';
import {analyze,DEFAULTS,weightValue,evaluate,splitSentences,tokenize,norm,gram} from '../src/engine.js';
import {examples} from '../src/examples.js';
const close=(a,b,e=1e-9)=>assert.ok(Math.abs(a-b)<e,`${a} != ${b}`);
const base={...DEFAULTS,weight:'tf',normalize:false,stopwords:false,kMode:'fixed',fixedK:2,topicWeight:0};
test('SVD reconstructs tall, wide and rank-deficient matrices and preserves orthonormal vectors',()=>{
 for(const [text,options] of [
  ['alpha beta gamma\nbeta gamma delta\nalpha delta epsilon',base],
  ['alpha\nbeta\nalpha beta\nbeta beta\nalpha alpha',{...base,vocab:'all'}],
  ['alpha beta\nalpha beta\nalpha beta',base],
  [examples[0].text,DEFAULTS]
 ]){const r=analyze(text,[],options);assert.ok(r.reconstructionError<1e-10);assert.ok(r.orthogonalError<1e-10);assert.equal(r.U.length,r.vocab.length);assert.equal(r.V.length,r.lines.length);assert.equal(r.VT.length,r.rank);
  const B=gram(r);for(let i=0;i<r.rank;i++)for(let j=0;j<r.lines.length;j++){const lhs=B[j].reduce((s,x,l)=>s+x*r.V[l][i],0);close(lhs,r.sigma[i]**2*r.V[j][i]);}
 }
});
test('word weighting uses exact PDF formulas without IDF smoothing',()=>{
 close(weightValue(2,2,4,'tfidf'),2*Math.log(2));close(weightValue(2,4,4,'tfidf'),0);close(weightValue(4,2,4,'logtf'),Math.log(5));assert.equal(weightValue(2,2,4,'binary'),1);assert.equal(weightValue(0,2,4,'binary'),0);
 const r=analyze('alpha alpha beta\nalpha beta beta',[],{...base,weight:'entropy'});r.entropy.forEach(x=>assert.ok(x>=0&&x<=1));
 const uniform=analyze('alpha\nalpha',[],{...base,weight:'entropy'});close(uniform.entropy[0],0);assert.equal(uniform.rank,0);
});
test('full-rank S-J score is the column norm, not its square; alpha uses SVD-input norm',()=>{
 const r=analyze('alpha alpha beta\nbeta beta gamma\ngamma gamma delta',[],{...base,fixedK:100,alpha:1});
 for(let j=0;j<r.lines.length;j++){close(r.baseScores[j],norm(r.columns[j]));close(r.adjusted[j],1);}
 const z=analyze('alpha alpha beta\nbeta beta gamma\ngamma gamma delta',[],{...base,fixedK:100,normalize:true});z.baseScores.forEach(s=>close(s,1));assert.deepEqual(z.selected,[0,1,2]);
});
test('energy mode retains the smallest k satisfying the threshold',()=>{const r=analyze(examples[0].text,[],{...DEFAULTS,energy:.9});assert.ok(r.energy[r.k-1]>=.9-1e-12);if(r.k>1)assert.ok(r.energy[r.k-2]<.9);});
test('Gong-Liu chooses unused maximum absolute loading for each dimension',()=>{
 const r=analyze(examples[2].text,['học_sinh'],{...base,method:'gl',count:3,mmr:true,alpha:1});assert.equal(r.k,3);
 const used=new Set();r.trace.forEach((t,i)=>{const remaining=r.V.map((row,j)=>({j,score:Math.abs(row[i])})).filter(x=>!used.has(x.j)).sort((a,b)=>Math.abs(b.score-a.score)<1e-10?a.j-b.j:b.score-a.score);assert.equal(t.index,remaining[0].j);used.add(t.index);});
 assert.equal(r.beta,0);assert.equal(new Set(r.selected).size,3);assert.ok(r.notices.some(x=>x.includes('Gong–Liu')));
});
test('MMR normalizes importance and accounts for maximum cosine redundancy at each round',()=>{
 const r=analyze('alpha beta alpha\nalpha beta alpha\ngamma delta gamma\nalpha epsilon',[],{...base,fixedK:100,mmr:true,lambda:.4,count:3});
 for(const t of r.trace){assert.equal(t.index,t.rows[0].index);for(const row of t.rows)close(row.value,.4*row.importance-.6*row.redundancy);}
 assert.ok(r.selectionOrder[1]!==1,'second identical sentence should be penalized');
});
test('zero matrix and empty vocabulary fall back to source position',()=>{
 for(const [text,cfg] of [['alpha\nalpha',{...base,weight:'tfidf'}],['và là\ncủa để',{...DEFAULTS}]] ) {const r=analyze(text,[],cfg);assert.equal(r.rank,0);assert.equal(r.k,0);assert.deepEqual(r.selected,[0,1]);assert.ok(r.notices.length);assert.ok(r.finalScores.every(s=>s===0));}
});
test('Gong-Liu c > rank declares and fills the fallback extension',()=>{const r=analyze('alpha\nalpha\nalpha',[],{...base,method:'gl',count:3});assert.equal(r.rank,1);assert.equal(r.selected.length,3);assert.ok(r.notices.some(s=>s.includes('lớn hơn rank')));});
test('topic fold-in uses the same vocabulary and projection; unknown topic disables blend',()=>{
 const text='alpha beta\nalpha gamma\ndelta epsilon\ndelta zeta';
 const r=analyze(text,['alpha'],{...base,fixedK:100,topicWeight:1});assert.equal(r.beta,1);assert.ok(r.relevance[0]>r.relevance[2]);
 const unknown=analyze(text,['omega'],{...base,topicWeight:1});assert.equal(unknown.beta,0);assert.ok(unknown.notices.some(s=>s.includes('không có trọng số')));
});
test('tokenization preserves Vietnamese NFC and joined words, sentence splitting preserves decimals',()=>{assert.deepEqual(tokenize('Học_Sinh, CÔNG_nghệ!'),['học_sinh','công_nghệ']);assert.deepEqual(splitSentences('Giá 3.14 đồng. Câu khác!\nDòng cuối'),['Giá 3.14 đồng.','Câu khác!','Dòng cuối']);});
test('ROUGE matches the exact worked example in the PDF',()=>{
 const m=evaluate('học_sinh dùng trí_tuệ_nhân_tạo để học_tập hiệu_quả','học_sinh sử_dụng trí_tuệ_nhân_tạo để học_tập hiệu_quả hơn');
 assert.equal(m.rouge1.match,5);assert.equal(m.rouge2.match,3);assert.equal(m.rougeL.match,5);close(m.rouge1.f1,10/13);close(m.rouge2.f1,6/11);close(evaluate('','').rougeL.f1,0);
});
test('invalid and oversized inputs fail before allocation',()=>{assert.throws(()=>analyze('chỉ một câu'),/ít nhất 2/);assert.throws(()=>analyze(Array(121).fill('alpha').join('\n')),/120 câu/);});
