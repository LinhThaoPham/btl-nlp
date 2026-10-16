import {esc,fmt,pct,matrixTable} from './views.js';
import {selectionMetrics} from './research.js';
import {evaluationRows,nlp} from './nlp-client.js';
import {parseVietNews} from './vietnews-engine.js';

const $=id=>document.getElementById(id);
let external=null,current=null,comparison=null,revision=0,vietnewsRecords=[];
let worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'}),serial=0;
const pending=new Map();
worker.onmessage=({data})=>{const task=pending.get(data.id);if(!task)return;if(data.progress){task.progress?.(data.progress);return;}pending.delete(data.id);data.error?task.reject(new Error(data.error)):task.resolve(data.result);};
worker.onerror=()=>{for(const task of pending.values())task.reject(new Error('Mô-đun thực nghiệm gặp lỗi. Tải lại trang.'));pending.clear();};
const request=(data,progress)=>new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject,progress});worker.postMessage({...data,id});});
export const getExternal=()=>external;
export const download=(name,data)=>{const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);};
const table=(heads,rows)=>`<div class="table-scroll"><table class="data-table"><thead><tr>${heads.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
const score=x=>x==null?'—':fmt(x,3);
function gold(){return [...document.querySelectorAll('[name="gold-sentence"]:checked')].map(x=>Number(x.value));}
function setMessage(id,text,error=false){$(id).textContent=text;$(id).classList.toggle('research-error',error);}
async function busy(button,status,fn){button.disabled=true;setMessage(status,'Đang xử lý…');try{await fn();}catch(error){setMessage(status,error.message,true);}finally{button.disabled=false;}}

export function mountResearch({config,dirty,run}){
  const grid=document.querySelector('.config-grid');
  grid.insertAdjacentHTML('afterbegin',`<label>Biểu diễn đầu vào<select id="representation"><option value="terms">Ma trận từ–câu</option><option value="embedding">Word2Vec / PhoW2V</option></select></label><label>Tách từ<select id="tokenizer"><option value="basic">Cơ bản · âm tiết / token có sẵn</option><option value="underthesea">Underthesea · từ tiếng Việt + POS</option></select></label><label>Giới hạn top-m<input id="topM" type="number" min="1" max="1200" value="200"></label>`);
  const chooser=document.createElement('div');chooser.className='demo-chooser';
  chooser.innerHTML='<h3>Demo tóm tắt văn bản bằng SVD</h3><div class="demo-selects"></div><p id="demo-description"></p>';
  document.querySelector('.example-row').before(chooser);
  chooser.querySelector('.demo-selects').append($('representation').closest('label'),$('tokenizer').closest('label'));
  $('representation').previousSibling.textContent='Chọn phương pháp';
  $('representation').options[0].textContent='TF-IDF → SVD';
  $('representation').options[1].textContent='Word2Vec / PhoW2V → SVD';
  function describe(){const embedding=$('representation').value==='embedding';$('demo-description').textContent=embedding?'Word2Vec/PhoW2V: tách từ → tra vector từ → trung bình vector câu → SVD → chọn câu.':'TF-IDF: đếm từ → trọng số TF-IDF → SVD → chọn câu.';}
  function choose(embedding){$('representation').value=embedding?'embedding':'terms';$('tokenizer').value=embedding?'underthesea':'basic';$('vocab').value='all';if(!embedding)$('weight').value='tfidf';const url=new URL(location.href);url.searchParams.set('demo',embedding?'word2vec':'tfidf');history.replaceState(null,'',url);$('representation').dispatchEvent(new Event('input',{bubbles:true}));$('representation').dispatchEvent(new Event('change',{bubbles:true}));describe();run(true);}
  $('representation').addEventListener('change',event=>{if(event.isTrusted)choose($('representation').value==='embedding');});
  $('representation').addEventListener('change',describe);
  if(new URLSearchParams(location.search).get('demo')==='word2vec'){$('representation').value='embedding';$('tokenizer').value='underthesea';}
  describe();
  $('vocab').insertAdjacentHTML('beforeend','<option value="pos">Danh từ, động từ, tính từ (POS)</option><option value="top-tf">Top-m theo tổng TF</option><option value="top-tfidf">Top-m theo tổng TF-IDF</option><option value="external">Từ vựng + IDF từ kho ngoài</option>');
  document.querySelector('.config').insertAdjacentHTML('beforeend',`<div class="research-resource"><details><summary>Mô hình ngôn ngữ và kho ngữ liệu ngoài</summary><p>Underthesea và mô hình nhúng chạy trên máy này. Chọn mô hình Word2Vec đã giải nén; PhoW2V 100 chiều tạo ma trận 100 × số câu.</p><p><a href="https://github.com/datquocnguyen/PhoW2V" target="_blank" rel="noopener">Nguồn PhoW2V và điều kiện sử dụng ↗</a></p><div class="research-controls"><label>Tệp mô hình<input id="model-file" type="file" accept=".bin,.vec,.txt"></label><button type="button" class="secondary" id="upload-model">Nạp mô hình</button><label>Mô hình đã có<select id="model-select"><option value="">Chưa tải danh sách</option></select></label><button type="button" class="secondary" id="load-model">Dùng mô hình</button><button type="button" class="secondary" id="refresh-model">Làm mới</button></div><p id="model-status" role="status"></p><hr><label>Kho ngoài · JSON mảng các văn bản<input type="file" id="corpus-file" accept=".json"></label><p>Mỗi phần tử là một tài liệu. df đếm số tài liệu chứa từ; IDF = ln(N/df). Dùng một kho độc lập với tập dev/test.</p><button type="button" class="secondary" id="load-corpus">Dựng từ vựng và IDF ngoài</button><p id="corpus-status" role="status">Chưa có kho ngoài.</p></details></div>`);
  document.querySelector('.page-footer').insertAdjacentHTML('beforebegin',`<section id="research" class="learning-section"><div class="section-heading"><span class="step-number">07</span><div><span class="overline">SO SÁNH & THỰC NGHIỆM</span><h2>Đo chất lượng của bản tóm tắt</h2><p>So sánh trên cùng văn bản và số câu, giải thích điểm số, chọn tham số trên tập phát triển.</p></div></div>
    <div class="panel"><h3>Đồng chọn câu</h3><p>Chọn các câu tham chiếu trích rút. P/R/F1 ở đây đo trùng chỉ số câu.</p><div id="gold-sentences" class="gold-list"></div><button type="button" class="secondary" id="score-selection">Tính P/R/F1 theo câu</button><div id="selection-output" aria-live="polite"></div></div>
    <div class="panel"><h3>So sánh các phương pháp</h3><p>So sánh ba cách trích rút: LEAD, TF-IDF và Word2Vec/PhoW2V.</p><div class="research-controls"><label>Bài VietNews<select id="research-vietnews"><option value="">Chọn bài có tham chiếu ROUGE</option></select></label></div><p id="research-vietnews-status" class="note">Chọn bài để tự nạp nội dung và tham chiếu.</p><details><summary>Thiết lập so sánh</summary><p>Cùng văn bản và số câu. Word2Vec/PhoW2V dùng Underthesea để tách từ; cần nạp mô hình ở mục mô hình ngôn ngữ.</p></details><button type="button" class="primary" id="compare-methods">So sánh trên văn bản hiện tại</button><p id="compare-status" role="status"></p><div id="compare-output"></div></div>
</section>`);
  document.querySelector('nav').insertAdjacentHTML('beforeend','<a href="#research" class="step-link"><span>07</span>So sánh & thực nghiệm</a>');
  async function refresh(){const response=await fetch('/api/nlp');const data=await response.json();if(data.error)throw new Error(data.error);$('model-select').innerHTML='<option value="">Chọn mô hình</option>'+data.models.map(n=>`<option value="${esc(n)}">${esc(n)}</option>`).join('');setMessage('model-status',`Underthesea ${data.underthesea||'chưa cài'} · ${data.model?`${data.model.name}, ${data.model.dim} chiều, ${data.model.count.toLocaleString()} từ`:'Chưa nạp vector.'}`);return data;}
  $('refresh-model').onclick=()=>busy($('refresh-model'),'model-status',refresh);
  $('upload-model').onclick=()=>busy($('upload-model'),'model-status',async()=>{const file=$('model-file').files[0];if(!file)throw new Error('Chọn tệp mô hình trước.');setMessage('model-status','Đang đọc và lập chỉ mục mô hình. Tệp lớn có thể mất vài phút…');const response=await fetch('/api/nlp/upload?name='+encodeURIComponent(file.name),{method:'POST',body:file});const data=await response.json();if(data.error)throw new Error(data.error);await nlp({action:'load',name:data.name});await refresh();dirty();});
  $('load-model').onclick=()=>busy($('load-model'),'model-status',async()=>{if(!$('model-select').value)throw new Error('Chọn mô hình trong danh sách.');await nlp({action:'load',name:$('model-select').value});await refresh();dirty();});
  $('load-corpus').onclick=()=>busy($('load-corpus'),'corpus-status',async()=>{
    const file=$('corpus-file').files[0];if(!file||file.size>2_000_000)throw new Error('Chọn JSON kho ngoài dưới 2 MB.');const docs=JSON.parse(await file.text());if(!Array.isArray(docs)||docs.length<2||docs.length>500||docs.some(s=>typeof s!=='string'||!s.trim()))throw new Error('Kho cần 2–500 văn bản không rỗng.');
    const tokenizer=config().tokenizer,rows=await evaluationRows(docs,tokenizer),df=Object.create(null);for(const row of rows)for(const token of new Set(row))df[token]=(df[token]||0)+1;
    external={name:file.name,documentCount:docs.length,df,tokenizer};dirty();setMessage('corpus-status',`${docs.length} tài liệu · ${Object.keys(df).length} từ · ${tokenizer}. Kho giữ trong phiên.`);
  });
  $('score-selection').onclick=()=>{if(!current)return;const chosen=gold();if(!chosen.length){$('selection-output').textContent='Chọn ít nhất một câu tham chiếu.';return;}const m=selectionMetrics(current.selected,chosen,current.lines.length);$('selection-output').innerHTML=table(['Số câu trùng','Precision','Recall','F1'],[[m.match,score(m.precision),score(m.recall),score(m.f1)]]);};
  $('research-vietnews').onchange=()=>{
    const record=vietnewsRecords.find(item=>item.id===$('research-vietnews').value);
    if(!record){$('research-vietnews-status').textContent='Chọn bài để tự nạp nội dung và tham chiếu.';return;}
    $('text').value=record.text;$('reference').value=record.reference;
    // Keep main.js's referenceText in sync; renderRanking rebuilds this field after analysis.
    $('reference').dispatchEvent(new Event('input',{bubbles:true}));
    for(const box of document.querySelectorAll('[name="gold-sentence"]'))box.checked=false;
    $('research-vietnews-status').textContent=`Đã nạp “${record.title}” cùng tham chiếu. Đang phân tích; xong bạn có thể so sánh phương pháp.`;
    $('text').dispatchEvent(new Event('input',{bubbles:true}));$('text').scrollIntoView({behavior:'smooth',block:'center'});
    run(true);
  };
  $('compare-methods').onclick=()=>busy($('compare-methods'),'compare-status',async()=>{
    if(!current)throw new Error('Phân tích văn bản trước.');const version=revision,r=current,reference=$('reference').value;
    const data=await request({action:'compare',text:r.lines.join('\n'),options:r.cfg,external,reference,gold:gold().length?gold():null});if(version!==revision)throw new Error('Đầu vào đã đổi. Hãy phân tích và so sánh lại.');comparison=data;
    const referencePanel=reference.trim()
      ?`<div class="vn-reference compare-reference"><span class="eyebrow">BẢN THAM CHIẾU CÓ SẴN</span><h4>Bản tóm tắt chuẩn để đối chiếu</h4><p class="compare-reference-text">${esc(reference.trim())}</p><small>Đây là bản dùng để tính ROUGE; các bản bên dưới do từng phương pháp trích từ văn bản tạo ra.</small></div>`
      :'<p class="note">Chưa có bản tóm tắt tham chiếu. Hãy chọn bài VietNews hoặc nhập tham chiếu để tính ROUGE.</p>';
    $('compare-output').innerHTML=referencePanel+table(['Phương pháp','Câu chọn','ROUGE-1 F1','ROUGE-2 F1','ROUGE-L F1','F1 theo câu','Tính toán (ms)'],data.rows.map(row=>[esc(row.method),row.selected.map(i=>'C'+(i+1)).join(', '),score(row.rouge?.rouge1.f1),score(row.rouge?.rouge2.f1),score(row.rouge?.rougeL.f1),score(row.selection?.f1),fmt(row.elapsedMs,2)]))+data.rows.map(row=>`<details><summary>${esc(row.method)} · xem bản tóm tắt</summary><p>${esc(row.summary)}</p></details>`).join('');
    setMessage('compare-status',data.rows.some(row=>row.rouge)?'Đã so sánh LEAD, TF-IDF và Word2Vec/PhoW2V theo cùng bản tham chiếu.':'Đã so sánh. Nhập tham chiếu ROUGE hoặc chọn câu tham chiếu để có điểm chất lượng.');
  });
  refresh().catch(e=>setMessage('model-status',e.message,true));
  fetch('/data/vietnews.json').then(response=>{if(!response.ok)throw new Error('Không tải được dữ liệu VietNews.');return response.json();}).then(data=>{
    vietnewsRecords=data.items.map(item=>parseVietNews(item.raw,item));
    $('research-vietnews').innerHTML='<option value="">Chọn bài có tham chiếu ROUGE</option>'+vietnewsRecords.map(item=>`<option value="${esc(item.id)}">${esc(item.id)} · ${esc(item.title)}</option>`).join('');
    $('research-vietnews-status').textContent=`Có sẵn ${vietnewsRecords.length} bài VietNews kèm tham chiếu. Chọn bài để điền dữ liệu vào demo.`;
  }).catch(error=>setMessage('research-vietnews-status',error.message,true));
}

export function invalidateResearch(){
  revision++;current=null;comparison=null;
  for(const id of ['compare-methods','score-selection'])if($(id))$(id).disabled=true;
  // Never leave summaries from a previous input beside a newly selected article.
  if($('compare-output'))$('compare-output').replaceChildren();
  if($('selection-output'))$('selection-output').replaceChildren();
  if($('compare-status'))setMessage('compare-status','Đầu vào đã thay đổi; kết quả cũ đã được xóa. Phân tích lại trước khi so sánh.');
}
export function updateResearch(r,elapsed){
  revision++;current=r;comparison=null;
  for(const id of ['compare-methods','score-selection'])$(id).disabled=false;
  $('gold-sentences').innerHTML=r.lines.map((line,i)=>`<label><input name="gold-sentence" type="checkbox" value="${i}"><b>C${i+1}</b> ${esc(line)}</label>`).join('');
  $('selection-output').innerHTML='';$('compare-output').innerHTML='';setMessage('compare-status','Sẵn sàng so sánh kết quả mới.');
  const selectedRecord=vietnewsRecords.find(item=>item.id===$('research-vietnews')?.value);
  if(selectedRecord&&$('text').value===selectedRecord.text)setMessage('research-vietnews-status',`Đã phân tích xong “${selectedRecord.title}”. Bấm “So sánh trên văn bản hiện tại” để xem kết quả.`);
  const info=r.resourcesInfo||{};
  $('analysis-meta').insertAdjacentHTML('beforeend',`<span class="performance-note"> · ${fmt(elapsed,0)} ms tổng · mảng số ≈ ${fmt((info.numericBytes||0)/1024,1)} KiB${info.processRssBytes?` · Python RSS ${fmt(info.processRssBytes/1024**2,1)} MiB`:''}</span>`);
  $('preprocess-content').insertAdjacentHTML('beforeend',`<p class="note">Thời gian gồm chuẩn bị NLP và tính SVD, chưa gồm vẽ giao diện. Bộ nhớ mảng số là ước lượng 8 byte/phần tử của các ma trận đang giữ, chưa gồm overhead JavaScript. Python RSS là bộ nhớ tiến trình NLP tại thời điểm đo, có cả mô hình dùng chung; không phải bộ nhớ riêng của văn bản.</p>`);
  if(r.cfg.tokenizer==='underthesea')$('preprocess-content').insertAdjacentHTML('beforeend',`<details class="panel"><summary>Xem nhãn từ loại Underthesea</summary>${table(['Câu','Từ và nhãn POS'],r.tagged.map((row,j)=>['C'+(j+1),row.map(t=>`${esc(t.word)} <small>(${esc(t.pos)})</small>`).join(' · ')]))}</details>`);
}
export async function explainRouge(candidate,reference,tokenizer){
  const version=revision,data=await request({action:'rouge',candidate,reference,tokenizer});if(version!==revision)throw new Error('Kết quả đã thay đổi, hãy tính ROUGE lại.');
  return table(['Độ đo','Số khớp','Precision','Recall','F1'],Object.entries(data.metrics).map(([name,m])=>[name==='rougeL'?'ROUGE-L':name==='rouge1'?'ROUGE-1':'ROUGE-2',m.match,score(m.precision),score(m.recall),score(m.f1)]))+`<p>Tokenizer: ${esc(tokenizer)}. Giữ stopword, nối câu; ROUGE-L trên cả chuỗi. Ứng viên ${data.candidate.length} token, tham chiếu ${data.reference.length} token.</p>`+[['Token đơn',data.unigrams],['Bigram',data.bigrams]].map(([label,rows])=>`<details><summary>${label}: cách đếm số khớp</summary>${table(['Token / n-gram','Ứng viên','Tham chiếu','min(hai số đếm)'],rows.map(r=>[esc(r.token),r.candidate,r.reference,r.match]))}</details>`).join('')+`<details open><summary>LCS · ${data.alignment.length} token</summary><div class="token-chips">${data.alignment.map(x=>`<span>${esc(x.token)} [${x.candidate+1}↔${x.reference+1}]</span>`).join('')||'Không có token chung.'}</div><p>Chỉ số là vị trí token (bắt đầu từ 1) ở ứng viên ↔ tham chiếu. LCS giữ thứ tự, có thể bỏ qua token ở giữa.</p><p>Ứng viên: ${esc(data.candidate.join(' | '))}</p><p>Tham chiếu: ${esc(data.reference.join(' | '))}</p></details>`;
}

export function embeddingMatrix(r,register){
  const e=r.embedding,cols=r.lines.map((_,i)=>'C'+(i+1));
  return `<div class="panel"><h3>Ma trận nhúng ${e.dim} × ${r.lines.length}</h3><p>Mô hình: ${esc(e.model.name)}. Mỗi hàng Eᵢ là một chiều nhúng, mỗi cột là trung bình vector của các lần xuất hiện từ có vector. Trọng số TF-IDF không áp dụng cho ma trận nhúng.</p><p>Tra trực tiếp; nếu từ ghép chưa có, lấy trung bình các thành phần tra được. Từ hoàn toàn không có vector bị bỏ khỏi mẫu số; câu không có vector giữ cột 0. Độ phủ: ${e.used}/${e.total} lần xuất hiện (${pct(e.total?e.used/e.total:0)}).</p>${table(['Câu','Token giữ','Trực tiếp','Ghép từ thành phần','Không có vector'],e.sentences.map((s,i)=>['C'+(i+1),s.total,s.direct,s.fallback,esc(s.missing.join(', '))||'—']))}${register('A',r.A,r.vocab,cols,'Aᴱ — chiều nhúng × câu')}<div class="cell-inspector" id="cell-inspector">Bấm ô để xem trung bình vector và chuẩn hóa.</div><details><summary>Vector câu trước chuẩn hóa L2</summary>${register('rawA',r.rawA,r.vocab,cols,'Aᴱ gốc')}</details><details><summary>Tra cứu từ và xử lý từ ngoài từ điển</summary>${table(['Từ','Cách tra','Thành phần dùng'],Object.entries(e.detail).map(([word,d])=>[esc(word),d.route==='direct'?'Trực tiếp':d.route==='parts'?'Trung bình thành phần':'Không có vector',esc(d.parts.join(', '))||'—']))}</details></div>`;
}
export function embeddingCell(r,i,j){const s=r.embedding.sentences[j],terms=s.words.filter(w=>r.embedding.detail[w].vector).map(w=>`${esc(w)}: ${fmt(r.embedding.detail[w].vector[i],6)}`);return `<h4>Chiều E${i+1} · câu C${j+1}</h4><p>${terms.join(' + ')||'Không có vector'}.</p><p>Trung bình ${s.used} lần xuất hiện = ${fmt(r.rawA[i][j],7)}. Chuẩn cột = ${fmt(r.rawNorms[j],7)}. ${r.cfg.normalize?'Sau chuẩn hóa L2':'Không chuẩn hóa'}: ${fmt(r.A[i][j],7)}.</p>`;}
