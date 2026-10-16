import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parseVietNews,scoreVietNews,vietnewsOptions} from '../src/vietnews-engine.js';
import {validateDataset} from '../src/research.js';
import {splitSentences} from '../src/engine.js';

test('anonymized names do not leave detached VietNews sentence fragments',()=>{
 const lines=splitSentences('Tiến và chị Phạm_Thị_Huyền_Tr .\n( SN 1997 ) kết_hôn .\nTiến đánh chị Tr .\ntới_tấp .\nChị Tr .\nđã tử_vong .');
 assert.equal(lines.length,3);
 assert.ok(lines[1].includes('Tr . tới_tấp'));
 assert.ok(lines[2].includes('Tr . đã tử_vong'));
 assert.equal(splitSentences('Ông Bảo_Th . , ngụ Hải_Châu .\nÔng Lê_Quốc_Th .\n400 triệu đồng .').length,2);
});

test('VietNews parser excludes title/reference and final caption block by default',()=>{
 const r=parseVietNews('Tiêu đề\n\nTóm tắt riêng.\n\nalpha beta.\nGamma delta.\n\nChú thích ảnh.',{id:'x',split:'dev'});
 assert.equal(r.title,'Tiêu đề');assert.equal(r.reference,'Tóm tắt riêng.');assert.ok(!r.text.includes('Tóm tắt riêng'));assert.ok(!r.text.includes('Chú thích ảnh.'));assert.equal(r.excludeTrailing,true);assert.equal(r.text,'alpha beta.\nGamma delta.');
 assert.throws(()=>parseVietNews('Chỉ một khối'),/3 khối/);
});
test('shipped subset is pinned, has disjoint dev/test records and no reference leakage',()=>{
 const data=JSON.parse(readFileSync(new URL('../public/data/vietnews.json',import.meta.url),'utf8'));
 assert.match(data.commit,/^[a-f0-9]{40}$/);const rows=data.items.map(r=>parseVietNews(r.raw,r));
 assert.equal(rows.filter(r=>r.split==='dev').length,10);assert.equal(rows.filter(r=>r.split==='test').length,20);validateDataset(rows);
 for(const r of rows){assert.ok(r.source.includes(data.commit));assert.notEqual(r.text,r.reference);assert.ok(r.sha256);}
});
test('reference is used for scoring only; changing it does not alter selected sentences',()=>{
 const r={text:'alpha beta\nbeta gamma\ndelta epsilon',reference:'alpha beta'};
 const options=vietnewsOptions(2),a=scoreVietNews(r,{},options),b=scoreVietNews({...r,reference:'delta epsilon'}, {},options);
 assert.deepEqual(a.selected,b.selected);assert.equal(a.selected.length,2);assert.equal(a.options.topicWeight,0);
});
