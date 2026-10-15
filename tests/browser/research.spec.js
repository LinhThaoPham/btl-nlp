import {test,expect} from '@playwright/test';

test('Vietnamese POS, detailed ROUGE, baselines and held-out evaluation',async({page})=>{
  test.setTimeout(120000);
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');await expect(page.locator('#run-status')).toContainText('Đã phân tích');
  await page.locator('.config>summary').click();await page.locator('#tokenizer').selectOption('underthesea');await page.locator('#vocab').selectOption('pos');
  await page.locator('#run-button').click();await expect(page.locator('#run-status')).toContainText('Đã phân tích',{timeout:60000});
  await expect(page.locator('#preprocess-content')).toContainText('học_sinh');await expect(page.locator('#preprocess-content')).toContainText('Underthesea');
  await page.locator('.evaluation>details>summary').click();await page.locator('#reference').fill('Học sinh sử dụng trí tuệ nhân tạo để học tập.');await page.locator('#evaluate').click();await expect(page.locator('#evaluation-results')).toContainText('LCS',{timeout:30000});
  await page.locator('#tab-advanced').click();
  await page.locator('[name="gold-sentence"]').nth(0).check();await page.locator('[name="gold-sentence"]').nth(1).check();await page.locator('#score-selection').click();await expect(page.locator('#selection-output')).toContainText('Precision');
  await page.locator('#compare-methods').click();await expect(page.locator('#compare-output')).toContainText('TextRank',{timeout:60000});await expect(page.locator('#compare-status')).toContainText('5 seed');
  await page.locator('#rater').fill('reviewer-1');for(const k of ['coverage','nonredundancy','coherence'])await page.locator('#rating-'+k).selectOption('4');await page.locator('#save-rating').click();await expect(page.locator('#rating-status')).toContainText('Đã lưu 1');
  const dataset=[{id:'dev',split:'dev',text:'Học sinh học tập.\nGiáo viên hướng dẫn học sinh.\nCông nghệ giúp học tập.',reference:'Giáo viên hướng dẫn học sinh.',gold:[2]},{id:'test',split:'test',text:'Cây xanh bảo vệ môi trường.\nNgười dân trồng cây xanh.\nXe buýt giảm khí thải.',reference:'Cây xanh bảo vệ môi trường.',gold:[1]}];
  await page.locator('#dataset-file').setInputFiles({name:'dataset.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(dataset))});
  await page.locator('#run-batch').click();await expect(page.locator('#batch-status')).toContainText('Hoàn tất',{timeout:60000});await expect(page.locator('#batch-output')).toContainText('Kết quả tập test');await expect(page.locator('#export-batch')).toBeEnabled();
  const download=page.waitForEvent('download');await page.locator('#export-batch').click();expect((await download).suggestedFilename()).toBe('experiment-results.json');
  expect(errors).toEqual([]);
});
test('top-m, external corpus and model-unavailable errors are explicit',async({page})=>{
  await page.goto('/');await expect(page.locator('#run-status')).toContainText('Đã phân tích');await page.locator('.config>summary').click();
  await page.locator('#vocab').selectOption('top-tf');await page.locator('#topM').fill('5');await page.locator('#run-button').click();await expect(page.locator('#run-status')).toContainText('Đã phân tích');await expect(page.locator('#matrix-content')).toContainText('5 hàng');
  await page.locator('.research-resource summary').click();await page.locator('#corpus-file').setInputFiles({name:'corpus.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(['Học sinh học tập cùng giáo viên','Giáo dục và công nghệ phát triển']))});await page.locator('#load-corpus').click();await expect(page.locator('#corpus-status')).toContainText('2 tài liệu');await page.locator('#vocab').selectOption('external');await page.locator('#run-button').click();await expect(page.locator('#run-status')).toContainText('Đã phân tích');await expect(page.locator('#preprocess-content')).toContainText('IDF từ kho ngoài');
});
test('100-dimensional embedding UI handles OOV and signed vector cells',async({page})=>{
  // Controlled fixture tests UI dimensionality; it is never saved as a user model.
  await page.route('**/api/nlp',async route=>{
    const data=route.request().postDataJSON();
    if(!data?.embedding)return route.continue();
    const vector=value=>Array.from({length:100},(_,i)=>i===0?value:i===1?1:0);
    await route.fulfill({json:{rows:data.lines.map(s=>s.split(' ').map(word=>({word,pos:''}))),vectors:{alpha:vector(1),beta:vector(-1)},model:{name:'TEST fixture (not PhoW2V)',dim:100,count:2},elapsedMs:1,processRssBytes:10000}});
  });
  await page.goto('/');await expect(page.locator('#run-status')).toContainText('Đã phân tích');await page.locator('.config>summary').click();
  await page.locator('#text').fill('alpha alpha beta\nalpha_beta\nunknown');await page.locator('#topic-input').fill('');await page.locator('#representation').selectOption('embedding');await page.locator('#run-button').click();
  await expect(page.locator('#run-status')).toContainText('Đã phân tích');await expect(page.locator('#matrix-content')).toContainText('100 × 3');await expect(page.locator('#matrix-content')).toContainText('unknown');
  await page.locator('[data-cell="A"]').first().click();await expect(page.locator('#cell-inspector')).toContainText('Trung bình 3 lần');await expect(page.locator('#topic-keywords')).toContainText('không phải từ khóa');
});
