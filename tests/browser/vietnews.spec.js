import {test,expect} from '@playwright/test';

test('VietNews tab isolates content, compares a real article and exports provenance',async({page})=>{
 test.setTimeout(120000);const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/?tab=vietnews');await expect(page.locator('#vn-dataset-status')).toContainText('10 bài dev');
 await expect(page.locator('#demo-panel')).toBeHidden();await expect(page.locator('#advanced-panel')).toBeHidden();
 await expect(page.locator('#vn-title')).toContainText('Alibaba');await expect(page.locator('#vn-reference-preview')).toContainText('Viện kiểm sát');
 const body=await page.locator('#vn-text').inputValue();expect(body.startsWith('Ngày 22/6')).toBeTruthy();
 await expect(page.locator('#vn-trailing-option')).toBeVisible();await page.locator('#vn-exclude').check();expect(await page.locator('#vn-text').inputValue()).not.toContain('Nguyễn_Huỳnh_Tú_Trinh tại Cơ_quan');
 await page.locator('#vn-run-single').click();await expect(page.locator('.vn-results-pair article')).toHaveCount(2,{timeout:60000});
 await expect(page.locator('#vn-single-results')).toContainText('ROUGE-L F1');await expect(page.locator('.vn-results-pair')).toContainText('PhoW2V');
 const download=page.waitForEvent('download');await page.locator('#vn-export-single').click();expect((await download).suggestedFilename()).toBe('vietnews-single.json');
 await page.locator('#vn-mode-batch').click();await page.locator('#vn-batch-dev').fill('1');await page.locator('#vn-batch-test').fill('2');await page.locator('#vn-run-batch').click();
 await expect(page.locator('#vn-status')).toContainText('Hoàn tất 1 dev / 2 test',{timeout:60000});await expect(page.locator('#vn-batch-results')).toContainText('Kết quả trên 2 bài test');
 await page.locator('#tab-demo').click();await expect(page.locator('#vietnews-panel')).toBeHidden();await expect(page.locator('#input')).toBeVisible();
 await page.locator('#tab-advanced').click();await expect(page.locator('#research')).toBeVisible();await expect(page.locator('#demo-panel')).toBeHidden();expect(errors).toEqual([]);
});
test('mobile VietNews tabs and selectors fit the viewport',async({page})=>{
 await page.setViewportSize({width:390,height:844});await page.goto('/?tab=vietnews');await expect(page.locator('#vn-title')).toContainText('Alibaba');
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
 await page.locator('#vn-split').selectOption('test');await expect(page.locator('#vn-article-stat')).toContainText('Test');
 await page.locator('#tab-demo').focus();await page.keyboard.press('ArrowRight');await expect(page.locator('#tab-vietnews')).toBeFocused();
});
