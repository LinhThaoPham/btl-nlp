import {test,expect} from '@playwright/test';
test('complete interactive walkthrough and edge cases',async({page})=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.locator('#run-status')).toContainText('Đã phân tích');
 await expect(page.locator('#analysis-results')).toBeVisible();
 await expect(page.locator('#cell-inspector')).toContainText('Token');
 await page.locator('[data-cell="A"]').nth(1).click();await expect(page.locator('#cell-inspector')).toContainText('Ô (1, 2)');
 for(const tab of ['Sigma','VT','V','U']){await page.locator(`[data-svd-tab="${tab}"]`).click();await expect(page.locator(`[data-matrix="${tab}"]`)).toBeVisible();}
 await page.locator('[data-svd-tab="Sigma"]').click();await page.locator('[data-cell="Sigma"]').first().click();await expect(page.locator('#svd-cell-inspector')).toContainText('σ1');
 await page.locator('#topic-dimension').selectOption('1');await expect(page.locator('#topic-keywords')).toContainText('T2');
 await page.locator('[data-set-k="2"]').click();await expect(page.locator('#run-status')).toContainText('Đã phân tích');await expect(page.locator('#topics-content')).toContainText('k = 2');
 await page.locator('#sentence-detail').selectOption('2');await expect(page.locator('#score-detail blockquote')).toContainText('C3');
 await page.locator('#iteration').selectOption('1');await expect(page.locator('#selection-trace')).toContainText('VÒNG 2');
 await page.locator('.evaluation summary').click();await page.locator('#reference').fill('Trí tuệ nhân tạo thay đổi giáo dục.');await page.locator('#evaluate').click();await expect(page.locator('#evaluation-results')).toContainText('ROUGE-L');
 const download=page.waitForEvent('download');await page.locator('#export').click();expect((await download).suggestedFilename()).toBe('svd-report.json');
 await page.locator('#text').fill('alpha\nalpha');await page.locator('#topic-input').fill('');await page.locator('#run-button').click();await expect(page.locator('#run-status')).toContainText('Đã phân tích 2 câu');await expect(page.locator('#ranking-content')).toContainText('Ma trận A toàn 0');await expect(page.locator('.summary-list article')).toHaveCount(2);
 await page.locator('#text').fill('Một câu duy nhất');await page.locator('#run-button').click();await expect(page.locator('#form-error')).toContainText('ít nhất 2 câu');
 expect(errors).toEqual([]);
});
test('mobile layout fits viewport and selects with Gong-Liu',async({page})=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));await page.setViewportSize({width:390,height:844});await page.goto('/');await expect(page.locator('#run-status')).toContainText('Đã phân tích');
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
 await page.locator('.config>summary').click();await page.locator('#method').selectOption('gl');await expect(page.locator('#kMode')).toBeDisabled();await page.locator('#run-button').click();await expect(page.locator('#analysis-meta')).toContainText('Gong–Liu');await expect(page.locator('.summary-list article')).toHaveCount(3);expect(errors).toEqual([]);
});
