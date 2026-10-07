// Final functional acceptance of the local viewer. No real actor writes.
// NODE_PATH must expose Playwright; an installed Chromium is required.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');

(async () => {
  const out = process.argv[2];
  const replay = process.argv[3];
  if (!out || fs.existsSync(out)) throw Error('Provide a fresh output directory');
  fs.mkdirSync(out, {recursive: true});
  const browser = await chromium.launch({headless: true, executablePath: process.env.FORWARD_CHROMIUM || chromium.executablePath()});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
    if (replay) {
      for (const [route, file] of [['context', 'context'], ['evaluate', 'evaluation'], ['guide', 'guidance']])
        await page.route('**/api/' + route, r => r.fulfill({contentType: 'application/json', body: fs.readFileSync(path.join(replay, file + '.json'), 'utf8')}));
    }
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto('http://127.0.0.1:43128/');
    const captured = page.waitForResponse(r => r.url().endsWith('/api/context'));
    await page.locator('#context-button').click();
    const context = await (await captured).json();
    await page.waitForFunction(() => !document.getElementById('control').disabled);
    assert.equal(await page.locator('#control option').count(), 59);
    await page.locator('#kind').selectOption('abmx');
    assert.equal(await page.locator('#control option').count(), 30);
    await page.locator('#kind').selectOption('native');
    await page.locator('#control').selectOption('native_58');
    await page.locator('#range').selectOption('installed');
    const current = Number(await page.locator('#current').inputValue());
    const bound = .77;
    await page.locator('#candidate').fill(String(bound));
    const computed = page.waitForResponse(r => r.url().endsWith('/api/evaluate'));
    await page.locator('#evaluate-button').click();
    const evaluation = await (await computed).json();
    await page.waitForFunction(() => !document.getElementById('result').hidden);
    assert.equal(evaluation.live_state_unchanged, true);
    assert.equal(evaluation.surfaces.length, context.surfaces.length);
    assert.equal(await page.locator('#surface option').count(), context.surfaces.length);
    const surface = evaluation.surfaces.find(s => s.mesh_name === 'o_head');
    const target = Math.max(...surface.delta.map(d => Math.hypot(...d))) / 2;
    assert(target > 0);
    await page.locator('#guide-surface').selectOption(surface.renderer_path);
    await page.locator('#target-max').fill(String(target));
    await page.locator('#direction').selectOption(bound > current ? '1' : '-1');
    await page.locator('#bound').fill(String(bound));
    const guided = page.waitForResponse(r => r.url().endsWith('/api/guide'), {timeout: 90000});
    await page.locator('#guide-button').click();
    const guidance = await (await guided).json();
    await page.waitForFunction(() => !document.getElementById('guidance-result').hidden);
    assert.equal(guidance.guidance.status, 'target_bracketed');
    assert.equal(guidance.evaluation.live_state_unchanged, true);
    assert.equal(Number(await page.locator('#candidate').inputValue()), guidance.guidance.candidate_value);
    assert.equal(errors.length, 0, errors.join('\n'));
    await page.screenshot({path: path.join(out, 'viewer.png'), fullPage: true});
    fs.writeFileSync(path.join(out, 'context.json'), JSON.stringify(context));
    fs.writeFileSync(path.join(out, 'evaluation.json'), JSON.stringify(evaluation));
    fs.writeFileSync(path.join(out, 'guidance.json'), JSON.stringify(guidance));
    fs.writeFileSync(path.join(out, 'review.json'), JSON.stringify({passed: true, native_controls: 59, abmx_channels: 30,
      surfaces: context.surfaces.length, guidance_status: guidance.guidance.status, errors,
      scope: replay ? 'Browser replay of preserved actual proxy receipts; no new game calls' : 'Actual local browser/proxy/isolated source-method calculations; no actor writes and no new live parity samples'}, null, 2));
    console.log('Viewer: 59 controls, 30 ABMX channels, surfaces, three projections and actual bounded source guide passed');
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
