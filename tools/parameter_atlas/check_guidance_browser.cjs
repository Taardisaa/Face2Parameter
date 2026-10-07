#!/usr/bin/env node
/* One headless browser pass over all packaged selectors and held-out directions. */
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { pathToFileURL } = require("node:url");
const hash = filename => crypto.createHash("sha256").update(fs.readFileSync(filename)).digest("hex");

async function main() {
  const args = {};
  for (let i = 2; i < process.argv.length; i += 2) {
    assert(["--index", "--review", "--out"].includes(process.argv[i]) && process.argv[i + 1], "Use --index --review --out absolute paths");
    assert(!args[process.argv[i]], "Duplicate argument");
    args[process.argv[i]] = process.argv[i + 1];
  }
  for (const name of ["--index", "--review", "--out"]) assert(args[name] && path.isAbsolute(args[name]), name + " must be absolute");
  const index = path.resolve(args["--index"]), reviewPath = path.resolve(args["--review"]), out = path.resolve(args["--out"]);
  assert(!fs.existsSync(out), "QA output must be fresh");
  const review = JSON.parse(fs.readFileSync(reviewPath, "utf8"));
  fs.mkdirSync(out, { recursive: true });
  const report = { schema_version: 1, passed: false, started_utc: new Date().toISOString(), index: { path: index, sha256: hash(index) }, review: { path: reviewPath, sha256: hash(reviewPath) },
    source_receipts: [__filename, "explorer_template.py", "attach_guidance_review.py"].map(name => { const filename = path.isAbsolute(name) ? name : path.join(__dirname, name); return { path: filename, sha256: hash(filename) }; }),
    entry_checks: [], direction_checks: [], screenshots: [], page_errors: [], console_errors: [], scope: "All 148 selectors and 296 sign-specific held-out UI witnesses; no new game sampling and no whole-interval validation" };
  let browser;
  try {
    if (!process.env.PLAYWRIGHT_BROWSERS_PATH && process.env.LOCALAPPDATA) process.env.PLAYWRIGHT_BROWSERS_PATH = path.join(process.env.LOCALAPPDATA, "ms-playwright");
    const bundled = "C:/Users/13666/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright";
    const moduleName = process.env.PLAYWRIGHT_MODULE || (fs.existsSync(bundled) ? bundled : "playwright");
    const { chromium } = require(moduleName);
    let executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;
    if (!executablePath && !fs.existsSync(chromium.executablePath())) {
      const cache = process.env.PLAYWRIGHT_BROWSERS_PATH;
      executablePath = fs.readdirSync(cache).filter(name => /^chromium_headless_shell-\d+$/.test(name))
        .sort((a, b) => Number(b.split("-").pop()) - Number(a.split("-").pop()))
        .map(name => path.join(cache, name, "chrome-headless-shell-win64", "chrome-headless-shell.exe")).find(fs.existsSync);
    }
    browser = await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
    report.runtime = { node: process.version, chromium: browser.version(), module: moduleName, executable: executablePath, headless: true };
    const page = await browser.newPage({ viewport: { width: 1540, height: 1000 }, deviceScaleFactor: 1 });
    page.on("pageerror", error => report.page_errors.push(String(error)));
    page.on("console", message => { if (message.type() === "error") report.console_errors.push(message.text()); });
    await page.goto(pathToFileURL(index).href, { waitUntil: "load" });
    await page.waitForFunction(() => typeof payload !== "undefined" && payload !== null);
    const catalog = await page.evaluate(() => globalThis.eval("catalog"));
    assert.deepEqual(catalog, JSON.parse(fs.readFileSync(path.join(path.dirname(index), "catalog.json"), "utf8")));
    assert.equal(catalog.entries.length, 148);
    assert.equal(review.entries.length, 296);
    assert.equal(review.skipped.length, 0);
    assert.equal(catalog.provenance.guidance_overlay.review_receipt.sha256, hash(reviewPath));
    const reviewed = new Map(review.entries.map(row => [row.entry_id + "/" + row.sign, row]));
    assert.equal(reviewed.size, 296);
    for (const entry of catalog.entries) {
      assert(/^data\/[A-Za-z0-9_.-]+\.js$/.test(entry.file));
      assert.equal(hash(path.join(path.dirname(index), entry.file)), entry.sha256, "Payload hash mismatch");
      await page.locator("#baseline").selectOption(entry.baseline_name);
      await page.locator("#kind").selectOption(entry.kind);
      await page.locator("#control").selectOption(entry.id);
      await page.waitForFunction(id => payload?.id === id && !document.getElementById("content").hidden, entry.id);
      const state = await page.evaluate(() => ({ id: payload.id, kind: payload.kind, baseline: payload.baseline_name, estimate_verified: payload.local_response.estimate_verified,
        samples: payload.deltas.length, options: document.getElementById("sample").options.length, witnesses: payload.local_response.heldout_validation_by_sign,
        canvases: ["xy", "zy", "xz"].map(id => { const c = document.getElementById(id), pixels = c.getContext("2d").getImageData(0, 0, c.width, c.height).data; let occupied = 0; for (let i = 3; i < pixels.length; i += 4) if (pixels[i]) occupied++; return { id, occupied }; }) }));
      assert.equal(state.id, entry.id);
      assert.equal(state.kind, entry.kind);
      assert.equal(state.baseline, entry.baseline_name);
      assert.equal(state.estimate_verified, false, "Global estimates must never be promoted");
      assert.equal(state.samples, state.options);
      assert.equal(state.samples, entry.kind === "native" && entry.baseline_name === "card_input" ? 6 : 2);
      assert(state.canvases.every(c => c.occupied > 500));
      report.entry_checks.push({ id: entry.id, kind: entry.kind, baseline: entry.baseline_name, canvases: state.canvases, sample_count: state.samples });
      await page.locator("#target").fill("0.1");
      for (const sign of [-1, 1]) {
        const row = reviewed.get(entry.id + "/" + sign);
        assert(row && row.input_trusted === true);
        assert.deepEqual(state.witnesses[String(sign)], row, "Attached witness differs from independent review");
        for (const field of ["heldout_value", "signed_step", "actual_max", "target_units", "max_vector_error", "relative_target_error", "error_budget"]) assert(Number.isFinite(row[field]), "Nonfinite witness metric");
        assert.equal(Math.sign(row.signed_step), sign);
        await page.locator("#sign").selectOption(String(sign));
        const sampleIndex = await page.evaluate(direction => { const role = direction < 0 ? "local_minus" : "local_plus"; const nativeIndex = payload.roles.indexOf(role); return nativeIndex >= 0 ? nativeIndex : payload.levels.findIndex(value => Math.sign(value - payload.baseline_level) === direction); }, sign);
        assert(sampleIndex >= 0);
        await page.locator("#sample").selectOption(String(sampleIndex));
        const ui = await page.evaluate(() => ({ status: document.getElementById("heldout-status").textContent, error: document.getElementById("heldout-status").classList.contains("error"),
          suggestion: document.getElementById("suggestion").textContent, metric_count: document.getElementById("heldout-metrics").children.length,
          receipt: JSON.parse(document.getElementById("heldout-receipt").textContent), invalid: /(?:NaN|Infinity)/.test(document.body.innerText) }));
        assert.deepEqual(ui.receipt, row);
        assert.equal(ui.error, !row.verified_prediction);
        assert(ui.status.includes(row.verified_prediction ? "具体点预测通过" : "预测未通过"));
        assert(ui.status.includes("实际测试值") && ui.status.includes("仅适用于此点"));
        assert(ui.suggestion.includes("任意目标的估算尚未由独立采样验证"));
        assert.equal(ui.metric_count, 6);
        assert(!ui.invalid);
        report.direction_checks.push({ entry_id: entry.id, sign, verified_prediction: row.verified_prediction, heldout_value: row.heldout_value, ui_status: ui.status });
        if ((entry.kind === "native" && entry.control === 57 && entry.baseline_name === "card_input" && sign === -1) || (entry.kind === "abmx" && entry.bone === "cf_J_Eye_s_L" && entry.channel === "rotation" && entry.axis === 0 && sign === 1)) {
          const filename = path.join(out, entry.kind === "native" ? "ear57_failed_minus.png" : "abmx_verified_point.png");
          await page.screenshot({ path: filename, fullPage: true });
          report.screenshots.push({ entry_id: entry.id, sign, verified_prediction: row.verified_prediction, path: filename, sha256: hash(filename) });
        }
      }
      if (report.entry_checks.length % 30 === 0) process.stdout.write(`Checked ${report.entry_checks.length}/148 entries, ${report.direction_checks.length}/296 directions\n`);
    }
    const failed = report.direction_checks.filter(row => !row.verified_prediction);
    assert.equal(failed.length, 9, "Preserve all nine failed predictions");
    assert.equal(report.screenshots.length, 2);
    assert.equal(await page.locator("#failures li").count(), 3);
    assert(await page.locator("#failure-details").evaluate(e => e.open));
    assert.equal(report.page_errors.length, 0);
    assert.equal(report.console_errors.length, 0);
    report.prediction_pass_count = 287;
    report.prediction_fail_count = failed.length;
    report.failed_directions = failed;
    report.original_eye_failures_preserved = 3;
    report.passed = true;
  } catch (error) {
    report.error = { message: String(error), stack: error.stack };
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
    report.finished_utc = new Date().toISOString();
    const reportPath = path.join(out, "guidance_browser_report.json");
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n", "utf8");
    process.stdout.write(JSON.stringify({ passed: report.passed, entries: report.entry_checks.length, directions: report.direction_checks.length, report: reportPath, error: report.error?.message }) + "\n");
  }
}
main().catch(error => { process.stderr.write(String(error) + "\n"); process.exitCode = 1; });
