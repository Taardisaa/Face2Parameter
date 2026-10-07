#!/usr/bin/env node
/* Reproducible headless Chromium QA of the generated local-file explorer. */
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { pathToFileURL } = require("node:url");

function argumentsFrom(argv) {
  const result = {};
  for (let i = 0; i < argv.length; i += 2) {
    assert(["--index", "--out"].includes(argv[i]) && argv[i + 1],
      "Usage: node check_explorer_browser.cjs --index ABSOLUTE/index.html --out FRESH_DIRECTORY");
    assert(!result[argv[i]], "Duplicate CLI argument");
    result[argv[i]] = argv[i + 1];
  }
  assert(result["--index"] && result["--out"], "Both --index and --out are required");
  assert(path.isAbsolute(result["--index"]) && path.isAbsolute(result["--out"]), "Paths must be absolute");
  return { index: path.resolve(result["--index"]), out: path.resolve(result["--out"]) };
}

function fingerprint(filename) {
  const data = fs.readFileSync(filename);
  return { path: filename, bytes: data.length, sha256: crypto.createHash("sha256").update(data).digest("hex") };
}

async function main() {
  const { index, out } = argumentsFrom(process.argv.slice(2));
  assert(fs.statSync(index).isFile(), "Index must already exist; do not run during export");
  assert(!fs.existsSync(out), "Report directory must be fresh");
  fs.mkdirSync(out, { recursive: true });
  const report = {
    schema_version: 1, started_utc: new Date().toISOString(), passed: false,
    input: fingerprint(index), runtime: { node: process.version, platform: process.platform },
    source_fingerprints: [fingerprint(__filename)], data_fingerprints: [],
    entries_checked: [], sample_checks: [], screenshots: [], page_errors: [], console_errors: [],
    scope: "Browser rendering and UI behavior for every packaged entry; does not establish game geometry accuracy or held-out step validation",
  };
  let browser;
  const reportFile = path.join(out, "browser_report.json");
  try {
    if (!process.env.PLAYWRIGHT_BROWSERS_PATH && process.env.LOCALAPPDATA) {
      process.env.PLAYWRIGHT_BROWSERS_PATH = path.join(process.env.LOCALAPPDATA, "ms-playwright");
    }
    const bundled = "C:/Users/13666/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright";
    const moduleName = process.env.PLAYWRIGHT_MODULE || (fs.existsSync(bundled) ? bundled : "playwright");
    report.runtime.playwright_module = moduleName;
    const { chromium } = require(moduleName);
    let executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;
    if (!executablePath && !fs.existsSync(chromium.executablePath())) {
      const cache = process.env.PLAYWRIGHT_BROWSERS_PATH;
      const candidates = fs.existsSync(cache || "") ? fs.readdirSync(cache)
        .filter(name => /^chromium_headless_shell-\d+$/.test(name))
        .sort((a, b) => Number(b.split("-").pop()) - Number(a.split("-").pop()))
        .map(name => path.join(cache, name, "chrome-headless-shell-win64", "chrome-headless-shell.exe"))
        .filter(filename => fs.existsSync(filename)) : [];
      executablePath = candidates[0];
    }
    if (executablePath) report.runtime.chromium_executable = fingerprint(executablePath);
    browser = await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
    report.runtime.chromium = browser.version();
    const page = await browser.newPage({ viewport: { width: 1540, height: 1000 }, deviceScaleFactor: 1 });
    page.on("pageerror", e => report.page_errors.push(String(e)));
    page.on("console", message => { if (message.type() === "error") report.console_errors.push(message.text()); });
    await page.goto(pathToFileURL(index).href, { waitUntil: "load" });
    await page.waitForFunction(() => typeof catalog !== "undefined" && typeof payload !== "undefined" && payload !== null);
    const inventory = await page.evaluate(() => ({ entries: catalog.entries, failures: catalog.failures, baselines: catalog.baselines }));
    assert.equal(inventory.entries.length, 148, "Expected 59 native × 2 baselines plus 30 selected ABMX channels");
    assert.equal(inventory.entries.filter(e => e.kind === "native" && e.baseline_name === "card_input").length, 59);
    assert.equal(inventory.entries.filter(e => e.kind === "native" && e.baseline_name === "mixed_input").length, 59);
    assert.equal(inventory.entries.filter(e => e.kind === "abmx" && e.baseline_name === "card_input").length, 30);
    assert.equal(inventory.entries.filter(e => e.kind === "abmx" && e.baseline_name === "mixed_input").length, 0);
    assert.equal(new Set(inventory.entries.map(e => e.id)).size, 148, "Entry IDs must be unique");
    report.catalog = { entry_count: inventory.entries.length, failures: inventory.failures, baselines: Object.keys(inventory.baselines) };
    const catalogFile = path.join(path.dirname(index), "catalog.json");
    report.source_fingerprints.push(fingerprint(catalogFile));
    const onDiskCatalog = JSON.parse(fs.readFileSync(catalogFile, "utf8"));
    assert.deepEqual(await page.evaluate(() => catalog), onDiskCatalog, "Embedded catalog differs from packaged catalog.json");
    report.catalog_matches_embedded = true;
    report.template_sha256 = fingerprint(path.join(__dirname, "explorer_template.py")).sha256;
    for (const filename of ["explorer_template.py", "explore_live_response.py"]) {
      report.source_fingerprints.push(fingerprint(path.join(__dirname, filename)));
    }
    for (const entry of inventory.entries) {
      assert(/^data\/[A-Za-z0-9_.-]+\.js$/.test(entry.file), "Unsafe package data path");
      const record = fingerprint(path.join(path.dirname(index), entry.file));
      assert.equal(record.sha256, entry.sha256, "Data file hash differs from embedded catalog: " + entry.id);
      report.data_fingerprints.push({ id: entry.id, ...record });
    }

    async function selectEntry(entry) {
      await page.locator("#baseline").selectOption(entry.baseline_name);
      await page.locator("#kind").selectOption(entry.kind);
      await page.locator("#control").selectOption(entry.id);
      await page.waitForFunction(id => payload?.id === id && !document.getElementById("content").hidden, entry.id);
      const actual = await page.evaluate(() => ({
        id: payload.id, kind: payload.kind, baseline: payload.baseline_name,
        samples: payload.deltas.length, sample_options: document.getElementById("sample").options.length,
        roles: payload.roles, levels: payload.levels,
        status: document.getElementById("status").textContent,
        status_error: document.getElementById("status").classList.contains("error"),
        canvases: ["xy", "zy", "xz"].map(id => {
          const c = document.getElementById(id), pixels = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
          let occupied = 0;
          for (let i = 3; i < pixels.length; i += 4) if (pixels[i]) occupied++;
          return { id, width: c.width, height: c.height, nontransparent_pixels: occupied };
        }),
        invalid_numbers: /(?:NaN|Infinity)/.test(document.body.innerText),
      }));
      assert.equal(actual.id, entry.id);
      assert.equal(actual.kind, entry.kind);
      assert.equal(actual.baseline, entry.baseline_name);
      const expected = entry.kind === "native" && entry.baseline_name === "card_input" ? 6 : 2;
      assert.equal(actual.samples, expected, "Unexpected sample count: " + entry.id);
      assert.equal(actual.sample_options, expected);
      assert(actual.status.includes(entry.label) && !actual.status_error, "Status must identify the loaded entry");
      assert(!actual.invalid_numbers, "Rendered text contains NaN / Infinity");
      assert(actual.canvases.every(c => c.width > 0 && c.height > 0 && c.nontransparent_pixels > 500), "Empty canvas: " + entry.id);
      return actual;
    }

    for (const entry of inventory.entries) {
      const actual = await selectEntry(entry);
      report.entries_checked.push({ id: entry.id, kind: entry.kind, baseline_name: entry.baseline_name, sample_count: actual.samples, canvases: actual.canvases });
      if (report.entries_checked.length % 20 === 0) process.stdout.write(`Checked ${report.entries_checked.length}/148 entries\n`);
    }
    const find = predicate => { const entry = inventory.entries.find(predicate); assert(entry, "Required representative entry missing"); return entry; };
    const native30 = find(e => e.kind === "native" && e.baseline_name === "card_input" && e.control === 30);
    const native57 = find(e => e.kind === "native" && e.baseline_name === "card_input" && e.control === 57);
    const mixed58 = find(e => e.kind === "native" && e.baseline_name === "mixed_input" && e.control === 58);
    const scale = find(e => e.kind === "abmx" && e.channel === "scale" && e.bone === "cf_J_ChinTip_s");
    const rotation = find(e => e.kind === "abmx" && e.channel === "rotation" && e.bone === "cf_J_Eye_s_L");
    for (const entry of [native30, native57, mixed58, scale, rotation]) {
      const actual = await selectEntry(entry);
      for (let i = 0; i < actual.samples; i++) {
        await page.locator("#sample").selectOption(String(i));
        const check = await page.evaluate(() => ({ sample: Number(document.getElementById("sample").value), note: document.getElementById("sample-note").textContent, metrics: document.getElementById("metrics").textContent }));
        assert.equal(check.sample, i);
        assert(!/(?:NaN|Infinity)/.test(check.note + check.metrics));
        assert(check.note.includes("实测"));
        if (actual.roles[i].startsWith("extended_")) assert(check.note.includes("SliderUnlocker"));
        if (entry.kind === "abmx") assert(check.note.includes("ABMX") && check.note.includes("o_head"));
        report.sample_checks.push({ id: entry.id, role: actual.roles[i], sample_index: i, level: actual.levels[i], note: check.note });
      }
    }
    await selectEntry(native30);
    await page.locator("#target").fill("10000");
    for (const sign of ["1", "-1"]) {
      await page.locator("#sign").selectOption(sign);
      const text = await page.locator("#suggestion").innerText();
      assert(text.includes("已限制") && text.includes("尚未由独立采样验证"));
    }
    report.large_target_bounded = true;
    const noiseTarget = inventory.baselines.card_input.repeat_drift_normalized * 100 / 2;
    assert(noiseTarget > 0, "Noise test needs a recorded nonzero repeat drift");
    await page.locator("#target").fill(String(noiseTarget));
    assert((await page.locator("#suggestion").innerText()).includes("不提供步长"));
    report.below_noise_target_refused = true;
    await page.locator("#target").fill("0.1");
    await page.locator("#sign").selectOption("1");
    await page.locator("#vectors").check();
    await page.locator("#gain").fill("10");
    await page.locator("#sample").selectOption(String((await page.evaluate(() => payload.roles)).indexOf("local_plus")));
    const nativeShot = path.join(out, "native30.png");
    await page.screenshot({ path: nativeShot, fullPage: true });
    report.screenshots.push({ kind: "native", id: native30.id, ...fingerprint(nativeShot) });
    await selectEntry(rotation);
    const rotationImpact = await page.evaluate(() => ({ max_distance: payload.max_distance, noise_floor: catalog.baselines[payload.baseline_name].repeat_drift_normalized * catalog.baselines[payload.baseline_name].head_diagonal }));
    let screenshotEntry = rotation;
    if (rotationImpact.max_distance <= rotationImpact.noise_floor) {
      await selectEntry(scale);
      screenshotEntry = scale;
    }
    report.abmx_screenshot_choice = { requested_id: rotation.id, observed_rotation_impact: rotationImpact, selected_id: screenshotEntry.id, fallback_reason: screenshotEntry.id !== rotation.id ? "Eye rotation head-surface impact is below repeat-drift floor; ChinTip scale is shown" : null };
    await page.locator("#gain").fill("1");
    const abmxShot = path.join(out, "abmx_response.png");
    await page.screenshot({ path: abmxShot, fullPage: true });
    report.screenshots.push({ kind: "abmx", id: screenshotEntry.id, ...fingerprint(abmxShot) });

    await page.locator("#baseline").selectOption("mixed_input");
    const absent = await page.evaluate(() => ({
      disabled: document.getElementById("control").disabled,
      hidden: document.getElementById("content").hidden,
      payload_null: payload === null,
      status: document.getElementById("status").textContent,
      sample_disabled: document.getElementById("sample").disabled,
    }));
    assert(absent.disabled && absent.hidden && absent.payload_null && absent.sample_disabled && absent.status.includes("不会借用"));
    report.missing_mixed_abmx_clears_stale_data = absent;
    assert.equal(inventory.failures.length, 3);
    assert.equal(await page.locator("#failures li").count(), 3);
    assert(await page.locator("#failure-details").evaluate(e => e.open));
    assert((await page.locator("#failure-summary").innerText()).includes("3"));
    report.retained_failure_count = 3;
    assert.equal(report.page_errors.length, 0, "Browser page errors occurred");
    assert.equal(report.console_errors.length, 0, "Browser console errors occurred");
    report.passed = true;
  } catch (e) {
    report.error = { message: String(e), stack: e.stack };
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
    report.finished_utc = new Date().toISOString();
    fs.writeFileSync(reportFile, JSON.stringify(report, null, 2) + "\n", "utf8");
    process.stdout.write(JSON.stringify({ passed: report.passed, checked: report.entries_checked.length, report: reportFile, screenshots: report.screenshots.map(s => s.path), error: report.error?.message }) + "\n");
  }
}

main().catch(e => { process.stderr.write(String(e) + "\n"); process.exitCode = 1; });
