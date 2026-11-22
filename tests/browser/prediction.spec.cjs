const { test, expect } = require("@playwright/test");

const HAM = "See you at lunch tomorrow";
const SPAM = "Congratulations! You won a free prize. Call now to claim.";

function analyze(page) {
  return page.locator("#analyzeBtn").click();
}

function upload(page, text, name = "message.txt") {
  return page.locator('input[type="file"]').setInputFiles({
    name,
    mimeType: "text/plain",
    buffer: Buffer.from(text),
  });
}

function gate() {
  let open;
  const wait = new Promise((resolve) => { open = resolve; });
  return { open, wait };
}

async function delayPrediction(page) {
  const arrived = gate();
  const release = gate();
  const done = gate();
  await page.route("/predict", async (route) => {
    arrived.open();
    await release.wait;
    try {
      await route.fulfill({ json: { label: "spam", spam_probability: 1 } });
    } finally {
      done.open();
    }
  }, { times: 1 });
  return { arrived, release, done };
}

test.beforeEach(async ({ page }) => {
  await page.goto("/?model=v2");
});

test("English SMS predictions render without safety claims or confidence scores", async ({ page }) => {
  await expect(page).toHaveTitle("TRACE — English SMS Classifier");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Every message deserves a closer look.");
  await expect(page.locator("body")).not.toContainText(/email|high accuracy/i);

  for (const [text, label, title] of [[HAM, "ham", "Likely Ham"], [SPAM, "spam", "Likely Spam"]]) {
    await page.getByRole("textbox").fill(text);
    const response = page.waitForResponse("/predict");
    await analyze(page);
    const result = await response;
    expect(result.request().postDataJSON()).toEqual({ text });
    expect((await result.json()).label).toBe(label);
    await expect(page.locator("#resultCard")).toContainText(title);
    await expect(page.getByRole("status")).toContainText(title);
    await expect(page.locator("#resultCard")).not.toContainText(/confidence|%|\bsafe\b|\bunsafe\b/i);
    await expect(page.getByRole("progressbar")).toHaveCount(0);
  }
});

test("zero-feature SMS is announced as inconclusive", async ({ page }) => {
  await page.getByRole("textbox").fill("12345");
  await analyze(page);
  await expect(page.locator("#resultCard")).toContainText("Inconclusive");
  await expect(page.getByRole("status")).toContainText("Inconclusive");
});

for (const [text, reason, explanation] of [
  ["12345", "no_letters", /no words to analyze/i],
  ["مرحبا كيف حالك", "unsupported_script", /only supports English SMS/i],
  ["qzxwvj", "no_recognized_features", /no words or phrases the model recognizes/i],
]) {
  test(`inconclusive ${reason} is explained to the user`, async ({ page }) => {
    await page.getByRole("textbox").fill(text);
    const response = page.waitForResponse("/predict");
    await analyze(page);
    const body = await (await response).json();
    expect(body).toEqual({ label: "inconclusive", spam_probability: null, reason, model_version: "2.0.0" });
    await expect(page.locator("#resultCard")).toContainText("Inconclusive");
    await expect(page.locator("#resultCard")).toContainText(explanation);
  });
}

test("unknown inconclusive reasons fall back to the generic explanation", async ({ page }) => {
  await page.route("/predict", (route) => route.fulfill({ json: { label: "inconclusive", spam_probability: null, reason: "future_reason" } }));
  await page.getByRole("textbox").fill(HAM);
  await analyze(page);
  await expect(page.locator("#resultCard")).toContainText(/no words or phrases the model recognizes/i);
});

test("blank and over-limit text are blocked before a request", async ({ page }) => {
  const requests = [];
  page.on("request", (request) => {
    if (new URL(request.url()).pathname === "/predict") requests.push(request);
  });
  const input = page.getByRole("textbox");
  for (const text of ["", " \t\n ", "\u0085\u001c"]) {
    await input.fill(text);
    await analyze(page);
    await expect(page.getByRole("alert")).toContainText("Enter an SMS message");
    await expect(input).toHaveAttribute("aria-invalid", "true");
  }

  const atLimit = HAM + "🙂".repeat(5000 - HAM.length);
  await input.fill(atLimit + "🙂");
  await expect(page.locator("#charCount")).toHaveText("5001");
  await analyze(page);
  await expect(page.getByRole("alert")).toContainText("5,000 characters");
  expect(requests).toHaveLength(0);

  await input.fill(atLimit);
  await expect(page.locator("#charCount")).toHaveText("5000");
  await analyze(page);
  await expect(page.locator("#resultCard")).toContainText("Likely Ham");
  expect(requests).toHaveLength(1);
});

test("oversized uploads are rejected without replacing the current text", async ({ page }) => {
  const input = page.getByRole("textbox");
  await input.fill(HAM);
  await analyze(page);
  await expect(page.locator("#resultCard")).toBeVisible();

  for (const inputText of ["x".repeat(5001), "🙂".repeat(5001)]) {
    await upload(page, inputText);
    await expect(page.getByRole("alert")).toContainText("5,000 characters");
    await expect(input).toHaveValue(HAM);
    await expect(page.locator("#resultCard")).toBeHidden();
  }

  const atLimit = SPAM + "🙂".repeat(5000 - SPAM.length);
  await upload(page, atLimit);
  await expect(input).toHaveValue(atLimit);
  await expect(page.getByRole("alert")).toBeEmpty();
  await expect(page.locator("#charCount")).toHaveText("5000");
});

test("an outdated file read cannot replace a newer edit", async ({ page }) => {
  await page.evaluate(() => {
    File.prototype.text = () => new Promise((resolve) => {
      window.finishTestRead = () => resolve("Earlier file contents");
    });
  });
  await upload(page, SPAM);
  await expect(page.getByRole("status")).toContainText(/reading sms file/i);
  await expect(page.locator("#analyzeBtn")).toBeDisabled();
  const input = page.getByRole("textbox");
  await input.fill(HAM);
  await page.evaluate(() => window.finishTestRead());
  await expect(input).toHaveValue(HAM);
  await expect(page.getByRole("status")).toBeEmpty();
  await expect(page.locator("#analyzeBtn")).toBeEnabled();
});

for (const [action, change] of [
  ["edit", (page) => page.getByRole("textbox").fill(HAM)],
  ["upload", (page) => upload(page, HAM)],
  ["clear", (page) => page.getByRole("button", { name: /clear/i }).click()],
]) {
  test(`${action} invalidates both displayed and pending predictions`, async ({ page }) => {
    const input = page.getByRole("textbox");
    await input.fill(SPAM);
    await analyze(page);
    await expect(page.locator("#resultCard")).toBeVisible();
    await change(page);
    await expect(page.locator("#resultCard")).toBeHidden();

    const { arrived, release, done } = await delayPrediction(page);
    try {
      await input.fill(SPAM);
      await analyze(page);
      await arrived.wait;
      await expect(page.getByRole("status")).toContainText(/analyzing/i);
      await change(page);
      await expect(input).toHaveValue(action === "clear" ? "" : HAM);
    } finally {
      release.open();
    }
    await done.wait;
    await expect(page.locator("#resultCard")).toBeHidden();
    await expect(page.locator("#analyzeBtn")).toBeEnabled();
  });
}

for (const [kind, respond, expected] of [
  ["validation", (route) => route.fulfill({ status: 422, json: { detail: [{ type: "string_too_long" }] } }), /5,000 characters/],
  ["network", (route) => route.abort("failed"), /could not reach/i],
  ["server", (route) => route.fulfill({ status: 500 }), /prediction service.*problem/i],
]) {
  test(`${kind} failures give actionable inline feedback`, async ({ page }) => {
    await page.route("/predict", respond);
    await page.getByRole("textbox").fill(HAM);
    await analyze(page);
    await expect(page.getByRole("alert")).toContainText(expected);
    await expect(page.locator("#resultCard")).toBeHidden();
    await expect(page.locator("#analyzeBtn")).toBeEnabled();
  });
}

test("the textbox has a label, description and visible keyboard focus on narrow screens", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 800 });
  const textbox = page.getByRole("textbox", { name: /SMS message/ });
  await expect(textbox).toHaveAccessibleDescription(/5,000 characters/);
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Skip to message" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(textbox).toBeFocused();
  await expect(textbox).toHaveCSS("outline-style", "solid");
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(320);
});

for (const [width, height] of [[320, 800], [390, 844], [768, 1024], [1440, 900]]) {
  test(`${width}px: full flow without overflow, page errors or CSP violations`, async ({ page }, testInfo) => {
    const problems = [];
    page.on("pageerror", (error) => problems.push(`pageerror: ${error.message}`));
    page.on("console", (message) => {
      if (message.type() === "error") problems.push(`console: ${message.text()}`);
    });
    await page.setViewportSize({ width, height });
    await page.goto("/?model=v2");
    await page.screenshot({ path: testInfo.outputPath(`trace-${width}-empty.png`), fullPage: true, animations: 'disabled' });
    await page.getByRole("textbox").fill(SPAM);
    await analyze(page);
    await expect(page.locator("#resultCard")).toContainText("Likely Spam");
    await page.locator("#resultCard").scrollIntoViewIfNeeded();
    await expect(page.locator("#resultCard")).toBeInViewport();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
    await page.getByRole('button', { name: 'Try a variation' }).click();
    await page.getByRole('textbox').fill(HAM);
    await analyze(page);
    await expect(page.locator('#revisionLabel')).toHaveText('Likely Ham');
    await expect(page.locator('#originalLabel')).toHaveText('Likely Spam');
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
    expect(problems).toEqual([]);
    await page.screenshot({ path: testInfo.outputPath(`trace-${width}-result.png`), fullPage: true, animations: 'disabled' });
  });
}

async function startComparison(page, text = SPAM) {
  await page.getByRole('textbox').fill(text);
  await analyze(page);
  await expect(page.locator('#resultCard')).toBeVisible();
  await page.getByRole('button', { name: 'Try a variation' }).click();
  await expect(page.getByRole('textbox')).toBeFocused();
}

test('real revision requests preserve the original and compare changed and unchanged labels', async ({ page }) => {
  await startComparison(page);
  for (const [text, label, note] of [[HAM, 'Likely Ham', 'Classification changed.'], [SPAM, 'Likely Spam', 'Classification unchanged.'], ['12345', 'Inconclusive', 'Classification changed.']]) {
    await page.getByRole('textbox').fill(text);
    await expect(page.locator('#revisionLabel')).toHaveText('Not analyzed');
    const response = page.waitForResponse('/predict');
    await analyze(page);
    expect((await response).request().postDataJSON()).toEqual({ text });
    await expect(page.locator('#revisionLabel')).toHaveText(label);
    await expect(page.locator('#originalMessage')).toHaveText(SPAM);
    await expect(page.locator('#originalLabel')).toHaveText('Likely Spam');
    await expect(page.locator('#comparisonNote')).toContainText(note);
  }
  await expect(page.locator('#revisionDescription')).toHaveText('The message has no words to analyze.');
  await page.getByRole('button', { name: 'Start fresh' }).click();
  await expect(page.getByRole('textbox')).toHaveValue('');
  await expect(page.getByRole('textbox')).toBeFocused();
  await expect(page.locator('#smsWorkspace')).toHaveAttribute('data-stage', 'compose');
  await expect(page.locator('#originalMessage')).toBeEmpty();
  await expect(page.locator('#resultPane')).toHaveAttribute('inert', '');
});

for (const [action, change] of [
  ['edit', page => page.getByRole('textbox').fill(HAM)],
  ['upload', page => upload(page, HAM)],
  ['clear', page => page.getByRole('button', { name: 'Clear', exact: true }).click()],
  ['start fresh', page => page.getByRole('button', { name: 'Start fresh' }).click()],
]) {
  test(`revision ${action} invalidates displayed and pending results`, async ({ page }) => {
    await startComparison(page);
    await analyze(page);
    await expect(page.locator('#revisionLabel')).toHaveText('Likely Spam');
    await change(page);
    await expect(page.locator('#revisionLabel')).toHaveText('Not analyzed');
    if (action === 'start fresh') await startComparison(page);
    await page.getByRole('textbox').fill(SPAM);
    const { arrived, release, done } = await delayPrediction(page);
    try {
      await analyze(page);
      await arrived.wait;
      await expect(page.locator('#revisionLabel')).toHaveText('Analyzing revision…');
      await expect(page.locator('#originalLabel')).toHaveText('Likely Spam');
      await change(page);
    } finally { release.open(); }
    await done.wait;
    await expect(page.locator('#revisionLabel')).toHaveText('Not analyzed');
    await expect(page.locator('#analyzeBtn')).toBeEnabled();
    if (action !== 'start fresh') {
      await expect(page.locator('#originalMessage')).toHaveText(SPAM);
      await expect(page.locator('#originalLabel')).toHaveText('Likely Spam');
    }
  });
}

test('start fresh during a file read clears busy state and ignores its late content', async ({ page }) => {
  await startComparison(page);
  await page.evaluate(() => { File.prototype.text = () => new Promise(resolve => { window.finishTestRead = () => resolve('stale upload'); }); });
  await upload(page, HAM);
  await expect(page.locator('#analyzeBtn')).toBeDisabled();
  await page.getByRole('button', { name: 'Start fresh' }).click();
  await page.evaluate(() => window.finishTestRead());
  await expect(page.getByRole('textbox')).toHaveValue('');
  await expect(page.locator('#analyzeBtn')).toBeEnabled();
  await page.getByRole('textbox').fill(HAM);
  await analyze(page);
  await expect(page.locator('#resultLabel')).toHaveText('Likely Ham');
});

for (const [status, message] of [[413, /5,000 characters/], [503, /unavailable/], [500, /problem/]]) {
  test(`revision HTTP ${status} preserves the baseline and permits retry`, async ({ page }) => {
    await startComparison(page, '12345');
    await page.route('/predict', route => route.fulfill({ status }), { times: 1 });
    await page.getByRole('textbox').fill(HAM);
    await analyze(page);
    await expect(page.getByRole('alert')).toContainText(message);
    await expect(page.locator('#revisionLabel')).toHaveText('Not analyzed');
    await expect(page.locator('#originalDescription')).toHaveText('The message has no words to analyze.');
    await expect(page.locator('#analyzeBtn')).toBeEnabled();
    await analyze(page);
    await expect(page.locator('#revisionLabel')).toHaveText('Likely Ham');
  });
}

for (const reducedMotion of ['no-preference', 'reduce']) {
  test(`real pending state and cancellation with motion preference ${reducedMotion}`, async ({ page }) => {
    await page.emulateMedia({ reducedMotion });
    const { arrived, release, done } = await delayPrediction(page);
    try {
      await page.getByRole('textbox').fill(SPAM);
      await analyze(page);
      await arrived.wait;
      await expect(page.locator('#loadingPanel')).toBeVisible();
      await expect(page.locator('#tryVariationBtn')).toBeHidden();
      await expect(page.locator('.loading-dots span').first()).toHaveCSS('animation-name', reducedMotion === 'reduce' ? 'none' : 'dot-pulse');
      await page.getByRole('textbox').fill(HAM);
      await expect(page.locator('#loadingPanel')).toBeHidden();
    } finally { release.open(); }
    await done.wait;
    await expect(page.locator('#resultCard')).toBeHidden();
    await analyze(page);
    await expect(page.locator('#resultCard')).toBeVisible();
    await expect(page.locator('#resultBox')).toHaveCSS('animation-name', reducedMotion === 'reduce' ? 'none' : 'result-reveal');
  });
}

test('keyboard workflow keeps focus visible through analysis, comparison and reset', async ({ page }) => {
  await page.keyboard.press('Tab');
  await page.keyboard.press('Enter');
  await page.keyboard.type(HAM);
  for (const id of ['clearBtn', 'uploadBtn', 'analyzeBtn']) {
    await page.keyboard.press('Tab');
    await expect(page.locator('#' + id)).toBeFocused();
    await expect(page.locator('#' + id)).toHaveCSS('outline-style', 'solid');
  }
  await page.keyboard.press('Enter');
  await expect(page.locator('#resultCard')).toBeVisible();
  await page.keyboard.press('Tab');
  await expect(page.locator('#tryVariationBtn')).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('textbox')).toBeFocused();
});

test('silver pointer light follows, settles, respects preferences and adds no inline styles', async ({ page }) => {
  const problems = [];
  page.on('console', message => { if (message.type() === 'error') problems.push(message.text()); });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.mouse.move(100, 100);
  const light = page.locator('#pointerLight');
  await expect(light).toHaveClass(/is-visible/);
  await page.mouse.move(800, 300);
  await expect.poll(async () => light.evaluate(el => {
    const matrix = new DOMMatrix(getComputedStyle(el).transform);
    return Math.abs(matrix.m41 - 475) < 1 && Math.abs(matrix.m42 + 25) < 1;
  })).toBe(true);
  const settled = await light.evaluate(el => getComputedStyle(el).transform);
  await page.waitForTimeout(100);
  expect(await light.evaluate(el => getComputedStyle(el).transform)).toBe(settled);
  await expect(light).toHaveCSS('pointer-events', 'none');
  await expect(page.locator('[style], style, script:not([src])')).toHaveCount(0);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.mouse.move(400, 200);
  await expect(light).toBeHidden();
  await expect(light).not.toHaveClass(/is-visible/);
  // The spotlight resets on the media-query change event, which can arrive
  // after emulateMedia resolves. Move the pointer after that reset.
  await page.evaluate(() => {
    const query = matchMedia("(prefers-reduced-motion: reduce)");
    window.spotlightPreferenceChanged = new Promise(resolve => {
      query.addEventListener("change", () => resolve(), { once: true });
    });
  });
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.evaluate(() => window.spotlightPreferenceChanged);
  await page.mouse.move(401, 201);
  await expect(light).toHaveClass(/is-visible/);
  expect(problems).toEqual([]);
});

test('touch devices never activate pointer lighting', async ({ browser }) => {
  const context = await browser.newContext({ hasTouch: true, viewport: { width: 768, height: 1024 } });
  const page = await context.newPage();
  await page.goto(process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:8765');
  await page.mouse.move(400, 300);
  await expect(page.locator('#pointerLight')).toBeHidden();
  await expect(page.locator('#pointerLight')).not.toHaveClass(/is-visible/);
  await context.close();
});

test('versioned assets match the working tree on cold and warm loads', async ({ page }) => {
  const fs = require('node:fs');
  const path = require('node:path');
  for (const [selector, attr, filename] of [['link[rel="stylesheet"]', 'href', 'style.css'], ['script[src]', 'src', 'script.js']]) {
    const url = await page.locator(selector).getAttribute(attr);
    expect(url).toMatch(/\?v=trace-/);
    const response = await page.request.get(url);
    expect(response.ok()).toBeTruthy();
    expect((await response.text()).replace(/\r\n/g, '\n')).toBe(fs.readFileSync(path.resolve(__dirname, '../../frontend', filename), 'utf8').replace(/\r\n/g, '\n'));
  }
  await page.reload();
  await expect(page.locator('body')).toHaveCSS('background-color', 'rgb(5, 5, 5)');
  await expect(page.locator('[data-example]')).toHaveCount(0);
  await page.getByRole('textbox').fill(HAM);
  await analyze(page);
  await expect(page.locator('#resultLabel')).toHaveText('Likely Ham');
});

test('invalid revision upload preserves text, baseline and input limits', async ({ page }) => {
  await startComparison(page);
  await page.getByRole('textbox').fill(HAM);
  await analyze(page);
  await expect(page.locator('#revisionLabel')).toHaveText('Likely Ham');
  for (const [text, name, error] of [['x'.repeat(5001), 'long.txt', /5,000 characters/], ['bad', 'message.csv', /Choose a .txt/]]) {
    await upload(page, text, name);
    await expect(page.getByRole('alert')).toContainText(error);
    await expect(page.getByRole('textbox')).toHaveValue(HAM);
    await expect(page.locator('#originalMessage')).toHaveText(SPAM);
    await expect(page.locator('#revisionLabel')).toHaveText('Not analyzed');
  }
});

test('a revision network failure leaves the original available and clears pending feedback', async ({ page }) => {
  await startComparison(page);
  await page.route('/predict', route => route.abort('failed'), { times: 1 });
  await page.getByRole('textbox').fill(HAM);
  await analyze(page);
  await expect(page.getByRole('alert')).toContainText(/Could not reach/);
  await expect(page.locator('#revisionLabel')).toHaveText('Not analyzed');
  await expect(page.locator('#originalLabel')).toHaveText('Likely Spam');
  await analyze(page);
  await expect(page.locator('#revisionLabel')).toHaveText('Likely Ham');
});

test("verdict cues distinguish all outcomes without color and keep the Ham limitation in the result", async ({ page }) => {
  const result = page.locator("#resultCard");
  for (const [text, outcome, title, glyph, color] of [
    [SPAM, "spam", "Likely Spam", "!", "rgb(215, 125, 118)"],
    [HAM, "ham", "Likely Ham", "✓", "rgb(130, 185, 161)"],
    ["12345", "inconclusive", "Inconclusive", "?", "rgb(217, 181, 124)"],
  ]) {
    await page.getByRole("textbox").fill(text);
    await analyze(page);
    const heading = result.getByRole("heading", { name: title, exact: true });
    await expect(heading).toHaveCSS("color", "rgb(240, 240, 240)");
    const verdict = result.locator(`.verdict[data-outcome="${outcome}"]`);
    await expect(verdict.locator(".verdict-icon")).toHaveText(glyph);
    await expect(verdict.locator(".verdict-icon")).toHaveAttribute("aria-hidden", "true");
    await expect(verdict.locator(".verdict-indicator")).toHaveCSS("background-color", color);
    if (outcome === "ham") await expect(result).toContainText("does not guarantee message safety");
    if (outcome === "inconclusive") await expect(result).toContainText("no words to analyze");
  }
  await expect(page.locator("footer")).toHaveText(/Predictions can be wrong/);
  await expect(page.locator("footer")).not.toContainText(/guarantee/);
});

test("cancel variation restores the cached verdict and aborts a pending revision", async ({ page }) => {
  const requests = [];
  page.on("request", request => {
    if (new URL(request.url()).pathname === "/predict") requests.push(request.postDataJSON());
  });
  await startComparison(page, HAM);
  await page.getByRole("textbox").fill(SPAM);
  const { arrived, release, done } = await delayPrediction(page);
  try {
    await analyze(page);
    await arrived.wait;
    await expect(page.locator("#revisionLabel")).toHaveText("Analyzing revision…");
    const aborted = page.waitForEvent("requestfailed", request => new URL(request.url()).pathname === "/predict");
    await page.getByRole("button", { name: "Cancel variation" }).click();
    expect((await aborted).failure().errorText).toContain("ABORTED");
  } finally {
    release.open();
  }
  await done.wait;
  await expect(page.getByRole("textbox")).toHaveValue(HAM);
  await expect(page.locator("#resultLabel")).toHaveText("Likely Ham");
  await expect(page.locator("#resultCard")).toBeVisible();
  await expect(page.locator("#comparisonPanel")).toBeHidden();
  await expect(page.getByRole("button", { name: "Try a variation" })).toBeFocused();
  expect(requests).toEqual([{ text: HAM }, { text: SPAM }]);
  await page.getByRole("button", { name: "Try a variation" }).click();
  await analyze(page);
  await expect(page.locator("#revisionLabel")).toHaveText("Likely Ham");
  await expect(page.locator("#comparisonPanel").getByText(/does not guarantee message safety/)).toHaveCount(1);
  await page.getByRole("button", { name: "Cancel variation" }).click();
  await expect(page.getByRole("textbox")).toHaveValue(HAM);
  await expect(page.locator("#resultLabel")).toHaveText("Likely Ham");
  expect(requests).toEqual([{ text: HAM }, { text: SPAM }, { text: HAM }]);
});

test("experimental comparison renders the prediction returned by the V3 endpoint", async ({ page }) => {
  await page.goto("/?model=v3");
  await expect(page.locator("#experimentalNotice")).toBeVisible();
  const cases = [
    { label: "phishing", spam_probability: 0.3, model_version: "3.0.0-dev.1", experimental: true },
    { label: "ham", spam_probability: 0.1, model_version: "3.0.0-dev.1", experimental: true },
  ];
  await page.route("/predict/v3", async route => {
    expect(route.request().method()).toBe("POST");
    await route.fulfill({ json: cases.shift() });
  });
  await page.getByRole("textbox").fill("Please verify your account.");
  await analyze(page);
  await expect(page.getByRole("heading", { name: "Potential Phishing" })).toBeVisible();
  await page.getByRole("button", { name: "Try a variation" }).click();
  await page.getByRole("textbox").fill(HAM);
  await analyze(page);
  await expect(page.locator("#originalLabel")).toHaveText("Potential Phishing");
  await expect(page.locator("#revisionLabel")).toHaveText("Likely Ham");
  expect(cases).toHaveLength(0);
});

for (const width of [390, 1440]) {
  test(`packaged V3 predicts and compares through the real API at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
    await page.goto("/");
    await expect(page.locator("#experimentalNotice")).toBeVisible();
    const meta = await page.request.get("/model/v3");
    expect((await meta.json()).independent_phishing_validation).toBe(false);
    const input = page.getByRole("textbox");
    for (const [text, label] of [
      ["12345", "Inconclusive"],
      [SPAM, "Likely Spam"],
      ["Please verify your account at https://example.invalid/verify", "Potential Phishing"],
    ]) {
      await input.fill(text);
      await analyze(page);
      await expect(page.locator("#resultLabel")).toHaveText(label);
    }
    await page.getByRole("button", { name: "Try a variation" }).click();
    await input.fill(HAM);
    await analyze(page);
    await expect(page.locator("#originalLabel")).toHaveText("Potential Phishing");
    await expect(page.locator("#revisionLabel")).toHaveText("Likely Ham");
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
    expect(errors).toEqual([]);
    await page.screenshot({ path: testInfo.outputPath(`trace-v3-${width}.png`), fullPage: true, animations: "disabled" });
  });
}
