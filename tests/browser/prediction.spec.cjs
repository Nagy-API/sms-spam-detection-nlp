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
  await page.goto("/");
});

test("English SMS predictions render without safety claims or confidence scores", async ({ page }) => {
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("English SMS Spam Detector");
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
  const textbox = page.getByRole("textbox", { name: "SMS message" });
  await expect(textbox).toHaveAccessibleDescription(/5,000 characters/);
  await page.keyboard.press("Tab");
  await expect(textbox).toBeFocused();
  await expect(textbox).toHaveCSS("outline-style", "solid");
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(320);
});

for (const [width, height] of [[390, 844], [768, 1024], [1440, 900]]) {
  test(`${width}px: full flow without overflow, page errors or CSP violations`, async ({ page }) => {
    const problems = [];
    page.on("pageerror", (error) => problems.push(`pageerror: ${error.message}`));
    page.on("console", (message) => {
      if (message.type() === "error") problems.push(`console: ${message.text()}`);
    });
    await page.setViewportSize({ width, height });
    await page.goto("/");
    await page.getByRole("textbox").fill(SPAM);
    await analyze(page);
    await expect(page.locator("#resultCard")).toContainText("Likely Spam");
    await page.locator("#resultCard").scrollIntoViewIfNeeded();
    await expect(page.locator("#resultCard")).toBeInViewport();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
    expect(problems).toEqual([]);
  });
}
