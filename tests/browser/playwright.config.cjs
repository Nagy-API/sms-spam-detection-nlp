const path = require("node:path");
const { defineConfig } = require("@playwright/test");

// Default: start the local FastAPI app. Set PLAYWRIGHT_BASE_URL to run the same suite against a
// deployed (e.g. protected Vercel Preview) URL; VERCEL_AUTOMATION_BYPASS_SECRET, if set, is sent
// as Vercel's protection-bypass header on every request. See docs/DEPLOYMENT.md.
const hostedURL = process.env.PLAYWRIGHT_BASE_URL;
const baseURL = hostedURL || "http://127.0.0.1:8765";
const bypass = process.env.VERCEL_AUTOMATION_BYPASS_SECRET;

module.exports = defineConfig({
  testDir: ".",
  use: {
    baseURL,
    extraHTTPHeaders: bypass ? { "x-vercel-protection-bypass": bypass } : {},
    launchOptions: { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE },
  },
  webServer: hostedURL
    ? undefined
    : {
        command: "python -B -m uvicorn app.main:app --host 127.0.0.1 --port 8765",
        cwd: path.resolve(__dirname, "../.."),
        url: `${baseURL}/health`,
        timeout: 60000,
      },
});
