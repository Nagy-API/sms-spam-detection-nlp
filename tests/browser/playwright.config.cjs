const path = require("node:path");
const { defineConfig } = require("@playwright/test");

const baseURL = "http://127.0.0.1:8765";

module.exports = defineConfig({
  testDir: ".",
  use: {
    baseURL,
    launchOptions: { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE },
  },
  webServer: {
    command: "python -B -m uvicorn app.main:app --host 127.0.0.1 --port 8765",
    cwd: path.resolve(__dirname, "../.."),
    url: `${baseURL}/health`,
    timeout: 60000,
  },
});
