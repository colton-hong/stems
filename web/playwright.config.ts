import path from "node:path";

import { defineConfig } from "@playwright/test";

// The browser tests run the real thing: the API with its maintenance thread and the
// local worker provider, the built web app, and Chrome. The only stand-in is the
// separator (STEMS_WORKER_FAKE_MODELS), so no weights are needed.
const REPO = path.resolve(__dirname, "..");
const WORK = path.join(__dirname, ".e2e");
const API_PORT = 8787;
const WEB_PORT = 3939;
const API_URL = `http://localhost:${API_PORT}`;
const PYTHON = process.env.STEMS_PYTHON ?? path.join(REPO, ".venv", "bin", "python");

export default defineConfig({
  testDir: "./e2e",
  timeout: 180_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  outputDir: path.join(WORK, "results"),
  use: {
    baseURL: `http://localhost:${WEB_PORT}`,
    // Locally the installed Chrome is used, so nothing is downloaded. CI installs
    // Playwright's own Chromium and sets STEMS_E2E_CHANNEL to an empty string.
    channel: process.env.STEMS_E2E_CHANNEL ?? "chrome",
    acceptDownloads: true,
    trace: "retain-on-failure",
  },
  webServer: [
    {
      // Every run starts from an empty database and empty storage. They are cleared here,
      // before the API starts, and never while it is running.
      command: `rm -rf "${WORK}/storage" "${WORK}/e2e.db"* && mkdir -p "${WORK}" && exec "${PYTHON}" -m stems_api --port ${API_PORT}`,
      cwd: REPO,
      url: `http://127.0.0.1:${API_PORT}/api/health`,
      reuseExistingServer: false,
      timeout: 120_000,
      // The API's log goes into the test output: a 500 is explained where it is seen.
      stdout: "pipe",
      stderr: "pipe",
      env: {
        // No .env: the tests never see the owner's keys, so nothing here calls a paid service.
        STEMS_DOTENV_PATH: path.join(WORK, "no-such-file.env"),
        STEMS_WORKER_FAKE_MODELS: "1",
        STEMS_WORKER_PROVIDER: "local",
        STEMS_STORAGE_BACKEND: "local",
        STEMS_STORAGE_LOCAL_DIR: path.join(WORK, "storage"),
        STEMS_DATABASE_URL: `sqlite:///${path.join(WORK, "e2e.db")}`,
        STEMS_WEB_ORIGIN: `http://localhost:${WEB_PORT}`,
      },
    },
    {
      command: `npm run build && npm run start -- --port ${WEB_PORT}`,
      cwd: __dirname,
      url: `http://localhost:${WEB_PORT}`,
      reuseExistingServer: false,
      timeout: 300_000,
      env: { NEXT_PUBLIC_API_URL: API_URL },
    },
  ],
});
