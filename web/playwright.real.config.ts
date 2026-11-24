import path from "node:path";

import { defineConfig } from "@playwright/test";

// One job with the real models: the API, its local worker, the real registry and the
// built web app, with no stand-in anywhere. It needs the model weights on disk, takes
// minutes, and is not part of `make check`, `make web-e2e` or CI. Run it with
// `make web-e2e-real`. The worker picks its device as it always does (STEMS_DEVICE, or
// the best one this machine has); nothing here reads `.env`, so no key is used and the
// decisions are made by the fallbacks.
const REPO = path.resolve(__dirname, "..");
const WORK = path.join(__dirname, ".e2e-real");
const API_PORT = 8797;
const WEB_PORT = 3949;
const API_URL = `http://localhost:${API_PORT}`;
const PYTHON = process.env.STEMS_PYTHON ?? path.join(REPO, ".venv", "bin", "python");

export default defineConfig({
  testDir: "./e2e-real",
  timeout: 60 * 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  outputDir: path.join(WORK, "results"),
  use: {
    baseURL: `http://localhost:${WEB_PORT}`,
    channel: process.env.STEMS_E2E_CHANNEL ?? "chrome",
    acceptDownloads: true,
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: `rm -rf "${WORK}/storage" "${WORK}/real.db"* && mkdir -p "${WORK}" && exec "${PYTHON}" -m stems_api --port ${API_PORT}`,
      cwd: REPO,
      url: `http://127.0.0.1:${API_PORT}/api/health`,
      reuseExistingServer: false,
      timeout: 120_000,
      stdout: "pipe",
      stderr: "pipe",
      env: {
        STEMS_DOTENV_PATH: path.join(WORK, "no-such-file.env"),
        // Empty means off: the real models, whatever the caller's environment says.
        STEMS_WORKER_FAKE_MODELS: "",
        STEMS_WORKER_PROVIDER: "local",
        STEMS_STORAGE_BACKEND: "local",
        STEMS_STORAGE_LOCAL_DIR: path.join(WORK, "storage"),
        STEMS_DATABASE_URL: `sqlite:///${path.join(WORK, "real.db")}`,
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
