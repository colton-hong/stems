import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { expect, test } from "@playwright/test";

import { playerReady } from "../e2e/helpers";

const WORK = path.resolve(__dirname, "..", ".e2e-real");
const PYTHON =
  process.env.STEMS_PYTHON ?? path.resolve(__dirname, "..", "..", ".venv", "bin", "python");
const FIXTURE = path.join(WORK, "Synthetic Fixture.wav");
const SECONDS = 12;
const STEMS = ["Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"];

type Run = {
  model_id: string;
  device: string;
  stage: string;
  outputs: string[];
  audio_seconds: number;
  wall_seconds: number;
  cached: boolean;
};

test.beforeAll(() => {
  fs.mkdirSync(WORK, { recursive: true });
  // The synthetic fixture: tones and noise made by code. Never real music.
  const script = [
    "import sys",
    "from stems_common.synthetic import generate_track",
    "from stems_worker.audio import save_audio",
    `track = generate_track(seed=41, duration_seconds=${SECONDS}.0)`,
    'save_audio(sys.argv[1], track.mix, track.sample_rate, "wav")',
  ].join("\n");
  execFileSync(PYTHON, ["-c", script, FIXTURE]);
});

test("a song goes through the real models and comes back as six stems that play", async ({
  page,
}) => {
  const started = Date.now();
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles(FIXTURE);
  for (const stem of STEMS) {
    await page.getByRole("checkbox", { name: stem, exact: true }).check();
  }
  await page.getByRole("button", { name: "Upload and separate" }).click();
  await page.waitForURL(/\/jobs\/[0-9a-f-]{36}$/);
  const id = page.url().split("/").pop() ?? "";
  await expect(page.getByTestId("status")).toHaveText("Ready", { timeout: 55 * 60_000 });
  const untilReady = (Date.now() - started) / 1000;

  // Six stems, in display order, each with a waveform.
  const rows = page.locator("ul.stems > li");
  await expect(rows.locator(".stem-name")).toHaveText(STEMS);
  await expect(page.locator(".wave")).toHaveCount(6);

  // They load and start on one clock.
  await playerReady(page);
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect
    .poll(() => page.evaluate(() => window.__stemsEngine?.schedule.length ?? 0))
    .toBe(6);
  const schedule = await page.evaluate(() => window.__stemsEngine?.schedule ?? []);
  expect(new Set(schedule.map((entry) => entry.when)).size).toBe(1);
  await expect
    .poll(() => page.evaluate(() => window.__stemsEngine?.position ?? 0))
    .toBeGreaterThan(0.2);
  await page.getByRole("button", { name: "Pause", exact: true }).click();

  // A stem downloads, under the song's name, and is a file of audio and not an error page.
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download Drums" }).click();
  const download = await pending;
  expect(download.suggestedFilename()).toBe("Synthetic Fixture (drums).wav");
  const saved = await download.path();
  expect(fs.statSync(saved).size).toBeGreaterThan(SECONDS * 44100 * 2);

  // What ran, read from the job's own result file in local storage (objects/<key>).
  const result = JSON.parse(
    fs.readFileSync(path.join(WORK, "storage", "objects", "jobs", id, "result.json"), "utf8"),
  );
  expect(result.state).toBe("ready");
  const runs: Run[] = result.timings;
  expect(runs.length).toBeGreaterThan(0);
  for (const run of runs) {
    expect(run.model_id).not.toMatch(/^fake_/);
  }
  const summary = {
    job_id: id,
    finished_at: result.finished_at,
    fixture: { seconds: SECONDS, kind: "synthetic (generate_track, seed 41)" },
    device: result.device,
    seconds_from_upload_to_ready: Math.round(untilReady * 10) / 10,
    stems: result.stems.map(
      (stem: { name: string; model_ids: string[]; mode: string; attempts?: number }) => ({
        name: stem.name,
        model_ids: stem.model_ids,
        mode: stem.mode,
      }),
    ),
    warnings: await rows.locator(".stem-warning").allInnerTexts(),
    runs: runs.map((run) => ({
      model_id: run.model_id,
      stage: run.stage,
      outputs: run.outputs,
      device: run.device,
      audio_seconds: run.audio_seconds,
      wall_seconds: run.wall_seconds,
      cached: run.cached,
    })),
    track_features: result.track_features,
  };
  fs.writeFileSync(path.join(WORK, "real-job.json"), `${JSON.stringify(summary, null, 2)}\n`);
});
