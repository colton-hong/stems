import fs from "node:fs";

import { expect, test, type Page } from "@playwright/test";

import { chooseFile, playerReady, toneWav, uploadAndWait } from "./helpers";

const UNCHECKED = "This stem could not be quality-checked automatically.";

/** The page's own error box. (Next.js keeps an empty alert of its own for route changes.) */
const errorIn = (page: Page) => page.locator('.notice[role="alert"]');

test("upload, watch it finish, listen, download, delete", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText(/Up to 250 MB and 20 minutes/)).toBeVisible();
  const upload = page.getByRole("button", { name: "Upload and separate" });
  await expect(upload).toBeDisabled();
  await expect(page.getByText("Choose a file first.")).toBeVisible();

  await chooseFile(page, "Test Tone.wav", toneWav());
  await expect(page.getByText("Tick at least one stem.")).toBeVisible();
  await page.getByRole("checkbox", { name: "Vocals", exact: true }).check();
  await page.getByRole("checkbox", { name: "Bass", exact: true }).check();
  await page.getByLabel("File format").selectOption("flac");
  await expect(page.getByText(/Only this browser can see the result. It is deleted after 7 days/)).toBeVisible();
  await upload.click();

  // The job page follows the job to Ready by itself: no reload anywhere in this test.
  await page.waitForURL(/\/jobs\/[0-9a-f-]{36}$/);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Test Tone.wav");
  await expect(page.getByText("Asked for: Vocals, Bass, as FLAC")).toBeVisible();
  const stages = page.locator("ol.stages > li");
  await expect(stages).toHaveCount(9);
  await expect(page.getByTestId("status")).toHaveText("Ready", { timeout: 120_000 });
  await expect(page.locator('ol.stages > li[data-state="done"]')).toHaveCount(9);
  await expect(stages.first()).toContainText("Uploaded");
  await expect(stages.last()).toContainText("Ready");

  // Warnings: the stand-in models are not listened to, so every stem carries the note.
  await expect(page.getByText("Every stem has a note.")).toBeVisible();
  const rows = page.locator("ul.stems > li");
  await expect(rows).toHaveCount(2);
  for (const row of await rows.all()) {
    await expect(row.locator(".stem-warning")).toContainText(UNCHECKED);
  }
  await expect(rows.nth(0).locator(".stem-meta")).toContainText("FLAC");

  // The player: both stems on one clock.
  await playerReady(page);
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect(page.getByRole("button", { name: "Pause", exact: true })).toBeVisible();
  await expect
    .poll(() => page.evaluate(() => window.__stemsEngine?.position ?? 0))
    .toBeGreaterThan(0.2);
  const first = await page.evaluate(() => window.__stemsEngine?.schedule ?? []);
  expect(first.map((entry) => entry.name)).toEqual(["vocals", "bass"]);
  expect(first[0]?.when).toBe(first[1]?.when);
  expect(first[0]?.offset).toBe(first[1]?.offset);

  // Solo and mute change what is heard and restart nothing.
  await page.getByRole("button", { name: "Solo Vocals" }).click();
  await expect(page.getByRole("button", { name: "Solo Vocals", pressed: true })).toBeVisible();
  await page.getByRole("button", { name: "Mute Bass" }).click();
  await page.getByRole("button", { name: "Solo Vocals" }).click();
  const states = await page.evaluate(() => window.__stemsEngine?.stemStates ?? []);
  expect(states).toEqual([
    { name: "vocals", muted: false, solo: false },
    { name: "bass", muted: true, solo: false },
  ]);
  expect(await page.evaluate(() => window.__stemsEngine?.schedule.length)).toBe(2);

  // Seeking restarts both stems together at the new place.
  await page.getByRole("slider", { name: "Position" }).fill("2");
  await expect
    .poll(() => page.evaluate(() => window.__stemsEngine?.schedule.length))
    .toBe(4);
  const after = await page.evaluate(() => window.__stemsEngine?.schedule.slice(2) ?? []);
  expect(after.map((entry) => entry.offset)).toEqual([2, 2]);
  expect(after[0]?.when).toBe(after[1]?.when);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(page.getByRole("button", { name: "Play", exact: true })).toBeVisible();

  // Downloads: a stem and the zip, named after the upload, in the format asked for.
  const stemDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download Vocals" }).click();
  const stem = await stemDownload;
  expect(stem.suggestedFilename()).toBe("Test Tone (vocals).flac");
  const stemBytes = fs.readFileSync(await stem.path());
  expect(stemBytes.subarray(0, 4).toString("latin1")).toBe("fLaC");
  expect(stemBytes.length).toBeGreaterThan(1000);

  const zipDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download all stems (zip)" }).click();
  const zip = await zipDownload;
  expect(zip.suggestedFilename()).toBe("Test Tone (stems).zip");
  const zipBytes = fs.readFileSync(await zip.path());
  expect(zipBytes.subarray(0, 2).toString("latin1")).toBe("PK");
  expect(zipBytes.includes(Buffer.from("vocals.flac"))).toBe(true);
  expect(zipBytes.includes(Buffer.from("bass.flac"))).toBe(true);

  // Deletion: the countdown, then a delete that takes two clicks.
  await expect(page.getByTestId("countdown")).toContainText(/6 days 23 hours|7 days/);
  await page.getByRole("button", { name: "Delete now" }).click();
  await expect(page.getByText("Delete the upload and all stems? This cannot be undone.")).toBeVisible();
  await expect(page.getByTestId("status")).toHaveText("Ready"); // nothing happened yet
  await page.getByRole("button", { name: "Cancel" }).click();
  await page.getByRole("button", { name: "Delete now" }).click();
  await page.getByRole("button", { name: "Yes, delete" }).click();
  await expect(page.getByTestId("status")).toHaveText("Deleted");
  await expect(page.getByText("This job was deleted.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Play", exact: true })).toHaveCount(0);
});

test("a request in words fills in the ticks, and unclear words block the upload", async ({
  page,
}) => {
  await page.goto("/");
  await chooseFile(page, "Test Tone.wav", toneWav());
  const words = page.getByLabel("Or describe it in words (optional)");
  await words.fill("just the vocals and the drums please, as flac");
  await page.getByRole("button", { name: "Fill in from my words" }).click();
  await expect(page.getByText(/Understood as: Vocals, Drums\./)).toBeVisible();
  for (const [stem, ticked] of [
    ["Vocals", true],
    ["Drums", true],
    ["Bass", false],
    ["Guitar", false],
    ["Piano", false],
    ["Other", false],
  ] as const) {
    const box = page.getByRole("checkbox", { name: stem, exact: true });
    await (ticked ? expect(box).toBeChecked() : expect(box).not.toBeChecked());
  }
  await expect(page.getByLabel("File format")).toHaveValue("flac");
  await expect(page.getByRole("button", { name: "Upload and separate" })).toBeEnabled();

  // Words that name no stem: the page says so and will not upload.
  await words.fill("make it sound nicer");
  await page.getByRole("button", { name: "Fill in from my words" }).click();
  await expect(page.getByText("I could not tell for sure what you asked for.")).toBeVisible();
  await expect(page.getByText("No stem was recognised in your words.")).toBeVisible();
  await expect(page.getByRole("checkbox", { name: "Vocals", exact: true })).not.toBeChecked();
  await expect(page.getByRole("button", { name: "Upload and separate" })).toBeDisabled();
  await expect(page.getByText("Tick at least one stem.")).toBeVisible();
  // Ticking a stem by hand is the confirmation.
  await page.getByRole("checkbox", { name: "Piano", exact: true }).check();
  await expect(page.getByRole("button", { name: "Upload and separate" })).toBeEnabled();
});

test("a file the page can see is wrong is refused before it is sent", async ({ page }) => {
  let uploads = 0;
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().endsWith("/api/jobs")) {
      uploads += 1;
    }
  });
  await page.goto("/");
  await page
    .locator('input[type="file"]')
    .setInputFiles({ name: "notes.pdf", mimeType: "application/pdf", buffer: Buffer.from("%PDF") });
  await page.getByRole("checkbox", { name: "Vocals", exact: true }).check();
  await expect(errorIn(page)).toContainText("This file type is not supported.");
  await expect(errorIn(page)).toContainText("wav");
  await expect(page.getByRole("button", { name: "Upload and separate" })).toBeDisabled();

  // Too large: the limit is whatever the API says it is. Here the API's answer is given
  // a small limit, since a browser test cannot hand over a 250 MB file.
  await page.route("**/api/config", async (route) => {
    const response = await route.fetch();
    const config = await response.json();
    config.uploads.max_bytes = 100_000;
    await route.fulfill({ response, json: config });
  });
  await page.reload();
  await expect(page.getByText(/Up to 97\.7 KB and 20 minutes/)).toBeVisible();
  await chooseFile(page, "Test Tone.wav", toneWav());
  await page.getByRole("checkbox", { name: "Vocals", exact: true }).check();
  await expect(errorIn(page)).toHaveText("This file is 517 KB. The limit is 97.7 KB.");
  await expect(page.getByRole("button", { name: "Upload and separate" })).toBeDisabled();
  expect(uploads).toBe(0);
});

test("a refusal by the API is shown in the API's words", async ({ page }) => {
  await page.goto("/");
  // The name passes; the content is not audio, which only the server can tell.
  const fake = Buffer.from("%PDF-1.7 this is not audio at all ".repeat(400));
  await chooseFile(page, "renamed.mp3", fake);
  await page.getByRole("checkbox", { name: "Vocals", exact: true }).check();
  await page.getByRole("button", { name: "Upload and separate" }).click();
  await expect(errorIn(page)).toHaveText(
    "The file could not be read as audio. It may be damaged or not an audio file.",
  );
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("button", { name: "Upload and separate" })).toBeEnabled();
});

test("a failed job shows why, and how far it got", async ({ page }) => {
  // The stand-in models do not fail, so the API's answer for this one job is replaced.
  const id = "0b9c2f6e-58d4-4c58-9d3b-6f3d1c8a7e21";
  const failed = {
    id,
    status: "failed",
    stage: "failed",
    name: "Broken Song.wav",
    request: {
      stems: ["vocals"],
      text: null,
      split_vocals: false,
      dereverb: false,
      format: null,
      quality_mode: "unspecified",
    },
    created_at: "2026-11-24T12:00:00.000+00:00",
    updated_at: "2026-11-24T12:01:00.000+00:00",
    expires_at: "2099-01-01T00:00:00.000+00:00",
    seconds_until_expiry: 600000,
    source: { bytes: 1000, duration_seconds: 3, sample_rate_hz: 44100, channels: 2 },
    progress: null,
    stems: [],
    error: "RuntimeError: the GPU fell over",
  };
  await page.route(`**/api/jobs/${id}`, (route) => route.fulfill({ json: failed }));
  await page.goto(`/jobs/${id}`);
  await expect(page.getByTestId("status")).toHaveText("Failed");
  await expect(errorIn(page)).toContainText("This job failed.");
  await expect(errorIn(page)).toContainText("RuntimeError: the GPU fell over");
  await expect(page.locator('ol.stages > li[data-state="current"]')).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Play", exact: true })).toHaveCount(0);
});

test("a job that is not this browser's is not found", async ({ page, browser }) => {
  const id = await uploadAndWait(page);
  // Another browser, with no cookie, asks for the same job.
  const stranger = await browser.newContext();
  const other = await stranger.newPage();
  await other.goto(`/jobs/${id}`);
  await expect(other.getByRole("heading", { name: "Job not found" })).toBeVisible();
  await expect(errorIn(other)).toContainText("only be opened in the browser that uploaded it");
  await other.goto("/jobs");
  await expect(other.getByTestId("no-jobs")).toBeVisible();
  await stranger.close();
});

test("the jobs page lists this browser's jobs and can delete everything", async ({ page }) => {
  await uploadAndWait(page, { name: "First.wav" });
  await uploadAndWait(page, { name: "Second.wav" });
  await page.goto("/jobs");
  const list = page.getByRole("list", { name: "Jobs" });
  await expect(list.getByRole("listitem")).toHaveCount(2);
  await expect(list.getByRole("listitem").first()).toContainText("Second.wav"); // newest first
  await expect(list.getByRole("listitem").first()).toContainText("Deleted in 6 days 23 hours");
  await page.getByRole("button", { name: "Delete everything" }).click();
  await expect(list.getByRole("listitem")).toHaveCount(2); // one click deletes nothing
  await page.getByRole("button", { name: "Yes, delete everything" }).click();
  await expect(page.getByTestId("no-jobs")).toBeVisible();
  await expect(page.getByText("Everything from this browser has been deleted")).toBeVisible();
  await page.reload();
  await expect(page.getByTestId("no-jobs")).toBeVisible();
});

test("terms and privacy are placeholders, say so first, and are linked from every page", async ({
  page,
}) => {
  const id = await uploadAndWait(page);
  for (const path of ["/", "/jobs", `/jobs/${id}`, "/terms", "/privacy"]) {
    await page.goto(path);
    const footer = page.getByRole("contentinfo");
    await expect(footer.getByRole("link", { name: "Terms" })).toHaveAttribute("href", "/terms");
    await expect(footer.getByRole("link", { name: "Privacy" })).toHaveAttribute("href", "/privacy");
  }
  for (const [path, title] of [
    ["/terms", "Terms"],
    ["/privacy", "Privacy"],
  ] as const) {
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1, name: title })).toBeVisible();
    // The first thing after the heading says what the page is not.
    const first = page.locator("main > div > *").nth(1);
    await expect(first).toContainText("This page is a placeholder.");
    await expect(first).toContainText(/not legal text|not a privacy policy/);
  }
});

test("every page is sent with the headers that keep it out of frames and caches", async ({
  page,
  request,
}) => {
  for (const path of ["/", "/jobs", "/terms", "/jobs/0b9c2f6e-58d4-4c58-9d3b-6f3d1c8a7e21"]) {
    const response = await page.goto(path);
    const headers = response?.headers() ?? {};
    expect(headers["x-frame-options"], path).toBe("DENY");
    expect(headers["x-content-type-options"], path).toBe("nosniff");
    expect(headers["referrer-policy"], path).toBe("no-referrer");
    expect(headers["content-security-policy"], path).toContain("frame-ancestors 'none'");
    expect(headers["permissions-policy"], path).toContain("microphone=()");
  }
  // The API's answers about jobs are never cached and never sniffed.
  const api = await request.get("http://localhost:8787/api/jobs");
  expect(api.headers()["cache-control"]).toBe("no-store");
  expect(api.headers()["x-content-type-options"]).toBe("nosniff");
});
