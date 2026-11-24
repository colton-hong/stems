import assert from "node:assert/strict";
import { test } from "node:test";

import {
  extensionOf,
  fileProblem,
  formatBytes,
  formatClock,
  formatRemaining,
  qualityMatters,
  secondsUntil,
  stemLabel,
} from "./format.ts";

test("sizes", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(1023), "1023 B");
  assert.equal(formatBytes(1024), "1.0 KB");
  assert.equal(formatBytes(1_500_000), "1.4 MB");
  assert.equal(formatBytes(262_144_000), "250 MB");
  assert.equal(formatBytes(5 * 1024 ** 3), "5.0 GB");
});

test("clock times", () => {
  assert.equal(formatClock(0), "0:00");
  assert.equal(formatClock(75.9), "1:15");
  assert.equal(formatClock(3725), "1:02:05");
  assert.equal(formatClock(-3), "0:00");
});

test("time remaining uses the two largest units", () => {
  assert.equal(formatRemaining(7 * 86400), "7 days");
  assert.equal(formatRemaining(6 * 86400 + 23 * 3600 + 59 * 60), "6 days 23 hours");
  assert.equal(formatRemaining(86400 + 3600), "1 day 1 hour");
  assert.equal(formatRemaining(3 * 3600 + 60), "3 hours 1 minute");
  assert.equal(formatRemaining(2 * 3600), "2 hours");
  assert.equal(formatRemaining(61), "1 minute");
  assert.equal(formatRemaining(59), "less than a minute");
  assert.equal(formatRemaining(-5), "less than a minute");
});

test("seconds until a timestamp never goes below zero", () => {
  const now = Date.parse("2026-11-24T12:00:00.000+00:00");
  assert.equal(secondsUntil("2026-11-24T12:01:30.000+00:00", now), 90);
  assert.equal(secondsUntil("2026-11-24T11:00:00.000+00:00", now), 0);
});

test("stem names", () => {
  assert.equal(stemLabel("vocals"), "Vocals");
  assert.equal(stemLabel("vocals_dry"), "Vocals, de-reverbed");
  assert.equal(stemLabel("something_new"), "something new");
});

test("a file is checked against the API's limits before it is sent", () => {
  const limits = { max_bytes: 1000, allowed_extensions: ["wav", "mp3", "flac"] };
  assert.equal(fileProblem({ name: "Song.WAV", size: 1000 }, limits), null);
  assert.match(fileProblem({ name: "notes.pdf", size: 10 }, limits) ?? "", /not supported.*flac, mp3, wav/);
  assert.match(fileProblem({ name: "noextension", size: 10 }, limits) ?? "", /not supported/);
  assert.match(fileProblem({ name: "a.mp3", size: 1001 }, limits) ?? "", /The limit is 1000 B/);
  assert.equal(fileProblem({ name: "a.mp3", size: 0 }, limits), "This file is empty.");
  assert.equal(extensionOf("archive.tar.FLAC"), "flac");
  assert.equal(extensionOf("none"), "");
});

test("the quality menu matters only when the settings can differ", () => {
  const modes = ["unspecified", "fastest", "balanced", "best_quality"];
  const same = {
    unspecified: "single_best",
    fastest: "single_best",
    balanced: "single_best",
    best_quality: "single_best",
  };
  // The rules plan the job and every setting gives one thing: nothing to choose.
  assert.equal(
    qualityMatters({ quality_modes: modes, quality_gives: same, decided_by: "rules" }),
    false,
  );
  // One setting gives something else.
  assert.equal(
    qualityMatters({
      quality_modes: modes,
      quality_gives: { ...same, best_quality: "full_ensemble" },
      decided_by: "rules",
    }),
    true,
  );
  // Jev plans the job: it may give a setting a blend, whatever the table says.
  assert.equal(
    qualityMatters({ quality_modes: modes, quality_gives: same, decided_by: "jev" }),
    true,
  );
  // An API that does not say, or does not say for every setting: show the menu.
  assert.equal(qualityMatters({ quality_modes: modes }), true);
  assert.equal(
    qualityMatters({
      quality_modes: modes,
      quality_gives: { fastest: "single_best" },
      decided_by: "rules",
    }),
    true,
  );
});
