import assert from "node:assert/strict";
import { test } from "node:test";

import { stageLabel, stagesFor, statusLabel, stemStateLabel } from "./stages.ts";

const states = (status: string, stage: string, reached?: string) =>
  stagesFor(status, stage, reached).map((entry) => `${entry.key}:${entry.state}`);

test("a running job has one current stage, the earlier ones done", () => {
  const view = stagesFor("running", "separating");
  assert.deepEqual(
    view.map((entry) => entry.state),
    ["done", "done", "done", "done", "done", "current", "pending", "pending", "pending"],
  );
  assert.equal(view.filter((entry) => entry.state === "current")[0]?.label, "Separating stems");
});

test("a queued job is waiting for a worker", () => {
  assert.deepEqual(states("queued", "queued").slice(0, 3), [
    "uploaded:done",
    "queued:current",
    "transcoding:pending",
  ]);
});

test("retrying is shown as part of checking", () => {
  const current = stagesFor("running", "retrying").find((entry) => entry.state === "current");
  assert.equal(current?.key, "assessing");
  assert.equal(stageLabel("retrying"), "Trying a stem again");
});

test("a ready job has every stage done", () => {
  assert.ok(stagesFor("ready", "ready").every((entry) => entry.state === "done"));
});

test("a failed job has no current stage: what it reached is done, the rest is not", () => {
  const view = states("failed", "failed", "separating");
  assert.deepEqual(view.slice(4, 7), ["planning:done", "separating:pending", "assessing:pending"]);
  assert.ok(!view.some((entry) => entry.endsWith(":current")));
  assert.ok(stagesFor("failed", "failed").every((entry) => entry.state === "pending"));
});

test("labels", () => {
  assert.equal(statusLabel("running"), "Working");
  assert.equal(statusLabel("unheard_of"), "unheard_of");
  assert.equal(stemStateLabel("delivered_with_warning"), "done, with a note");
  assert.equal(stageLabel("packaging"), "Packaging downloads");
});
