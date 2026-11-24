import assert from "node:assert/strict";
import { test } from "node:test";

import {
  clampSeek,
  gainsFor,
  positionAt,
  START_DELAY,
  StemEngine,
  type ContextLike,
  type GainLike,
  type SourceLike,
} from "./player.ts";

class FakeSource implements SourceLike {
  buffer: unknown = null;
  onended: (() => void) | null = null;
  started: Array<{ when: number; offset: number }> = [];
  stopped = 0;
  connectedTo: unknown = null;
  connect(destination: unknown): unknown {
    this.connectedTo = destination;
    return destination;
  }
  start(when: number, offset: number): void {
    this.started.push({ when, offset });
  }
  stop(): void {
    this.stopped += 1;
  }
  disconnect(): void {}
}

class FakeGain implements GainLike {
  ramps: number[] = [];
  gain = {
    value: 1,
    setTargetAtTime: (value: number) => {
      this.ramps.push(value);
      this.gain.value = value;
    },
  };
  connect(destination: unknown): unknown {
    return destination;
  }
}

class FakeContext implements ContextLike {
  currentTime = 10;
  destination = { name: "speakers" };
  state = "running";
  sources: FakeSource[] = [];
  gains: FakeGain[] = [];
  resumed = 0;
  createGain(): GainLike {
    const gain = new FakeGain();
    this.gains.push(gain);
    return gain;
  }
  createBufferSource(): SourceLike {
    const source = new FakeSource();
    this.sources.push(source);
    return source;
  }
  async resume(): Promise<void> {
    this.resumed += 1;
    this.state = "running";
  }
}

function engineWith(names: string[], duration = 200): { engine: StemEngine; context: FakeContext } {
  const context = new FakeContext();
  const engine = new StemEngine(context);
  for (const name of names) {
    engine.addStem(name, { duration });
  }
  return { engine, context };
}

test("every stem is started at the same time and the same offset", async () => {
  const { engine, context } = engineWith(["vocals", "drums", "bass", "guitar", "piano", "other"]);
  await engine.play();
  assert.equal(context.sources.length, 6);
  const starts = context.sources.map((source) => source.started[0]);
  assert.deepEqual(new Set(starts.map((start) => start?.when)), new Set([10 + START_DELAY]));
  assert.deepEqual(new Set(starts.map((start) => start?.offset)), new Set([0]));
  assert.ok(context.sources.every((source) => source.started.length === 1));
  assert.deepEqual(
    engine.schedule.map((entry) => entry.name),
    ["vocals", "drums", "bass", "guitar", "piano", "other"],
  );
});

test("pause remembers the position and play resumes every stem from it", async () => {
  const { engine, context } = engineWith(["vocals", "drums"]);
  await engine.play();
  context.currentTime = 10 + START_DELAY + 42.5;
  assert.equal(engine.position, 42.5);
  engine.pause();
  assert.equal(engine.isPlaying, false);
  assert.ok(context.sources.every((source) => source.stopped === 1));
  context.currentTime += 100; // paused time does not move the position
  assert.equal(engine.position, 42.5);
  await engine.play();
  const resumed = context.sources.slice(2).map((source) => source.started[0]);
  assert.deepEqual(resumed, [
    { when: context.currentTime + START_DELAY, offset: 42.5 },
    { when: context.currentTime + START_DELAY, offset: 42.5 },
  ]);
});

test("seeking while playing restarts every stem together at the new offset", async () => {
  const { engine, context } = engineWith(["vocals", "drums", "bass"]);
  await engine.play();
  context.currentTime = 15;
  await engine.seek(120);
  assert.equal(engine.isPlaying, true);
  assert.ok(context.sources.slice(0, 3).every((source) => source.stopped === 1));
  const restarted = context.sources.slice(3).map((source) => source.started[0]);
  assert.equal(restarted.length, 3);
  assert.deepEqual(new Set(restarted.map((start) => start?.offset)), new Set([120]));
  assert.deepEqual(new Set(restarted.map((start) => start?.when)), new Set([15 + START_DELAY]));
});

test("seeking while paused only moves the position", async () => {
  const { engine, context } = engineWith(["vocals"]);
  await engine.seek(30);
  assert.equal(engine.position, 30);
  assert.equal(context.sources.length, 0);
  await engine.seek(9999);
  assert.equal(engine.position, 200);
  await engine.seek(-5);
  assert.equal(engine.position, 0);
});

test("solo and mute change gains and never restart a source", async () => {
  const { engine, context } = engineWith(["vocals", "drums", "bass"]);
  await engine.play();
  const before = context.sources.length;
  engine.setMuted("drums", true);
  assert.deepEqual(context.gains.map((gain) => gain.gain.value), [1, 0, 1]);
  engine.setSolo("vocals", true);
  assert.deepEqual(context.gains.map((gain) => gain.gain.value), [1, 0, 0]);
  engine.setSolo("bass", true);
  assert.deepEqual(context.gains.map((gain) => gain.gain.value), [1, 0, 1]);
  engine.setSolo("vocals", false);
  engine.setSolo("bass", false);
  engine.setMuted("drums", false);
  assert.deepEqual(context.gains.map((gain) => gain.gain.value), [1, 1, 1]);
  assert.equal(context.sources.length, before);
  assert.ok(context.sources.every((source) => source.stopped === 0 && source.started.length === 1));
});

test("the end of the audio stops playback and reports it", async () => {
  const { engine, context } = engineWith(["vocals", "drums"], 5);
  let ended = 0;
  engine.onEnded = () => {
    ended += 1;
  };
  await engine.play();
  context.currentTime = 10 + START_DELAY + 5;
  context.sources[0]?.onended?.();
  assert.equal(ended, 1);
  assert.equal(engine.isPlaying, false);
  assert.equal(engine.position, 5);
  // Play again starts from the beginning.
  await engine.play();
  assert.equal(context.sources[2]?.started[0]?.offset, 0);
});

test("a suspended context is resumed before anything is scheduled", async () => {
  const { engine, context } = engineWith(["vocals"]);
  context.state = "suspended";
  await engine.play();
  assert.equal(context.resumed, 1);
  assert.equal(context.sources.length, 1);
});

test("the gain rule", () => {
  const stem = (name: string, muted = false, solo = false) => ({ name, muted, solo });
  assert.deepEqual(gainsFor([stem("a"), stem("b")]), { a: 1, b: 1 });
  assert.deepEqual(gainsFor([stem("a", true), stem("b")]), { a: 0, b: 1 });
  assert.deepEqual(gainsFor([stem("a", false, true), stem("b")]), { a: 1, b: 0 });
  // Mute wins over solo on the same stem.
  assert.deepEqual(gainsFor([stem("a", true, true), stem("b")]), { a: 0, b: 0 });
});

test("position and seek limits", () => {
  assert.equal(positionAt(false, 12, 0, 999, 100), 12);
  assert.equal(positionAt(true, 12, 50, 49, 100), 12); // scheduled, not yet sounding
  assert.equal(positionAt(true, 12, 50, 53, 100), 15);
  assert.equal(positionAt(true, 90, 50, 80, 100), 100);
  assert.equal(clampSeek(Number.NaN, 100), 0);
  assert.equal(clampSeek(150, 100), 100);
});
