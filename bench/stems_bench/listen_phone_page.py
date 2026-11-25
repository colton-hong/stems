"""The listening page for a phone: one page that a file host serves beside the clips.

``stems_bench.listen pack`` writes it out with the kit's steps in place of ``__KIT__``.
Unlike the page of ``listen_page``, which talks to this machine's own server, this one
fetches its clips by relative address (``audio/<track>/<stem>/<name>.mp3``) and keeps the
marks itself: in the browser's own storage, and in the host's document store where the
host offers one to the page as ``claude.use("db")``. It can also copy the marks out as
text. Either way ``stems_bench.listen take`` brings them back.

It is written as a host that wraps pages takes it: the content alone, with no doctype
or ``<html>`` of its own. ``pack`` also writes it inside a whole document.

Phones need three things the other page does not: sound may start only from a tap; an
iPhone mutes a page's own sound with its silent switch unless the page is also playing a
media element (so this one plays a silent file while a clip plays); and there is no
keyboard. The test tone and the level bar are there to tell "the page is making no sound"
from "the phone is not playing it".
"""

PAGE = """<title>Stems Listening Test</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap">
<style>
  /* Layout: one column for a phone. The transport bar sticks to the top of a step and the
     marks scroll under it, so a version can be switched while it is being marked. */
  :root {
    --bg: #f2f5f4;
    --surface: #ffffff;
    --line: #d3dbd9;
    --fg: #16201f;
    --muted: #586866;
    --accent: #0d6e6a;
    --accent-fg: #ffffff;
    --accent-soft: #d9eeeb;
    --warn: #a3301d;
    --font-display: "Space Grotesk", "Avenir Next", "Segoe UI", system-ui, sans-serif;
    --font-body: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0f1514; --surface: #182120; --line: #2c3a38; --fg: #e7efed; --muted: #9bacaa;
      --accent: #5fcac1; --accent-fg: #052321; --accent-soft: #1c3533; --warn: #ff9c88;
      color-scheme: dark;
    }
  }
  :root[data-theme="dark"] {
    --bg: #0f1514; --surface: #182120; --line: #2c3a38; --fg: #e7efed; --muted: #9bacaa;
    --accent: #5fcac1; --accent-fg: #052321; --accent-soft: #1c3533; --warn: #ff9c88;
    color-scheme: dark;
  }
  body {
    background: var(--bg); color: var(--fg);
    font: 16px/1.5 var(--font-body);
    padding-block: 16px 56px;
    padding-left: max(16px, env(safe-area-inset-left, 0px));
    padding-right: max(16px, env(safe-area-inset-right, 0px));
  }
  body::before {
    content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px);
    z-index: 10; background: var(--bg);
  }
  .page { max-width: 36rem; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }
  h1 { font: 600 1.35rem/1.2 var(--font-display); margin: 0; text-wrap: balance; }
  h2 { font: 600 1.15rem/1.25 var(--font-display); margin: 0; text-wrap: balance; }
  p { margin: 0; }
  .muted { color: var(--muted); }
  .small { font-size: .875rem; }
  .stack { display: flex; flex-direction: column; gap: 12px; }
  .where { display: flex; flex-direction: column; gap: 6px; }
  .where p { font-variant-numeric: tabular-nums; }
  .bar { height: 6px; border-radius: 3px; background: var(--line); overflow: hidden; }
  .bar > span { display: block; height: 100%; width: 0; background: var(--accent); }
  .card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
  button {
    font: 500 1rem/1.2 var(--font-body); color: var(--fg); background: var(--surface);
    border: 1px solid var(--line); border-radius: 10px; padding: 0 16px; min-height: 48px;
    cursor: pointer; touch-action: manipulation;
  }
  button.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-fg); font-weight: 600; }
  button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: var(--accent-fg); }
  button:disabled { opacity: .5; cursor: default; }
  button:focus-visible, textarea:focus-visible, input:focus-visible + span {
    outline: 2px solid var(--accent); outline-offset: 2px;
  }
  .transport {
    position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5;
    background: var(--bg); padding-block: 10px; border-bottom: 1px solid var(--line);
    display: flex; flex-direction: column; gap: 10px;
  }
  .row { display: flex; gap: 8px; align-items: center; }
  #play { flex: 0 0 6.5rem; }
  .clock { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; gap: 4px; }
  .clock p { font-variant-numeric: tabular-nums; }
  input[type="range"] { width: 100%; margin: 0; accent-color: var(--accent); min-height: 24px; }
  .choices { display: grid; grid-auto-flow: column; grid-auto-columns: 1fr; gap: 8px; }
  .choices button { padding: 0 4px; font-family: var(--font-display); font-weight: 600; }
  .meter { height: 4px; border-radius: 2px; background: var(--line); overflow: hidden; }
  .meter > span { display: block; height: 100%; width: 0; background: var(--accent); }
  fieldset { margin: 0; min-width: 0; }
  fieldset.card { display: flex; flex-direction: column; gap: 14px; }
  legend { float: left; width: 100%; padding: 0; }
  .version { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .version h3 { font: 600 1.1rem/1.2 var(--font-display); margin: 0; }
  .version button { min-height: 40px; }
  .mark { display: flex; flex-direction: column; gap: 6px; }
  .mark .label { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
  .mark .name { font-weight: 600; }
  .scale { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; }
  .scale label { position: relative; display: block; }
  .scale input { position: absolute; inset: 0; opacity: 0; margin: 0; width: 100%; height: 100%; cursor: pointer; }
  .scale span {
    display: flex; align-items: center; justify-content: center; min-height: 46px;
    border: 1px solid var(--line); border-radius: 10px; background: var(--bg);
    font: 600 1.05rem/1 var(--font-display); font-variant-numeric: tabular-nums;
  }
  .scale input:checked + span { background: var(--accent); border-color: var(--accent); color: var(--accent-fg); }
  textarea {
    font: inherit; color: var(--fg); background: var(--bg); border: 1px solid var(--line);
    border-radius: 10px; width: 100%; min-height: 4.5rem; padding: 10px; box-sizing: border-box;
  }
  .nav { display: grid; grid-template-columns: 1fr 2fr; gap: 8px; }
  .problem { color: var(--warn); font-weight: 500; }
  .problem:empty, .note:empty { display: none; }
  ul.plain { margin: 0; padding-left: 1.1rem; display: flex; flex-direction: column; gap: 6px; }
  textarea.out { min-height: 9rem; font: .8rem/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }
</style>

<div class="page">
  <header class="where">
    <h1>Listening test</h1>
    <p class="muted small" id="where">Loading…</p>
    <div class="bar" role="progressbar" aria-label="Steps marked" aria-valuemin="0" aria-valuenow="0" id="progress"><span></span></div>
  </header>

  <section class="card stack" id="intro">
    <p>Each step is one stem of one song, in two or three versions called A, B and C. Which
    is which is hidden, and it changes from step to step. One of them may be the true stem
    from the recording session.</p>
    <ul class="plain">
      <li>Press play and switch between the versions as often as you like. They stay in time.</li>
      <li><b>Song</b> plays the same half minute of the full mix. It is for reference and is not marked.</li>
      <li>Give every version an <b>overall</b> mark. Bleed and artifacts are optional.</li>
      <li>Use headphones, at one volume for the whole test.</li>
    </ul>
    <p class="muted small" id="size"></p>
    <p class="muted small">Your marks are saved after every step, so you can stop and come back.
</p>
    <button id="tone">Play a test tone</button>
    <p class="muted small" id="tone-note">Try this first: you should hear a short tone.</p>
    <button class="primary" id="start">Start</button>
  </section>

  <section class="stack" id="step" hidden>
    <h2 id="title"></h2>
    <div class="transport">
      <div class="row">
        <button id="play" aria-pressed="false" disabled>Play</button>
        <div class="clock">
          <input type="range" id="seek" min="0" max="30" step="0.1" value="0" aria-label="Position in the clip">
          <p class="muted small" id="clock" aria-live="off">Loading the audio…</p>
        </div>
      </div>
      <div class="choices" id="choices" role="group" aria-label="Which version is playing"></div>
      <div class="meter" aria-hidden="true"><span id="level"></span></div>
      <p class="muted small note" id="sound"></p>
    </div>
    <div class="stack" id="marks"></div>
    <div class="card stack">
      <label for="note"><b>Anything you noticed</b> <span class="muted">(optional)</span></label>
      <textarea id="note" maxlength="2000"></textarea>
    </div>
    <p class="problem" id="problem" role="alert"></p>
    <div class="nav">
      <button id="back">Back</button>
      <button class="primary" id="next">Save and next</button>
    </div>
    <p class="muted small note" id="sync"></p>
  </section>

  <section class="card stack" id="done" hidden>
    <h2>That was the last one</h2>
    <p id="done-text"></p>
    <p class="muted small" id="done-sync"></p>
    <button id="retry" hidden>Send the marks again</button>
    <button id="review">Go back through the steps</button>
    <button id="copy">Copy my marks</button>
    <textarea class="out" id="out" readonly hidden aria-label="Your marks, as text"></textarea>
  </section>
</div>

<audio id="unmute" src="silence.wav" loop playsinline preload="auto" x-webkit-airplay="deny"></audio>
<script type="application/json" id="kit">__KIT__</script>
<script>
"use strict";
const KIT = JSON.parse(document.getElementById("kit").textContent);
const $ = (id) => document.getElementById(id);
const LOCAL_KEY = "stems-listening-" + KIT.run;
const REQUIRED = "overall";

let saved = {};            // item id -> {item, ratings, note, saved_at, synced}
let index = 0;
let db = null;             // the shared store, once it answers
let store = "waiting";     // waiting | ready | absent
let context = null, buffers = {}, sources = [], gains = {};
let playing = false, startedAt = 0, offset = 0, chosen = null, loading = 0, seeking = false;
let wake = null;
let master = null, analyser = null, meter = null;

// -- marks: this phone first, then the store ------------------------------------------------

function readLocal() {
  try {
    const text = localStorage.getItem(LOCAL_KEY);
    const value = text ? JSON.parse(text) : {};
    return value && typeof value === "object" ? value : {};
  } catch (error) { return {}; }
}
function writeLocal() {
  try { localStorage.setItem(LOCAL_KEY, JSON.stringify(saved)); } catch (error) { /* private mode */ }
}
function docId(id) { return id.replace("/", "-"); }
function clean(item, body) {
  // Marks from the store are data, not markup: keep whole numbers from 1 to 5 and text only.
  if (!body || typeof body !== "object" || !body.ratings || typeof body.ratings !== "object") return null;
  const ratings = {};
  for (const sample of item.samples) {
    const given = body.ratings[sample];
    if (!given || typeof given !== "object") return null;
    ratings[sample] = {};
    for (const rating of KIT.ratings) {
      const value = given[rating];
      ratings[sample][rating] = Number.isInteger(value) && value >= 1 && value <= 5 ? value : null;
    }
    if (ratings[sample][REQUIRED] === null) return null;
  }
  return {
    item: item.id, ratings,
    note: typeof body.note === "string" ? body.note.slice(0, 2000) : "",
    saved_at: typeof body.saved_at === "string" ? body.saved_at : "",
  };
}
function within(promise, ms) {
  return Promise.race([promise, new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), ms))]);
}
async function push(id) {
  const entry = saved[id];
  if (!db || !entry || entry.synced) return Boolean(entry && entry.synced);
  try {
    await within(db.doc("marks/" + docId(id)).set({
      item: entry.item, ratings: entry.ratings, note: entry.note, saved_at: entry.saved_at,
    }), 10000);
    if (saved[id] === entry) { entry.synced = true; writeLocal(); }
    return true;
  } catch (error) { return false; }
}
async function pushAll() {
  for (const item of KIT.items) {
    if (saved[item.id] && !saved[item.id].synced) await push(item.id);  // one write at a time
  }
  showWhere();
}
function unsynced() { return KIT.items.filter((item) => saved[item.id] && !saved[item.id].synced).length; }
function marked() { return KIT.items.filter((item) => saved[item.id]).length; }

async function connect() {
  let space = null;
  try { space = window.claude && window.claude.use ? await window.claude.use("db") : null; }
  catch (error) { space = null; }
  if (!space) { store = "absent"; showWhere(); return; }
  try {
    const snapshot = await within(space.collection("marks").get(), 10000);
    const byId = Object.fromEntries(KIT.items.map((item) => [item.id, item]));
    for (const doc of snapshot.docs) {
      const body = doc.data();
      const item = body && byId[body.item];
      const entry = item ? clean(item, body) : null;
      // Marks made on this phone and not sent yet are newer than what the store holds.
      if (entry && !(saved[item.id] && saved[item.id].synced === false)) {
        saved[item.id] = { ...entry, synced: true };
      }
    }
    db = space; store = "ready";
    writeLocal();
  } catch (error) { db = space; store = "ready"; }
  showWhere();
  if ($("step").hidden) offerStart();
  await pushAll();
}

// -- the page ------------------------------------------------------------------------------

function showWhere() {
  const total = KIT.items.length, count = marked();
  let text = `${total} steps, ${count} marked`;
  if (!$("step").hidden) text = `Step ${index + 1} of ${total} · ${count} marked`;
  if (store === "waiting") text += " · checking for saved marks…";
  $("where").textContent = text;
  const bar = $("progress");
  bar.setAttribute("aria-valuemax", String(total));
  bar.setAttribute("aria-valuenow", String(count));
  bar.firstElementChild.style.width = `${(100 * count) / total}%`;
  const waiting = unsynced();
  let note = "";
  if (store === "absent") note = "Marks are being kept on this phone only. At the end, copy them and send them to Claude.";
  else if (waiting) note = `${waiting} ${waiting === 1 ? "step is" : "steps are"} saved on this phone and not yet sent. They are sent again at the next save.`;
  $("sync").textContent = note;
}
function firstOpen() {
  const first = KIT.items.findIndex((item) => !saved[item.id]);
  return first < 0 ? 0 : first;
}
function offerStart() {
  const count = marked();
  $("start").textContent = count === 0 ? "Start" : count === KIT.items.length
    ? "Go through the steps again" : `Continue at step ${firstOpen() + 1}`;
}

function names(item) { return ["song", ...item.samples]; }
function label(name) { return name === "song" ? "Song" : name; }
function address(item, name) {
  return `audio/${item.track}/${item.stem}/${name}.${name === "song" ? KIT.song_format : KIT.clip_format}`;
}
function length() {
  const all = Object.values(buffers).map((buffer) => buffer.duration);
  return all.length ? Math.min(...all) : 0;
}
function position() {
  if (!playing) return offset;
  const whole = length();
  return whole ? (offset + context.currentTime - startedAt) % whole : 0;
}
function unlock() {
  // Called inside every tap that should make sound. A phone lets a page make sound only
  // from a tap, and an iPhone also mutes a page's own sound with the silent switch unless
  // the page is playing a media element; the one here plays silence.
  if (!context) {
    const Context = window.AudioContext || window.webkitAudioContext;
    context = new Context();
    master = context.createGain();
    analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    meter = new Uint8Array(analyser.fftSize);
    master.connect(analyser);
    analyser.connect(context.destination);
  }
  try { if (navigator.audioSession) navigator.audioSession.type = "playback"; } catch (error) { /* older phones */ }
  try {
    const quiet = $("unmute");
    if (quiet.paused) { const begun = quiet.play(); if (begun && begun.catch) begun.catch(() => {}); }
  } catch (error) { /* no media element: the page's own sound may still play */ }
  try { const resumed = context.resume(); if (resumed && resumed.catch) resumed.catch(() => {}); } catch (error) { /* old */ }
}
function quiet() {
  try { $("unmute").pause(); } catch (error) { /* nothing to pause */ }
}
function loudness() {
  if (!analyser) return 0;
  analyser.getByteTimeDomainData(meter);
  let sum = 0;
  for (const value of meter) { const x = (value - 128) / 128; sum += x * x; }
  return Math.sqrt(sum / meter.length);
}
function stop() {
  if (playing) offset = position();
  for (const source of sources) { try { source.stop(); } catch (error) { /* not started */ } }
  sources = []; playing = false;
  $("play").textContent = "Play";
  $("play").setAttribute("aria-pressed", "false");
  if (wake) { wake.release().catch(() => {}); wake = null; }
  quiet();
}
function play() {
  const whole = length();
  if (!whole) return;
  stop();
  unlock();
  const when = context.currentTime + 0.06;
  gains = {};
  for (const [name, buffer] of Object.entries(buffers)) {
    const source = context.createBufferSource();
    source.buffer = buffer;
    // Every version loops over the same span and starts at the same instant: only gains change.
    source.loop = true; source.loopStart = 0; source.loopEnd = whole;
    const gain = context.createGain();
    gain.gain.value = name === chosen ? 1 : 0;
    source.connect(gain).connect(master);
    source.start(when, offset % whole);
    sources.push(source);
    gains[name] = gain;
  }
  startedAt = when; playing = true;
  $("play").textContent = "Pause";
  $("play").setAttribute("aria-pressed", "true");
  if (navigator.wakeLock && navigator.wakeLock.request) {
    navigator.wakeLock.request("screen").then((lock) => { if (playing) wake = lock; else lock.release(); }).catch(() => {});
  }
}
function choose(name) {
  chosen = name;
  for (const [other, gain] of Object.entries(gains)) {
    gain.gain.setTargetAtTime(other === name ? 1 : 0, context.currentTime, 0.01);
  }
  for (const button of document.querySelectorAll("[data-choose]")) {
    button.setAttribute("aria-pressed", String(button.dataset.choose === name));
  }
  tick();
}
function tick() {
  if ($("step").hidden) return;
  const whole = length();
  if (!whole) return;
  const at = position();
  const what = chosen === "song" ? "the song" : `version ${chosen}`;
  $("clock").textContent = `${at.toFixed(0)} s of ${whole.toFixed(0)} s · ${playing ? "playing" : "paused on"} ${what}`;
  if (!seeking) $("seek").value = String(at);
  $("level").style.width = `${playing ? Math.min(100, Math.round(Math.sqrt(loudness()) * 250)) : 0}%`;
  $("sound").textContent = !playing ? ""
    : context.state !== "running"
      ? `The phone has not let this page play yet (${context.state}). Tap Pause, then Play.`
      : "No sound while the bar above moves? Turn off silent mode, turn the volume up, and check Bluetooth.";
}
function decode(data) {
  return new Promise((resolve, reject) => {
    const answer = context.decodeAudioData(data, resolve, reject);
    if (answer && answer.catch) answer.catch(reject);
  });
}
async function loadAudio(item) {
  stop();
  buffers = {}; gains = {}; offset = 0;
  const mine = ++loading;
  $("play").disabled = true; $("seek").disabled = true; $("seek").value = "0";
  const wanted = names(item);
  let ready = 0;
  $("clock").textContent = `Loading the audio… 0 of ${wanted.length}`;
  const decoded = await Promise.all(wanted.map(async (name) => {
    const response = await fetch(address(item, name));
    if (!response.ok) throw new Error(`${label(name)}: ${response.status}`);
    const buffer = await decode(await response.arrayBuffer());
    ready += 1;
    if (mine === loading) $("clock").textContent = `Loading the audio… ${ready} of ${wanted.length}`;
    return buffer;
  }));
  if (mine !== loading) return false;  // the listener has moved to another step
  wanted.forEach((name, i) => { buffers[name] = decoded[i]; });
  $("seek").max = String(length());
  $("play").disabled = false; $("seek").disabled = false;
  return true;
}

function scale(sample, rating, value) {
  const wrap = document.createElement("div");
  wrap.className = "mark";
  const top = document.createElement("div");
  top.className = "label";
  const name = document.createElement("span");
  name.className = "name";
  name.textContent = rating === REQUIRED ? "Overall" : rating[0].toUpperCase() + rating.slice(1);
  const help = document.createElement("span");
  help.className = "muted small";
  help.textContent = KIT.short[rating];
  top.append(name, help);
  const row = document.createElement("div");
  row.className = "scale";
  row.setAttribute("role", "radiogroup");
  row.setAttribute("aria-label", `Version ${sample}: ${KIT.help[rating]}`);
  for (let mark = 1; mark <= 5; mark += 1) {
    const box = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = `${sample}-${rating}`;
    input.id = `mark-${sample}-${rating}-${mark}`;
    input.value = String(mark);
    input.checked = value === mark;
    input.dataset.was = input.checked ? "1" : "";
    input.addEventListener("click", () => {
      // An optional mark can be taken back by tapping it again.
      if (rating !== REQUIRED && input.dataset.was === "1") input.checked = false;
      for (const other of row.querySelectorAll("input")) other.dataset.was = other.checked ? "1" : "";
    });
    const face = document.createElement("span");
    face.textContent = String(mark);
    box.append(input, face);
    row.append(box);
  }
  wrap.append(top, row);
  return wrap;
}

async function show(i) {
  index = i;
  $("intro").hidden = true; $("done").hidden = true; $("step").hidden = false;
  $("problem").textContent = "";
  const item = KIT.items[index];
  const before = saved[item.id] || { ratings: {}, note: "" };
  $("title").textContent = `Song ${Number(item.track.slice(1))} · ${item.stem.replace("_", " ")}`;
  $("back").disabled = index === 0;
  $("next").textContent = index + 1 < KIT.items.length ? "Save and next" : "Save and finish";
  $("note").value = before.note || "";
  const choices = $("choices");
  choices.replaceChildren();
  for (const name of names(item)) {
    const button = document.createElement("button");
    button.dataset.choose = name;
    button.textContent = label(name);
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => choose(name));
    choices.append(button);
  }
  const marks = $("marks");
  marks.replaceChildren();
  for (const sample of item.samples) {
    const card = document.createElement("fieldset");
    card.className = "card";
    const legend = document.createElement("legend");
    const head = document.createElement("div");
    head.className = "version";
    const name = document.createElement("h3");
    name.textContent = `Version ${sample}`;
    const listen = document.createElement("button");
    listen.type = "button";
    listen.dataset.choose = sample;
    listen.textContent = "Listen";
    listen.setAttribute("aria-pressed", "false");
    listen.setAttribute("aria-label", `Listen to version ${sample}`);
    listen.addEventListener("click", () => { choose(sample); if (!playing) play(); });
    head.append(name, listen);
    legend.append(head);
    card.append(legend);
    for (const rating of KIT.ratings) {
      card.append(scale(sample, rating, (before.ratings[sample] || {})[rating]));
    }
    marks.append(card);
  }
  showWhere();
  window.scrollTo(0, 0);
  try {
    if (await loadAudio(item)) choose(item.samples[0]);
  } catch (error) {
    $("clock").textContent = "No audio";
    $("problem").textContent = `The audio for this step could not be loaded (${error.message}). Check the connection, then go back and forward to try again.`;
  }
}

function collect(item) {
  const ratings = {};
  for (const sample of item.samples) {
    ratings[sample] = {};
    for (const rating of KIT.ratings) {
      const picked = document.querySelector(`input[name="${sample}-${rating}"]:checked`);
      ratings[sample][rating] = picked ? Number(picked.value) : null;
    }
  }
  return { ratings, note: $("note").value };
}
async function save() {
  const item = KIT.items[index];
  const body = collect(item);
  const missing = item.samples.filter((sample) => body.ratings[sample][REQUIRED] === null);
  if (missing.length) {
    const which = missing.length > 1
      ? `versions ${missing.slice(0, -1).join(", ")} and ${missing[missing.length - 1]}`
      : `version ${missing[0]}`;
    $("problem").textContent = `Give an overall mark to ${which} first.`;
    return false;
  }
  const before = saved[item.id];
  const same = before && JSON.stringify([before.ratings, before.note]) === JSON.stringify([body.ratings, body.note]);
  if (!same) {
    saved[item.id] = { item: item.id, ...body, saved_at: new Date().toISOString(), synced: false };
    writeLocal();
  }
  await pushAll();
  return true;
}
function finish() {
  stop();
  $("step").hidden = true; $("done").hidden = false;
  const waiting = unsynced();
  $("done-text").textContent = `All ${KIT.items.length} steps have their marks. Tell Claude you have finished, and it will write the report with the versions' names revealed.`;
  $("done-sync").textContent = store === "ready" && !waiting
    ? "Every mark has been sent."
    : store === "ready"
      ? `${waiting} ${waiting === 1 ? "step has" : "steps have"} not been sent yet. Try again, or copy the marks and paste them to Claude.`
      : "The marks are on this phone only. Copy them and paste them to Claude.";
  $("retry").hidden = !(store === "ready" && waiting);
  showWhere();
}

$("start").addEventListener("click", async () => {
  unlock();
  quiet();
  await show(marked() === KIT.items.length ? 0 : firstOpen());
});
$("tone").addEventListener("click", () => {
  unlock();
  const tone = context.createOscillator();
  const gain = context.createGain();
  const at = context.currentTime + 0.05;
  tone.frequency.value = 440;
  gain.gain.setValueAtTime(0.0001, at);
  gain.gain.exponentialRampToValueAtTime(0.2, at + 0.05);
  gain.gain.setValueAtTime(0.2, at + 0.8);
  gain.gain.exponentialRampToValueAtTime(0.0001, at + 1.0);
  tone.connect(gain).connect(master);
  tone.start(at);
  tone.stop(at + 1.05);
  tone.onended = () => { if (!playing) quiet(); };
  setTimeout(() => {
    $("tone-note").textContent = context.state === "running"
      ? "A tone was played. If you did not hear it, turn off silent mode, turn the volume up, and check that the sound is not going to a Bluetooth speaker. Then try again."
      : `The phone has not let this page make sound yet (${context.state}). Tap the button again.`;
  }, 500);
});
$("play").addEventListener("click", () => (playing ? stop() : play()));
$("seek").addEventListener("pointerdown", () => { seeking = true; });
$("seek").addEventListener("input", () => { seeking = true; });
$("seek").addEventListener("change", () => {
  const was = playing;
  stop();
  offset = Number($("seek").value) || 0;
  seeking = false;
  if (was) play(); else tick();
});
$("back").addEventListener("click", () => { stop(); show(index - 1); });
$("review").addEventListener("click", () => show(0));
$("retry").addEventListener("click", async () => { await pushAll(); finish(); });
$("next").addEventListener("click", async () => {
  $("next").disabled = true;
  try {
    if (!(await save())) return;
    stop();
    if (index + 1 < KIT.items.length) await show(index + 1); else finish();
  } finally { $("next").disabled = false; }
});
$("copy").addEventListener("click", async () => {
  const text = JSON.stringify({ run: KIT.run, items: Object.fromEntries(
    KIT.items.filter((item) => saved[item.id]).map((item) => {
      const { ratings, note, saved_at } = saved[item.id];
      return [item.id, { ratings, note, saved_at }];
    })) });
  const out = $("out");
  out.value = text; out.hidden = false;
  try { await navigator.clipboard.writeText(text); $("copy").textContent = "Copied"; }
  catch (error) { out.focus(); out.select(); $("copy").textContent = "Select the text below and copy it"; }
});
document.addEventListener("visibilitychange", () => { if (document.hidden && playing) stop(); });

saved = readLocal();
$("size").textContent = KIT.size_note;
showWhere();
offerStart();
setInterval(tick, 250);
connect();
</script>
"""
