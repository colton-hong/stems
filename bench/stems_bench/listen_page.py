"""The listening page: one HTML document, served by ``stems_bench.listen``.

Kept as a string so that the page travels with the module however the project is
installed. It talks to four routes of the same server and to nothing else.
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listening test</title>
<style>
  :root { color-scheme: light dark; --line: #8884; --accent: #2563eb; }
  body { font: 16px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 46rem;
         padding: 1.5rem 1rem 4rem; }
  h1 { font-size: 1.4rem; margin: 0 0 .25rem; }
  .muted { opacity: .75; }
  .card { border: 1px solid var(--line); border-radius: .6rem; padding: 1rem; margin: 1rem 0; }
  button { font: inherit; padding: .55rem .9rem; border-radius: .5rem; cursor: pointer;
           border: 1px solid var(--line); background: transparent; color: inherit;
           min-height: 2.75rem; min-width: 2.75rem; }
  button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent);
                                color: white; }
  button.primary { background: var(--accent); border-color: var(--accent); color: white; }
  button:disabled { opacity: .5; cursor: default; }
  .row { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; }
  fieldset { border: 0; padding: 0; margin: .5rem 0 0; }
  legend { padding: 0; font-weight: 600; }
  .scale { display: flex; flex-wrap: wrap; gap: .25rem .9rem; align-items: center; }
  .scale span.name { min-width: 7.5rem; }
  .scale label { display: inline-flex; gap: .25rem; align-items: center; min-height: 2rem; }
  textarea { font: inherit; width: 100%; min-height: 3.5rem; box-sizing: border-box; }
  progress { width: 100%; }
  .error { color: #b91c1c; }
  kbd { border: 1px solid var(--line); border-radius: .25rem; padding: 0 .3rem; }
</style>
</head>
<body>
<h1>Listening test</h1>
<p class="muted" id="where">Loading…</p>
<progress id="progress" value="0" max="1" aria-label="How far through the test"></progress>

<section class="card" id="intro" hidden>
  <p>Each step is one stem of one song, made in several ways. The versions are called
  A, B, C and so on; which is which is hidden, and changes from step to step. One of them
  may be the true stem from the recording session.</p>
  <p>Press play, switch between the versions as often as you like (they stay in time),
  and give each one a mark. "Song" plays the full mix, for reference; it is not marked.
  Overall quality is needed for every version; the other two marks are optional.</p>
  <p>Keys: <kbd>Space</kbd> play or pause, <kbd>A</kbd> <kbd>B</kbd> <kbd>C</kbd> … choose
  a version, <kbd>S</kbd> the song. Your marks are saved after every step, so you can stop
  and come back.</p>
  <button class="primary" id="start">Start</button>
</section>

<section id="step" hidden>
  <div class="card">
    <h2 id="title" style="font-size:1.15rem;margin:0 0 .5rem"></h2>
    <div class="row" role="group" aria-label="Playback">
      <button id="play" aria-pressed="false">Play</button>
      <span id="choices" class="row" role="group" aria-label="Which version is playing"></span>
    </div>
    <p class="muted" id="clock" aria-live="off"></p>
  </div>
  <div id="marks"></div>
  <div class="card">
    <label for="note">Anything you noticed (optional)</label>
    <textarea id="note" maxlength="2000"></textarea>
  </div>
  <p class="error" id="problem" role="alert"></p>
  <div class="row">
    <button id="back">Back</button>
    <button class="primary" id="next">Save and next</button>
  </div>
</section>

<section class="card" id="done" hidden>
  <h2 style="font-size:1.15rem;margin:0 0 .5rem">That was the last one</h2>
  <p>Every step has its marks, and they are saved. You can go back and change any of
  them, or close this page.</p>
  <p>To see what the marks say, with the names of the versions revealed:</p>
  <pre>.venv/bin/python -m stems_bench.listen report</pre>
  <button id="review">Go back through the steps</button>
</section>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
let kit = null, saved = {}, index = 0;
let context = null, buffers = {}, sources = [], gains = {};
let playing = false, startedAt = 0, offset = 0, chosen = null, timer = null;

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

function names(item) { return ["song", ...item.samples]; }
function label(name) { return name === "song" ? "Song" : name; }

function stop() {
  for (const source of sources) { try { source.stop(); } catch (error) { /* not started */ } }
  sources = [];
  if (playing) offset = position();
  playing = false;
  $("play").textContent = "Play";
  $("play").setAttribute("aria-pressed", "false");
}

function duration() {
  const first = Object.values(buffers)[0];
  return first ? first.duration : 0;
}

function position() {
  if (!playing) return offset;
  const length = duration();
  return length ? (offset + context.currentTime - startedAt) % length : 0;
}

function play() {
  if (!Object.keys(buffers).length) return;
  stop();
  const when = context.currentTime + 0.05;
  for (const [name, buffer] of Object.entries(buffers)) {
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.loop = true;
    const gain = context.createGain();
    gain.gain.value = name === chosen ? 1 : 0;
    source.connect(gain).connect(context.destination);
    // Every version starts at the same instant and the same place: only gains change.
    source.start(when, offset % buffer.duration);
    sources.push(source);
    gains[name] = gain;
  }
  startedAt = when;
  playing = true;
  $("play").textContent = "Pause";
  $("play").setAttribute("aria-pressed", "true");
}

function choose(name) {
  chosen = name;
  for (const [other, gain] of Object.entries(gains)) {
    gain.gain.setTargetAtTime(other === name ? 1 : 0, context.currentTime, 0.01);
  }
  for (const button of $("choices").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.name === name));
  }
}

function tick() {
  const length = duration();
  const at = position();
  const what = chosen === "song" ? "the song" : `version ${chosen}`;
  $("clock").textContent = length
    ? `${at.toFixed(0)} s of ${length.toFixed(0)} s, ${playing ? "playing" : "paused on"} ${what}`
    : "Loading the audio…";
}

async function loadAudio(item) {
  stop();
  buffers = {}; gains = {}; offset = 0;
  $("play").disabled = true;
  const wanted = names(item);
  const decoded = await Promise.all(wanted.map(async (name) => {
    const url = name === "song"
      ? `/audio/${item.track}/mix.wav`
      : `/audio/${item.track}/${item.stem}/${name}.wav`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status}`);
    return context.decodeAudioData(await response.arrayBuffer());
  }));
  wanted.forEach((name, i) => { buffers[name] = decoded[i]; });
  $("play").disabled = false;
}

function scale(sample, rating, help, value) {
  const wrap = document.createElement("div");
  wrap.className = "scale";
  wrap.setAttribute("role", "radiogroup");
  wrap.setAttribute("aria-label", `${sample}: ${help}`);
  const name = document.createElement("span");
  name.className = "name";
  name.textContent = rating === "overall" ? "Overall" : rating[0].toUpperCase() + rating.slice(1);
  name.title = help;
  wrap.append(name);
  for (let mark = 1; mark <= 5; mark += 1) {
    const box = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = `${sample}-${rating}`;
    input.value = String(mark);
    input.checked = value === mark;
    box.append(input, String(mark));
    wrap.append(box);
  }
  return wrap;
}

async function show(i) {
  index = i;
  $("done").hidden = true;
  $("problem").textContent = "";
  const item = kit.items[index];
  const before = saved[item.id] || { ratings: {}, note: "" };
  $("where").textContent = `Step ${index + 1} of ${kit.items.length}`;
  $("progress").max = kit.items.length;
  $("progress").value = Object.keys(saved).length;
  $("title").textContent = `Song ${item.track.slice(1)}: the ${item.stem.replace("_", " ")} stem`;
  $("back").disabled = index === 0;
  $("note").value = before.note || "";
  const choices = $("choices");
  choices.replaceChildren();
  for (const name of names(item)) {
    const button = document.createElement("button");
    button.dataset.name = name;
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
    legend.textContent = `Version ${sample}`;
    card.append(legend);
    for (const rating of kit.ratings) {
      const value = (before.ratings[sample] || {})[rating];
      card.append(scale(sample, rating, kit.help[rating], value));
    }
    marks.append(card);
  }
  $("step").hidden = false;
  try {
    await loadAudio(item);
    choose(item.samples[0]);
  } catch (error) {
    $("problem").textContent = `The audio for this step could not be loaded (${error.message}).`;
  }
}

function collect(item) {
  const ratings = {};
  for (const sample of item.samples) {
    ratings[sample] = {};
    for (const rating of kit.ratings) {
      const picked = document.querySelector(`input[name="${sample}-${rating}"]:checked`);
      ratings[sample][rating] = picked ? Number(picked.value) : null;
    }
  }
  return { ratings, note: $("note").value };
}

async function save() {
  const item = kit.items[index];
  const body = collect(item);
  const missing = item.samples.filter((sample) => body.ratings[sample].overall === null);
  if (missing.length) {
    $("problem").textContent = `Give an overall mark to version ${missing.join(", ")} first.`;
    return false;
  }
  const response = await fetch(`/api/ratings/${item.id}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    $("problem").textContent = `Not saved: ${(await response.json()).error}`;
    return false;
  }
  saved[item.id] = body;
  return true;
}

$("start").addEventListener("click", async () => {
  context = new AudioContext();
  $("intro").hidden = true;
  const first = kit.items.findIndex((item) => !saved[item.id]);
  await show(first < 0 ? 0 : first);
});
$("play").addEventListener("click", () => (playing ? stop() : play()));
$("back").addEventListener("click", () => { stop(); show(index - 1); });
$("review").addEventListener("click", () => show(0));
$("next").addEventListener("click", async () => {
  if (!(await save())) return;
  stop();
  if (index + 1 < kit.items.length) {
    await show(index + 1);
  } else {
    $("step").hidden = true;
    $("done").hidden = false;
    $("where").textContent = `All ${kit.items.length} steps are marked`;
    $("progress").value = kit.items.length;
  }
});
document.addEventListener("keydown", (event) => {
  if (!kit || $("step").hidden || event.target.tagName === "TEXTAREA") return;
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  const key = event.key.toLowerCase();
  const item = kit.items[index];
  if (key === " ") {
    event.preventDefault();
    if (playing) stop(); else play();
  } else if (key === "s") {
    choose("song");
  } else if (item.samples.map((sample) => sample.toLowerCase()).includes(key)) {
    choose(key.toUpperCase());
  }
});

(async () => {
  try {
    kit = await getJson("/api/kit");
    saved = (await getJson("/api/ratings")).items;
    $("where").textContent =
      `${kit.items.length} steps, ${Object.keys(saved).length} already marked`;
    $("progress").max = kit.items.length;
    $("progress").value = Object.keys(saved).length;
    $("intro").hidden = false;
    timer = setInterval(tick, 250);
    window.__listening = { position, choose, get playing() { return playing; },
                           get chosen() { return chosen; } };
  } catch (error) {
    $("where").textContent = `The test could not be loaded (${error.message}).`;
  }
})();
</script>
</body>
</html>
"""
