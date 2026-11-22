# PLAN

v1 of a web-based audio stem separator: upload a song, get back vocals, drums, bass, guitar,
piano and other, with optional lead/backing vocal split and optional de-reverb. Personal,
non-commercial. The aim is the best separation quality open models can give, shown with
reproducible benchmarks.

This plan rests on `docs/research-report.pdf`. Where the report and the project brief
disagree, the brief wins. Where the report was wrong or out of date, the fix is in
`docs/corrections.md` and this plan uses the corrected fact.

**Status:** Phases 0 and 1 are complete (see `PROGRESS.md`). The `separate` command works
with real models. No benchmark has been run and no quality number exists yet.

## 1. Fixed decisions

These came with the brief and are not reopened here.

| # | Decision | How the design applies it |
|---|---|---|
| 1 | Models are chosen on quality alone | The registry's `licence_note` is text only. No code reads it. |
| 2 | Six stems, plus optional vocal split and de-reverb | Eight *slots* in `config/defaults.yaml`: the six stems, `vocal_split`, `dereverb`. |
| 3 | One model registry file | `models/registry.yaml`, loaded only through `stems_models.registry`. |
| 4 | Three ensemble modes per stem | Every slot defines `single_best`, `two_model_blend`, `full_ensemble`. |
| 5 | Jev (pinned `jev-1.14.0`) plus local taggers decide | Jev answers choice/score/noul questions; our code assembles and validates the three schemas. |
| 6 | BEATs primary, CLAP second opinion above 8 kHz | Clef-omni only gets a test harness (Phase 3). |
| 7 | Every Jev decision has a deterministic fallback | Same output schema from rules when Jev is absent, failing, or under 0.8 confidence. |
| 8 | Gate: pass ≥ 0.8, score ≥ 2, confidence ≥ 0.7, at most 2 retries, then deliver with a warning | Thresholds live in `config/defaults.yaml` under `gate`. |
| 9 | No dependence on Azure GPU quota | Worker and storage sit behind interfaces: local and Modal workers; local, S3 and Azure Blob storage. |
| 10 | Careful upload handling | Private to the uploader, signed expiring links, 7-day auto-delete with countdown, one-click delete, no URL import, no sharing, no song recognition. |

## 2. Architecture

```
 Browser (web/, Next.js)
    │  upload, poll/SSE, signed links
    ▼
 API (api/, FastAPI) ── owns the database, never loads a model
    │  writes source audio + job spec          ▲ reads status.json / result.json
    ▼                                           │
 Storage interface (local disk │ S3-compatible │ Azure Blob)
    ▲                                           │
    │  reads spec + source, writes stems        │
 Worker (worker/) ── started through the provider interface (local │ Modal │ Azure seam)
    transcode → tagger → plan → separate → measure → gate → retry or deliver
```

The API and the worker never call each other directly. They share only the storage bucket:
the API writes `jobs/{id}/spec.json`, the worker writes `status.json` as it moves through
stages and `result.json` at the end, and the API polls those. This is what lets the worker
run on Modal while the API runs on a laptop that Modal cannot reach, and it means a slow
job can never block a request.

### Job stages

`uploaded → queued → transcoding → analysing → planning → separating → assessing →
(retrying) → packaging → ready`, with `failed`, `expired` and `deleted` as end states.
The web UI shows these stages by name, and `separating` and `assessing` report per stem.

### Repository layout

| Path | Package | Holds |
|---|---|---|
| `common/` | `stems_common` | Stem vocabulary, settings, paths. Added to the brief's layout because API, worker and bench all need it. |
| `models/` | `stems_models` | `registry.yaml`, registry loader, source checker, download scripts (Phase 1). |
| `worker/` | `stems_worker` | Audio I/O, chunking, model adapters, ensembling, taggers, decision layer, job runner. |
| `bench/` | `stems_bench` | Datasets, metrics, benchmark runner, Clef-omni test, listening-test kit. |
| `api/` | `stems_api` | FastAPI app, database, storage and provider interfaces, sweeper. |
| `web/` | | Next.js app. |
| `config/` | | `defaults.yaml`: every tunable value. |
| `runs/` | | Logs, status files and results of long jobs. Git-ignored. |
| `docs/` | | The report, `corrections.md`, later the deployment guide and benchmark reports. |

One Python distribution (`pyproject.toml` at the root) with extras `worker`, `bench`, `api`
and `dev`, so the worker image installs only what it needs.

## 3. Separation core (Phase 1)

**Sample rate and channels.** Every registry separator runs at 44.1 kHz stereo. Input is
decoded to float32, resampled to 44.1 kHz with a high-quality resampler, and mono is
duplicated to two channels. Stems are delivered at 44.1 kHz stereo; the source rate is
recorded in the job. Stereo is never summed anywhere in the path. In code, audio is always a
float32 array shaped `(channels, samples)`.

**Chunked inference.** Models see fixed-length chunks (length from each model's own
config). Chunks overlap; each output is multiplied by a window and summed, then divided by
the summed windows, so overlapping regions crossfade and the gain is exactly one
everywhere. The ends are reflect-padded so the first and last samples get full overlap.
Tests use an identity model and a deliberately chunk-position-dependent fake model to
prove there are no seams and no length drift.

**Model adapters.** One small `Separator` interface (`load`, `separate(chunk batch)`,
`output names`), with an adapter per `loader` value in the registry:

| Loader | Covers | Backend |
|---|---|---|
| `msst` | RoFormers, SCNet, MDX23C, PolarFormer | ZFTurbo's Music-Source-Separation-Training model classes, vendored at a pinned commit (decided in P1.8, see below). |
| `demucs` | htdemucs, htdemucs_ft, htdemucs_6s | The official `demucs` 4.1.0 package for the network; our own safe loading, chunking and whole-track normalisation. No shift trick. |
| `uvr_mdx_onnx` | UVR and kuielab MDX-Net | `onnxruntime` on CPU (CUDA where installed; there is no MPS provider). STFT settings come from UVR's model data, looked up by UVR's hash of the file. |
| `uvr_vr` | UVR VR-arch karaoke, de-echo, de-noise | UVR's network and multi-band spectrogram code, unpacked from the pinned `audio-separator` 0.48.0 wheel (those files only). Band resampling fixed to scipy polyphase. |

Chunking, overlap and ensembling are ours for every adapter, so all models are compared
under the same inference procedure.

**The `msst` backend (decided 2026-11-22, task P1.8).** The `msst` 0.1.0 package on PyPI is
ZFTurbo's own, but it requires Python below 3.14 and torch below 2.12, and this project
runs Python 3.14 with torch 2.15. So `stems_models.vendor` fetches the upstream repository
at commit `1523206d12a5` (MIT licence), pins the archive's SHA-256 in
`models/checksums.lock.json`, and unpacks only its `models/` tree into `third_party/`
(git-ignored). Only the network definitions are used. Config parsing, weight loading,
chunking and overlap are ours. Proven on `mel_roformer_kim_vocals`, which loads and runs on
both MPS and CPU here.

**Loading safety.** Checkpoints are pickle files from community sources. They are loaded
with `torch.load(..., weights_only=True)`, the SHA-256 of every downloaded file is pinned
in `models/checksums.lock.json` on first download and checked on every load, and model
YAML configs go through a safe loader that accepts the one non-standard tag they use
(`!!python/tuple`) and nothing else.

**"Other".** Two things are called "other" and they differ:

- A 4-stem model's `other` still contains guitar and piano.
- The product's `other` is what is left after all five named stems are removed.

The product's `other` always means the second thing, whichever stems a job asks for. So
when `other` comes from a 4-stem model, the composer subtracts the guitar and piano stems
from it, and runs the guitar and piano models to get them even if the job did not request
those stems (they are computed but not delivered). A 6-stem model's `other` is used as it
is. The alternative is the pure residual, `mix − (vocals + drums + bass + guitar + piano)`,
selected with `separation.other_method: residual`. Which one ships (or a blend) is a
benchmark question.

Separately, the **residual check** reports the level of `mix − sum(all six stems)` relative
to the mix for every six-stem job, and names any stem that came back silent. A poor level
means content was lost or counted twice, which happens when the models serving different
stems disagree about what a sound is. It feeds the gate. Its thresholds are in
`config/defaults.yaml` and are uncalibrated.

**Ensembling.** When several models produce the same stem the combiner supports waveform
averaging and STFT-domain averaging, each with per-model weights. Method and weights are
chosen per stem in Phase 2 on a tuning split and reported on the frozen test tracks.

**CLI.** `separate <file> --stems vocals,drums,... [--split-vocals] [--dereverb]
[--format wav|flac|mp3_320|mp3_192] [--model SLOT=ID]` writes the stems plus a
`manifest.json` with the models used, the device, wall-clock seconds per model and the
residual report. Model outputs are cached under `runs/cache/`.

**Optional stages.** Vocal split and de-reverb both run on the vocals stem and add files
(`lead_vocals`, `backing_vocals`, `vocals_dry`) without replacing it. The split models emit
the lead and the rest of their input, so lead plus backing equals the vocals stem.

## 4. Model registry

`models/registry.yaml` has 53 entries: every checkpoint in report sections 1 and 5, plus 9
found while verifying sources (marked `origin: added`). 48 can be downloaded and run, 4 are
paper-only models tracked for a later re-check, and Clef-omni is hosted.

Each entry has: `id`, `name`, `kind`, `architecture`, `loader`, `stems` (outputs),
`candidate_for` (slots it is benchmarked for), `tier`, `status`, `sample_rate_hz`,
`source_page`, `files` (URL, filename, verified size), `config` (what the loader needs),
`licence_note`, and `unverified` (facts still to check when the model is first loaded).

- Every download URL was checked with an HTTP HEAD on 2026-11-22 (85 URLs, all answered)
  and sizes were written into the registry by `python -m stems_models.check_sources
  --write`. The only unsized file is BEATs, which is behind a OneDrive share link.
- Total download for all 48 loadable entries: 16.66 GB (candidates 8.43 GB, baselines
  2.82 GB, extras 5.41 GB). Clef-omni's open weights (about 70.7 GB) are never downloaded.
- No published SDR figures are stored in the registry, because they use incompatible
  protocols.
- `config/defaults.yaml` names the model(s) per slot and mode. It is marked
  `provisional: true`: the values are the first-listed candidates from the brief, not
  benchmark results. Phase 2 replaces them.

## 5. Listening and decision layers (Phases 3 and 4)

**Listening.** BEATs (16 kHz, 527 AudioSet classes, 10-second windows) gives per-class
probabilities, mapped to stems by a committed class-to-stem table. LAION-CLAP music
(48 kHz) scores text prompts and covers content above 8 kHz, which BEATs cannot hear.
Together they produce the `track_features` JSON for the mix and, for each output stem,
presence and bleed signals. MERT with a trained probe and the Essentia instrument model
are registered and get compared only if time allows; either is promoted if it tests better.

**Decisions.** Jev never hears audio and never returns our JSON. Three calls per job:

| Call | State sent to Jev | Questions | Assembled object | Fallback without Jev |
|---|---|---|---|---|
| Parse | The user's free-text request | One noul per stem, choice for format, choice for quality mode, noul for vocal split | `ParsedStemRequest` | Keyword rules. Skipped entirely when the user used checkboxes. |
| Plan | `track_features` | One choice per requested stem (options are registry ids), choice for de-reverb, noul "is an ensemble worth the time" | `EnsemblePlan` | `config/defaults.yaml` plus simple rules (for example no vocals detected: skip vocal models). |
| Gate | Per-stem metrics and tagger signals | Score on the 0-4 rubric, noul "acceptable to deliver", choice of retry action | `StemQualityAssessment` | Rule-based scorer over the same metrics. |

Details that matter:

- The model is always `jev-1.14.0`. Settings validation rejects an alias.
- Schema (b)'s placeholder model enums are generated from the registry: for each slot the
  allowed values are the loadable candidates for that slot plus `none`.
- A Jev `noul` answer has a probability but **no confidence field** (correction C1). Its
  confidence is derived as `max(p, 1 − p)`, which matches the report's rule of acting only
  when `p ≤ 0.2` or `p ≥ 0.8`.
- A Jev `score` is a probability-weighted average and can be fractional. The gate compares
  that number with `score_min`; the label comes from the most probable level.
- Fallback triggers: no key, network or HTTP failure after retries, or confidence below
  0.8. Every call, including fallbacks, is logged with provider, responding `model` field,
  token usage and all probabilities. Keys are never logged.
- Retry policy: the gate's chosen action maps to the stem plan's fallback model, the next
  ensemble mode, or adding de-reverb. After 2 retries the best-scoring attempt is delivered
  with a warning.
- A recorded-response mock transport makes the whole pipeline testable with no key.

**Clef-omni** is tested exactly as pre-registered in report section 5 (Parts A and B,
thresholds A1-A3, B1-B3, C). Thresholds, track ids, windows and the class-to-stem mapping
are committed before the first request, every request and response is logged, and the
output is a pass/fail report and a recommendation.

## 6. Worker and storage interfaces (Phase 5)

```python
class WorkerProvider(Protocol):
    def submit(self, spec: JobSpec) -> ProviderHandle: ...
    def status(self, handle: ProviderHandle) -> ProviderStatus: ...
    def cancel(self, handle: ProviderHandle) -> None: ...


class Storage(Protocol):
    def put(self, key: str, data: BinaryIO, content_type: str) -> None: ...
    def open(self, key: str) -> BinaryIO: ...
    def exists(self, key: str) -> bool: ...
    def delete_prefix(self, prefix: str) -> int: ...
    def signed_url(self, key: str, ttl_seconds: int, filename: str | None = None) -> str: ...
```

- **Local worker**: runs the job in a subprocess on whatever device is present (CUDA, then
  MPS, then CPU), one job at a time.
- **Modal worker**: the same `run_job` entry point inside a Modal function, weights on a
  Modal Volume.
- **Azure Container Apps**: a documented class that raises `NotImplementedError`, with the
  two methods to fill in.
- **Local storage** signs links with an HMAC over key and expiry and serves them from the
  API with range support. **S3** uses presigned URLs, **Azure Blob** uses SAS tokens.

## 7. Data model

SQLite by default (`STEMS_DATABASE_URL`), Postgres-compatible SQL throughout.

| Table | Key columns |
|---|---|
| `owners` | `id`, `token_hash`, `created_at`, `last_seen_at` |
| `jobs` | `id` (uuid), `owner_id`, `status`, `stage`, `request` (stems, vocal split, de-reverb, format, quality mode), `source_key`, `source_duration_s`, `source_sample_rate_hz`, `source_channels`, `track_features`, `plan`, `provider`, `provider_handle`, `error`, `created_at`, `updated_at`, `expires_at`, `deleted_at` |
| `stems` | `id`, `job_id`, `name`, `object_key`, `format`, `bytes`, `model_ids`, `ensemble_mode`, `attempts`, `assessment`, `warning` |
| `job_events` | `id`, `job_id`, `at`, `stage`, `stem`, `message`, `progress` |
| `decision_calls` | `id`, `job_id`, `kind` (parse, plan, gate), `provider` (jev, fallback, mock), `model`, `input_tokens`, `output_tokens`, `latency_ms`, `answers`, `fallback_reason` |
| `timings` | `id`, `job_id`, `model_id`, `stem`, `device`, `audio_seconds`, `wall_seconds` |

**Ownership.** There are no accounts. The first visit sets a random secret in an HttpOnly,
SameSite cookie; the database stores only its hash. Every job row carries the owner id and
every query filters on it. Losing the cookie loses access, and the files still auto-delete.
(This is my choice, not the brief's: see BLOCKED.md decision D1.)

Object keys are random (`jobs/{uuid}/source`, `jobs/{uuid}/stems/{name}.{ext}`); the
original filename is kept only as a display label and in the download `Content-Disposition`.
Audio tags are not read and nothing fingerprints the upload.

## 8. API routes

All under `/api`. All job routes are scoped to the owner cookie and return 404 for anyone else.

| Method and path | Purpose |
|---|---|
| `GET /health` | Liveness. |
| `GET /config` | Public settings: stems, formats, size and duration limits, retention days. |
| `POST /requests/parse` | Free text to `ParsedStemRequest` (Jev or fallback). The UI shows it for confirmation when `needs_review` is true. |
| `POST /jobs` | Multipart upload plus options. Checks size, decodes the header to confirm it is audio within the duration limit, stores the source, queues the job. Returns the job. |
| `GET /jobs` | The owner's jobs. |
| `GET /jobs/{id}` | Status, stage, per-stem state and warnings, `expires_at`. |
| `GET /jobs/{id}/events` | Server-sent events for progress. |
| `POST /jobs/{id}/links` | Fresh signed links for each stem and for the zip. |
| `GET /jobs/{id}/stems/{name}/peaks` | Waveform peaks for the player. |
| `DELETE /jobs/{id}` | Delete source and stems now. |
| `DELETE /me` | Delete everything belonging to this browser. |
| `GET /files/{token}` | Local-storage downloads only: verifies the HMAC and expiry, supports range requests. |

Rate limits apply per owner and per IP on uploads and on `/requests/parse`. A sweeper
deletes expired jobs' objects and marks the rows; it is idempotent and runs on a timer and
at startup.

## 9. Web UI (Phase 6)

Upload page (drop zone, stem checkboxes, optional free-text request, format, quality mode),
job page (named stages with per-stem progress, per-stem player with solo, mute and
waveform kept in sync, quality warnings where the gate was unsure, download per stem or as
a zip, deletion countdown, delete button), and placeholder Terms and Privacy pages that say
they are placeholders.

## 10. Benchmark (Phase 2)

- **Metrics**: museval SDR, SIR and SAR per stem on 1-second windows, median over windows
  then median over tracks. Also silence fraction (below −60 dBFS), RMS and peak, duration
  match, and the ratio of estimate to reference energy above 8 kHz.
- **Every stem is scored as a pair** (target, mix minus target), for single-stem and
  multi-stem models alike, so SIR and SAR are defined and comparable between them. SDR is
  the same as museval's joint evaluation; SIR and SAR are not the joint four-source ones.
  On the synthetic fixture only SDR is meaningful (pure tones make museval's projection
  ill-conditioned).
- **Datasets**: MUSDB18-HQ test (50 tracks, 4 stems) and a frozen stratified 50-track
  MoisesDB subset (the same ids the Clef-omni test uses). Reported separately, never
  averaged. MoisesDB is the only ground truth for guitar, piano and lead vs backing.
  "Other" is scored as 4-stem other on MUSDB18-HQ and 6-stem other on MoisesDB.
- **MoisesDB subset**: chosen by a seeded script under the report's strata (at least 8 with
  lead vocals, 8 instrumental, 10 guitar, 10 piano, 10 prominent drums and bass, 6 genres)
  plus at least 8 tracks with both lead and backing vocals. "Present" means above −60 dBFS
  whole-track RMS; "prominent" means within 12 dB of the mix. The mix is the sum of the
  dataset's sources, grouped by its own six-stem mapping.
- **Frozen before any run**: track-id lists, seeds and the protocol version are committed
  first, and the commit hash is written into every result.
- **Comparison**: every registry model on its stems, then single best vs two-model blend vs
  full ensemble, with seconds per song and the device next to every score. Blend weights
  are tuned on tracks outside the frozen test lists.
- **Resumable**: results are cached per (model, weights hash, track, protocol version), runs
  start in the background and write a status file under `runs/bench/`.
- **Blends in the benchmark**: `bench run --ensembles` scores each slot of the config in
  its three modes from its members' kept audio, plus the composed six-stem other (the
  four-stem blend minus the same mode's guitar and piano). A blend's time is the sum of
  the times of every model it needs.
- **Repeat runs**: `--trial N` does the same jobs again under separate keys, and
  `bench compare` sets run-to-run differences beside device-to-device differences.
- **Reports** are written only from a run's result files (`bench report`), one per
  dataset, with the command, commit, frozen-list hash, device and seconds beside every
  score. `docs/benchmarks/README.md` has the protocol and the commands.
- **Synthetic fixture**: sine and noise mixes with known stems, generated from a seed. The
  whole harness runs on it now, so it is ready when the datasets arrive.
- **Device sensitivity**: the same checkpoint gives measurably different output on CPU and
  MPS (found in P1.8 on the synthetic fixture, where a 1e-6 change to the input moved the
  Kim vocals model's output at about −42 dB relative). Task P2.14 measures how much scores
  move between devices on real tracks before any cross-device comparison is made.
- **Rules**: every number in a report comes from a run in this repo with its command.
  Published figures are never copied in or compared against.

This machine is an Apple M5 Pro with 24 GB of memory and no NVIDIA GPU, so local timings
are MPS or CPU timings and are labelled as such. Timings on a T4 or L4 need Modal.

## 11. What each phase delivers

| Phase | Delivers | Needs from you |
|---|---|---|
| 0 | This plan, scaffold, tooling, CI file, registry, PROGRESS.md, BLOCKED.md | Nothing |
| 1 | Download scripts, chunked inference, adapters, `separate` CLI, per-model timing, synthetic fixture and tests | Nothing (about 17 GB of downloads) |
| 2 | Metrics, dataset loaders, resumable benchmark runner, ensemble comparison, recommended defaults | MUSDB18-HQ and MoisesDB paths; Modal for GPU timings |
| 3 | BEATs and CLAP wrappers, `track_features`, pre-registered Clef-omni test and report | Cloudflare token, MoisesDB |
| 4 | Jev client, three schemas, answer mapping, fallbacks, gate with retries, recorded mock | Jev key only for a live check |
| 5 | API, queue, storage backends, signed links, sweeper, rate limits, Modal worker | Modal token; bucket credentials only for live checks |
| 6 | The web app | Nothing |
| 7 | Calibrated gate, end-to-end tests, cost per song, deployment guide, listening-test kit | Listeners; benchmark data |

`PROGRESS.md` breaks each phase into tasks with acceptance checks. `BLOCKED.md` lists what
is needed from you in the order it is needed.

## 12. Environment facts checked on 2026-11-22

- Python 3.14.5 is the only interpreter here. `pip index versions` finds releases of
  torch, torchaudio, onnxruntime, demucs, museval, librosa, laion-clap, transformers,
  fastapi, modal, typesafe-sdk and audio-separator for it. Whether each has a working
  3.14 wheel on Apple Silicon is confirmed at install time in the phase that needs it.
- No ffmpeg, Homebrew or uv. Decoding will use `soundfile` and PyAV wheels (which bundle
  their codecs), so ffmpeg is not required.
- Node.js v24.22.3 LTS is installed inside the repo at `.tools/node` by
  `tools/install_node.sh`. The web scaffold (Next 16.4, React 19.3, TypeScript 6.0,
  ESLint 9) installs, lints, type-checks and builds.
- 760 GB of free disk.
- The python.org build has no system CA certificates, so HTTPS code uses `certifi`.

## 13. Choices I made that you may want to change

1. **No accounts**; ownership is an anonymous browser cookie (BLOCKED.md D1).
2. **Upload limits** of 250 MiB and 20 minutes (BLOCKED.md D2).
3. **Stems are delivered at 44.1 kHz stereo** whatever the source rate was.
4. **A `common/` directory** was added to the layout in the brief.
5. **Nine checkpoints were registered beyond the report**, including three newer karaoke
   models and a downloadable 4-stem BS-RoFormer. They are tier `extra` and are benchmarked
   after the brief's candidates.
6. **API and worker talk only through storage**, with no callback URL.
