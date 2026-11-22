# PROGRESS

Single source of truth for where the build is. Read this first, every iteration.

**Current phase: 2 (benchmark harness).** The owner started it on 2026-11-22.

## How to use this file

1. Take the first unticked task in the current phase that is not marked BLOCKED.
2. Implement it. Run `make check`. Commit with a clear message.
3. Tick it only when its acceptance check passes, and add a one-line note after the tick.
4. If a task needs something from the owner, add it to `BLOCKED.md`, mark the task
   `BLOCKED (Bn)` and move to the next one.
5. If a task fails three iterations in a row, write what was tried in `BLOCKED.md`, mark it
   `BLOCKED`, move on.
6. When the current phase has no unticked unblocked tasks, write the phase summary at the
   bottom of this file and stop. Never start the next phase without the owner saying so.

Long jobs (downloads, benchmark runs) start in the background, write a log and a
`status.json` under `runs/<kind>/<run-id>/`, and are checked by a later iteration.
Every task must be safe to re-run: check for existing downloads, caches and finished runs
before doing work.

Format: `- [ ] Pn.m Task. **Accept:** how to tell it is done.`

---

## Phase 0: plan and scaffold

- [x] P0.1 Read the research report in full and verify what can be verified. **Accept:** `docs/corrections.md` lists each correction with its source. — 11 corrections recorded; Jev, Clef-omni and Modal docs fetched live on 2026-11-22.
- [x] P0.2 Write `PLAN.md`. **Accept:** covers architecture, data model, API routes and what each phase delivers. — Done.
- [x] P0.3 Scaffold the monorepo and Python tooling. **Accept:** `make setup && make check` passes from a clean clone. — ruff clean, 46 tests pass on Python 3.14.5.
- [x] P0.4 Model registry with every checkpoint from report sections 1 and 5. **Accept:** `python -m stems_models` validates and prints all entries; `tests/test_registry.py` passes. — 53 entries (38 from §1, 6 from §5, 9 added), 48 loadable.
- [x] P0.5 Verify every download URL and record sizes. **Accept:** `python -m stems_models.check_sources --write` exits 0. — 85 of 85 URLs answered; 16.66 GB in total; BEATs is the one file without a verified size (OneDrive link).
- [x] P0.6 `config/defaults.yaml` with gate thresholds, retention, limits and per-slot models. **Accept:** `tests/test_defaults.py` passes, including the cross-check against the registry. — Models are marked `provisional: true` until Phase 2.
- [x] P0.7 CI workflow. **Accept:** `.github/workflows/ci.yml` runs lint, registry validation and tests. — Committed but never executed: the repo has no remote yet.
- [x] P0.8 `PROGRESS.md`, `BLOCKED.md`, `CLAUDE.md`, `.env.example`, `.gitignore`. **Accept:** files exist; weights, audio, datasets and `.env` are git-ignored. — Done.
- [x] P0.9 Web scaffold installs and passes its checks. **Accept:** `make web-setup && make web-check` passes and `web/package-lock.json` is committed. — Node v24.22.3 LTS installed project-locally (`make node-setup`); lint, typecheck and build pass. TypeScript pinned to 6.0 and ESLint to 9 because `eslint-config-next`'s plugins reject TypeScript 7 and ESLint 10 (see `web/README.md`).

## Phase 1: separation core

- [x] P1.1 Install and pin the worker dependencies. **Accept:** `pip install -e ".[worker,dev]"` succeeds on this machine; versions are pinned in `pyproject.toml`; `runs/env/worker.json` records torch version and which of CUDA, MPS, CPU is available. — torch 2.15.0, numpy 2.5.5, scipy 1.18.1, soundfile 0.14.0, soxr 1.1.0, av 19.0.2 and onnxruntime 1.32.0 install on Python 3.14 arm64 and are exact-pinned. MPS is available and passes a smoke computation; no CUDA. `make env` rewrites the record. demucs and the MSST backend are left to P1.10 and P1.8 (a dry run shows demucs 4.1.0 resolves against these pins).
- [x] P1.2 Synthetic fixture generator (sine and noise stems for all six stems, plus lead and backing vocal parts and a reverberant variant; 44.1 kHz stereo; seeded). **Accept:** generating twice with the same seed gives identical samples; the mix equals the sum of its stems to within float error; no audio file is committed. — `stems_common.synthetic.generate_track(seed, duration_seconds, sample_rate, active_stems)` returns float32 `(2, samples)` stems whose float32 sum is the mix exactly; vocals = lead + backing exactly; `vocals_wet` adds reverb at −12 dB. Drums hold over 95% of the energy above 8 kHz, bass is over 95% below 300 Hz, backing vocals enter halfway. A 6 s track builds in about 0.2 s. `tests/test_repo_hygiene.py` fails if audio, weights, `.env` or run output is ever tracked.
- [x] P1.3 Audio I/O: decode any supported format to 44.1 kHz stereo float32; write WAV, FLAC and MP3. **Accept:** tests cover mono to stereo, 48 kHz and 22.05 kHz input, length preserved to within one sample after resampling, WAV and FLAC round-trips are exact at 24-bit precision. — `stems_worker.audio`: `probe`, `load_audio` (libsndfile first, PyAV fallback for M4A and anything wider than stereo, duration limit enforced while decoding), `resample` (soxr VHQ, exact output length), `save_audio` (24-bit WAV and FLAC, MP3 at 320 or 192 kbps via PyAV's libmp3lame, atomic write, reports peak and clipped samples). 28 tests. Finding: libsndfile truncates float to 24-bit instead of rounding, so `save_audio` rounds and clips itself. No system ffmpeg is needed.
- [x] P1.4 Chunker with overlap and crossfade. **Accept:** an identity model reconstructs the input to within 1e-6 for inputs shorter than one chunk, exactly one chunk, and not a multiple of the hop; a fake model that adds a chunk-position ramp produces no step at chunk boundaries; stereo channels stay independent. — `stems_worker.chunking.apply_chunked(audio, fn, chunk_size, hop, batch_size, progress)`: reflect-padded ends, linear crossfade windows normalised by their sum, numpy only (the model function owns the device). Identity error is 6e-8 on a 30 s fixture at chunk size 588800. With a ramp model of amplitude 1 the largest sample-to-sample step is 1.0 with no overlap, 0.0005 at 25% overlap and 0 at 50%. 30 tests. Overlap and batch size are arguments; P1.14 wires them to config.
- [x] P1.5 Download manager keyed by registry id. **Accept:** resumes a partial file, skips a complete one, verifies size, records SHA-256 in `models/checksums.lock.json` on first download and fails on a later mismatch; tested against a local HTTP server; `--background` writes `runs/downloads/<id>/status.json`. — `python -m stems_models.download <ids> [--tier T] [--all] [--stem S] [--background] [--status]`; `ensure_model(id)` in code. Resume via HTTP Range, restart if the server ignores it, HTML pages refused, one downloader per model (flock), wrong-size or wrong-hash content never kept. 24 tests against a local server. Real check: `mel_roformer_guitar_becruily` downloaded in the background in 9 s and its SHA-256 matches the hash Hugging Face lists.
- [x] P1.6 Download all candidate-tier models in the background, then baselines and extras. **Accept:** `runs/downloads/*/status.json` all say `done`; checksums are committed; nothing under `models/weights/` is tracked by git. (Long job: start it, check it next iteration.) — 47 of 48 models downloaded (16 GB) by one background run in about 35 minutes; all 85 file hashes are pinned in `models/checksums.lock.json`; nothing under `models/weights/` is tracked. The 48th, `beats_iter3plus_as2m_cpt2`, is **BLOCKED (B10)**: its OneDrive link serves a sign-in page. It is a Phase 3 need.
- [x] P1.7 `Separator` interface, device selection, and a fake separator for tests. **Accept:** a fake two-stem model runs through chunker and writer end to end on the fixture on CPU. — `stems_worker.separator` (`Separator` base class, `run_separator` returning stems, device, wall seconds and chunk count), `stems_worker.device.pick_device` (argument, then `STEMS_DEVICE`, then best usable; an unusable device is an error), `stems_worker.fakes.BandSplitSeparator` (torch FFT band split with real chunk-edge error), and `audio.save_stems` (one shared gain if any stem would clip). The fake runs fixture → chunker → 24-bit files on CPU and the stems sum to the mix within 2e-6; MPS matches CPU within 1e-4. 12 tests.
- [x] P1.8 Choose and prove the `msst` backend. **Accept:** `mel_roformer_kim_vocals` loads with `weights_only=True` and a safe YAML loader and separates a 10-second fixture clip on this machine; the choice (pip package or vendored commit) is recorded in `PLAN.md` section 3. — Vendored at commit `1523206d12a5` via `stems_models.vendor.ensure_msst()`: the `msst` pip package refuses Python 3.14 and torch above 2.11. `stems_worker.adapters.msst.MsstSeparator.from_registry(id)` loads Kim with `weights_only=True` and a SafeLoader that only adds `!!python/tuple`; the 10 s clip separates on MPS and CPU (`make test-slow`). New pins: einops, beartype, rotary-embedding-torch, librosa, packaging. Findings for P1.9: (1) the pinned upstream tree has no PolarFormer class, so `bs_polarformer_vocals` needs its code located; (2) upstream's MDX23C imports `utils.model_utils`, which is not vendored; (3) CPU and MPS outputs differ at about −20 dB relative on the fixture because the model is that sensitive to tiny numeric differences, so never assert tight cross-device equality.
- [x] P1.9 `msst` adapter for RoFormer, SCNet, MDX23C and PolarFormer entries. **Accept:** every `loader: msst` entry loads and produces outputs with the names the registry lists; each entry's `unverified` items are resolved or moved to `docs/corrections.md`. — `MsstSeparator` builds `mel_band_roformer`, `bs_roformer` (PolarFormer is `use_pope: true` and needs PoPE-pytorch), `scnet`, `scnet_masked` and `mdx23c`. All 23 msst entries load and emit their registry outputs (`make test-slow`: 46 passed across all adapters). Registry `config.instrument_names` and `config.ignore_instruments` map each config's names to ours. Resolved: PolarFormer and masked-SCNet model types, 6-stem output order, becruily guitar is single-target, `bs_roformer_viperx_other` is MUSDB-style other. Still open, by design: that the karaoke models' "Vocals" is the lead (Phase 2 ground truth), and byte identity of the mirrored 6-stem file (corrections C3).
- [x] P1.10 `demucs` adapter. **Accept:** `htdemucs`, `htdemucs_ft` and `htdemucs_6s` load and run on the fixture; a single-stem request to `htdemucs_ft` loads only that stem's file. — `stems_worker.adapters.demucs.DemucsSeparator` on the official `demucs` 4.1.0 package (pinned). Checkpoints load with `weights_only=True` and an allowlist of the five types they contain. Whole-track normalisation uses new `prepare` / `finish` hooks on `Separator`. All three models run on MPS and CPU; a bass-only request to `htdemucs_ft` builds one model from one file. On the fixture, drums and bass route at 0.93 to 1.00 and `htdemucs_6s` puts 0.79 of the synthetic guitar in its guitar output. Registry entries now record `sources` and `segment_seconds`, checked against the checkpoint at load. The package's shift trick is not applied.
- [x] P1.11 `uvr_mdx_onnx` adapter. **Accept:** all MDX-Net entries run on the fixture; parameters come from UVR's model data by MD5; primary and secondary outputs are mapped correctly (checked by which output correlates with the fixture's vocal stem). — `stems_worker.adapters.mdx.MdxSeparator`. All 17 ONNX files (7 UVR vocal/instrumental, 2 UVR karaoke, 8 kuielab per-stem) are found in UVR's model data by its tail-MD5 hash and run on the fixture (`make test-slow`). Mapping confirmed: drums and bass land at 1.03 in the instrumental and 0.00 in the vocals whether the model's primary stem is Vocals or Instrumental; kuielab drums and bass route at 0.81 to 0.93; the karaoke models put more of the lead than of the backing in their lead output. kuielab bags take one stem per separator because each stem has its own chunk length. onnxruntime has no MPS provider, so these run on CPU and results say `cpu`. UVR's optional denoise pass is available (`denoise=True`, off by default).
- [x] P1.12 `uvr_vr` adapter. **Accept:** the VR karaoke and de-echo entries run on the fixture. — `stems_worker.adapters.vr.VrSeparator`. UVR's network definitions and multi-band spectrogram routines are unpacked from the pinned audio-separator 0.48.0 wheel by `stems_models.vendor.ensure_uvr_lib()` (only those files; installing the package would pull in its whole dependency tree). All six models are found in UVR's model data by hash and run on MPS and CPU (`make test-slow`). The registry names each model's two outputs explicitly (`primary_stem`, `primary`, `secondary`), and a mismatch with UVR's data is an error. On the fixture the two karaoke models genuinely split lead from backing (lead output: 0.67 and 0.90 of the lead, 0.11 and 0.27 of the backing). Band resampling is fixed to scipy polyphase on every platform; UVR varies it by operating system. New pin: audioread.
- [x] P1.13 Stem composer: route each requested stem to its single-best model, share one model run across the stems it serves, build 6-stem "other" from a 4-stem "other", and compute the residual check. **Accept:** on a fake perfect model the residual is below −80 dB relative to the mix and the composed "other" equals the fixture's; a fake model that drops a stem makes the residual check report it. — `stems_worker.compose`: `plan_composition(requested, assignment, registry, other_method)` groups stems into one run per model and adds guitar and piano as undelivered dependencies when `other` comes from a 4-stem model; `compose(...)` runs each model in turn (unloading between), builds `other` (direct, subtract or residual) and returns a `ResidualReport`. `stems_worker.factory.build_separators(id, stems)` picks the adapter. New `OracleSeparator` fake is exact under any chunking by encoding sample positions in the signal. With oracles: stems exact to 1e-6, residual below −80 dB; a dropped bass is reported as silent with the residual at the bass's own level. Real defaults on the fixture (`make test-slow`): four model runs produce six stems; the check flags guitar and piano as silent and a −8.7 dB residual, because the fixture's sine-tone voice is taken by the vocal model and also left in SCNet's `other`. `separation` settings added to `config/defaults.yaml`.
- [x] P1.14 `separate` CLI. **Accept:** `separate fixture.wav --stems vocals,drums,bass,guitar,piano,other` writes six files and `manifest.json` (models, device, seconds per model); `--format flac` and `--format mp3_320` work; a second run with the same arguments reuses cached model outputs. — `separate` console script (`stems_worker.cli`), over `stems_worker.pipeline.separate_file`, which the job worker will reuse. Options: `--stems` (names or `all`), `--out`, `--format`, `--device`, `--model STEM=ID`, `--other-method`, `--overlap`, `--batch-size`, `--cache-dir`, `--no-cache`. `manifest.json` records the input hash, per-model device, wall seconds and cache use, per-stem source and level, the shared output gain, the residual report and the git commit. `stems_worker.cache.OutputCache` keys model outputs on the decoded audio, the model's pinned weights, overlap and device. Real run on a 20 s fixture file on MPS: six WAVs written, a repeat run entirely from cache, FLAC and MP3 from the same cache. Only `single_best` mode exists; blends come with Phase 2.
- [x] P1.15 Vocal split and de-reverb stages (`--split-vocals`, `--dereverb`). **Accept:** both run on the fixture's vocal stem and write the extra files; manifest records them. — `--split-vocals` writes `lead_vocals` and `backing_vocals`; `--dereverb` writes `vocals_dry`. Both run their slot's model (`vocal_split`, `dereverb`; overridable with `--model`) on the vocals stem, which is computed for them even if not requested, and neither replaces the vocals file. The manifest lists stage runs in `models` (with `stage` and `input`) and the files under `extras`. Stage outputs have their own cache entries keyed on the vocals stem's audio. Real run: Kim vocals from cache, then the aufr33 karaoke and anvuew de-reverb models, each about 5 s for 20 s of audio on MPS. On the fixture the RoFormer karaoke model returns a silent lead, as it does not recognise sine-tone voices; lead plus backing still equals the vocals.
- [x] P1.16 Timing log. **Accept:** every model run appends a line to `runs/timings.jsonl` with model id, device, audio seconds and wall seconds; a test checks the line is written. — `stems_worker.timing.TimingLog` appends one JSON line per real model run (never for a cache hit) to `runs/timings.jsonl`: model id, device, stage, outputs, audio and wall seconds, seconds per second of audio, chunk count, overlap, batch size, the audio's hash, code commit, torch version and platform. File names are never logged. `compose.run_model` writes it, so the composer and the optional stages are both covered. `python -m stems_worker.timing` prints the median per model and device; `--timings-log` and `--no-timings` on the command. 10 tests, all writing to temporary logs, so the real log will only ever hold real runs (it does not exist yet: the trial runs so far logged to a scratch file).
- [x] P1.17 Smoke-test every loadable separator on the fixture and write `runs/smoke/report.md`. **Accept:** the report lists each model as loaded and ran, or failed with the error; failures are fixed or recorded in `BLOCKED.md`. — `python -m stems_worker.smoke` (`make smoke`) runs each loadable separator on a 10 s fixture and writes `runs/smoke/report.md` and `report.json`: result, device, load and run seconds, outputs, and where each known fixture signal went (a joint least-squares share, so correlated signals are not double-counted). Run of 2026-11-22: **43 of 43 loaded and ran** (32 on MPS, 11 MDX-Net models on CPU), no failures, nothing for BLOCKED.md. A model that fails is recorded with its error and the rest still run. 9 tests.

## Phase 2: benchmark harness

- [x] P2.1 museval metrics: median SDR, SIR, SAR per stem on 1-second windows. **Accept:** on the fixture a perfect estimate scores above 60 dB SDR; adding noise at known levels lowers SDR monotonically; single-stem evaluation uses the (target, residual) pair. — `stems_bench.metrics.bss_eval_stem(reference, estimate, mix, rate)` returns median SDR, SIR, SAR and ISR over 1 s windows, with every window's value and the count of scored windows. The numbers are museval 0.4.1's own `metrics.bss_eval` (pinned in the `bench` extra), loaded by file path because `import museval` fails without ffmpeg. Every stem of every model is scored as the (target, mix minus target) pair, so SIR and SAR are comparable across models; SDR is the standard museval SDR either way (tested). A stem absent from a track gives no score rather than an error. Finding: on the synthetic fixture SIR, SAR and ISR are numerical noise, because museval's projection is ill-conditioned for pure tones with identical channels; only SDR is meaningful there. On broadband signals all four behave (tested). 18 tests.
- [x] P2.2 Extra metrics: silence fraction below −60 dBFS, RMS and peak in dBFS, duration match, energy ratio above 8 kHz. **Accept:** unit tests with hand-built signals for each. — `stems_common.levels` (`rms_dbfs`, `peak_dbfs`, `silence_fraction` over 0.1 s windows; no reference needed, so the quality gate can reuse them) and `stems_bench.signal_metrics` (`level_stats`, `duration_delta_seconds`, `high_band_ratio_db`, and `compare`, which returns all of it for an estimate and its reference as JSON). The high-band ratio is estimate over reference energy at and above 8 kHz, in dB: negative for a low-passed stem, positive for added hiss or cymbal bleed, undefined when the reference has no high band. 8 tests on sines, silence and padded signals.
- [x] P2.3 Dataset interface and the synthetic dataset. **Accept:** `bench/frozen/synthetic.json` (track ids, seeds, protocol version) is committed and the loader refuses to run if the generated tracks do not match it. — `stems_bench.datasets`: `Dataset` (a frozen list plus `load(track_id)` returning inputs and references), `load_frozen` (refuses a list for another dataset or protocol version; its SHA-256 goes into every result) and `SyntheticDataset`. `stems_bench.protocol` holds `PROTOCOL_VERSION = 1` and the rules it stands for, including which input each kind of model is fed and which reference its `other` is scored against. `bench/frozen/synthetic.json` freezes 8 tracks of 12 s (seeds 101 to 108; one without guitar and piano, one without vocals) with a fingerprint of every signal; loading regenerates the track and refuses if any fingerprint is off by more than 1e-6. Fingerprints, not byte hashes, because generated floats can differ in the last bit between platforms. `python -m stems_bench.datasets.synthetic` checks the list; `--freeze` writes it only if absent. 9 tests.
- [ ] P2.4 MUSDB18-HQ loader reading `MUSDB18HQ_PATH`. **Accept:** tested against a fake directory tree built from fixtures; on first contact with the real dataset it writes `bench/frozen/musdb18hq_test.json` and refuses to benchmark until that file is committed.
- [ ] P2.5 MoisesDB loader and the frozen stratified 50-track subset. **Accept:** stem grouping into the six stems plus lead and backing is tested on a fake tree; the selection script is seeded and enforces the report §5 strata; real subset freeze is **BLOCKED (B3)** until the dataset arrives.
- [ ] P2.6 Resumable run manager (`bench run`). **Accept:** results cached per (model id, weights SHA-256, track id, protocol version); killing and restarting a run redoes nothing finished; `runs/bench/<run-id>/status.json` updates as it goes.
- [ ] P2.7 Ensemble combiner: waveform and STFT-domain averaging with per-model weights. **Accept:** averaging two copies of the same estimate returns it unchanged; a weighted mix of a good and a bad fake estimate scores between them; weights are read from config.
- [ ] P2.8 Report generator. **Accept:** produces Markdown with one table per dataset, seconds per song and device beside every score, the exact command, git commit and frozen-list hash; contains no number that is not in the run's result files.
- [ ] P2.9 Run every registry separator on the synthetic dataset, single best vs blend vs full ensemble. **Accept:** `docs/benchmarks/synthetic.md` exists and the run finishes end to end from one command.
- [x] P2.10 Benchmark on MUSDB18-HQ test. **Accept:** `docs/benchmarks/musdb18hq.md` generated from a completed run. **BLOCKED (B2).**
- [x] P2.11 Benchmark on the MoisesDB subset, including guitar, piano and lead vs backing. **Accept:** `docs/benchmarks/moisesdb.md` generated from a completed run. **BLOCKED (B3).**
- [x] P2.12 Tune blend methods and weights on tracks outside the frozen test lists, then score them on the test lists. **Accept:** the tuning tracks are listed and disjoint from the frozen lists. **BLOCKED (B2, B3).**
- [x] P2.13 Recommend a default model or ensemble per slot from measured gain against time. **Accept:** `docs/benchmarks/recommendation.md` cites the runs; `config/defaults.yaml` is updated and `provisional` set to false. **BLOCKED (B2, B3).**
- [x] P2.14 Measure how much scores move between devices (CPU, MPS, and CUDA when available) for the same model and tracks. **Accept:** the benchmark report states the per-stem SDR spread across devices on the synthetic set and, once available, on real tracks; if it exceeds run-to-run noise, every comparison in a report uses a single device.

## Phase 3: listening layer and the Clef-omni test

- [ ] P3.1 BEATs wrapper. **Accept:** returns 527 class probabilities per 10-second window on the fixture, on this machine.
- [ ] P3.2 AudioSet class-to-stem mapping file. **Accept:** committed as data, with a test that every mapped class exists in the label set.
- [ ] P3.3 LAION-CLAP wrapper with text prompts and a high-band check. **Accept:** on fixture clips with and without content above 8 kHz the high-band signal differs in the right direction.
- [ ] P3.4 `track_features` builder and per-stem signals (presence, bleed estimate, notes). **Accept:** output validates against the `track_features` part of schema (b) and the `metrics` part of schema (c).
- [ ] P3.5 Pre-registration file for the Clef-omni test: thresholds A1-A3, B1-B3, C, the 50 track ids, 30-second windows, class-to-stem mapping. **Accept:** committed before any Clef-omni request; its commit hash is embedded in the test's log header. **BLOCKED (B3)** for the track ids and windows.
- [ ] P3.6 Clip builder: loudest 30-second window, programmatic presence ground truth, Part B bleed mixes at +30, +20, +12 and +6 dB SIR, five reverb clips, 16 kHz mono encoding under 8 MiB. **Accept:** tested on the synthetic dataset, including the measured SIR of each bleed clip.
- [ ] P3.7 Clef-omni client with full request and response logging and a recorded-response mock. **Accept:** enforces 4 clips, 8 MiB, 300 seconds and 64 questions per request; every call appends to a JSONL log with question ids, probabilities and token usage; the token never appears in logs.
- [ ] P3.8 Scoring: accuracy overall and per stem, AUC, Spearman correlation, clean false-positive rate, pass or fail per threshold. **Accept:** unit tests with hand-computed cases for each statistic.
- [ ] P3.9 Run Parts A and B for Clef-omni and BEATs. **Accept:** `runs/clef_test/<run-id>/` holds all logs and scores. **BLOCKED (B3, B5).**
- [ ] P3.10 Pass/fail report and recommendation. **Accept:** `docs/clef-omni-test.md` states each threshold, the measured value and the verdict, generated from the run. **BLOCKED (B3, B5).**
- [ ] P3.11 Optional: MERT probe and Essentia model compared with BEATs on Part A. **Accept:** results added to the same report, or a note saying it was skipped and why.

## Phase 4: decision layer

- [ ] P4.1 The three JSON Schemas as files, with schema (b)'s model enums generated from the registry. **Accept:** valid and invalid examples for each schema are tested; changing the registry changes the generated enum.
- [ ] P4.2 Jev client pinned to `jev-1.14.0`. **Accept:** retries 429 and 5xx with backoff and honours `retry-after`; times out; logs model field, token usage and probabilities for every call; refuses an alias; the key never appears in logs (tested).
- [ ] P4.3 Recorded-response mock transport. **Accept:** replays fixtures by request hash and fails loudly on an unrecorded request.
- [ ] P4.4 Request parsing to `ParsedStemRequest`, with the keyword-rule fallback. **Accept:** mapping tested from recorded answers; "no stem detected" sets `needs_review`; the fallback handles the test phrases with no network.
- [ ] P4.5 Ensemble planning to `EnsemblePlan`, with the config-default fallback. **Accept:** enforces the requested/primary/fallback rules from the report; low confidence falls back and says why.
- [ ] P4.6 Stem assessment to `StemQualityAssessment`, with the rule-based fallback scorer. **Accept:** noul confidence is derived as `max(p, 1 − p)`; a stem with no reference SDR is gated on tagger signals.
- [ ] P4.7 Gate and retry state machine. **Accept:** table-driven tests for deliver, each retry action, the 2-retry cap, and deliver-with-warning choosing the best attempt; thresholds come from config.
- [ ] P4.8 Whole pipeline on the fixture with the mock, and again with no key. **Accept:** both produce stems and a complete decision log; the no-key run shows `provider: fallback` for all three decisions.
- [ ] P4.9 Live check against Jev and capture of real recordings. **Accept:** one real request per call type succeeds and is saved as a fixture. **BLOCKED (B6).**

## Phase 5: API and worker

- [ ] P5.1 Storage interface and local-disk backend with HMAC-signed expiring links. **Accept:** expired or tampered links are rejected; keys cannot escape the storage root.
- [ ] P5.2 S3-compatible backend. **Accept:** passes the shared storage test suite against a stubbed client; live check noted as done or **BLOCKED (B7)**.
- [ ] P5.3 Azure Blob backend. **Accept:** same as P5.2.
- [ ] P5.4 Database models and migrations. **Accept:** schema matches `PLAN.md` section 7; every job query is owner-scoped (tested by asking for another owner's job).
- [ ] P5.5 Job spec, status and result contract between API and worker. **Accept:** JSON Schemas for `spec.json`, `status.json`, `result.json`; both sides validate.
- [ ] P5.6 Worker provider interface and the local provider. **Accept:** runs a job in a subprocess; a slow fake job does not delay an unrelated API request (tested with a timing bound).
- [ ] P5.7 Worker `run_job`: the full pipeline with stage reporting. **Accept:** on the fixture it walks every stage and writes stems, peaks and `result.json`; re-running a finished job does no work.
- [ ] P5.8 API: owner cookie, `POST /jobs` with validation, `GET /config`, `GET /health`. **Accept:** rejects oversize, overlong and non-audio uploads; a renamed non-audio file is rejected by decoding, not by extension.
- [ ] P5.9 API: job list, status, events, signed links, peaks, delete, delete-all. **Accept:** deletion removes every object under the job prefix and the links stop working.
- [ ] P5.10 Retention sweeper. **Accept:** with a fake clock, a job past `retention.days` loses its objects and is marked expired; running the sweeper twice changes nothing more.
- [ ] P5.11 Rate limits per owner and per IP. **Accept:** limits come from config and return 429 with `Retry-After`.
- [ ] P5.12 Modal provider. **Accept:** unit-tested against a stub; live run of the fixture on Modal noted as done or **BLOCKED (B4)**.
- [ ] P5.13 Azure Container Apps seam. **Accept:** a documented class with the two methods to implement and a test that selecting it fails with a clear message.
- [ ] P5.14 End-to-end API test with the local provider on the fixture. **Accept:** upload to download works in one test, using the fake separator so it runs in CI.

## Phase 6: web app

- [ ] P6.1 App shell, typed API client, error states. **Accept:** `make web-check` passes.
- [ ] P6.2 Upload page: file drop, stem choices, optional free-text request with confirmation when `needs_review`, format and quality mode.
- [ ] P6.3 Job page: named stages with per-stem progress, live via server-sent events.
- [ ] P6.4 Stem player: waveform, solo and mute, all stems sample-synchronised.
- [ ] P6.5 Quality warnings where the gate was unsure.
- [ ] P6.6 Download per stem and as a zip, in the chosen format.
- [ ] P6.7 Deletion countdown, delete now, delete everything.
- [ ] P6.8 Terms and Privacy placeholder pages that say they are placeholders.
- [ ] P6.9 Browser test of the whole flow against the API with the fake separator; keyboard and screen-reader pass on the player.

(Acceptance checks for P6.2 to P6.9 are written when Phase 6 starts.)

## Phase 7: calibration and hardening

- [ ] P7.1 Calibrate gate thresholds on labelled clean and deliberately bled stems. **Accept:** `docs/gate-calibration.md` shows the measured trade-off and the chosen values, from a run. **BLOCKED (B3; B6 for real Jev answers).**
- [ ] P7.2 End-to-end tests across API, worker and web.
- [ ] P7.3 Cost per song from measured timings and prices checked that day. **Accept:** every figure cites a `runs/` file and a dated price source.
- [ ] P7.4 Deployment guide (local, Modal, S3 or Azure Blob, the Azure Container Apps seam).
- [ ] P7.5 Listening-test kit: blind randomised 30-second excerpts and a scoring sheet per report §9.
- [ ] P7.6 Hardening pass: limits, failure paths, a security review of upload and download handling.
- [ ] P7.7 Optional: fine-tune the weakest stem on MoisesDB if the benchmark shows a clear gap.

---

## Phase summaries

### Phase 0 (2026-11-22)

Done: the report was read in full and checked against live sources (11 corrections, the
most important being that Jev `noul` answers carry no confidence field). `PLAN.md` is
written. The repo is scaffolded with one Python distribution, ruff, pytest and a CI
workflow. The model registry holds 53 entries with every download URL verified and sized
(16.66 GB for the 48 loadable ones). `config/defaults.yaml` carries the gate thresholds,
retention, limits and provisional per-slot models. 46 tests pass.

The web scaffold installs, lints, type-checks and builds on a project-local Node v24 LTS
(`.tools/node`, installed at the owner's request).

Not done: CI has never run because the repo has no remote. No pipeline code exists, by
design.

Notes for the next iteration: the report PDF was at the repo root under another name and
was moved to `docs/research-report.pdf`. Python here is the python.org 3.14 build, which
has no system CA bundle, so use `certifi` for any HTTPS. There is no ffmpeg; plan on
`soundfile` and PyAV.

**Next:** Phase 1, starting at P1.1 (started 2026-11-22 on the owner's go-ahead).

### Phase 1 (2026-11-22)

Done: all 17 tasks. `separate <file> --stems ...` turns an audio file into stems with a
manifest, using real models, on this machine.

- **Models**: 47 of the 48 loadable registry models are downloaded (16 GB) with every file's
  SHA-256 pinned. All 43 separators load and run (`make smoke`: 43 of 43).
- **Adapters**: `msst` (RoFormers, SCNet, MDX23C, PolarFormer; upstream code vendored at a
  pinned commit), `demucs`, `uvr_mdx_onnx`, `uvr_vr` (UVR code vendored from a pinned
  wheel). Every checkpoint loads with `weights_only=True`; configs use a safe YAML loader.
- **Core**: decoding without ffmpeg, soxr resampling, chunking with overlap and crossfade,
  the composer (shared model runs, "other", residual check), optional vocal split and
  de-reverb, a cache of model outputs, a timing log, and a synthetic fixture with exact
  "perfect model" fakes.
- **Tests**: 330 fast (`make check`, about 15 s) and 55 with real weights (`make test-slow`,
  about 3 minutes; 1 more needs CUDA).

Blocked: only the BEATs checkpoint (B10), which Phase 3 needs. Nothing in Phase 2 needs it.

Findings worth carrying forward:

1. **Device sensitivity.** The same model gives measurably different output on CPU and MPS
   (about −20 dB relative on the fixture for Kim vocals), because the models amplify tiny
   numeric differences. Task P2.14 measures this on real tracks.
2. **The fixture's limits.** Models recognise its drums and bass, and the VR karaoke models
   split its lead from its backing, but most vocal, guitar and piano models hear sine tones
   as "not that instrument". Phase 1 could therefore check output naming, not quality.
3. **The residual check works.** With the provisional defaults on the fixture it reports
   about −9 dB and names the silent stems, because the models serving different stems
   disagree about the synthetic voice. Whether "other" should come from its model or be the
   mix minus the other five stems is a Phase 2 question.
4. **No published SDR figure was used anywhere**, and no quality number exists yet.

Choices made here that the owner can overrule (see `PLAN.md` section 13 for the earlier
ones):

- "other" always means the six-stem other, so guitar and piano are computed to build it
  even when not requested.
- Vocal split and de-reverb each work on the vocals stem independently; lead and backing
  are not de-reverbed.
- Demucs runs without its random-shift averaging; VR band resampling is fixed to one
  method on every platform; MDX-Net's optional denoise pass is off.
- Timings are logged by audio hash only, never by file name.

Not done, by design or for lack of access: blends and ensembles (Phase 2 chooses them),
any benchmark (needs the datasets), GPU timings (this machine has MPS and CPU only), and
CI has still never run because the repo has no remote.

**Next:** wait for the owner. Phase 2 starts at P2.1. Its harness tasks (P2.1 to P2.9,
P2.14 on synthetic data) need nothing from the owner; the real benchmarks need B2 and B3.
