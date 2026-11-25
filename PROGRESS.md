# PROGRESS

Single source of truth for where the build is. Read this first, every iteration.

**Current phase: 14 (what the six stems lose), started 2026-11-24.** Phase 13 (a listening test the owner can do alone) has one open task and it is the owner's: the kit is built and checked (P13.4): `make listen`, then http://127.0.0.1:8765; P13.5 is the listening itself, and its report follows when the marks are there. Phase 12 (the edges) is complete: see its summary. The owner asked on 2026-11-24 for each phase to follow the last without waiting, and for new phases to be added when none is left; Phases 8 to 14 were added that way. Phases 0, 1, 3 and 4 to 12 are complete apart from checks that need the owner's accounts (S3 and Azure, B7; Modal, B4; the panel listening test, B8). Still open from earlier phases: Phase 2's real benchmarks, running in the background (`tools/run_real_benchmarks.sh`; P2.10 and P2.11 are done; the MoisesDB tuning stage is done and reported; the MUSDB18-HQ tuning stage started at 18:27Z on 2026-11-24 and was at 173 of 3,000 jobs at 00:20Z on the 25th, having shared the GPU with other runs for most of that time; the MoisesDB stage took 8.8 hours for 30 tracks, so 100 tracks alone would take about 30; P2.12, P2.13, P2.14b: each later iteration checks `bench status`, commits a report when its stage has ended, and ticks the task). **This Mac cannot hold three GPU jobs at once**: beside the benchmark, run one heavy job at a time.

**Asked for by the owner on 2026-11-24, during Phase 12:** "after this next phase finishes, i'd like to do a listening test myself". So the phase after 12 is a listening test the owner can do alone: no panel, nothing bought (B8 asks for both; this is the part that needs neither).

## How to use this file

1. Take the first unticked task in the current phase that is not marked BLOCKED.
2. Implement it. Run `make check`. Commit with a clear message.
3. Tick it only when its acceptance check passes, and add a one-line note after the tick.
4. If a task needs something from the owner, add it to `BLOCKED.md`, mark the task
   `BLOCKED (Bn)` and move to the next one.
5. If a task fails three iterations in a row, write what was tried in `BLOCKED.md`, mark it
   `BLOCKED`, move on.
6. When the current phase has no unticked unblocked tasks, write the phase summary at the
   bottom of this file, then start the next phase. If no phase is left, add one here with
   tasks and acceptance checks, and work it. (Owner's instruction of 2026-11-24. Until then
   the rule was to stop and wait for a go-ahead.)

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
- [x] P0.7 CI workflow. **Accept:** `.github/workflows/ci.yml` runs lint, registry validation and tests. — Committed but never executed: the repo has no remote yet. (Since 2026-11-22 it has: see the Phase 2 summary. First green run: commit `0b357e6`.)
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
- [x] P2.4 MUSDB18-HQ loader reading `MUSDB18HQ_PATH`. **Accept:** tested against a fake directory tree built from fixtures; on first contact with the real dataset it writes `bench/frozen/musdb18hq_test.json` and refuses to benchmark until that file is committed. — `stems_bench.datasets.musdb.Musdb18HqDataset(split)` (`musdb18hq_test`, and `musdb18hq_train` for tuning). The path comes from `MUSDB18HQ_PATH` in the environment or `.env` (new `stems_common.env.get_env`). On first contact it freezes the split's track folders with a SHA-256 of all five files each, and refuses a folder count other than the released 50 or 100. `Dataset.check_ready()` refuses to benchmark unless the frozen list is tracked by git with no uncommitted changes (it applies to every dataset). Loading verifies each file's hash, requires 44.1 kHz stereo, and maps MUSDB's `other` to `other_4stem`. Tested on a fake tree of synthetic tracks written as 16-bit WAVs, including the commit gate in a scratch git repository. 11 tests, plus 2 for the `.env` reader. **Not tested on the real dataset: it is not on this machine (B2).**
- [x] P2.5 MoisesDB loader and the frozen stratified 50-track subset. **Accept:** stem grouping into the six stems plus lead and backing is tested on a fake tree; the selection script is seeded and enforces the report §5 strata; real subset freeze is **BLOCKED (B3)** until the dataset arrives. — `stems_bench.datasets.moisesdb`: reads the dataset's own layout (`<provider>/<track>/data.json` plus `<stem>/<source>.wav`; checked against the official loader's code at commit 1fbc5c78, not against the data), groups its eleven stem types into our six by the dataset's own six-stem mapping (percussion, other keys, strings and wind go to `other`), and takes lead and backing vocals from the source types inside the vocals stem. The mix is the sum of the sources; short sources are padded, not trimmed, so the six stems sum to the mix exactly. `select_subset` puts candidates in a seeded random order (seed 2307) and enforces the report's strata: at least 8 with lead vocals, 8 instrumental, 10 with guitar, 10 with piano, 10 with prominent drums and bass, 6 genres; plus 8 with both lead and backing, which the lead-versus-backing comparisons need. It says which requirement cannot be met if one cannot. Two definitions are mine, since the report gives none: a stem is present above −60 dBFS whole-track RMS, and drums and bass are prominent when each is within 12 dB of the mix. `--freeze` scans, selects and writes `bench/frozen/moisesdb_subset.json` with a hash of every file; loading verifies them. 13 tests on a fake tree. The real freeze is task P2.5b.
- [x] P2.5b Freeze the real MoisesDB subset: `python -m stems_bench.datasets.moisesdb --freeze`, then commit `bench/frozen/moisesdb_subset.json`. **Accept:** the file is committed and `python -m stems_bench.datasets.moisesdb` passes. — Done 2026-11-22 after the owner asked for the dataset to be downloaded (`moisesdb.zip`, 88.8 GB, SHA-256 matches the authors'). `bench/frozen/moisesdb_subset.json`: 50 of the 240 tracks, 10 genres, 49 with lead vocals, 26 with lead and backing, 45 with guitar, 20 with piano, 45 with prominent drums and bass. **The instrumental quota of 8 cannot be met: the dataset has 1 track without vocals** (corrections C15); the selection takes it and records the shortfall. Also frozen: `moisesdb_tuning.json`, 30 tracks outside the subset. Two facts from the real data: source types are spelled with underscores (C14; the loader accepts both), and the "prominent drums and bass" rule (within 12 dB of the mix) holds for 230 of 240 tracks, so it barely constrains anything. First real job (`htdemucs_6s` on one track) scored plausibly, and a repeat gave the same scores to four decimals.
- [x] P2.6 Resumable run manager (`bench run`). **Accept:** results cached per (model id, weights SHA-256, track id, protocol version); killing and restarting a run redoes nothing finished; `runs/bench/<run-id>/status.json` updates as it goes. — `bench run --dataset D (--models a,b | --all) [--tracks] [--device] [--overlap] [--run-id] [--keep-estimates MODELS] [--background]` and `bench status [RUN]` (`stems_bench.cli`, `stems_bench.run`; also `python -m stems_bench`). One job is one model on one track. Its result is a JSON file under `runs/bench/results/<dataset>/<model>/`, named by a hash of the protocol version, dataset, frozen-list hash, track, model, the model's pinned weights and config, the device asked for and the overlap; a job with a result file is never run again by any run. Each run folder holds `run.json` (request, commit, package versions), `status.json` (rewritten after every job; a running status whose process is gone reads as `dead`) and `results.jsonl` (every job's outcome and result path). A result holds, per output, the reference it was scored against, the BSS Eval scores with every window, the level and high-band metrics, plus device, seconds and chunk count. Models load once for all their pending tracks; a failing model or track is recorded and retried next time, and the exit code is 1. A model the dataset cannot score (no such input or reference) is listed as not applicable and not run. A six-stem model is also scored on four-stem `other` as the sum of its guitar, piano and other. The run refuses to start on an uncommitted frozen list. Kept estimates are optional and off by default: on real datasets they are about 90 MB per output per track. 19 tests with weightless fake models, including a run killed mid-way and resumed.
- [x] P2.7 Ensemble combiner: waveform and STFT-domain averaging with per-model weights. **Accept:** averaging two copies of the same estimate returns it unchanged; a weighted mix of a good and a bad fake estimate scores between them; weights are read from config. — `stems_worker.ensemble.blend(estimates, weights, method)` and `blend_stems` (it lives in the worker because the product pipeline needs it too). Methods: `waveform` (weighted mean; stems that summed to the mix still do), `stft_magnitude` (weighted mean of magnitudes with the phase of the weighted sum, so out-of-phase estimates do not cancel), and `stft_min` / `stft_max` (per-bin choice, no weights). Config: `ensembles.blend_method` with a per-slot override, and per-slot `two_model_weights` and `full_ensemble_weights`; unset means equal weights, which is what is committed, since no tuning has been run. Checked: copies of one estimate come back within 2e-6 for every method; a good/bad blend scores between its members and improves with weight on the good one (also by museval); two equally noisy estimates average to 3 dB better. 22 tests, plus the config ones.
- [x] P2.8 Report generator. **Accept:** produces Markdown with one table per dataset, seconds per song and device beside every score, the exact command, git commit and frozen-list hash; contains no number that is not in the run's result files. — `bench report [RUN] [--out FILE]` (`stems_bench.report`). The report is a pure function of the run folder and the result files it lists: it names the run, the exact command, the commit of the run and of every result (flagging uncommitted changes), the protocol version, the frozen list's SHA-256, the device, overlap and package versions, then one table per target with each model's median SDR, SIR and SAR, the number of tracks the score rests on, seconds per track, seconds per minute of audio and the device on the same row, the high-band ratio, and notes for wrong lengths or mostly silent output. Failures and models the dataset cannot score are listed. Metrics a dataset marks unreliable are left out with the reason (SIR and SAR on the synthetic set). `report.json` beside it holds the same rows and the SHA-256 of every result file. A test recomputes every allowed figure independently and checks the tables contain no other number. One report covers one dataset, so datasets are never averaged. 10 tests. Trial on two real models and two synthetic tracks worked end to end; its results were deleted so P2.9 starts clean.
- [x] P2.9 Run every registry separator on the synthetic dataset, single best vs blend vs full ensemble. **Accept:** `docs/benchmarks/synthetic.md` exists and the run finishes end to end from one command. — `make bench-synthetic` (`bench run --dataset synthetic --all --ensembles --run-id synthetic-v1`, then `bench report`). Run of 2026-11-22 on commit `905a0cf`, MPS: **560 of 560 jobs computed, none failed**: 43 separators on 8 tracks (344 jobs) and 27 systems on 8 tracks (216): every slot's single best, two-model blend and full ensemble, plus the composed six-stem other. `docs/benchmarks/synthetic.md` and `.json` are generated from it. `bench run --ensembles` keeps only the audio the blends read, scores each blend from it, and counts a blend's time as the sum of its members'. Checks that held on real models: each slot's single best as a system scores what that model scores on its own (vocals 6.11 dB both ways), and models that return silence on sine-tone voices score 0 dB and are flagged as silent. **What the report is not:** a quality ranking (synthetic audio; SDR only), or a clean timing (the Mac slept five times during the run, which pauses but does not inflate job times, and the test suite ran alongside several times, which does). 31 tests in `tests/test_bench_run.py`, 13 in `tests/test_bench_report.py`.
- [x] P2.10 Benchmark on MUSDB18-HQ test. **Accept:** `docs/benchmarks/musdb18hq.md` generated from a completed run. — Done: run `musdb18hq-v1` finished 2026-11-23T17:01Z after 12.5 hours on MPS: 1,000 jobs computed, 0 failed, 1,050 not applicable (MUSDB18-HQ has no guitar or piano references and no inputs for the vocal split or de-reverb). `docs/benchmarks/musdb18hq.md` and its `.json` were written by `bench report` from that run. It covers the 14 models named in `config/defaults.yaml` and the 12 applicable blends; stage 5 of `tools/run_real_benchmarks.sh` adds the other separators and rewrites the report. The report records that the run started from commit `3cb7b64` with uncommitted changes in the tree (Phase 3 work in progress, none of it in the benchmark's code path as far as I know; not proven). Not checked: whether any of these checkpoints was trained on MUSDB18-HQ's test tracks, which would flatter its score here. No conclusion is drawn yet: P2.13 does that from this report, MoisesDB and the tuning runs together.
- [x] P2.11 Benchmark on the MoisesDB subset, including guitar, piano and lead vs backing. **Accept:** `docs/benchmarks/moisesdb.md` generated from a completed run. — Done: run `moisesdb-v1` finished 2026-11-24T04:53Z, 11.9 hours after its restart on MPS: 2,050 jobs, 1,699 computed in this run, 101 kept from the earlier start, 0 failed, 250 not applicable (the subset has no reverberant vocals, so nothing de-reverb is scored). `docs/benchmarks/moisesdb.md` and its `.json` were written by `bench report` from that run. It covers the 14 models named in `config/defaults.yaml` and the applicable blends, with guitar (45 of 50 tracks have it), piano (20), six-stem other (33) and lead and backing vocals (26); stage 6 of `tools/run_real_benchmarks.sh` adds the other separators and rewrites the report. Results were computed on three commits (the report lists them). Not checked: whether any of these checkpoints was trained on MoisesDB tracks that are in the subset, which would flatter its score here. No conclusion is drawn yet: P2.13 does that.
- [x] P2.12a Blend tuner, proven on synthetic data. **Accept:** `bench tune` refuses any list that is not a tuning list or that shares a track with its test list; on a tuning list it tries every method and a grid of weights per slot and mode, and what it picks equals a brute-force search in a test. — `bench tune --dataset LIST [--slots] [--out FILE]` (`stems_bench.tune`). Tuning lists and the test lists each must be disjoint from are declared in one table (`synthetic_tuning` / `synthetic`, `musdb18hq_train` / `musdb18hq_test`, `moisesdb_tuning` / `moisesdb_subset`), and the check runs before any model does. It runs the members with their audio kept, then tries, per slot and mode, the four methods and weights in steps of 0.1 (two models) or 0.2 (three), 22 and 44 candidates, scoring each by median SDR over tracks; scores are cached per track, so it resumes. A slot gets one method for both modes and each mode its weights; a candidate must beat an earlier, simpler one by more than 0.001 dB. Output: `tuning.json`, a Markdown table (equal-weight baseline, choice, gain) listing the tuning tracks and the lists they were checked against, and the choice as a config snippet that is **not applied**. Candidates are compared with `metrics.sdr_frames`, the plain SDR definition, tested equal to museval's SDR. Also new: `bench/frozen/synthetic_tuning.json` (4 tracks, seeds 201 to 204), `python -m stems_bench.datasets.moisesdb --freeze-tuning` (30 seeded tracks outside the subset), and `ensemble.Blender`, which computes spectrograms once for many candidate blends. 12 tests. Run once with real models (`synthetic-tuning-v1`, commit `0b357e6`, MPS): 120 of 120 jobs, 16 blends tuned, written to `docs/benchmarks/tuning-synthetic.md`. That run proves the tool end to end and nothing else: on sine tones it mostly learns to drop the models that return silence, and nothing from it is in the config.
- [ ] P2.12 Tune blend methods and weights on tracks outside the frozen test lists, then score them on the test lists. **Accept:** the tuning tracks are listed and disjoint from the frozen lists. *Half done. The tuner finished on the MoisesDB tuning tracks on 2026-11-24T18:27Z and `docs/benchmarks/tuning-moisesdb.md` is its report: with weights free to move, the blends mostly turn back into one model (guitar, piano and the vocal split put all the weight on one; bass 0.9 on `bs_roformer_sw_6stem`), and the best any blend does against the single default model chosen in P10.2 is vocals, 11.55 dB against 11.48 with three models. Its vocal-split score (19.02) is over all 30 tracks, most of which have no backing vocals, so it is not the figure of the benchmark report, which scores the 14 tracks that have both. The MUSDB18-HQ stage (`musdb18hq-tuning-v1`, 3,000 jobs) started at 18:27Z and will take more than a day. Still to do when it ends: apply what the two reports choose and score it on the test lists.*
- [ ] P2.13 Recommend a default model or ensemble per slot from measured gain against time. **Accept:** `docs/benchmarks/recommendation.md` cites the runs; `config/defaults.yaml` is updated and `provisional` set to false. *Waits for P2.10 to P2.12 to finish.*
- [x] P2.14 Measure how much scores move between devices (CPU, MPS, and CUDA when available) for the same model and tracks. **Accept:** the benchmark report states the per-stem SDR spread across devices on the synthetic set and, once available, on real tracks; if it exceeds run-to-run noise, every comparison in a report uses a single device. — `bench run --trial N` repeats jobs under their own keys and `bench compare` sets the differences side by side (`make bench-device-spread`). On the synthetic set, the provisional default model of each slot (6 models, 8 tracks, 150 scores per comparison), runs of 2026-11-22: **running again on the same device changed nothing** (largest difference 0.000 dB on MPS and on CPU: inference is deterministic on both), **moving from MPS to CPU changed one track's score by up to 0.817 dB** (`bs_roformer_sw_6stem` piano, `synth-03`) and moved a model's median by up to 0.221 dB (`mel_roformer_kim_vocals` vocals, 6.108 to 6.329). The per-output table is `docs/benchmarks/device-spread-synthetic.md`. The device effect exceeds run-to-run noise, so scores from different devices are not compared: a run asks for one device and refuses to be resumed on another, a report is built from one run, and every report now says so. The MDX-Net models run on CPU whatever is asked (no MPS in onnxruntime); the device column shows it. No CUDA here. Real tracks are task P2.14b.
- [ ] P2.14b Device spread on real tracks: the same repeat and cross-device runs on MUSDB18-HQ (and on CUDA through Modal if there is a token). **Accept:** `docs/benchmarks/device-spread-musdb18hq.md` generated by `bench compare`. *MPS and CPU runs wait for the benchmark queue to free the machine (two heavy runs at once do not fit in 24 GB). CUDA is **BLOCKED (B4)**.*
- [x] P2.15 Blend modes in the product pipeline: `separate --mode two_model_blend` and `--mode full_ensemble`. **Accept:** on the fixture with weightless models, each mode runs every listed model once, blends per stem with the configured method and weights, and the manifest names the members and weights; `single_best` is unchanged. — `separate --mode two_model_blend | full_ensemble`. `default_assignment` turns each slot's model list, weights and method from config into a `Blend`; `plan_composition` accepts a model id or a `Blend` per stem and still plans one run per model, however many stems and blends it serves; `compose` blends per stem with `stems_worker.ensemble`. Six-stem `other` in a blend mode is the blended four-stem other minus the blended guitar and piano, and a blend that mixes four-stem and six-stem `other` models is refused. Vocal split and de-reverb are blended the same way. `--model SLOT=ID` replaces that slot's blend with one model. The manifest gives `models`, `weights` and `blend_method` for every blended stem and file; a `single_best` manifest is byte-for-byte the shape it was. Cached single-model outputs are reused by blends. 11 tests with weightless models. Not run with real models yet: which blend is worth its time is what P2.10 to P2.13 measure.

## Phase 3: listening layer and the Clef-omni test

- [x] P3.1 BEATs wrapper. **Accept:** returns 527 class probabilities per 10-second window on the fixture, on this machine. — Done 2026-11-22: the owner said to fetch the checkpoint through the browser pane; the OneDrive link served it there with no sign-in (363,145,291 bytes, SHA-256 pinned in `models/checksums.lock.json`; Microsoft publishes no checksum, so the pin is of that download). `stems_worker.listening.beats.BeatsTagger` loads it with `weights_only=True` and returns `(windows, 527)` in AudioSet's order (the checkpoint's own order differs and is rearranged by class id). Model code is Microsoft's at commit `732d834db70e`, with torchaudio's Kaldi filterbank file (torchaudio has no build for this torch), each file hash-pinned. On the 25 s fixture: 3 windows, 0.74 s on CPU, MPS within 6e-6 of CPU. 7 fast and 4 slow tests. *A look at 3 MoisesDB test tracks (not a measurement):* isolated drums, bass and guitar stems score 0.2 to 0.8 for their own stem, but on full mixes every stem scores under 0.1 while `Music` is at 0.94, and an isolated vocal stem scores 0.05 to 0.16 (its mass goes to `Mantra`, `Chant`, `A capella`, which the stem map does not list). So the pre-registered 0.5 threshold will make BEATs call almost everything absent on mixes; the report's best-F1 figure is the fairer view of it, and the `listening` thresholds in config are far too high for mixes. Nothing pre-registered was changed after this look.
- [x] P3.2 AudioSet class-to-stem mapping file. **Accept:** committed as data, with a test that every mapped class exists in the label set. — `models/listening/audioset_stem_map.yaml` (version 1) with AudioSet's own 527-class list beside it (`audioset_class_labels.csv`, hash-pinned in the map). 78 classes are mapped: vocals 13, drums 9, bass 2, guitar 6, piano 2, other 46; a stem's score is the largest probability among its classes. The report's core classes are where it says, widened to their neighbours; unpitched percussion that is not a drum kit goes to `other`, as in MoisesDB's own grouping. Every class from "Music" to "Scratching" is either mapped or listed as deliberately unmapped with a reason (14, such as "Keyboard (musical)"). `stems_worker.listening.audioset.load_stem_map()` validates it (unknown class, class in two stems, wrong label file) and gives `stem_scores` by index or by class id, so a tagger's own class order does not matter. Its SHA-256 is exposed for the Clef-omni pre-registration. 10 tests. Done before P3.1, which is blocked on the BEATs file (B10).
- [x] P3.3 LAION-CLAP wrapper with text prompts and a high-band check. **Accept:** on fixture clips with and without content above 8 kHz the high-band signal differs in the right direction. — `stems_worker.listening.clap.ClapTagger` (`laion_clap_music`, 48 kHz, 10 s windows, 512-dimension embeddings). `embed_audio`, `embed_text`, `similarity`, `score_prompts` (zero-shot: softmax over prompts at the model's own temperature) and `high_band`: one minus the cosine similarity between a clip and the same clip low-passed at 8 kHz. On the fixture with the real checkpoint: **0.047 for the mix against 0.0002 for the low-passed mix; 0.098 for the drums stem against 0.0000004 for the bass stem**, and the prompts name the drums and bass stems correctly (0.99 each). The `laion_clap` package would have downgraded numpy and installed a training stack, so its three model files are taken from the pinned 1.1.7 wheel (`ensure_clap_lib`; hash equals PyPI's) with small stubs for the rest. The checkpoint loads with `weights_only=True` and a three-type numpy allowlist, strictly into the model. The text tower is built from RoBERTa-base's config and the tokenizer from its `tokenizer.json`, both pinned to a commit in the registry, so nothing is fetched from a hub at run time. Runs on CPU and MPS (embeddings agree to 2e-7). New pins: transformers 5.24.0, tokenizers 0.23.3, torchlibrosa 0.1.0. The model's mel front end stops at 14 kHz, so "above 8 kHz" means 8 to 14 kHz. 8 fast tests, 4 with the real weights.
- [x] P3.4 `track_features` builder and per-stem signals (presence, bleed estimate, notes). **Accept:** output validates against the `track_features` part of schema (b) and the `metrics` part of schema (c). *Note before starting (2026-11-22, a look at 3 MoisesDB tracks, one 30 s window each, not a measurement):* with plain prompts ("the sound of drums"), CLAP's cosine similarity singles out isolated drums, bass and guitar stems (0.33 to 0.45 against 0.2 or less for the wrong prompts), but is weak for vocals and piano stems (about 0.15, not separated from other prompts) and weak on full mixes (0.02 to 0.24). So CLAP alone is not a presence detector for mixes; `track_features` needs the AudioSet tagger (BEATs, blocked on B10) as planned, with CLAP as the second opinion and the high-band check. Build the feature builder against a tagger interface, with thresholds in config and marked uncalibrated until measured on MoisesDB. — Done: `stems_worker.listening.features` (`build_track_features`, `stem_signals`) reads any tagger that returns 527 AudioSet probabilities per window; schemas transcribed from the report into `common/stems_common/schemas/` and every output is validated against them; BS.1770 loudness in `listening/loudness.py`, checked against the standard's 48 kHz coefficients and its -3.01 LKFS reference tone; class lists for backing vocals, strings, synth, brass, genres and reverb in `models/listening/audioset_feature_classes.yaml`; thresholds in `config/defaults.yaml` under `listening`, `calibrated: false`. With no tagger the hearing fields are `unknown`/null and levels and loudness still hold. **Not yet shown:** how it behaves with a real tagger. Every tagger-dependent test uses a scripted one, because BEATs is still missing (B10); 39 tests.
- [x] P3.5 Pre-registration file for the Clef-omni test: thresholds A1-A3, B1-B3, C, the 50 track ids, 30-second windows, class-to-stem mapping. **Accept:** committed before any Clef-omni request; its commit hash is embedded in the test's log header. *Unblocked 2026-11-22: the 50 track ids are frozen in `bench/frozen/moisesdb_subset.json`.* — Done: `bench/preregistration/clef-omni-v1.json`, written by `python -m stems_bench.clef.prereg --freeze` from the 50 frozen tracks. It fixes thresholds A1-A3, B1-B3 and C, the decision rule, every question's wording, the 30-second window of each track, the presence answers (300 mix decisions; vocals absent from 6 of 50 excerpts, piano present in 14), Part B's 20 clean stems and contaminants, the 5 reverb clips, 23 lead-or-backing clips, and the stem map's hash. `prereg.load()` returns the file with the commit that last changed it and refuses a file that is not committed as it stands; a test fails if code, stem map or frozen list drift from it. No Clef-omni request has been made. *Amended once, before any request (2026-11-23): added the request layout (state text, one clip per request, question ids); the file lists the amendment.*
- [x] P3.6 Clip builder: loudest 30-second window, programmatic presence ground truth, Part B bleed mixes at +30, +20, +12 and +6 dB SIR, five reverb clips, 16 kHz mono encoding under 8 MiB. **Accept:** tested on the synthetic dataset, including the measured SIR of each bleed clip. — Done, ahead of P3.5 because the pre-registration needs its window and presence rules: `stems_bench.clef.clips` (loudest window on a one-second grid, the -40 dBFS / 5 s presence rule, `mix_at_sir`, seeded Part B selection with no track used twice, 16 kHz mono PCM16 encoding that only ever turns a clip down) and `clef.clipset` (`make_plan`, `build_clips` with a manifest). On the synthetic set every bleed clip's measured SIR equals the requested one to 1e-6 dB, and the contaminant recovered from the written files rises by 24 dB (within 3) from +30 to +6 dB SIR. The reverb probe uses Part B's clean vocal stems. 16 tests.
- [x] P3.7 Clef-omni client with full request and response logging and a recorded-response mock. **Accept:** enforces 4 clips, 8 MiB, 300 seconds and 64 questions per request; every call appends to a JSONL log with question ids, probabilities and token usage; the token never appears in logs. — Done: `stems_bench.clef.client`. Request and answer shapes are the ones on Cloudflare's model page (noul gives `noul`; choice gives `choice`, `probabilities`, `confidence`; score adds `score` and `legend`; audio as `{content_type, base64}`). Limits come from the registry entry and a request over any of them is refused before sending. Each call appends a JSONL line (question ids and wording, clip id, size and SHA-256 but never audio, status, attempts, every probability, token usage); the first line is a header naming the pre-registration commit. Retries on 408/429/5xx and network errors with 1, 2, 4 s waits. `RecordingTransport` saves exchanges and `ReplayTransport` answers from them with no network. **Not live-tested:** the saved responses in the tests are hand-written to the documented format, and whether the REST endpoint wraps the body in `{result, success}` is unconfirmed (both are accepted). 35 tests.
- [x] P3.8 Scoring: accuracy overall and per stem, AUC, Spearman correlation, clean false-positive rate, pass or fail per threshold. **Accept:** unit tests with hand-computed cases for each statistic. — Done: `stems_bench.clef.scoring`: `accuracy`, `auc` (pair counting, ties half), `spearman` (on mean ranks), `best_f1_threshold` (secondary), `score_part_a` (A1-A3, plus what always answering 'present' would score on A3), `score_part_b` (B1-B3; a clean clip is flagged by score >= 1 or probability >= 0.5), `compare` (C, in points, needed on both A1 and B1) and `verdict` (the pre-registered decision rule). A statistic that cannot be computed fails its threshold rather than passing. 14 tests, each against a case worked by hand.
- [x] P3.9 Run Parts A and B for Clef-omni and BEATs. **Accept:** `runs/clef_test/<run-id>/` holds all logs and scores. *Unblocked 2026-11-24: the Cloudflare account id and token are in `.env` and Cloudflare reports the token active; the Clef-omni half is the next thing to run.* *BEATs half run 2026-11-22* (`python -m stems_bench.clef.run --only beats`, run `clef-omni-v1`, CPU, all 300 mix decisions and 100 Part B clips): A1 26.0% (78 of 300) at the pre-registered 0.5; per stem vocals 12%, drums 4%, bass 4%, guitar 12%, piano 72%, other 52%; B1 AUC 0.643 pooled and 0.598 at +30 dB; B2 Spearman 0.231; B3 10% (2 of 20 clean clips flagged). On mixes the median stem score is under 0.1 for every stem, so at 0.5 BEATs answers 'absent' almost everywhere; the best-F1 cut-offs are 0.02 or lower, where it is close to answering 'present' every time. The median bleed score does rise with every level (0.022 clean, then 0.047, 0.062, 0.078, 0.126), but clean bass stems score 0.24 to 0.61 for guitar (AudioSet files bass guitar under guitar), which buries it. These are BEATs with the pre-registered mapping and thresholds, not a tuned detector. The task stays open for the Clef-omni half.* *Ready to run (2026-11-23): `python -m stems_bench.clef.run` builds the clips from the pre-registration, asks every pre-registered question, runs the tagger on the same clips, scores both and writes the run folder; it resumes from its log. Proven end to end on the synthetic set with stand-ins that answer from the known truth (`tests/test_clef_run.py`). Nothing has been sent to Clef-omni and no real tagger has run.* — Done: Clef-omni half run 2026-11-24 (`--only clef`, run `clef-omni-v1`): 183 of 183 requests answered on the first attempt, 137,930 input tokens, logged in `clef-log.jsonl` with no token or account id in it. A further run of both halves sent nothing new and recomputed BEATs, so `runs/clef_test/clef-omni-v1/` holds `run.json` (both ran, clean tree at commit `83d7a20`), the request log, the tagger scores and `scores.json` with both systems and the verdict.
- [x] P3.10 Pass/fail report and recommendation. **Accept:** `docs/clef-omni-test.md` states each threshold, the measured value and the verdict, generated from the run. *The generator exists: `python -m stems_bench.clef.run --report docs/clef-omni-test.md` writes the report from a run folder's files.* — Done: `docs/clef-omni-test.md`, generated by `python -m stems_bench.clef.run --run-id clef-omni-v1 --report`: every threshold with its measured value and result for both systems, the margin test, and the verdict. Clef-omni: rejected (A1 81.3% against 85%; piano 76% and other 58%; bleed AUC at +30 dB 0.757 against 0.80; 4 of 20 clean clips called bleeding). It beat BEATs by 55 points on A1 and 26 on B1, which the margin test passes but which does not rescue a system that misses its own thresholds. Listening layer: BEATs with CLAP as second opinion, as PLAN.md decision 6 already had it.
- [x] P3.11 Optional: MERT probe and Essentia model compared with BEATs on Part A. **Accept:** results added to the same report, or a note saying it was skipped and why. *Waits for P3.10: it adds to that report, and compares against BEATs.* — Skipped, and this is the note (written 2026-11-24, in P11.5). The comparison was to say which listening model names instruments best on short clips, in case another should replace BEATs. Since then the question it served has been answered another way: on real separator output nothing a tagger of this kind hears is something the gate acts on (P7.1, P8.2), it cannot name instruments in a full mix (P8.4), and the tagger is off by default (P9.3). A better namer of instruments in clips would not change what the product does. Neither MERT nor an Essentia model was downloaded or installed, so whether they run on this machine is not known, and no figure for them exists.

## Phase 4: decision layer

- [x] P4.1 The three JSON Schemas as files, with schema (b)'s model enums generated from the registry. **Accept:** valid and invalid examples for each schema are tested; changing the registry changes the generated enum. — Done: `parsed_stem_request`, `ensemble_plan` and `stem_quality_assessment` in `common/stems_common/schemas/`, transcribed from the report; (b)'s per-stem lists are `none` plus the loadable candidates for that stem and are generated by `stems_models.plan_schema` (`make schemas`), with a test that fails when the committed file no longer matches the registry. `uuid` formats are enforced. 35 tests (3 valid examples, 27 invalid ones, registry-change cases).
- [x] P4.2 Jev client pinned to `jev-1.14.0`. **Accept:** retries 429 and 5xx with backoff and honours `retry-after`; times out; logs model field, token usage and probabilities for every call; refuses an alias; the key never appears in logs (tested). — Done: `stems_worker.decision.jev`: plain HTTPS (no SDK, so the pinned model and the retries are ours to control and log). Refuses any model that is not an exact version, and a reply from a different model. Retries 408, 429, 5xx, timeouts and network errors up to `decision.max_attempts`, waiting the reply's `retry-after` (seconds or a date) or `backoff_seconds` doubling; a `retry-after` over `max_retry_wait_seconds` ends the call so the fallback answers. Each call appends one JSONL entry: responding model, token usage, every probability, each attempt's status. The key is held only by the transport and is scrubbed from logs and error text. 55 tests, two against a real HTTP server on localhost (headers, a stalled server timing out). Not yet run against TypeSafe itself (P4.9, B6).
- [x] P4.3 Recorded-response mock transport. **Accept:** replays fixtures by request hash and fails loudly on an unrecorded request. — Done: `stems_worker.decision.mock`: `ReplayTransport` answers only a request recorded with exactly that model, state and question wording (SHA-256 of the body), from `.jsonl`/`.json` files or folders; anything else raises `UnrecordedRequest`, which is not a `JevError`, so the rule-based fallback cannot hide it. Each recording carries its `origin` (`jev`, `handwritten` or `scripted`) and is refused if its request no longer matches its hash. `RecordingTransport` saves settled replies (not retried ones, and no headers beyond `retry-after`) for P4.9. A client on it reports provider `mock`. 11 tests.
- [x] P4.4 Request parsing to `ParsedStemRequest`, with the keyword-rule fallback. **Accept:** mapping tested from recorded answers; "no stem detected" sets `needs_review`; the fallback handles the test phrases with no network. — Done: `stems_worker.decision.parse`: nine questions (a noul per stem, choices for format and quality, a noul for the vocal split); a stem is wanted from `decision.wanted_threshold` (0.7) up; `confidence` is the lowest among the answers used. Stem answers under `min_confidence` hand the request to the keyword rules and set `needs_review`; an unsure format or quality answer just becomes `unspecified`. No stem recognised sets `needs_review` with `no_stem_detected` on both paths and assumes nothing. The rules handle 28 test phrases with no network (synonyms, all/everything, exclusions, instrumental, acapella, formats, quality words, lead/backing); one known limit is pinned in a test (an exclusion placed after what it excludes). `from_selection` builds the object from checkboxes without parsing. Mapping is tested from hand-written wire-format answers in `tests/fixtures/jev/parse.handwritten.json` (labelled as such; real recordings come with P4.9). Every decision returns a log record (provider, model, tokens, latency, answers, fallback reason). 53 tests.
- [x] P4.5 Ensemble planning to `EnsemblePlan`, with the config-default fallback. **Accept:** enforces the requested/primary/fallback rules from the report; low confidence falls back and says why. — Done: `stems_worker.decision.plan`: a choice per requested stem among the registry's loadable candidates (each described with its place in `config/defaults.yaml`, so the measured default is marked), a choice of de-reverb model when de-reverb was asked for, and a noul on whether a blend is worth the time. `check_rules` enforces the report's requested/primary/fallback rules on every plan, on top of the schema generated for the registry in use. An answer under `min_confidence` is replaced by the configured default for that stem alone, `needs_review` is set, and the record names the answer and its confidence; with no usable answer, no key or a failed call the whole plan is the configuration. `estimated_seconds` comes only from measured rates in the timing log; unmeasured models add nothing and are named (a lower bound, never a guess). `assignment_for` turns a plan into what the pipeline runs. Nothing unrequested is added: de-reverb is planned only on request. New setting `decision.mode_for_quality`. Hand-written answers in `tests/fixtures/jev/plan.handwritten.json` (`tools/make_plan_fixture.py`). 21 tests.
- [x] P4.6 Stem assessment to `StemQualityAssessment`, with the rule-based fallback scorer. **Accept:** noul confidence is derived as `max(p, 1 − p)`; a stem with no reference SDR is gated on tagger signals. — Done: `stems_worker.decision.assess`: three questions per stem (score on the five rubric labels, noul "acceptable to deliver", choice of retry). `score` is Jev's fractional score, `score_label` its most probable level, `confidence` the score answer's; the noul has no confidence field (C1), so `max(p, 1 − p)` is used, and a score or noul under `min_confidence` hands the stem to the rule scorer. The rule scorer (`gate.rules` in config, uncalibrated) reads only silence, the tagger's bleed level and whether the stem's own source was heard; it never reads SDR, so a stem with no reference is gated on tagger signals, and the state sent to Jev leaves a missing SDR null. With no tagger it reports `low_confidence` instead of a judgement. `decision/gate.py` holds the gate rule (inclusive thresholds from config) and the verdict. Design choice recorded as BLOCKED.md D5: an unverifiable stem is delivered with a warning, not retried, and the warning says only that it was not checked. Hand-written answers in `tests/fixtures/jev/gate.handwritten.json`. 34 tests.
- [x] P4.7 Gate and retry state machine. **Accept:** table-driven tests for deliver, each retry action, the 2-retry cap, and deliver-with-warning choosing the best attempt; thresholds come from config. — Done: `stems_worker.decision.retry.StemGate`: one machine per stem (assessing → delivered, or → delivered_with_warning). An attempt is a `Spec` (models, mode, de-reverb); `retry_fallback_model` runs the plan's fallback alone, `retry_other_ensemble` the next larger configured blend, `retry_with_dereverb` the same attempt de-reverbed (vocals only). An action is offered only if it leads to an attempt not made yet; an impossible suggestion gives way to the next action in a fixed order. The cap (`gate.max_retries_per_stem`, 2) and all three thresholds come from config. When nothing passes, the best attempt is delivered: highest score, then pass probability, then the earliest. Jev's `flag_for_human_review` becomes a warning (no review queue in v1). Table-driven tests: deliver, each retry action, the cap at 0, 1 and 2, best-attempt choice with ties, a later pass, unverifiable stems, exhausted options, forged transitions. 24 tests.
- [x] P4.8 Whole pipeline on the fixture with the mock, and again with no key. **Accept:** both produce stems and a complete decision log; the no-key run shows `provider: fallback` for all three decisions. — Done: `stems_worker.decision.job.run_job` (and `python -m stems_worker.decision.job`): parse or checkboxes → track features → plan → separate → per stem assess, gate, retry → optional split and de-reverb → stems and `job.json` with every attempt and one log record per decision. Tested on the synthetic fixture with weightless models: (1) with a mock of Jev (a scripted stand-in, recorded and then replayed from the recording alone with identical decisions): all three kinds show provider `mock`; (2) with no key: all three show `fallback` with reason `no_key`, stems are written sample-exact, and because nothing listened to them each carries the warning that it was not quality-checked. Also tested: a silent stem retried with the fallback model, a retry that passes, the cap when everything fails, a calibrated stand-in tagger letting the fallback gate deliver cleanly, a request with no stem stopping at `needs_review` with no audio written, and the extras. `pipeline.run_stage` now serves both this and `separate`. 10 tests. Not run with real models or real Jev answers here.
- [x] P4.9 Live check against Jev and capture of real recordings. **Accept:** one real request per call type succeeds and is saved as a fixture. — Done: `python -m stems_worker.decision.live`, run 2026-11-24 with the owner's key: one request per decision type to `jev-1.14.0`, each answered on the first attempt with HTTP 200. Parse: 1,094 input and 241 output tokens, 186 ms. Plan: 2,316 and 592, 118 ms. Gate: 744 and 96, 108 ms. The three exchanges (request and reply, no key, no account headers) are `tests/fixtures/jev/{parse,plan,gate}.recorded.jsonl`, and tests replay them into valid documents. The live replies have exactly the documented shape: a noul carries only its probability (corrections C1), and the score came back fractional, 2.23 (C2). What Jev said, on made-up inputs with no audio behind them: the parse was used (vocals 0.98, drums 0.84, flac, lowest confidence 0.84); the plan and the gate answers were below `decision.min_confidence` (0.8), at 0.50 to 0.76 and 0.55 to 0.59, so the fallbacks decided both, as designed. Three calls say nothing about how often that happens on real jobs or whether Jev's choices are good; that is P7.1. A second run of the tool sends nothing. 7 tests.

## Phase 5: API and worker

- [x] P5.1 Storage interface and local-disk backend with HMAC-signed expiring links. **Accept:** expired or tampered links are rejected; keys cannot escape the storage root. — Done: `stems_api.storage`: the `Storage` protocol of PLAN.md section 6 (`put`, `open`, `exists`, `delete_prefix`, `signed_url`), JSON and bytes helpers, and `LocalStorage`. Keys are checked before any backend sees them: plain segments only, so `..`, absolute paths, hidden names and backslashes are refused, and the local backend also refuses a path that resolves outside its root through a symbolic link. A link is a token of base64url(payload) and an HMAC-SHA256 over it; the payload holds key, expiry and download name, so neither can be changed. Tested: a link is rejected from its expiry onwards, with any single character changed, with the payload swapped for another key or a later expiry, and when signed with another secret. Writes are atomic (a failed upload leaves the old object). With no `STEMS_SIGNING_SECRET` the secret is made once in `<dir>/signing-secret` (mode 600). The file route with range support is P5.9. 59 tests, 29 of them the contract every backend must pass (`tests/test_storage.py`).
- [x] P5.2 S3-compatible backend. **Accept:** passes the shared storage test suite against a stubbed client; live check noted as done or **BLOCKED (B7)**. — Done: `stems_api.storage.s3.S3Storage` on a boto3 client: `upload_fileobj`, `get_object`, `head_object`, paginated `list_objects_v2`, `delete_objects` in batches of 1,000, and presigned GET links that carry the download name. The 29 contract tests pass against an in-memory client (`tests/storage_fakes.py`). 11 more tests use the real boto3 1.44.10 client with no network: botocore's `Stubber` checks each request's parameters against the S3 service model, and the presigned link is a real SigV4 URL. Those tests found two things the fake could not: boto3 adds a checksum algorithm to uploads, and a link is only SigV4 when the client is configured for it (the factory does both; for a custom endpoint it also uses path-style addressing and turns the optional checksums off). `STEMS_STORAGE_BACKEND=s3` reads `S3_BUCKET`, `S3_ENDPOINT_URL` and the AWS credentials. boto3 is the first entry of the `api` extra. **Live check against a real bucket: BLOCKED (B7), not done.**
- [x] P5.3 Azure Blob backend. **Accept:** same as P5.2. — Done: `stems_api.storage.azure_blob.AzureBlobStorage` on an `azure-storage-blob` container client: `upload_blob`, `download_blob` (wrapped as a buffered binary file), `exists`, `list_blob_names`, `delete_blob`, and links that are the blob URL with a read-only, HTTPS-only SAS token carrying the expiry and the download name. Tested through the real SDK (12.31.1) with no network: `tests/azure_fake.py` is an azure-core transport that answers the SDK's HTTP requests from memory, so the 29 contract tests and 8 more run through the SDK's own request building and response parsing; SAS tokens are computed locally by the SDK. The fake does not check request signatures. `STEMS_STORAGE_BACKEND=azure_blob` reads `AZURE_STORAGE_CONNECTION_STRING` (which must hold the account key, since that signs the links) and `AZURE_STORAGE_CONTAINER`. **Live check against a real storage account: BLOCKED (B7), not done.**
- [x] P5.4 Database models and migrations. **Accept:** schema matches `PLAN.md` section 7; every job query is owner-scoped (tested by asking for another owner's job). — Done: `stems_api.db`: SQLAlchemy 2.1.9 models for the six tables of PLAN.md section 7, Alembic migrations run from code (`python -m stems_api.db.migrate`, no ini file), and queries. A test reads the table in PLAN.md and checks every table and column it names exists; the schema adds three columns to the plan's list (`jobs.display_name`, `jobs.source_bytes`, `job_events.seq`), and another test checks that the migration gives exactly the schema the models describe. Owner scoping: request code gets a `JobStore` bound to one owner, and each of its methods filters on that owner. Tested by asking as one owner for another's job through every read (job, stems, one stem, events, decision calls, timings: all answer as if it did not exist) and every write (delete does nothing; delete-everything removes only the caller's rows). Queries that act for no owner (poller, sweeper) are in a separate module, `db.system`. Only the SHA-256 of an owner's cookie secret is stored. Deleting or expiring a job scrubs its row (file name, analysis, plan, stems, events) and keeps the marker. Times are UTC and must carry a zone. SQLite is opened with foreign keys on and WAL. 18 tests, all on SQLite; Postgres has not been run.
- [x] P5.5 Job spec, status and result contract between API and worker. **Accept:** JSON Schemas for `spec.json`, `status.json`, `result.json`; both sides validate. — Done: `stems_common.jobs` and three schemas in `stems_common/schemas`: `job_spec` (written once by the API), `job_status` (rewritten by the worker, with the whole event history so a missed poll loses nothing) and `job_result` (written last; its presence ends the job). The validators add what a schema cannot say: timestamps parse and carry a zone, every key a document names lies inside its own job's folder, events are numbered 1, 2, 3 in order, a failed job says why. Both sides call them. The API builds the spec through `build_spec` and the provider validates it again before queueing; it validates every status and result it reads (an invalid status is ignored, an invalid result fails the job, one written for another job is not believed: tested over HTTP). The worker validates the spec it reads (a spec copied from another job is refused) and every status and result before writing. Job ids must be canonical UUIDs, because keys are built from them. 75 contract tests.
- [x] P5.6 Worker provider interface and the local provider. **Accept:** runs a job in a subprocess; a slow fake job does not delay an unrelated API request (tested with a timing bound). — Done: `stems_api.providers`: the `WorkerProvider` protocol (`submit`, `status`, `cancel`) and `LocalProvider`, which queues jobs and runs each as `python -m stems_worker.jobs JOB_ID` in its own subprocess, one at a time, from a dispatcher thread. `submit` validates the spec and returns at once. Timing, measured in the tests: with a 1.5-second worker running, `submit` and `status` for other jobs return within 0.25 s; and over HTTP, with the real local provider and a 2-second worker, the upload returned in under 1 s and `GET /health`, `/config` and `/jobs` each in under 0.5 s while that worker was still running. Also tested: order and one-at-a-time, a failing worker, a command that cannot start, cancel of a running and of a queued job, no double run. A queue that is lost when the API restarts is rebuilt from the database at startup (P5.9 tests that). 13 provider tests and 1 HTTP test.
- [x] P5.7 Worker `run_job`: the full pipeline with stage reporting. **Accept:** on the fixture it walks every stage and writes stems, peaks and `result.json`; re-running a finished job does no work. — Done: `stems_worker.jobs.run_job(job_id, storage)` and `python -m stems_worker.jobs JOB_ID`: reads and validates `spec.json`, downloads the source, runs the decision-layer pipeline, uploads the stems, a waveform-peaks file per stem (1,000 low/high pairs) and `stems.zip`, then writes `result.json` last. `status.json` is rewritten at every stage and stem change with the whole event history; progress alone rewrites it at most every 2 s. On the synthetic fixture with weightless models the history reads transcoding, analysing, planning, separating, assessing per stem, packaging, ready, and every status written on the way validates. A second run of a finished job makes one read (the result), builds no model and writes nothing. A failure still ends in a valid result: a non-audio upload, audio past the duration limit and an exception inside a model each give `state: failed` with a message. Text that names no stem gives `needs_review` and separates nothing. The pipeline gained `on_stage` and `max_duration_seconds`, and its report names each stem's model ids. `STEMS_WORKER_FAKE_MODELS=1` swaps in band-splitting stand-ins (`stems_worker.fakes.fake_pipeline`) for tests and a first local run; their stems are frequency bands, not music. 15 tests. Not run: a job with real models through this entry point.
- [x] P5.8 API: owner cookie, `POST /jobs` with validation, `GET /config`, `GET /health`. **Accept:** rejects oversize, overlong and non-audio uploads; a renamed non-audio file is rejected by decoding, not by extension. — Done: `stems_api.app.create_app` (FastAPI 0.165.0; `make api`). The owner cookie is made with the first accepted upload: 256 random bits, HttpOnly, SameSite=Lax, Path=/api, Secure over HTTPS, and only its SHA-256 in the database. `POST /jobs` checks, in order: rate limits, options, the file name's extension, the size, then the content by decoding (header parsed and the first second decoded with soundfile or PyAV). Refused with nothing kept (no row, no object, no owner, no cookie): a PDF renamed to .mp3, .wav, .flac or .m4a and a text file named .ogg (415 by decoding), a WAV header with no audio (415), a .pdf name (415), a file over the size limit by Content-Length and one streamed with no Content-Length (413, cut off while reading), an empty file (400), audio over the duration limit (422), bad options (422). A WAV named .mp3 is accepted, since the content decides. An accepted upload gets a row first, then its source and a validated `spec.json` under `jobs/{uuid}/`, then goes to the provider; the uploaded name is only a label and never part of a key. If storage or the provider fails, the objects are removed and the job is left failed for its owner to see. State-changing requests that name another site's Origin get 403; CORS allows only `STEMS_WEB_ORIGIN`. The API process does not import torch (tested). 25 tests. The other routes of PLAN.md section 8 are in the same module and are tested under P5.9.
- [x] P5.9 API: job list, status, events, signed links, peaks, delete, delete-all. **Accept:** deletion removes every object under the job prefix and the links stop working. — Done: All in `stems_api.app`, each scoped to the owner cookie. `GET /jobs` (own jobs, newest first), `GET /jobs/{id}` (status, stage, per-stem state while running, warnings and quality when ready, seconds until expiry), `GET /jobs/{id}/events` (server-sent events: the recorded history, a snapshot whenever the job changes, an end event; resumes from `Last-Event-ID`), `POST /jobs/{id}/links` (signed links per stem and for the zip, named after the upload), `GET /files/{token}` (local storage only; range requests answer 206 and 416), `GET /jobs/{id}/stems/{name}/peaks`, `DELETE /jobs/{id}`, `DELETE /me`, and `POST /requests/parse`. `stems_api.sync.JobSync` brings a job's row up to date from `status.json` and `result.json` whenever it is read, and fails a job whose worker is gone with no result; at startup unfinished jobs are handed back to the provider. Deletion, tested on a ready job: all 9 objects under its prefix are gone, its links answer 404 inside their 15 minutes, the row is scrubbed to `deleted`, and the owner's other job is untouched. Deleting a job in progress cancels its worker. Another browser gets, on every job route, exactly the 404 a nonexistent job gets. A link is refused once expired (410) or altered (404). A request whose text names no stem ends as failed with the parse shown, since the job table has no review state. 21 tests with the worker run in-process on weightless models.
- [x] P5.10 Retention sweeper. **Accept:** with a fake clock, a job past `retention.days` loses its objects and is marked expired; running the sweeper twice changes nothing more. — Done: `stems_api.sweeper.sweep`, run once at startup and then every `api.sweep_interval_seconds` (600) by the API's maintenance thread. Tested on a clock the test moves, with retention at the shipped 7 days: one second before a job's seven days nothing happens; at exactly seven days its 9 objects are deleted and its row is scrubbed to `expired` (file name, analysis, plan, stems, events and decision log gone; timings kept), while a job uploaded three days later is untouched. A second run reports no change and leaves storage, rows and the job list identical. Also tested: a link made before expiry answers 404 afterwards, an expired job still in progress has its worker cancelled, a storage failure leaves the job unexpired for the next sweep, the sweep at startup, and the thread doing both syncing and sweeping unprompted. For 24 hours after a job is deleted or expires its prefix is deleted again, which removes anything a worker wrote after the delete. 7 tests.
- [x] P5.11 Rate limits per owner and per IP. **Accept:** limits come from config and return 429 with `Retry-After`. — Done: `stems_api.ratelimit.RateLimiter`: a sliding window per key, all-or-nothing across the limits a request falls under, so a request refused by one does not use up another. The numbers are `rate_limits` in `config/defaults.yaml`: per browser 20 uploads an hour, 20 parses a minute and 3 jobs in progress; per address 40 uploads an hour and 40 parses a minute. These are proposals (BLOCKED.md D7), not measurements. A refusal is 429 with `Retry-After` in whole seconds until the oldest counted request leaves the window; tested with the limits overridden to small values, so the tests show the numbers come from config. Refused uploads count. The per-address limit holds when cookies are cleared (five fresh browsers from one address with a limit of 3: 201, 201, 201, 429, 429). Counted in memory in one process, and the address is the connection's: behind a proxy all users share it. 10 tests.
- [x] P5.12 Modal provider. **Accept:** unit-tested against a stub; live run of the fixture on Modal noted as done or **BLOCKED (B4)**. — Done: `stems_api.providers.modal.ModalProvider`: `submit` spawns a call of the deployed function `run_job` in the Modal app `stems-worker` and keeps the call id; `status` asks for the call's result with a zero timeout, so it never waits on a job; `cancel` cancels the call. `worker/stems_worker/modal_app.py` defines that app: the same `stems_worker.jobs` entry point the local provider runs, a T4, weights on a Modal Volume, storage settings from a Modal Secret. Unit-tested against a stub: the installed Modal SDK (1.6.4) with its two network entry points (`Function.from_name`, `FunctionCall.from_id`) replaced, and the SDK's own exception classes for still-running, not-found and failed calls. Selecting it needs `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` and a shared storage backend (s3 or azure_blob), and says so otherwise. 7 tests. **Live run of the fixture on Modal: BLOCKED (B4), not done.** Nothing here has touched Modal: the app has never been deployed, its image has never been built, and the GPU type and timeout are first choices.
- [x] P5.13 Azure Container Apps seam. **Accept:** a documented class with the two methods to implement and a test that selecting it fails with a clear message. — Done: `stems_api.providers.azure_aca.AzureContainerAppsProvider`: `submit`, `status` and `cancel` are written in terms of two methods that raise `NotImplementedError`, `_start_job(job_id)` and `_execution_state(execution_name)`, and the module's docstring says what each must do and with which Azure calls. Selecting it (`STEMS_WORKER_PROVIDER=azure_container_apps`) fails at startup with a message that names the two methods, the file and the alternatives (tested). A test finds exactly those two unwritten methods in the class, and another fills them in with stand-ins and gets a working provider, which shows the two are all that is missing. 3 tests. No Azure call exists or was made.
- [x] P5.14 End-to-end API test with the local provider on the fixture. **Accept:** upload to download works in one test, using the fake separator so it runs in CI. — Done: `tests/test_api_e2e.py`: one test through the real pieces wired as in production: the HTTP app with its maintenance thread, the local provider starting `python -m stems_worker.jobs` in a subprocess, local storage on disk and a SQLite file. Only the separator is a stand-in (`STEMS_WORKER_FAKE_MODELS=1`: band-splitting models with no weights, on CPU), so it runs in CI. It uploads a 2-second WAV asking for vocals and bass as FLAC, polls the job to `ready`, checks the event history is in stage order, downloads both stems and the zip through signed links and decodes them (44.1 kHz stereo, 2 s, not silent, different from each other), reads the peaks, then deletes the job and checks all 9 objects are gone and the links answer 404. It passed three times in a row here, about 2 seconds each. Separately, on 2026-11-24, the same path was driven by hand against `python -m stems_api` on a real port with `curl` (see the phase summary).

## Phase 6: web app

- [x] P6.1 App shell, typed API client, error states. **Accept:** `make web-check` passes. — Done: The app shell (`web/app/layout.tsx`: skip link, header, navigation, footer with Terms and Privacy on every page; one stylesheet, light and dark), the typed client (`web/lib/api.ts`: every route of PLAN.md section 8, the owner cookie sent on each call, uploads through XMLHttpRequest so progress can be shown), and error states: the API's own message is shown when it gives one, an unreachable server and a 404 each get their own wording, and a 429 says how long to wait. `make web-check` passes: ESLint, TypeScript, 21 unit tests run with Node's own test runner (formatting, the stage list, and the player engine's scheduling), and the Next.js build. The pages for P6.2 to P6.8 are written in the same commit; each is ticked once the browser test shows its acceptance check.
- [x] P6.2 Upload page: file drop, stem choices, optional free-text request with confirmation when `needs_review`, format and quality mode. **Accept:** in the browser test a chosen file with ticked stems becomes a job; a request in words fills the ticks from `POST /requests/parse`, and when the parse says `needs_review` the page says so and will not upload until stems are ticked; a file of the wrong type or over the size limit is refused in the page from the API's own limits; a refusal by the API is shown in the API's words. — Done: `web/components/UploadForm.tsx`. Browser test: a chosen file with Vocals and Bass ticked and FLAC selected becomes a job. "just the vocals and the drums please, as flac" ticks Vocals and Drums and sets FLAC through `POST /requests/parse`; "make it sound nicer" gives the could-not-tell notice with nothing ticked, and the upload button stays disabled until a stem is ticked by hand. A `.pdf` is refused in the page with the allowed types listed, and a file over the size limit with both sizes named (the limit is taken from `/config`; the test lowers it in the API's reply, since a browser test cannot hand over 250 MB); in both cases no upload request is sent. A PDF renamed to .mp3 is refused by the API and its own sentence is shown. Drag and drop is written but only the file picker is tested.
- [x] P6.3 Job page: named stages with per-stem progress, live via server-sent events. **Accept:** the job page names every stage as done, current or to come, with each stem's state, and follows the events stream to Ready without a reload (browser test); a failed job shows its error. — Done: `JobView` with `lib/useJob.ts`: nine named stages, each done, current or to come, said in words for a screen reader, with each stem's state while the job runs. The page follows the events stream and falls back to asking every two seconds if the stream breaks. In the browser test the page goes from upload to Ready with all nine stages done and no reload. A failed job shows the error and has no current stage; that test replaces the API's answer for one job, because the stand-in models do not fail.
- [x] P6.4 Stem player: waveform, solo and mute, all stems sample-synchronised. **Accept:** each stem's waveform is drawn from its peaks; play, pause and seek act on every stem from one audio clock (every source is started at the same context time and offset: unit-tested, and checked in the browser test); solo and mute change gains only and never restart a source. — Done: `lib/player.ts` and `StemPlayer`: every stem is decoded and started with one `when` and one `offset` on one AudioContext clock, so nothing can drift; pause and seek stop and restart all of them together; solo and mute move gains only. 9 unit tests of the engine against a stand-in audio context. In Chrome: after Play the two sources had the same start time and offset; solo and mute changed the states and started no new source; a seek to 2 s started both again at offset 2 with one start time. Waveforms are drawn from the peaks files. Not judged: how it sounds. The test tone and band-split stems prove the wiring, not that real stems line up audibly, though they are scheduled the same way.
- [x] P6.5 Quality warnings where the gate was unsure. **Accept:** a stem delivered with a warning shows that warning beside it and the page says how many stems carry one; a stem without one shows none (browser test, where the stand-in models give the not-checked warning). — Done: A stem's warning is shown in its row, and above the player the page says how many stems carry one and what a note means. In the browser test both stems show "could not be quality-checked automatically" and the page says every stem has a note, which is what the stand-in models give. A stem with no warning renders no note (the row only adds it when the API sends one); with stand-in models no such stem exists to show in a browser.
- [x] P6.6 Download per stem and as a zip, in the chosen format. **Accept:** each stem and the zip download through freshly signed links, named after the upload and in the format chosen; the browser test downloads a stem and the zip and checks they are audio and a zip. — Done: Each stem and the zip have a download button that asks for fresh signed links on every click. In the browser test "Download Vocals" saved `Test Tone (vocals).flac`, which starts with the FLAC signature, and the zip saved as `Test Tone (stems).zip`, a zip that names `vocals.flac` and `bass.flac`.
- [x] P6.7 Deletion countdown, delete now, delete everything. **Accept:** the job page shows the time left before automatic deletion, from `expires_at`; Delete now removes the job after a second, confirming click and the page says so; Delete everything on the jobs page empties it (browser test). — Done: The job page shows the time left from `expires_at`, refreshed every 30 s ("6 days 23 hours" in the test). Delete now asks first: after one click the job is still Ready, Cancel backs out, and only "Yes, delete" removes it, after which the page says Deleted and the player is gone. On the jobs page, Delete everything asks the same way and leaves the list empty, still empty after a reload.
- [x] P6.8 Terms and Privacy placeholder pages that say they are placeholders. **Accept:** `/terms` and `/privacy` exist, are linked from every page, and each starts by saying it is a placeholder and not legal text. — Done: `/terms` and `/privacy`. Each opens with a notice that the page is a placeholder and not legal text, then says in plain words how the software is built to behave. The browser test checks that notice is the first thing after the heading and that both pages are linked from the footer of the upload, jobs, job, terms and privacy pages.
- [x] P6.9 Browser test of the whole flow against the API with the fake separator; keyboard and screen-reader pass on the player. **Accept:** `make web-e2e` runs the browser tests against the real API with `STEMS_WORKER_FAKE_MODELS=1` and they pass; one test works the player with the keyboard alone; axe reports no violations on the upload and job pages; every player control is found by role with a name and state. A real screen reader is not run, and the note says so. — Done: `make web-e2e` (Playwright 1.65.1, the installed Chrome, `@axe-core/playwright` 4.14.0): 11 tests, all passing, about 30 s. They run the real API with its maintenance thread and local worker provider and `STEMS_WORKER_FAKE_MODELS=1`, and the built web app, on their own ports, database and storage with no `.env`. One test works the player with Tab, Space, Enter and the arrow keys only (play, seek, solo, mute, pause). axe reports no violations on the upload page, the job page with the player, the jobs page, and the job page with the delete question open. Every player control is found by role and name with its pressed or value state; the waveform is hidden from assistive technology and the position slider carries the time in words. **A real screen reader was not run**, and neither was any browser but Chrome, nor a phone. A CI job for these tests is written and has never run.

(Acceptance checks for P6.2 to P6.9 were written on 2026-11-24, when Phase 6 started.)

## Phase 7: calibration and hardening

- [x] P7.1 Calibrate gate thresholds on labelled clean and deliberately bled stems. **Accept:** `docs/gate-calibration.md` shows the measured trade-off and the chosen values, from a run. *Unblocked: the datasets (B3) and the key (B6) are both here. Scoring run `gate-v1` started 2026-11-24T10:18Z.* — Done: run `gate-v1` (`python -m stems_bench.calibrate collect`, finished 2026-11-24T10:45Z): 1,270 60-second excerpts of MoisesDB scored by the BEATs tagger on CPU, 30 tuning tracks to choose thresholds and 50 other tracks to check them; stems taken clean and with another stem of the same track mixed in at 30, 20, 12 and 6 dB below. The rules for choosing, and the bar for switching the gate on, were committed (`7d58a494a794`) before any clip was scored. `docs/gate-calibration.md` is the report. **Presence on a stem works:** with the chosen thresholds (present from 0.6466, absent up to 0.0207) the answer is right 97.1% of the time on the held-out tracks and is given 57.5% of the time; the old guesses (0.5, 0.2) answered 83.2% of the time and were right 85.2%. **Bleed does not:** AUC 0.644, balanced accuracy 61.4%, and half of the clean stems flagged; no threshold in the trade-off table does much better than chance. **Presence on a full mix does not either:** right 42.9% of the time when it answers (29.0% with the old guesses), because a playing instrument scores far lower in a mix than alone. 3 of the 5 checks failed, so `listening.calibrated` stays false and the gate still does not act on what the tagger hears. Applied to `config/defaults.yaml`: the two presence thresholds. Left as first guesses: the three bleed thresholds, since nothing better was found. Not done: the planned sample of Jev gate calls on these clips. Jev would be judging the same bleed scores that failed, so the calls were not made; how often Jev decides, and how well, is still unmeasured. 15 tests in `tests/test_calibrate.py`.
- [x] P7.2 End-to-end tests across API, worker and web. **Accept:** the browser tests include a job with the vocal split and de-reverb extras; one slow test runs a real checkpoint through the worker's storage entry point on the fixture when its weights are present; `make check`, `make web-check` and `make web-e2e` pass. — Done: a 13th browser test uploads a tone asking for vocals with the lead and backing split and the de-reverbed vocal, and checks the four stems come back in order, that only the gated stem carries a note, that all four start on one clock, and that an extra downloads under its own name. `tests/test_worker_jobs_real.py` (slow) puts an upload and a spec into storage and runs `stems_worker.jobs.run_job` with the real registry, defaults and `mel_roformer_kim_vocals` on CPU: the job ends ready with one FLAC stem of the right length that is not silence, peaks, a zip and a valid status, and a second run reuses the result. Run here on 2026-11-24: that slow test passed (the weights are on disk; it skips when they are not), `make web-e2e` 13 passed, `make web-check` passed. Still stand-ins everywhere else: no job with real models has gone through the API or the browser.
- [x] P7.3 Cost per song from measured timings and prices checked that day. **Accept:** every figure cites a `runs/` file and a dated price source. — Done: `docs/cost-per-song.md`, written by `python -m stems_bench.cost` from `config/prices.yaml` (each price with its page, read 2026-11-24) and from files of runs: the token counts of the live Jev check (`tests/fixtures/jev/*.recorded.jsonl`) and the per-model times of benchmark run `moisesdb-v1` (`docs/benchmarks/moisesdb.json`). For a 4-minute song with six stems: decisions cost $0.000285 (6,780 input tokens at $0.042 per million; output tokens are free), or $0.000660 if every stem is retried twice; a week of storage on Cloudflare R2 is at most $0.0029, sized by arithmetic as WAV, and downloads there are free. Separation takes 230 s in single-best mode on this Mac's GPU, 420 s as two-model blends and 530 s as full ensembles, with the provisional defaults. **The cost of that on a rented GPU is not given**: Modal's price is $0.000164 per second for a T4, but no job has run on one, so the seconds are unknown (B4). The report has no total for that reason. Amazon S3's prices were looked for and not found on its page. Token counts are from one call of each kind. 7 tests.
- [x] P7.4 Deployment guide (local, Modal, S3 or Azure Blob, the Azure Container Apps seam). **Accept:** `docs/deployment.md` gives the steps for each target, and marks every step as run here (with its date) or not run. — Done: `docs/deployment.md`: the four parts (API, worker, storage, web app) and how to run them on one machine, behind HTTPS, with a bucket, with workers on Modal, and what the Azure Container Apps seam needs. Every step says whether it was run here and when. Run here: install, checks, the API, the web app in development and built, and a whole job in the browser with stand-in models (2026-11-24). Not run: a job with real models through the API; anything on the internet (HTTPS, a proxy, Postgres); any bucket (B7); Modal (B4). The guide lists what the code assumes in each case (one API process, web app and API on the same site, a CORS rule on the bucket for the player) and what to check on a first real run. It is a guide written from the code, not from having deployed it.
- [x] P7.5 Listening-test kit: blind randomised 30-second excerpts and a scoring sheet per report §9. **Accept:** `python -m stems_bench.listening_kit` builds a kit from several systems' stems (the same 30-second excerpt per track for every system, level-matched, anonymous names, the key kept apart, a scoring sheet per listener) and scores returned sheets against the report's pass bar; tested on synthetic audio. Running the test itself needs B8. — Done: `python -m stems_bench.listening_kit`. `build` reads a manifest of tracks and each system's stems and writes the folder listeners get: per track and stem the same 30-second excerpt (the loudest of the mix) from every system, matched in level to within 0.2 dB and kept under a -1 dBFS peak, as A, B, C, D in an order that changes from stem to stem; a sheet per listener with the rows in an order of their own; and instructions. The key (which letter is which system, and the gains applied) is written outside that folder, and the tool refuses to put it inside. `score` unblinds returned sheets and applies the report's bar: ahead of the free reference by 0.3 points on at least 70% of tracks, ahead of every paid reference on mean overall quality, behind on no stem by more than 0.5, with at least 5 listeners and every reference present. Bad cells and missing rows are listed and left out. 14 tests on synthetic audio with four stand-in systems. **No listening test has been run**: it needs the other tools' stems, two paid plans and a panel (B8).
- [x] P7.6 Hardening pass: limits, failure paths, a security review of upload and download handling. **Accept:** `docs/security-review.md` lists what was reviewed, each finding with its fix and the test that holds it, and what was left as it is. — Done: `docs/security-review.md`: a review, by the author of the code, of what a stranger's upload can do to the server and what anyone but the uploader can reach. 12 findings fixed, each named with the test that holds it. The ones that mattered most: separated audio was kept in a shared cache the 7-day deletion never touched (the worker now caches inside the job's own temporary folder); a stopped worker left a copy of the upload in the temp directory; worker logs were never deleted; the media decoders ran inside the API process (the upload check now runs in a child process with a 20-second limit); any sample rate was accepted (now 8 kHz to 192 kHz, at most 8 channels); a failed job showed the exception's own message; nothing capped the queue overall (now 12 jobs). Also: owner-only permissions on a storage folder the code creates, no-store and nosniff headers on every API reply, and frame and permissions policies on the web pages. The review also lists what was found sound and what was left as it is and why (links work without the cookie by design, in-process rate limits, no cap on event streams, no dependency scan). 14 tests in `tests/test_api_hardening.py`, 3 more for the worker, and a browser test of the page headers. Not an independent audit and not a penetration test.
- [x] P7.7 Optional: fine-tune the weakest stem on MoisesDB if the benchmark shows a clear gap. **Accept:** done, or a note saying it was not and why. — Not done, and this is the note. The benchmark does show a gap: on the MoisesDB subset (run `moisesdb-v1`, `docs/benchmarks/moisesdb.md`) the best model scores 2.97 dB on the six-stem other, 5.80 dB on piano and 9.19 dB on guitar, against 10.82 dB on vocals and 12.95 dB on drums. Fine-tuning was not attempted, for three reasons. It is training of a large model: this machine has no NVIDIA GPU, its own GPU has been taken by the benchmark queue since 2026-11-23, and there is no rented one (B4). It needs a protocol of its own: tracks to train on that are outside both the frozen test subset and the tuning list, and it is not known which MoisesDB tracks the published checkpoints were already trained on. And the recommendation of a default model per stem (P2.13), which says what there is to improve on, is not finished. No training code was written.

## Phase 8: the gate on real output, and real jobs end to end

Added on 2026-11-24 under rule 6 (no phase was left). Phase 7 measured the listening layer
on reference stems with bleed mixed in by hand, and ran every job behind the API with
stand-in models. This phase measures it on what real separators produce, switches on
whatever passes, and puts real models through the API and the browser.

- [x] P8.1 A tool that measures the listening layer on real separator output, with its rules fixed first. **Accept:** `python -m stems_bench.gate_outputs collect` scores the outputs the benchmark kept (resumable, with a `status.json`), each with the SDR and SIR the benchmark measured for it; the questions, the rules tried, how thresholds are chosen and the bar each must clear are in the committed code before any output is scored; tested on made-up data with a stand-in tagger. — Done: `python -m stems_bench.gate_outputs` (`collect`, `report`). `collect` walks the outputs the benchmark kept under `runs/bench/estimates/`, keeps those that are one of the six product stems scored against their own reference, from models with an output for every track of the list, cuts each to the loudest minute of its track and writes a row: the tagger scores through the product code path, the level of the output and of the mix, the median of the benchmark SDR and SIR windows inside that minute, and whether the reference is present there. Resumable, with a `status.json`; the models that count are frozen in a plan file on the first run. Fixed in the code before anything is scored: two questions (can an empty stem be told from a real one; does the bleed score prefer the better of two outputs whose SDR differs by 3 dB or more), three rules for the first in order of preference (level under the mix, the tagger at the configured `absent_max`, the tagger at a threshold chosen on these outputs), thresholds chosen on outputs for the tuning tracks only, and the bars on the held-out outputs (empty: right 90%, found 50%; pairs: 80% of at least 30). A dry run of the bookkeeping, with no scoring, found 360 outputs for the tuning tracks (26 empty) and 700 for the held-out ones (111 empty). 20 tests on made-up rows and a made-up benchmark folder with a stand-in tagger. **Nothing has been scored yet**: that is P8.2.
- [x] P8.2 Run it and report. **Accept:** `docs/gate-on-real-output.md` generated from a finished run, with a verdict for each question (can an empty stem be told from a real one; does the bleed score prefer the better of two outputs). — Done: run `gate-outputs-v1` (2026-11-24, 11:14Z to 11:38Z, tagger on CPU): 1,060 kept outputs scored, 360 for the 30 tuning tracks (8 models, 26 empty) and 700 for the 50 held-out tracks (9 models, 111 empty). `docs/gate-on-real-output.md` is the report; the rules it applies are those of commit `5b63f0807f6e`. **An empty stem can be told from a real one, by level alone:** "the output is 19.8 dB or more below the mix in the loudest minute" (threshold chosen on the tuning tracks) called 103 held-out outputs empty and was right for 98 (95.1%), finding 88.3% of the 111 empty ones; AUC 0.997. The five it got wrong are piano outputs with a measured SDR of about 0 dB: the piano is in the song and the model did not find it, so the output is leftovers all the same. The gate's existing rule (98% silent) finds 69.4% of the empty ones and is never wrong. The tagger at the configured `absent_max` failed (right 87.9%, 90% needed); with a threshold chosen on these outputs it passed, behind the level rule in the order fixed beforehand. **The bleed score cannot pick the better of two outputs:** it preferred the one with the higher SDR in 64.8% of 91 pairs that differ by 3 dB or more (80% needed); its rank correlation with measured SIR is between -0.44 and 0.09 by stem, and with the guessed thresholds it calls 95 of 144 good bass outputs bled. So nothing the tagger says is a reason to retry. Note on the report's header: it names the commit the checkout was at when scoring ended (`4ac33f7`); the run started at `5b63f08`, and the scoring code (`gate_outputs.py`, `hear`, the tagger, the level helpers) is identical in both. Limits are in the report: one minute per track, only the models the benchmark kept, one dataset.
- [x] P8.3 Act on P7.1 and P8.2, one signal at a time. **Accept:** `listening.calibrated` becomes one switch per signal, each true only if its held-out check passed; the gate and the web app use exactly the signals that are on (a stem found empty gets a note that says the instrument was not found, not the generic one); with every switch off nothing changes; tests for both states; `make web-e2e` passes. — Done: one signal switched on, two left off, each by its held-out result. **On: an empty stem by level.** `gate.rules.empty_below_mix_db: -19.81` (the threshold P8.2 chose and checked): a stem that far below the mix over the mix's loudest minute is treated like a silent one, whatever the tagger's state, because it is a level measurement. It is tried again with the fallback model and then a blend, and if every attempt is as quiet it is delivered with a note of its own: "This stem is silent or very quiet compared with the song. The instrument may not be in the song, or it was not found." (the wording covers the five held-out cases where the instrument was there and the model missed it). The stem is never withheld; `null` turns the rule off (BLOCKED.md D12). The worker measures it with the same window rule the measurement used (`stems_common.levels.loudest_window`, tested equal to the benchmark's), and Jev is told the figure. **Off: the tagger's two signals.** `listening.calibrated` is now `{presence, bleed}`, and a verdict is trusted only if every signal it rests on is on (not-heard rests on presence, bleed on bleed, a clean pass on both). Presence passed on reference stems in P7.1 but on real outputs its "absent" was right 87.9% of the time (90% needed), so it stays off; bleed failed twice. With both off and the level rule set to `null`, behaviour is what it was. Tests: 4 for the level signal, 4 for the gate rule and switches, a job whose drums come back 20 dB down from both models (three attempts, then the note) and one where the fallback finds them, and a 14th browser test in which a stand-in "piano" stem 33 dB under the song shows the note and still downloads. `make web-e2e` 14 passed, `make web-check` passed. Two older tests had to be given other audio: with the stand-in models, which split by frequency, the band they asked for was 21 dB under the mix and the new rule retried it, as it should. Not measured: how often the rule fires on whole songs (it was measured on one minute per track), or what the retries cost in time. *Found afterwards, while closing P8.8:* this task changed what the gate sends to Jev (the level, and one switch per signal), so the recorded live gate request no longer matched and its replay test had been skipping itself since this commit. Re-recorded with one live request to `jev-1.14.0` (767 input and 96 output tokens, HTTP 200; its answers were at 0.54 confidence, so the rules decided, as before), and `docs/cost-per-song.md` was regenerated from it (decisions for six stems $0.000291, was $0.000285).
- [x] P8.4 Presence in a full mix: thresholds of its own from the scores of run `gate-v1`, or a stated dead end. **Accept:** `docs/gate-calibration.md` gains a section with thresholds chosen on the calibration mixes and checked on the held-out ones against a bar fixed first; `track_features` uses them if they pass, and otherwise reports "unknown" and no instruments instead of guesses. — Done: a dead end, stated. `docs/gate-calibration.md` has the new section, from the 30 calibration and 50 held-out mixes of run `gate-v1` (nothing was scored again), with its rules and bar committed first (`7f6c21e1a87d`; written after the stem results were known, which the section says). One pair of thresholds per instrument, by the same two choosing rules as for stems; an instrument is judged only with at least 10 mixes that have it and 10 that lack it in each set. **Vocals, drums, bass and guitar cannot be judged**: nearly every mix has them (0 to 3 of 30 calibration mixes lack each). **Piano and other were judged and no thresholds could be chosen**: on the calibration mixes no cut-off gave 10 answers that were 95% right. The piano score does carry something (held-out AUC 0.883; other 0.700), but its values in a mix are tiny (most under 0.01) and 30 mixes did not make a rule of it. So `listening.mix_presence` is empty and `track_features` now names no instrument in a mix and reports the vocal presence as "unknown", where before it gave guesses that P7.1 measured as right 29% to 43% of the time. Genre and reverb hints are unchanged and still unmeasured. More mixes without each instrument would be needed to do better; MoisesDB as sampled here does not have them. 4 more tests in `tests/test_calibrate.py`, 2 in `tests/test_track_features.py`.
- [x] P8.5 A real job through the API and the browser: real models, no stand-ins. **Accept:** one command runs it (the API, the local provider, the real registry, the synthetic fixture) and checks that all six stems arrive and play; run here once, with each model's time and its device stated; `docs/deployment.md` no longer says "not run" for it. — Done: `make web-e2e-real` (`web/playwright.real.config.ts`, `web/e2e-real/real.spec.ts`): the API with the real registry and no stand-ins, the local provider, the built web app and Chrome. It uploads a 12-second synthetic signal, asks for all six stems, waits for Ready, checks six stems with waveforms, that they start on one clock and play, that one downloads as audio under the song's name, and that no stand-in model ran; it writes what ran to `web/.e2e-real/real-job.json`. Run here on 2026-11-24 with `STEMS_DEVICE=cpu` (the benchmark had the GPU): passed, 103.3 s from upload to ready. Six models ran, each on CPU, for 12 s of audio: `mel_roformer_kim_vocals` 12.7 s, `scnet_xl_ihf` 49.2 s (drums and other), `htdemucs_ft` 3.9 s, `bs_roformer_sw_6stem` 17.5 s (guitar and piano), and as retries `mel_roformer_guitar_becruily` 5.3 s and `htdemucs_6s` 3.8 s. Guitar and piano came back more than 19.8 dB under the mix, so the new rule of P8.3 retried each twice (the blends reused cached outputs) and delivered them with its note; on tones no model was trained on, that says the rule runs, not that it is right. No key was used. `docs/deployment.md` has the run and its table in place of "not run". These times are single runs on a busy machine, not speed measurements. A first run of this command failed on my own mistake (the test looked for the result file in the wrong folder) after the job itself had finished; fixed and run again. Still not done: a real song, a long one, or the GPU through the API.
- [x] P8.6 Dependency audit. **Accept:** `pip-audit` and `npm audit` are run and their dated results are in `docs/security-review.md`, with what was changed and what was left and why. — Done: both run on 2026-11-24 and written into `docs/security-review.md` with their commands and times. `pip-audit` 2.10.1 over the packages installed in `.venv` (run from a separate environment, so nothing in `.venv` changed): 136 packages checked, no known vulnerability; the project itself skipped as not on PyPI. `npm audit` in `web/`: 402 packages, none with an advisory at any severity. Nothing needed changing. Left and said so: it is one day of known advisories, not a reading of anyone's code, and nothing reruns it; model files are outside both tools (their hashes are pinned at first download and every `torch.load` uses `weights_only=True`, which I checked at each call).
- [x] P8.7 The web app at phone width. **Accept:** upload to download, and the axe checks, pass at 375 px wide with no sideways scrolling, inside `make web-e2e`. — Done: `web/e2e/mobile.spec.ts`, two tests as Chrome emulates a phone held upright (375 by 812, touch). The first goes from upload to download by touch and checks on the upload, job, jobs, terms and privacy pages that nothing lies beyond the right edge, that every stem row is inside the screen, and that no button, menu or link-button showing is smaller than 24 by 24 pixels (with a count, so the check cannot pass on an empty page). The second runs axe on the upload, job and jobs pages. Both passed without any change to the app: the narrow-screen rule written in Phase 6 holds. `make web-e2e`: 16 passed. It is desktop Chrome at a phone's size, not a phone; Safari on iOS is untested.
- [x] P8.8 Rate limits behind a reverse proxy (BLOCKED.md D7). **Accept:** a setting names the proxies to trust; the forwarded address is used only when the connection comes from one of them; with the setting empty nothing changes; tests for a trusted proxy, an untrusted one and a forged header; `docs/deployment.md` says how to set it. — Done: `python -m stems_api --trusted-proxies 10.0.0.5,10.1.0.0/16` (or `STEMS_TRUSTED_PROXIES`). From those addresses, and no others, uvicorn takes the visitor's address from `X-Forwarded-For` and HTTPS from `X-Forwarded-Proto`; with nothing named the headers are ignored. That closes a small hole too: uvicorn's default, which the API had been running with, believes those headers from this machine. `*` and host names are refused. 7 tests in `tests/test_api_proxy.py`, through uvicorn's own middleware built from the same options: no proxy named (headers change nothing, even from 127.0.0.1); a trusted proxy by address, by network and in a list (each visitor has their own allowance); a visitor writing the header themselves, directly or through the proxy (the entry the proxy appended is the one used); and the Secure cookie only when a trusted proxy says HTTPS. `docs/deployment.md`, `.env.example` and the security review say how to set it. Not run behind a real proxy. Limits are still counted in one process (D7).

## Phase 9: the product against the benchmark

Added on 2026-11-24 under rule 6 (no phase was left). Phase 8 switched on one gate rule,
measured on one minute per track, and put real models through the API on a signal that
is not music. This phase checks that rule against whole songs, runs real songs through
the product's own path and sets the result beside the benchmark's, and says what the
tagger costs now that the gate acts on nothing it hears.

- [x] P9.1 The empty-stem rule against whole songs. **Accept:** with the rule for comparing fixed first, `docs/gate-on-real-output.md` gains a section from the kept outputs: for each output, whether the instrument is in the song at all (not only in the loudest minute), and how "far below the mix in the loudest minute" and "far below the mix over the whole song" each do against that; the config keeps or changes the rule by the result. — Done: the rule was replaced, by a result whose consequences were fixed first (commit `fcd175726d06`). `python -m stems_bench.gate_outputs levels` read the same 1,060 kept outputs again (no tagger; 12:34Z to 12:38Z on 2026-11-24) for each one's level over the whole song and how long its reference sounds anywhere, and `docs/gate-on-real-output.md` has the new section. Truth is now "the instrument sounds for at least 5 seconds somewhere in the song". Of 700 held-out outputs, 111 have a reference absent in the loudest minute but only 90 have an instrument that is nowhere in the song: 21 play only elsewhere, and the rule of P8.3 called 13 of those empty. Against the whole song that rule was right for 85 of the 103 outputs it called empty (82.5%; 90% needed): **failed**. The same level taken over the whole song, with its threshold chosen on the tuning tracks (-21.68 dB), called 92 empty and was right for 83 (90.2%), finding 92.2%: **passed**, narrowly. As fixed beforehand, it takes the other's place: `gate.rules.empty_below_mix_db` is -21.68, measured over the whole song, and the loudest-minute window is gone from the worker. So the rule P8.3 switched on was wrong about one flagged stem in six, and stood for about an hour of commits; no job outside the tests ran with it except the synthetic one of P8.5. BLOCKED.md D12 states the new figures. 6 more tests for the comparison, 3 for the level; `make web-e2e` 16 passed. This commit also carries what P9.3 will report on: each job now times the tagger (lines with stage `listening` in its timings) and `listening.enabled` can switch it off (2 tests). Still one dataset and only the models the benchmark kept; 90.2% is 83 of 92, so one or two outputs either way would move it across the bar.
- [x] P9.2 Real songs through the product, beside the benchmark. **Accept:** a resumable command runs tracks of the frozen MoisesDB subset through the worker's storage entry point with the real models (no stand-ins, no audio committed), scores the delivered stems against the references with the benchmark's own scorer, and writes `docs/product-vs-benchmark.md` from the run: per stem the product's SDR beside the benchmark's for the same model and track, which stems the gate flagged and whether that was right, and the wall time, labelled by device. — Done: run `product-v1` (`python -m stems_bench.product_check run --tracks 3 --device cpu`, 2026-11-24, 12:45Z to 14:07Z): the first three tracks of the frozen MoisesDB subset through `stems_worker.jobs.run_job` with the real models, all six stems, no Jev, the tagger on; delivered FLAC stems read back from storage and scored with the benchmark's scorer; the audio deleted after each track (the storage folder holds no object afterwards). `docs/product-vs-benchmark.md` is the report. **The product delivers what the benchmark measured:** 12 stems could be set beside the benchmark's figure for the same model and track, and all 12 are within 0.07 dB of it (CPU here, this Mac's GPU there). The six-stem `other` has no figure to compare with: the product makes it from a four-stem model, which the benchmark scores against a different target; it scored 3.40, 2.13 and 1.46 dB. **The gate:** of 3 stems whose instrument is not in the song it flagged 2 (two pianos, three attempts each); of 15 that are in the song, none. **The third is a fault in the gate, older than this phase:** on a song with no guitar the first model returned silence, the retry with `mel_roformer_guitar_becruily` returned something audible, and that was delivered with the ordinary note. A retry after "empty" rewards whichever model invents the most. Whether to retry such a stem at all is P9.6. **Time, CPU, busy machine:** 1,820 s, 1,298 s and 1,600 s for songs of 344 s, 249 s and 240 s, of which the models took all but 37 to 59 s. Three tracks show that the road works; they do not say how often. 7 tests.
- [x] P9.3 What the tagger costs a job. **Accept:** the worker's timing log records the seconds spent listening; `docs/cost-per-song.md` states them from a run; a setting turns the tagger off; the default is chosen by what was measured and listed in BLOCKED.md as a decision the owner can overrule. — Done: each job now records what the tagger cost (lines with stage `listening` in its timings: the mix, then every attempt at every stem), and `docs/cost-per-song.md` has a section on it from the three real jobs of run `product-v1`: 44 s, 30 s and 36 s for songs of 344 s, 249 s and 240 s (8 or 9 listens each, on CPU), a median of 7.7 s per minute of audio. For a 4-minute song that is 31 s, 13% on top of the 230 s the default models take on this Mac's GPU; the machine was busy, so it may be on the high side. What it buys today is nothing a user sees: the gate acts on neither of the tagger's signals, so a stem carries the same note with the tagger or without. **So `listening.enabled` is now false** (BLOCKED.md D13, for the owner to overrule): the worker and the `decision.job` command load no tagger unless it is switched on. With it on, its scores still go into each stem's notes and to Jev. 3 tests (the timing lines, the switch, the report section).
- [x] P9.4 What an absent instrument costs. **Accept:** `docs/cost-per-song.md` states, from measured model times, the extra seconds the empty-stem retries add for a song without guitar, without piano and without both. — Done: `docs/cost-per-song.md` has a section "When the song lacks an instrument", written by `python -m stems_bench.cost`. It plays the gate's own retry machine (`retry.StemGate`, the rules' fallback model, no Jev) on a stem that is found empty every time, and prices the models that adds with the per-model times of benchmark run `moisesdb-v1` (this Mac's GPU). For a 4-minute song in the default mode: no guitar adds `mel_roformer_guitar_becruily`, 26 s (11% of the job's 230 s); no piano adds `htdemucs_6s`, 11 s (5%); neither adds both, 36 s (16%). The third attempt, a blend of the first two, reuses their outputs and runs nothing. The chains are the ones the real-models job of P8.5 made. This is arithmetic on measured model times, not a measured job, and leaves out the tagger listening to each attempt. 3 tests.
- [x] P9.6 Should a stem found empty be retried? **Accept:** from the kept benchmark outputs and with the rule for deciding fixed first, `docs/gate-on-real-output.md` states how often a retry with the fallback model rescues an instrument that is in the song, and how often it replaces a rightly empty stem with something that is not; the gate retries or stops by the result. *Added 2026-11-24 from what P9.2 found.* — Done: no: the gate no longer retries a stem it finds empty. The counting rules and the consequence were committed first (`a537158b22f8`); `docs/gate-on-real-output.md` has the section. On the 50 held-out tracks, for the five stems whose default and fallback models both have kept outputs (250 track-and-stem pairs), the default model's output was empty by the rule in force 42 times. The fallback's was empty too in 38. In the other 4, all guitar, the song has no guitar and `mel_roformer_guitar_becruily` returned something 15 to 21 dB under the mix: leftovers that the gate would have delivered as the guitar, as it did in the real job of P9.2. **Rescues: 0. Wrong deliveries: 4.** So `gate.rules.retry_empty` is false: a silent or empty stem is delivered at once with its note and no other model runs for it (`true` brings the retry back; BLOCKED.md D12). This also removes the cost P9.4 priced; the cost report now says "nothing extra" and keeps the old figures as history. The silent-stem retry had been in the gate since Phase 4, where it was designed against a model that fails outright, which is not what real models did here. Tests: 4 for the counting, and the gate, job and cost tests now cover both settings of the switch (a job as shipped makes one attempt and never runs the fallback). `make web-e2e` 16 passed. Limits: 42 cases, one dataset, the fallback models of today's config; the six-stem `other` has no kept outputs to count; a stem that is 98% silent but loud in the rest was not counted separately.
- [x] P9.5 A page for the owner. **Accept:** `README.md` says in one screen what works, how to run it, what has been measured (with the report for each figure) and what waits on the owner; every command in it is run on the day it is written. — Done: `README.md` rewritten (it still said there was no web app and no quality number): what works, how to run it, a table of what has been measured with the report behind each figure, what is not known, and what waits on the owner. Every command on it was run on 2026-11-24: `make setup`, `make node-setup` (already installed, nothing fetched), `make web-setup`, `make api` and the new `make web-dev` (both answered and were stopped), `separate` on the synthetic fixture, `make check` (1,326 passed), `make web-check`, `make web-e2e` (16 passed), `make web-e2e-real` (passed again at 14:34Z with the new defaults: four models, no retries, no tagger, 126.4 s on a busy CPU) and `bench status`. Each figure in its table was read from the committed report it names.

## Phase 10: the default models, chosen on the tuning tracks

Added on 2026-11-24 under rule 6 (no phase was left). The models the product uses by
default are still the provisional ones picked before any benchmark existed, and the two
test-list reports show that at least drums and bass are served by a weaker model than
the best one measured. P2.13 was to settle this after blend tuning; the tuning run has
now scored every single model on the 30 MoisesDB tuning tracks, which is what a choice
of single models needs. This phase makes that choice on the tuning tracks only, by a
rule fixed first, checks it on the test lists, and puts real songs through the result.
Blends, and with them the rest of P2.13, still wait for the tuner.

- [x] P10.1 The single models on the tuning tracks. **Accept:** `docs/benchmarks/tuning-models-moisesdb.md` generated by `bench report` from run `moisesdb-tuning-v1`: every model's score per stem on the tuning list, with its time. — Done: `docs/benchmarks/tuning-models-moisesdb.md` and its `.json`, written by `bench report moisesdb-tuning-v1` from the finished tuning run (30 tuning tracks, 780 jobs computed, 0 failed, 120 not applicable, on this Mac's GPU): every single model and every untuned blend the config names, per stem, with its time. Generated and committed without its tables being read: the rule that chooses from them (P10.2, `stems_bench.recommend`) is in the same commit, so that the rule is on record before the scores are used. The tuner itself (`docs/benchmarks/tuning-moisesdb.md`, P2.12) was still searching when this was written.
- [x] P10.2 A default single model per stem, chosen by a rule fixed first. **Accept:** the rule (what is compared, on which tracks, how ties and time are weighed) is committed before the choice is computed; `docs/benchmarks/recommendation.md` states the choice per stem from the tuning list and what the two test lists then show for it, beside the model in use; `config/defaults.yaml` names the chosen models for `single_best`; `make check` and `make web-e2e` pass. — Done: `python -m stems_bench.recommend`, with its rule committed (`2e0030a`) before the tuning scores were read: on the 30 tuning tracks, a stem's best model by median SDR, any model within 0.3 dB of it acceptable, and the fewest models win. **The choice is one model for all six stems, `bs_roformer_sw_6stem`**: it is the best on the tuning tracks for every stem outright (vocals 11.48, drums 12.14, bass 15.64, guitar 7.85, piano 8.63, other 2.61 dB), so the margin never came into play. `docs/benchmarks/recommendation.md` has the table of every candidate. On the test lists, which decided nothing: drums +1.44 and +1.43 dB over the model in use (MoisesDB, MUSDB18-HQ), bass +1.90 and +2.52, other +0.98 over the composed stem the product had been delivering, guitar and piano unchanged (same model), and vocals -0.27 on MoisesDB and +0.34 on MUSDB18-HQ. One model instead of four: 18.3 s per minute of audio instead of 64.3 on this Mac's GPU. `config/defaults.yaml` now names it as `single_best` for the six stems. **Two things followed that I had not planned.** The config's blends (equal weights, never tuned) are worse than this single model on the tuning tracks for drums, bass, guitar and piano by 0.9 to 2.9 dB and gain 0.03 dB at most on vocals, so "best quality" would have been slower and worse: all three quality settings now give the single model until the tuner has chosen blends (`decision.mode_for_quality`, BLOCKED.md D14). And the consistency check that a blend must contain the single model is asked only once `ensembles.provisional` is false. **A fault found on the way:** one handwritten Jev reply in the plan fixtures had probabilities summing to 0.9825, so the parser rejected it and the test that Jev can ask for a blend had been passing through the config's old mapping, not through Jev's answer; the generator is fixed, one number in the fixture changed, and the test now asserts the reply was used. For the benchmark still running: its later stages start with this config, so their "single best" rows are this model; the MoisesDB tuning report in progress was started with the old one. Not known, and it matters for this choice: which songs each checkpoint was trained on. `make check` passed, `make web-e2e` 16 passed. 9 tests for the rule and report.
- [x] P10.3 Real songs through the product with the chosen defaults. **Accept:** `python -m stems_bench.product_check` run again under a new run id on the same three tracks, and `docs/product-vs-benchmark.md` regenerated from it; the six-stem `other`, which the first run could not compare, is set beside a benchmark figure or the report says why not. — Done: run `product-v2` (2026-11-24, 15:19Z to 15:41Z, CPU): the same three MoisesDB tracks through the worker's path with the config as it now is (one model, no tagger, no retries), and `docs/product-vs-benchmark.md` regenerated from it. **15 of 15 stems are within 0.02 dB of the benchmark's figure** for the same model and track; the six-stem `other` is now among them, since the model's own `other` is what is delivered (6.66, 5.80 and 2.63 dB). **The gate flagged all 3 stems whose instrument is not in the song, and none of the 15 that are;** every stem took one attempt. Beside the first run on the same songs (`product-v1`, the old defaults): drums 15.87 to 16.46, 11.52 to 13.18 and 9.54 to 10.01 dB; bass 16.15 to 16.89, 10.28 to 13.98 and 14.35 to 15.65; other 3.40 to 6.66, 2.13 to 5.80 and 1.46 to 2.63; vocals 11.25 to 11.26, 10.82 to 10.69 and 7.35 to 7.26; guitar and piano unchanged. Wall time 411 s, 303 s and 310 s where the first run took 1,820 s, 1,298 s and 1,600 s, both on a busy CPU. Three songs, as before: enough to see the road works with the new model, not to say how often. The audio was deleted after scoring.
- [x] P10.4 The figures that follow from the defaults. **Accept:** `docs/cost-per-song.md` and the table in `README.md` regenerated or corrected from the new config and reports, each figure traced to its file. — Done: `docs/cost-per-song.md` regenerated by `python -m stems_bench.cost` from the new config: separation of a 4-minute song in the default mode is 69 s on this Mac's GPU (one model at 17.14 s per minute of audio in `docs/benchmarks/moisesdb.json`), where the four old defaults took 230 s. Decisions are $0.000291 for six stems from the recordings, of which the plan call was recorded again (2,338 input and 592 output tokens): the plan request names the default models, so the config change had made the old recording stale and its replay test was skipping itself. One live request to `jev-1.14.0`; Jev's answers were under the confidence floor (0.63 to 0.75), so the rules planned, as before. `README.md`: the default model and what it replaced, 69 s, 15 of 15 stems within 0.02 dB, and what is now not known (whether a tuned blend beats the single model; what the checkpoints were trained on). The 13% in the notes on the tagger (config, D13) is gone: against one model's 69 s the tagger's 31 s would be 45%. Each figure was read from the file it names.

## Phase 11: more songs, and an app that says what it does

Added on 2026-11-24 under rule 6 (no phase was left). Three real songs have shown that
the product's path works with the new default model; three is not a rate. And the choice
of one model for every quality setting left the web app offering a "Speed or quality"
menu that changes nothing.

- [x] P11.1 Twenty real songs through the product. **Accept:** run `product-v2` extended to the first 20 tracks of the frozen MoisesDB subset (the three done are kept), and `docs/product-vs-benchmark.md` regenerated from it: how many stems agree with the benchmark, how many absent instruments the gate flagged and how many present ones it flagged wrongly, each wrong one named. — Done: run `product-v2` extended to the first 20 tracks of the frozen MoisesDB subset (15:48Z to 17:38Z on 2026-11-24, CPU; 4,492 s of audio in 5,534 s of wall time), and `docs/product-vs-benchmark.md` regenerated from it. **100 of 100 stems that can be compared are within 0.15 dB of the benchmark's figure** for the same model and track; the median difference is 0.00 dB. **The gate flagged all 20 stems whose instrument is not in the song.** It also flagged 7 of the 100 whose instrument is: six where the model found next to nothing (five `other` and one piano, SDR between -0.09 and 1.20 dB), for which "or it was not found" is the right half of the note, and one piano that is a good stem (5.54 dB) and simply quiet in its song. So of 27 flagged stems the instrument was absent in 20, missed by the model in 6 and wrongly flagged in 1. Every stem took one attempt; no audio is left in the run's storage. **One job of the twenty failed the first time, and the cause was this project's own tests:** the sweeper's tests drive it with clocks set days ahead, it cleans `stems-job-*` folders out of the temporary directory, and pointed at the real one a test run deleted the working folder of the job that was running. `tests/conftest.py` now gives the run and every process it starts a temporary directory of its own (commit `bc53163`, with a test that holds it); the track was run again, alongside a full test run, and finished. A job the owner had been running during any earlier `make check` could have died the same way. Twenty tracks of one dataset with one model: a rate for this dataset, not for music in general.
- [x] P11.2 The quality menu tells the truth. **Accept:** `GET /api/config` says what each quality setting gives; the upload page shows the menu only when the settings differ, and otherwise says in a line that one model is used; a request in words that asks for "best quality" still uploads; unit and browser tests for both cases. — Done: `GET /api/config` now carries `quality_gives` (the ensemble mode each setting gives when the rules plan a job; a quality named by nobody is planned as "balanced") and `decided_by` ("jev" when a key is set, since Jev may give a setting other than "fastest" a blend whatever the table says; otherwise "rules"). The upload page shows the "Speed or quality" menu only if the choice can change something (`qualityMatters` in `web/lib/format.ts`): not when the rules decide and every setting gives the same mode, which is the case as shipped. Then it shows one line in its place: "There is no speed or quality setting for now: one model does the separating, and nothing measured so far does better." An API that does not say, or Jev planning, keeps the menu. Tests: the config route; 1 unit test of the rule (5 cases); a 17th browser test in which the menu is absent and the line is there, a request in words asking for "best quality" still uploads and finishes, and with an API whose settings differ the menu is back with all four options. `make web-check` passed, `make web-e2e` 17 passed.
- [x] P11.3 The real-models job in the browser, with the new default. **Accept:** `make web-e2e-real` passes on the config as it is and `docs/deployment.md` describes that run. — Done: `make web-e2e-real` passed on the config as it is, at 15:58Z on 2026-11-24 with `STEMS_DEVICE=cpu`: one model, `bs_roformer_sw_6stem`, made all six stems of the 12-second synthetic signal in 28.5 s, and the job went from upload to ready in 35.8 s (the first run that day, with four models, retries and the tagger, took 103.3 s). Six stems arrived, played on one clock and one downloaded. Three of them (vocals, guitar, piano) carried the empty-stem note, which on tones and noise says nothing about the rule. `docs/deployment.md` has the run. Still not done: a real song in the browser, or the GPU through the API.
- [x] P11.4 The vocal-split model, by the same rule. **Accept:** `stems_bench.recommend` covers lead and backing vocals on the tuning tracks with the rule already committed, the report shows the result beside the model in use, and the config follows; or a note saying why the rule cannot be applied. — Done: the rule of P10.2, as committed, applied to the two outputs of the vocal split on the tuning tracks (14 of the 30 have both lead and backing vocals), and `docs/benchmarks/recommendation.md` has the section. It names `mel_roformer_karaoke_aufr33_viperx` for both: 12.32 dB on lead vocals (the others 10.99 and 10.06) and 4.92 dB on backing vocals (1.09 and 0.84). That is the model already in use, so the config does not change. On the MoisesDB test list it scores 10.53 and 4.28 dB. In the benchmark these models are fed the reference vocals, not a separated stem, so the figures say how well each splits a clean vocal and not what a user gets after the first separation. The de-reverb model still cannot be chosen this way: no list has reverberant vocals to score it on. 1 test (the slot is settled only if one model is named for both outputs).
- [x] P11.5 Close P3.11. **Accept:** the optional MERT and Essentia comparison is done, or ticked with the note its acceptance asks for. — Done: P3.11 ticked with its note (see there). Nothing was run for it.

## Phase 12: the edges

Added on 2026-11-24 under rule 6 (no phase was left). The product's ordinary road is now
measured on twenty songs, on CPU. Three things at its edges are not: the GPU, which is
the device a job uses by default on this machine and which every real job so far has
avoided because the benchmark held it; the longest upload the limits allow; and the
weakest stem, `other`, where the model often finds next to nothing.

- [x] P12.1 Real songs through the product on the GPU. **Accept:** `python -m stems_bench.product_check` run under its own run id with `--device mps` on the first three tracks of the frozen subset, and a report from it (`docs/product-on-gpu.md`): agreement with the benchmark, which ran on the same device, and the wall time beside the CPU run's. — Done: run `product-gpu-v1` (`python -m stems_bench.product_check run --run-id product-gpu-v1 --tracks 3 --device mps`, 2026-11-24, 17:49Z to 18:00Z), and `docs/product-on-gpu.md` from it. The first three tracks of the frozen subset through the worker's path on this Mac's GPU, the device a job uses here by default and that no real job had used. **15 of 15 stems are within 0.02 dB of the benchmark's figure**, which is from the same device; the same three stems were flagged as on CPU, and none wrongly. **Wall time 120 s, 94 s and 88 s** for songs of 344 s, 249 s and 240 s, where the CPU run of the same songs took 411 s, 303 s and 310 s; the model itself took 108 s, 84 s and 77 s, about 19 to 20 s per minute of audio, beside the benchmark's 17 to 18. The tuner was scoring on the CPU at the time. The report can now show another run's wall times beside its own (`report --beside RUN`). Three songs; the audio was deleted after scoring. 1 test.
- [x] P12.2 The longest upload. **Accept:** a synthetic signal as long as `uploads.max_duration_seconds` allows goes through the worker's storage entry point with the real default model; the job finishes, all six stems have the right length, and its wall time and the worker's peak memory are recorded with the device; the result is in `docs/deployment.md`. If it does not finish, the limit and what stopped it are written down instead. — Done: `python -m stems_bench.long_upload` (`bench/stems_bench/long_upload.py`): a synthetic signal exactly `uploads.max_duration_seconds` long (20 minutes; one minute of tones and noise, repeated) is put in storage as an upload and the worker is started as the local provider starts it, `python -m stems_worker.jobs JOB_ID`, in its own process, with the real default model. Run `long-upload-v1`, 2026-11-24 at 18:39Z, on this Mac's GPU while the benchmark was using it too: **the job finished**, 568 s from start to result (526 s in the model), all six stems exactly the upload's length, and the worker process held at most 6.46 GB as the operating system counts it (which may not include everything the GPU holds for it). `docs/deployment.md` has the run. **One thing it turned up:** twenty minutes of 24-bit stereo WAV is 318 MB, over the API's 250 MiB size limit, so the API would have refused this very file; the test put it straight into storage. The duration limit is reachable through the API only as 16-bit WAV (212 MB), FLAC or MP3. Nothing was changed about that: both limits are the owner's (BLOCKED.md D2). One job, one length, synthetic audio; the browser player with six 20-minute stems was not tried. 2 tests with the stand-in models.
- [x] P12.3 Is what is left over a better `other`? **Accept:** with the comparison and what counts as better fixed first, a run on the tuning tracks scores the default model's own `other` beside the mix minus its five other stems, and a report states the result and what the test tracks then show. PLAN.md decides that a six-stem model's `other` is used as it is, so nothing changes in the product: the result goes to the owner as a question in BLOCKED.md. — Done: the rule was committed first (`4bee9b0`: the leftover is better if its median SDR is at least 0.3 dB above the model's own and it is the higher on more than half of the tracks; worse is the mirror image; at least 10 tracks). Runs `leftover-tuning-v1` (the 30 tuning tracks, 2026-11-24 18:50Z to 20:38Z) and `leftover-test-v1` (the 20 test tracks, 23:07Z to 00:14Z), both `python -m stems_bench.product_check run --device mps --leftover` through the worker's own path; all 50 jobs finished. Report `docs/leftover-other.md`. **No clear difference.** Tuning, 27 tracks with an `other`: the model's own 2.61 dB, the leftover 2.56 dB (medians), the model's own the higher on 25. Test, 16 tracks, deciding nothing: 2.80 against 2.76, the model's own higher on 12. Where a song has no `other` (3 and 4 songs) the model's own stem is 78.3 and 81.3 dB under the song and the leftover 34.2 and 35.8 dB under: the leftover is never near silence. Nothing changed in the product (PLAN.md); the choice is put to the owner as D15 in BLOCKED.md, with the one thing SDR does not see: with the leftover the six stems add up to the song, so what another stem drops is in `other`. The tuning run's wall times are not timings: it stalled for most of an hour when I started a third GPU job beside it and the machine ran out of memory.
- [x] P12.5 Where the instrument plays and the product delivers silence. Added on 2026-11-24, after the listening kit's levels showed it in one song (P13.4): a bass stem that is there for a minute and missing for the rest, which neither SDR's one number nor the gate's whole-song level says. **Accept:** with what counts as dropped and as a long dropout fixed in code first, the six songs of the listening run are counted for the product now and before, and a report (`docs/dropouts.md`) gives the share of the true stems' sounding time that is dropped, every long dropout, and whether the product's own note would be on that stem. A count only: nothing in the product changes. — Done: `python -m stems_bench.dropouts --out docs/dropouts.md`, run on 2026-11-24 over the audio of the listening run `owner-v1` (six songs, 31 song-stems, each as the product makes it now and as it made it before). The definitions were committed first (`dc9ee6c`): a tenth of a second is dropped when the true stem reaches -60 dBFS and the product's does not; a dropout is one unbroken stretch of the product's silence and is long from 5 seconds of true-stem sound inside it. **The product now drops 5.8 of the 101.7 minutes in which the true stems sound (5.7%); before, 3.7 minutes (3.7%).** By stem, now: `other` 18.0% (before, with another model, 0.5%), bass 9.4% (before 8.9%), vocals 5.5% (6.2%), drums 1.8% (2.1%), piano 0.8%, guitar 0.2%. Five stems have a long dropout: song 4's `other` now (83.4 of 83.5 s; but its true stem is 35.5 dB under the song, so near-silence is close to right, and it is the one stem of the five that carries the product's note), song 6's bass now (110.0 of 144.0 s, longest 54.4 s) and before (93.7 s), song 3's `other` now (24.1 of 203.9 s) and song 2's drums before (12.2 of 231.0 s). The four that are not near-silent overall are delivered with no note: the gate measures a stem over the whole song, and they are 6.5 to 15.8 dB under it. Nothing in the product can catch this as it is, since the count needs the true stem. Six songs of one dataset: it shows the thing happens, not how often in general. A column with the true stem's own level was added to the table after the first count; no figure changed. 5 tests.
- [x] P12.4 The README and the cost report after these. **Accept:** every figure that changed is corrected from its file. — Done: README.md: four rows added to "What has been measured", each from its report (the GPU check, `docs/product-on-gpu.md`; the 20-minute upload, `docs/deployment.md`; the leftover as `other`, `docs/leftover-other.md`; dropouts, `docs/dropouts.md`); "How it sounds to people" now says that a blind test for one listener is ready and waits for the owner; `make listen` is among the commands; the decisions are fifteen (D15). The cost report was generated again (`python -m stems_bench.cost`) and came out the same to the byte: nothing Phase 12 measured feeds it, so it is unchanged.

## Phase 13: a listening test the owner can do alone

Asked for by the owner on 2026-11-24 ("after this next phase finishes, i'd like to do a
listening test myself"). The test the research report describes needs five listeners and
two paid tools (BLOCKED.md B8). This is the part that needs neither: one listener, blind,
comparing what the product delivers now with what it delivered before and with the true
stems. Its tooling was built while Phase 12's last runs were going, which take hours of
GPU time and need nobody; Phase 12 is still the current phase until those are in.

- [x] P13.1 The audio and the kit. **Accept:** `python -m stems_bench.listen make` puts the first N songs of the frozen test subset through the product's own job code as it is now and as it was before the recommendation of Phase 10, adds the true stems, and builds a blind kit with the existing kit builder (the same 30 seconds of each version, matched in level, letters that change from stem to stem, the key outside the folder that is served); resumable; tested on synthetic audio with stand-in models. — Done: `python -m stems_bench.listen make --tracks N --device D` (`bench/stems_bench/listen.py`). For each song: the stems whose instrument is in the song (the rest are left out for every version), the true stems from the dataset, and two runs of the product's own job code (`stems_worker.decision.job.run_job`, no Jev, no tagger): with the config as it is, and with each stem's single model set back to `docs/benchmarks/defaults-before-recommendation.json`. Then the existing kit builder makes the blind kit (three versions under A, B, C in an order that changes from stem to stem, level-matched 30-second excerpts, the key beside the kit and never in it), and the same excerpt of the full song is added for reference. A song already made is not made again. Tested on two synthetic songs with the stand-in models: every version present for every stem, nothing in the served folder names a version, the letters differ between stems, and a second `make` builds no model. **Not yet run on real songs**: that is P13.4.
- [x] P13.2 The page. **Accept:** `python -m stems_bench.listen serve` (or `make listen`) serves one page on this machine only: one stem at a time, every version and the song playing in step so that switching keeps the place, marks from 1 to 5, saved after every step, resumable; it serves the kit's audio and nothing else of the run; tested, and tried by hand in a browser on a synthetic kit. — Done: `python -m stems_bench.listen serve`, or `make listen`: a small server bound to 127.0.0.1 and one page (`bench/stems_bench/listen_page.py`, no outside resources). One stem per step; "Song", A, B, C all start at the same instant and only their volume switches, so changing version keeps the place; keys for play and for each version; an overall mark from 1 to 5 is needed for every version, bleed and artifacts are optional, and a note can be added; marks are saved to `ratings.json` after every step and the page resumes at the first unmarked step. The server serves the page, the kit's list, the marks and the kit's audio, and refuses anything else (the key, paths out of the audio folder, requests from another site or under another host name, marks that are not whole numbers from 1 to 5). 4 tests of the server. Tried by hand on 2026-11-24 in the app's browser with a synthetic kit: started, played (the position advanced by 1.00 s per second), switched by button and by key, was refused when moving on without marks, and saved a step's marks to disk. I heard nothing: a browser driven by me has no ears, so that the versions sound in step is what the code does, not something anyone has listened to yet.
- [x] P13.3 The report. **Accept:** `python -m stems_bench.listen report` unblinds the marks: mean marks per version and per stem, the product now against before with a sign test, and how often the true stem was marked highest; it says plainly that this is one listener; tested on made-up marks. — Done: `python -m stems_bench.listen report [--out FILE]`: joins the marks with the key and writes mean marks per version (overall, bleed, artifacts), mean overall per stem, the product now against before (steps higher, lower, the same, and an exact two-sided sign test on the steps that differ), how often the true stem had the highest mark, and any notes. It says in its first paragraph and again at the end that this is one listener who knows the project, and marks an unfinished test as unfinished. Tested on made-up marks with a known answer, on an empty marks file, and the sign test against hand-computed values.
- [x] P13.4 The real kit. **Accept:** the kit built from six songs with the real models on this machine, its audio only under `runs/` (never in the repository); the page serves it; the owner is told the one command that starts it. — Done: kit `owner-v1` (`python -m stems_bench.listen make --tracks 6 --device cpu`, 2026-11-24, 20:38Z to 23:06Z on this Mac's CPU, then rebuilt from the same audio at 23:13Z after the fixes below): the first six songs of the frozen subset, the true stems and the product's stems now and before, all under `runs/listening/owner-v1/` (git-ignored; nothing of it is in the repository). **30 steps, 83 clips.** The first build was checked by its levels before anyone listened, and three things were wrong with it; each is fixed in the builder and tested. (1) The 30 seconds were the song's loudest, and in two steps the true stem does not play there (song 2's piano, song 4's `other`): three silent clips. Each stem's 30 seconds are now where its true recording is loudest, and the same stretch of the song sits beside them (`window_by` in `listening_kit.build_kit`). (2) Guitar and piano have the same model now as before, so two of the three clips were the same audio, which gives the true stem away: such a version is now played once (7 steps have two clips), its mark counts for both versions, and "now against before" counts only the 23 steps where the models differ. The builder believes the config only where the files agree. (3) Song 4's true `other` is 35.5 dB under the song, and the product's own rule calls a stem empty at 21.68 dB under: that step is left out, by that rule, and the report says so (`left_out.json` beside the key). A kit is no longer rebuilt once marks exist. **One thing the levels show that is the product's, not the kit's:** in song 6 the true bass plays in seven of the song's eight half-minutes; the model the product uses now delivers bass in two of them and next to nothing (-75 dB or lower) in the rest, and the model it used before in three. In the step's 30 seconds the product's bass now is silence. The gate does not flag it (the whole stem is 15.5 dB under the song). It stays in the kit: it is what the product delivers. The page was tried on the real kit in the app's browser: the four files of a step load, the position advances 2.00 s in 2 s, versions switch by button and key, a step without marks is refused, and the key and `left_out.json` are not served; no mark was saved. The owner starts it with `make listen` and opens http://127.0.0.1:8765. 3 new tests, 2 changed; I heard none of it.
- [ ] P13.5 The listening itself, and its report. **Accept:** `docs/listening-owner.md` generated from the owner's marks. *Waits for the owner: only they can listen.*

## Phase 14: what the six stems lose

Added on 2026-11-24 under rule 6: Phase 13's one open task is the owner's, and no other
phase was left. Phase 12 counted how long a stem is silent while its instrument plays
(`docs/dropouts.md`, six songs) and put a choice to the owner (D15: is `other` the model's
own output, or what is left of the song?). What decides between the two is not known:
when a stem drops out, is the instrument in another of the six stems, where the user
still has it, or in none of them? And how much of a song is in none of its six stems at
all? Both can be measured on the twenty test songs with the true stems beside them.
Nothing in the product changes in this phase; it gives the owner what D15 needs.

- [x] P14.1 The measures, fixed first. **Accept:** `bench/stems_bench/lost.py` says what is measured (the share of a dropped true stem found in each of the six delivered stems and in none of them, with a check of that share where the stem is not dropped; the level, and the share of time, of what is in no stem), `product_check run --lost` records it for each song, and it is tested on synthetic audio where the answer is known. Committed before any real song is measured. — Done: `bench/stems_bench/lost.py` and `python -m stems_bench.product_check run --lost`. For each stem of a song: its dropped time as in P12.5; over that time, the share of the true stem found in each of the six delivered stems and in what is left of the song after the six (`<true, place> / <true, true>`); and, as a check, the same share for its own stem where it is not dropped. For the song: the level of what is in no stem against the song, and the share of the sounding time in which it is louder than the level the product calls very quiet. Tested on six tones where the answer is known (a bass missing for ten seconds, half of that time put into `other` and half into nothing: 50% and 50%, summing to 100%; five of the song's twenty seconds in no stem), on pooling over songs, and through the product check with the stand-in models, whose bands add back to the song. No real song has been measured: this commit comes first. 6 tests.
- [x] P14.2 Twenty songs. **Accept:** the 20 tracks of the frozen test subset through the worker's own path with `--lost`, on this Mac's GPU and alone beside the benchmark; `docs/what-is-lost.md` written from that run, naming the commit that fixed the measures. — Done: `docs/what-is-lost.md`, from the run `lost-v1`: the 20 songs through the worker's own path with the real default model on this Mac's GPU, 62 minutes beside the benchmark (00:33Z to 01:35Z on 2026-11-25), all 20 ready, made at the definitions commit `1790879` with nothing uncommitted. Dropped: 16.1 of the 292.9 minutes the true stems sound (5.5%); `other` 16.1%, piano 13.9%, vocals 4.0%, bass 3.5%, guitar 2.9%, drums 2.7%; 11 stems have a long dropout (6 `other`, 3 piano, 1 bass, 1 guitar). Over that dropped time 2% of the instrument is in none of the six stems; nearly all of it is in the six, mostly under another name (bass 79% in `other`, piano 65% in `other`, `other` 68% in drums). What is in no stem is a median 34.2 dB under the song (28.4 to 43.7), and louder than the level the product calls very quiet for 0.0% of a song's time (median; 1.8% at most). Of the 34 stems with 5 s or more dropped, 19 add up to within 10 points of 100% and hold 11.9 of the 16.1 minutes; the other 15 cannot be read share by share, and the report says so in a paragraph added after the run that changes no figure. A check against P12.5: the six songs of the listening test are among the twenty, and their 31 song-stems have the same sounding, dropped and longest seconds in both runs (CPU there, GPU here; `runs/listening/owner-v1/dropouts.json` beside this run's records). 8 tests.
- [ ] P14.3 What it means for D15. **Accept:** D15 in BLOCKED.md says, from the report, what a leftover `other` would give back and what it would not; the README has the result; nothing in the product is changed.

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

### Phase 2 (2026-11-22)

Done: every task that does not need a dataset (P2.1 to P2.9, P2.12a, P2.14, P2.15). The
benchmark harness runs end to end from one command, on synthetic audio. **No quality
number on music exists**, and no default model has been chosen: `config/defaults.yaml` is
still `provisional: true` with equal blend weights.

- **Metrics**: museval 0.4.1's BSS Eval v4 (SDR, SIR, SAR on 1-second windows, median of
  medians), plus silence fraction, RMS, peak, length match and energy above 8 kHz.
- **Datasets**: a frozen synthetic test list (8 tracks) and tuning list (4 tracks);
  loaders for MUSDB18-HQ and MoisesDB with hash-checked frozen lists, and the seeded
  stratified 50-track MoisesDB selection. A run refuses a list that is not committed.
- **Runner**: `bench run` keeps one result file per model and track, keyed on protocol,
  list, weights, device and overlap, so runs resume and never repeat work; `bench status`,
  `bench report`, `bench compare`, `bench tune`.
- **Blends**: four methods with per-model weights from config; in the benchmark
  (`--ensembles`: each slot's single best, two-model blend, full ensemble, and the composed
  six-stem other) and in the product (`separate --mode two_model_blend | full_ensemble`).
- **Reports** are generated only from result files, with the command, commit, frozen-list
  hash, device and seconds beside every score: `docs/benchmarks/synthetic.md` (560 of 560
  jobs, none failed), `device-spread-synthetic.md`, `tuning-synthetic.md` (a tool demo).
- **Tests**: 479 fast (`make check`, about 80 s) and 55 with real weights (`make
  test-slow`; 1 more needs CUDA).
- **GitHub**: at the owner's request the project is in the private repo
  `github.com/colton-hong/stems`. CI ran for the first time there: it failed on one test
  (the CPU-only torch wheel reports `2.15.0+cpu` against the pin), which was fixed; it now
  passes on Python 3.12 and 3.14 with 454 tests, skipping 23 that need downloaded model
  configs and 2 that need MPS.

Blocked, all on the datasets: P2.5b (freeze the MoisesDB subset), P2.10 and P2.11 (the real
benchmarks), P2.12 (tuning on real tracks), P2.13 (the recommendation and the config
update), P2.14b (device spread on real tracks). `docs/benchmarks/README.md` lists the
commands to run when they arrive; each is one line.

Findings worth carrying forward:

1. **Scores depend on the device, not on the run.** The same model on the same tracks
   scores identically when run again on the same device, and up to 0.817 dB differently
   on one track when moved from MPS to CPU (synthetic set, six models). Benchmarks must be
   compared within one device. Measured on synthetic audio only.
2. **Only SDR means anything on the synthetic set.** museval's SIR and SAR are numerical
   noise on pure tones with identical channels, so the synthetic reports leave them out.
   On broadband signals all three behave (tested).
3. **The synthetic scores are not a ranking.** Several strong vocal models return silence
   on sine-tone voices and score 0 dB. The run shows that all 43 separators and all 27
   blend systems go through the harness, and nothing about which is better on music.
4. **museval cannot be imported without ffmpeg** (its `musdb` dependency checks at
   import). Its metrics module is loaded by file path instead; the numbers are museval's.
5. **Blends that mix model kinds are refused.** A blend of a four-stem `other` with a
   six-stem `other` would average two different things, so the config check, the pipeline
   and the benchmark all reject it.

Mistakes made and corrected in this phase:

- The edit that ticked P2.1 also ticked P2.10 to P2.14 (a prefix match). They stayed
  wrongly ticked for six commits; commit `ec03b15` unticked them. Ticks are now edited by
  exact task id.
- The synthetic benchmark's timings are not clean: the test suite ran alongside it several
  times. The reports say what the seconds are and are not.

Choices made here that the owner can overrule:

- Every stem is scored as the pair (target, input minus target), for all models, so SIR
  and SAR are comparable between single-stem and multi-stem models. SDR is unaffected;
  SIR and SAR are not the joint four-source figures leaderboards print.
- Vocal-split and de-reverb models are benchmarked on the reference vocals, not on another
  model's output.
- MoisesDB: a stem is "present" above −60 dBFS whole-track RMS, and drums and bass are
  "prominent" within 12 dB of the mix (the report defines neither); the subset also
  requires 8 tracks with both lead and backing vocals; sources of unequal length are
  padded, not trimmed; percussion, other keys, strings and wind count as `other`, as in
  the dataset's own six-stem mapping.
- A blend's time is the sum of the times of every model it needs.
- A slot has one blend method for both of its blend modes; the tuner prefers the simpler
  candidate unless another beats it by more than 0.001 dB.
- The repo is private, and pushes use a deploy key that can write to that repo only.

Not done, for lack of access: any benchmark on music, blend tuning on music, the
recommended defaults, GPU timings (MPS and CPU only; Modal needs B4).

**Next:** wait for the owner. With the datasets, Phase 2 finishes at P2.5b, P2.10 to P2.13
and P2.14b. Phase 3 starts at P3.1, which needs the BEATs file (B10); P3.2 to P3.8 can be
built without it or the datasets.

### Phase 3 (2026-11-22, stopped at its blocked tasks)

Started on the owner's instruction of 2026-11-22 (benchmarks first, then the next phase),
while the real benchmarks run. Done: P3.2 to P3.8. Not done: P3.1 (BEATs wrapper) and
P3.9 to P3.11 (running the test and its report). **Nothing has been sent to Clef-omni and
no AudioSet tagger has run**, so there is no result of any kind yet for the test.

- **Listening** (P3.2 to P3.4): the AudioSet class-to-stem table; the CLAP wrapper with
  text prompts and a high-band check; `build_track_features` and `stem_signals`, which
  turn any tagger's 527 class probabilities into schema-valid `track_features` and
  per-stem metrics, with BS.1770 loudness. With no tagger the hearing fields say unknown
  and the rest still holds, which is the state today.
- **The Clef-omni test** (P3.5 to P3.8, plus the runner): pre-registration committed
  (`bench/preregistration/clef-omni-v1.json`: thresholds, wording, 50 tracks with their
  30-second windows and presence answers, Part B's 100 clips, the stem map's hash);
  clip builder; client with limits, full logging and replay; scoring with the decision
  rule; `python -m stems_bench.clef.run` to run it all and write the report.

What the data showed while building it:

- By the presence rule, vocals are absent from 6 of the 50 excerpts and piano is present
  in 14. A3 (vocals present or absent, 90%) therefore rests on 6 negatives; always
  answering "present" would score 88%. The report states this next to A3.
- CLAP's prompts single out isolated drums, bass and guitar but are weak on vocals, piano
  and full mixes (3 tracks, a look, not a measurement). CLAP cannot stand in for BEATs.

Choices made that the owner may want to change:

- Every listening threshold in `config/defaults.yaml` (`listening`) is a first guess,
  marked `calibrated: false`. They are measured in P7.1.
- `reverb_estimate` is never "dry": not hearing reverberation is not treated as evidence.
- Backing vocals are approximated by AudioSet's Choir, Chant and Humming with no lead
  class heard; AudioSet has no backing-vocals class.
- One clip per Clef-omni request (183 requests in all), so no question has to say which
  clip it means. The pre-registration was amended once to record this, before any request.
- The reverb probe uses Part B's five clean vocal stems; Part B's guitar-or-piano stems
  are three guitar and two piano.

Not confirmed against the live service: whether the REST endpoint wraps its answer in
`{result, success}` (both forms are accepted), and everything about rate limits.

**Next, for the owner:** the BEATs checkpoint (B10) unblocks P3.1 and the BEATs half of
P3.9; a Cloudflare token (B5) unblocks the Clef-omni half. With both, P3.9 and P3.10 are
one command each. Phase 4 waits for a go-ahead.

**Update, later on 2026-11-22:** the owner said to fetch BEATs through the browser pane.
It is downloaded, pinned and wrapped (P3.1 done, B10 closed), and the test runner now
loads it. What remains in Phase 3 needs only the Cloudflare token (B5). The first look at
BEATs on real mixes (see P3.1) says its instrument probabilities are far below 0.5 on
mixes, which matters for the test's fixed threshold and for every `listening` threshold.
The BEATs half of the test then ran (figures under P3.9): with the pre-registered mapping
and the 0.5 threshold, BEATs fails every threshold except B3. That is a result about this
simple use of BEATs, and it means the quality gate cannot rest on these scores as they are;
how to use the tagger (relative scores, a trained probe, a different mapping) is a design
question for Phase 4 and the calibration in P7.1, to be settled on tuning tracks, not on
the 50 test tracks.

### Phase 4 (2026-11-23, stopped at its blocked task)

Started on the owner's go-ahead ("Start Phase 4", 2026-11-22) while the real benchmarks
run. Done: P4.1 to P4.8. Not done: P4.9, the live check, which needs a TypeSafe key (B6).
**No request has been sent to Jev.** What the code knows about Jev is the API reference as
read on 2026-11-22; every Jev answer used in a test was written by hand or produced by a
scripted stand-in, and each fixture says so.

- **Schemas** (P4.1): the report's three contracts as files in
  `common/stems_common/schemas/`. The plan schema's model lists are generated from the
  registry (`make schemas`), and a test fails when the committed file goes stale.
- **Client and mock** (P4.2, P4.3): `decision/jev.py`, plain HTTPS rather than the SDK, so
  that the pinned model, the retries and the log are ours. It refuses any model name that
  is not an exact version and any reply from another model, retries 408, 429 and 5xx with
  backoff, honours `retry-after`, times out, and logs the responding model, token usage
  and every probability; the key is scrubbed from logs and errors. `decision/mock.py`
  replays recordings by request hash and raises on anything unrecorded, in a way the
  fallbacks cannot swallow.
- **The three decisions** (P4.4 to P4.6): `parse.py`, `plan.py`, `assess.py`. Each asks
  typed questions, assembles its schema object, validates it, and returns a log record
  (provider, model, tokens, latency, answers, fallback reason). Each has a deterministic
  fallback that gives the same schema: keyword rules, the configured defaults, and a rule
  scorer over the tagger's signals. An answer under `decision.min_confidence` is not
  acted on, and the record says which answer and how unsure it was.
- **Gate and retries** (P4.7): `gate.py` (decision 8's rule, thresholds from config) and
  `retry.py`, one state machine per stem with the 2-retry cap and best-attempt delivery.
- **The job** (P4.8): `decision/job.py`, the path the Phase 5 worker will wrap. It runs
  with Jev, with a mock, or with no key, and writes stems plus `job.json`.

243 tests were added (869 fast tests in all, `make check`).

One run outside the test suite, on this machine, 2026-11-23:
`python -m stems_worker.decision.job FIXTURE --request "just the vocals please, as flac"
--device cpu`, on a 12-second synthetic fixture with the real `mel_roformer_kim_vocals`
and BEATs and no key. It finished with status `ready`; all three decisions were made by
the fallbacks (`no_key`); separation took 13.9 s on CPU against a planned 9.6 s (the plan
uses the median CPU rate in the timing log, and the machine was busy with the benchmark).
BEATs scored the synthetic vocals 0.05 for vocals, which says nothing about real singing:
the fixture is made of test tones.

What that run changed: a stem the gate cannot vouch for now carries one neutral warning
("could not be quality-checked automatically") instead of naming what an untrusted tagger
suspects. The suspicion stays in the log as `reason_code`.

How the system behaves today, with no key and uncalibrated thresholds:

- Free text is read by keyword rules (28 test phrases; an exclusion placed after what it
  excludes is not understood, pinned in a test). The plan is `config/defaults.yaml`.
- The gate retries only a stem that is almost entirely silent, which is a level
  measurement. Every other verdict rests on tagger thresholds that are not calibrated
  (`listening.calibrated: false`), so those stems are delivered with the neutral warning
  and not retried (BLOCKED.md D5). This is deliberate: the measured BEATs half of the
  Clef-omni test (Phase 3) showed the uncalibrated mapping is not reliable. P7.1
  calibrates it, and the gate starts acting on it when `calibrated` is set to true.

Not verified, and why:

- Anything about the live service: the wire format beyond the documentation, real rate
  limits, latency, token counts, and whether Jev's answers are any good for these
  questions. P4.9 needs the key (B6).
- The job runner with real models on real music, with every stem and with retries. Only
  the single run above and the weightless-model tests exist.
- Every new number in `config/defaults.yaml` is a first guess, marked as such:
  `decision.wanted_threshold`, `gate.rules.*`, and `decision.mode_for_quality` (its
  `balanced` entry should follow P2.13's recommendation once there is one).
- `estimated_seconds` is only as good as the timing log: a model never timed on the
  device adds nothing and is named in the plan's record.

Choices made here that the owner can overrule are in BLOCKED.md: D5 (unverifiable stems
get a warning, not retries) and D6 (de-reverb is never added unasked).

For the owner: a TypeSafe key with credit (B6) unblocks P4.9. Phase 5 (API and worker)
waits for a go-ahead.

## Phase 5 summary (2026-11-24)

Done: P5.1 to P5.14. The system now runs from an HTTP upload to signed downloads.

- **Storage** (`stems_api.storage`): one interface, three backends. Local disk with
  HMAC-signed expiring links, S3-compatible with presigned URLs, Azure Blob with SAS
  tokens. All three pass the same 29 contract tests.
- **Database** (`stems_api.db`): the six tables of PLAN.md section 7, migrations, and
  queries that are bound to one owner.
- **Contract** (`stems_common.jobs`): `spec.json`, `status.json` and `result.json`, with
  schemas and validators that the API and the worker both call.
- **Worker** (`stems_worker.jobs`): a job from storage to storage, with stage reporting,
  waveform peaks and a zip. A finished job run again does no work.
- **Providers** (`stems_api.providers`): local (subprocess, one job at a time), Modal, and
  the Azure Container Apps seam.
- **API** (`stems_api.app`, `make api`): every route of PLAN.md section 8, the owner
  cookie, upload checks by decoding, status sync from storage, the retention sweeper and
  rate limits.

331 tests were added (1,200 fast tests in all, `make check`).

One run outside the test suite, on this machine, 2026-11-24: `python -m stems_api --port
8765` with `STEMS_WORKER_FAKE_MODELS=1` and its database and storage in a scratch folder,
driven with `curl`. A 4-second synthetic fixture was uploaded asking for vocals, drums
and bass as `mp3_192`. Polling four times a second saw `queued`, `transcoding`,
`separating`, `packaging`, `ready`. The three links each returned 200 `audio/mpeg` and a
file that decodes as 44.1 kHz, 4.0 s; a range request returned 206; the zip returned 200;
the job without the cookie returned 404; `DELETE` returned 204 and left 0 objects. The
server log had no traceback. The stems in that run are frequency bands from the stand-in
models, not separated music.

What the tests found that was then fixed:

- boto3 adds a checksum algorithm to uploads, and presigned links are only SigV4 when the
  client is configured for it. Both showed up only against the real boto3 client.
- A first upload that failed to start left a job its browser could not see, because the
  error reply did not carry the new cookie. It does now, and a refused upload no longer
  creates an owner at all.
- Deleting or expiring a job left its decision log, whose answers describe the request
  and the audio. Deletion and expiry now remove it.

Not verified, and why:

- **Real models through the API.** Every job in the tests and in the run above used the
  stand-in models. The worker entry point with real checkpoints has not been run: the
  machine's memory is taken by the benchmark (P2.12).
- **S3 and Azure Blob against a real bucket** (B7) and **Modal** (B4). The S3 and Azure
  code was exercised through the real client libraries with the network replaced, which
  checks request building and link signing but not that a service accepts them. The
  Modal app file has never been deployed and its image has never been built.
- **Postgres.** Every database test ran on SQLite.
- **A real browser.** Cookies, CORS and server-sent events were tested with an HTTP test
  client and with `curl`, not in a browser. Phase 6 does that.
- **Large uploads and load.** The size limit was tested at 1.5 MB, not at the shipped
  250 MiB; nothing measured memory or speed with a large file, or several users at once.
- **CI.** The workflow now installs the `api` extra. Whether it passes on Linux for
  these commits has not been checked from here.
- The rate-limit numbers are proposals, not measurements (BLOCKED.md D7).

How it behaves that the owner may want to change (BLOCKED.md D7 to D9):

- Rate limits are counted in memory in one process and by the connection's address.
  Behind a proxy, or with several API processes, they need a shared store and the
  proxy's forwarded address.
- A job whose request text names no stem ends as `failed` with the parse attached,
  because the job table has no review state. The web app is expected to call
  `POST /requests/parse` first and confirm with the user (Phase 6).
- After a job is deleted or expires, its row stays as a marker, with the file name,
  analysis, plan, stems, events and decision log removed, and its timing rows kept.
  `DELETE /me` removes every row.

For the owner: nothing is needed to use this locally (`make api`). Bucket credentials
(B7) and a Modal token (B4) unblock the three live checks. Phase 6 (web app) waits for a
go-ahead.

## Phase 3 addendum: the Clef-omni half (2026-11-24)

The Cloudflare token arrived, so the pre-registered test was completed (P3.9, P3.10).
`python -m stems_bench.clef.run --only clef --run-id clef-omni-v1` sent 183 requests to
Workers AI; all 183 were answered on the first attempt (137,930 input tokens). A second
run with both halves took the Clef-omni answers from the log, sent nothing, and
recomputed BEATs on CPU, so `scores.json` holds both. The report is
`docs/clef-omni-test.md`, generated from the run.

By the rule fixed before any score was seen, Clef-omni is **rejected** and the listening
layer stays BEATs with CLAP as second opinion. It missed five of its own thresholds:
overall presence accuracy 81.3% against 85%, piano 76% against 80%, other 58% against
70%, bleed AUC at +30 dB 0.757 against 0.80, and 4 of 20 clean clips called bleeding
against at most 2. It passed vocals, drums, bass and guitar presence, vocal absence, the
pooled bleed AUC (0.901) and the rank correlation with bleed level (0.703).

What the same report also shows, and the verdict does not say: on this test Clef-omni
was far ahead of BEATs as the listening layer uses it today. Presence accuracy 81.3%
against 26.0%; bleed AUC 0.901 against 0.642. The BEATs figures come from the
uncalibrated mapping at its pre-registered 0.5 cut-off, where it answers "absent" almost
everywhere; P7.1 calibrates that mapping. The verdict is the pre-registered one and
nothing was changed because of this comparison. Whether to look at Clef-omni again
after BEATs is calibrated is the owner's call.

Not known: what the run cost. No price was read from Cloudflare, and nothing was bought.
P3.11 (optional: MERT and Essentia against BEATs on Part A) is open.

## Phase 5 addendum: two faults found after the summary (2026-11-24)

Both came out of one failed `make check` and one failed run of the end-to-end test.

- **A link token had more than one spelling.** The last character of a base64 string can
  carry bits that decode to nothing, so a token with that character changed could carry
  the same signature bytes and still verify. It granted nothing new (same key, same
  expiry), but P5.1's note said any changed character is rejected, and about 1 run in 16
  of that test would have shown otherwise. Tokens are now accepted only in the one
  spelling the server writes; a test tries all 64 last characters of each half.
- **The poller and a request could record the same worker event twice.** The claim on a
  job was released before the transaction committed, leaving a window in which a second
  sync read the old history and wrote the same events, which the unique index then
  refused with an error. It failed the end-to-end test once in about 20 runs and could
  not be reproduced on demand; the cause was found by reading the code, so it is likely
  but not proven to be what failed. The claim is now held until the commit
  (`JobSync.refresh`), and a test holds one sync open and checks that a second steps
  aside and a request is still answered.

After both fixes the storage and API tests passed four times in a row (183 tests each).

## Phase 6 summary (2026-11-24)

Done: P6.1 to P6.9. There is now a web app in front of the API: upload a song, watch the
job, play the stems together, download them, delete them.

- **Upload** (`/`): file picker or drop, six stem ticks, an optional request in words that
  fills the ticks for you to check, format, speed or quality, vocal split and de-reverb.
- **Job** (`/jobs/[id]`): nine named stages followed live, each stem's state, the player,
  notes on stems the gate could not vouch for, downloads, time left, delete.
- **My jobs** (`/jobs`): this browser's jobs and Delete everything.
- **Terms and Privacy**: placeholders that say so first.

Checked by `make web-check` (ESLint, TypeScript, 21 unit tests, build) and `make web-e2e`
(11 browser tests in Chrome against the real API with stand-in models, about 30 seconds).
I also opened the upload and job pages in a browser and looked at them: a 6-second test
tone with all six stems went to Ready, and the waveforms, notes and deletion card were
laid out as intended.

What the browser tests found that was then fixed:

- With the owner's key now in `.env`, the test server's parse route was calling the real
  Jev service. `STEMS_DOTENV_PATH` now names the `.env` file to read; both test suites
  point it at a file that does not exist, so no test, and no process a test starts, sees
  a real key.
- The first run wiped the test database after the API had made its tables. That was the
  test setup's fault, not the app's; storage and database are now cleared before the API
  starts.

Not verified, and why:

- **Real music.** Every job behind these pages used the stand-in models, so the player has
  only ever played frequency bands of a test tone. That stems of a real song line up and
  sound right has not been heard by anyone.
- **A real screen reader, other browsers, a phone.** The accessibility evidence is axe, a
  keyboard-only test and roles read from the accessibility tree, all in Chrome on a
  desktop. The layout has a narrow-screen rule that nothing has exercised.
- **Long tracks.** The player decodes every stem in full (about 21 MB per minute per stem,
  by arithmetic). Tracks over 10 minutes wait for a click before loading; what a
  20-minute, six-stem job does to a browser has not been tried.
- **Drag and drop.** Written, not tested: the tests use the file picker.
- **Links to S3 or Azure.** The player fetches audio through the signed links. With local
  storage those come from the API, which allows the web app's origin. A bucket would need
  its own CORS rule for the app's origin; nothing sets or checks that yet (B7).
- **CI.** A job that runs the browser tests on Linux is in the workflow and has never run.
- The preview launcher of the desktop app could not start the servers here (it could not
  read the project folder), so the visual check used servers started from a shell.

Choices made here that the owner can overrule (BLOCKED.md D10, D11):

- A request in words is never uploaded as words. It fills in the ticks, the person sees
  and corrects them, and the job is sent with ticked stems.
- The player holds whole stems in memory to keep them on one clock, and makes long
  tracks opt in.

For the owner: `make api` in one terminal and `cd web && npm run dev` in another, then
http://localhost:3000. With no weights downloaded for a stem's model the first real job
will download them. Phase 7 (calibration and hardening) waits for a go-ahead.

## Phase 7 summary (2026-11-24)

Done: P7.1 to P7.6. P7.7 (optional fine-tuning) was not done; its note says why.

- **Gate calibration** (P7.1, `docs/gate-calibration.md`). 1,270 one-minute excerpts of
  MoisesDB scored by the tagger, thresholds chosen on 30 tracks and checked on 50 others,
  with the rules and the pass bar committed first.
- **End-to-end tests** (P7.2). A 13th browser test covers the vocal split and de-reverb
  extras; a slow test runs the real vocals model through the worker's storage entry point.
- **Cost per song** (P7.3, `docs/cost-per-song.md`). Decisions and storage are priced from
  measured usage and prices read that day. Separation on a rented GPU is not priced: no
  job has run on one.
- **Deployment guide** (P7.4, `docs/deployment.md`), every step marked run here or not.
- **Listening-test kit** (P7.5): builds a blind kit and scores returned sheets.
- **Security review** (P7.6, `docs/security-review.md`): 12 findings fixed, each held by a
  named test.

What the phase found:

- **The tagger can tell whether a stem holds its own instrument, and little else.** On
  held-out tracks its presence answer is right 97.1% of the time when it gives one (it
  gives one 57.5% of the time). Its bleed score barely separates clean stems from bled
  ones (AUC 0.644; half of the clean stems flagged), and in a full mix it is wrong more
  often than right about what is playing. Three of the five checks failed, so
  `listening.calibrated` stays false: stems are delivered with the "could not be
  quality-checked" note and nothing is retried on the tagger's word. The two presence
  thresholds in the config are now the measured ones; the bleed thresholds are still
  guesses.
- **Data from an upload outlived its deletion in three places**: separated audio in a
  shared cache, a copy of the upload in the temp folder of a stopped worker, and the
  job's log. All three are fixed and tested.

Not verified, and why:

- **The gate on real separator output.** The calibration used reference stems with bleed
  mixed in by hand. How the tagger scores what a separator really produces is Phase 8.
- **Jev as the judge.** The planned sample of Jev's gate answers on these clips was not
  run: Jev would be judging the same bleed scores that failed. How often Jev decides, and
  how well, is unmeasured beyond the three live calls of P4.9.
- **Real models behind the API and the browser.** One real model has run through the
  worker's entry point on a 6-second fixture. No real job has gone through the API.
- **Cost of separation on a rented GPU** (B4), **S3 and Azure** (B7), **the listening
  test** (B8): each waits for something only the owner can provide.
- The security review is the author's own; no outside audit, scan or attack.

Still open from earlier phases: P2.12, P2.13 and P2.14b (the tuning stage of the real
benchmarks is running; a later iteration commits its report and continues) and P3.11
(optional).

**Next:** Phase 8, added under rule 6: measure the listening layer on real separator
output, switch on the signals that pass, and run real models through the API.

## Phase 8 summary (2026-11-24)

Done: P8.1 to P8.8.

- **The listening layer on real separator output** (P8.1, P8.2,
  `docs/gate-on-real-output.md`). 1,060 outputs the benchmark had kept were scored by the
  tagger and set against the SDR and SIR the benchmark measured for them, with the two
  questions and their pass bars committed first.
- **One gate rule switched on** (P8.3): a stem 19.81 dB or more under the mix in the
  mix's loudest minute is treated as probably empty, retried, and delivered with a note
  of its own. The tagger's switch became one per signal; both are off.
- **Presence in a full mix** (P8.4): measured, and no instrument holds up, so a mix now
  reports no instruments and an unknown vocal presence instead of guesses.
- **A real job end to end** (P8.5): `make web-e2e-real` puts the real models through the
  API, the local worker and Chrome. 103.3 s on CPU for six stems of a 12-second synthetic
  signal.
- **Dependency audit** (P8.6): no known vulnerability in 136 Python or 402 npm packages
  on 2026-11-24.
- **Phone width** (P8.7): the flow and the axe checks pass at 375 px with no change to
  the app.
- **Reverse proxies** (P8.8): forwarded headers are believed only from proxies named in
  `STEMS_TRUSTED_PROXIES`, and from nobody by default.

What the phase found:

- **An empty stem shows in its level, not in what the tagger hears.** "At least 19.8 dB
  under the mix" was right for 95.1% of the held-out outputs it called empty and found
  88.3% of the empty ones. The five it got wrong were pianos the model failed to find,
  which are empty outputs too.
- **Nothing the tagger says is a reason to act.** Its bleed score preferred the better of
  two outputs in 64.8% of pairs (80% needed), after failing on hand-mixed stems in
  Phase 7; its "absent" answer was right 87.9% of the time on real outputs (90% needed);
  in a full mix it cannot name instruments. The tagger still runs in every job and its
  scores go into the notes and to Jev, but the gate acts on none of them. Whether it is
  worth its time is P9.3.
- **uvicorn's default believed forwarded headers from this machine.** The API had been
  running with that. It is now off unless a proxy is named.

Mistakes of mine in this phase, and how they were caught:

- P8.3 changed what the gate sends to Jev, which left the recorded live gate request
  stale. Its replay test skipped itself, and I did not look at the skip count until two
  commits later. Re-recorded with one live request; the cost report was regenerated.
- The first `make web-e2e-real` failed after the job had finished, because the test
  looked for the result file in the wrong folder.
- Two older tests fed the stand-in models audio whose requested band was 21 dB under the
  mix; the new rule retried it, correctly, and `make check` stopped the commit until the
  tests had other audio.

Not verified, and why:

- **A real song through the product.** Tests may only use the synthetic signal, so the
  one real-models job ran on tones. Real songs through the product's path, scored
  against references, are P9.2.
- **The empty-stem rule on whole songs.** It was measured on the loudest minute of each
  track, with "empty" meaning the reference is absent in that minute. An instrument that
  plays only elsewhere in the song would be flagged. P9.1.
- **What the retries cost** for a song that lacks an instrument. P9.4.
- **A real proxy, a real phone, Safari.** The proxy tests use uvicorn's middleware; the
  phone tests are desktop Chrome at a phone's size.
- **Jev as planner or judge.** Still unmeasured beyond single live calls.
- As before: S3 and Azure (B7), Modal and any GPU price (B4), the listening test (B8).

Still open from earlier phases: P2.12, P2.13 and P2.14b (the tuning stage of the real
benchmarks is running: 347 of 900 jobs at 12:21Z) and P3.11 (optional).

**Next:** Phase 9, added under rule 6.

## Phase 9 summary (2026-11-24)

Done: P9.1 to P9.6 (P9.6 was added during the phase, from what P9.2 found).

- **The empty-stem rule against whole songs** (P9.1). The rule of P8.3 looked at the
  mix's loudest minute and failed against the whole song: right for 82.5% of what it
  called empty, because it also caught instruments that play elsewhere. The level over
  the whole song passed (90.2% right, finds 92.2%) and replaced it.
- **Real songs through the product** (P9.2, `docs/product-vs-benchmark.md`). Three
  MoisesDB tracks through the worker's own path with the real models: all 12 stems that
  can be compared are within 0.07 dB of the benchmark's figure for the same model and
  track. The audio was deleted after scoring.
- **What the tagger costs** (P9.3). 31 s for a 4-minute song at the median of those
  three jobs, 13% on top of the separation, for nothing a user sees. It is now off by
  default (BLOCKED.md D13).
- **What an absent instrument costs** (P9.4): 11 to 36 s of retries for a 4-minute song,
  by arithmetic on measured model times. P9.6 then removed the retries.
- **Should an empty stem be retried?** (P9.6). No. In 42 cases on held-out tracks the
  fallback model rescued nothing and 4 times it would have delivered leftovers as a
  guitar the song does not have. An empty stem is now delivered at once with its note.
- **The README** (P9.5) says what works, how to run it and what has been measured.

What the phase found:

- **The product delivers what the benchmark measured.** That had never been checked: the
  benchmark calls the models directly, a job goes through storage, planning, the gate and
  FLAC. On three songs the two agree to within 0.07 dB.
- **The gate was making stems worse.** Its oldest rule, in place since Phase 4, sent a
  silent stem round again with another model. On real music a silent stem from the best
  model usually means the instrument is not there, and the second model's answer is
  leftovers. A real job showed it; the kept benchmark outputs confirmed it, 4 to 0.
- **After three phases of measuring, the gate does one thing:** it tells the user when a
  stem is silent or far below the song. It retries nothing and vouches for nothing. The
  tagger, the bleed levels and the retry machinery are all still in the code, switched
  off by measured results, each with a config value that turns it back on.
- **The default models are not the best measured ones.** The three real songs used
  `scnet_xl_ihf` for drums and `htdemucs_ft` for bass, as the provisional config says;
  both test-list reports put `bs_roformer_sw_6stem` ahead on those stems. Phase 10.

Mistakes of mine in this phase:

- The loudest-minute rule (P8.3) was switched on from a measurement whose "truth" was
  also the loudest minute. I wrote that limit down at the time and still shipped the
  rule; an hour later the whole-song check showed it wrong about one flagged stem in six.
- A scripted edit to the product report failed half-way and I regenerated the report
  before noticing; the next command applied it properly. Nothing wrong was committed.

Not verified, and why:

- **More than three songs, and any song in the browser by a person.** Three tracks show
  the road works. Tests may not use real music, and the browser run is the owner's.
- **The GPU path of the product.** All three real jobs ran on CPU because the benchmark
  holds the GPU; the benchmark's figures are from the GPU. They agreed anyway.
- **The six-stem `other` in the product.** It is made from a four-stem model and has no
  benchmark figure to stand beside. It scored 1.46 to 3.40 dB on the three songs.
- **A stem that is in the song but very quiet throughout.** The rule would flag it. Nine
  of the 92 held-out outputs it called empty had an instrument in the song; whether those
  were quiet parts or parts the model missed was not looked at.
- As before: Jev as planner or judge, S3 and Azure (B7), Modal and any GPU price (B4),
  the listening test (B8), a real proxy, a real phone.

Still open from earlier phases: P2.12, P2.13 and P2.14b (the tuner is searching on the
finished MoisesDB tuning run; the MUSDB18-HQ tuning stage follows) and P3.11 (optional).

**Next:** Phase 10, added under rule 6: choose the default single models on the tuning
tracks.

## Phase 10 summary (2026-11-24)

Done: P10.1 to P10.4.

- **The single models on the tuning tracks** (P10.1,
  `docs/benchmarks/tuning-models-moisesdb.md`).
- **The default model** (P10.2, `docs/benchmarks/recommendation.md`). By a rule committed
  before the tuning scores were read, one model, `bs_roformer_sw_6stem`, serves all six
  stems: it was the best on the 30 tuning tracks for each of them.
- **Real songs with it** (P10.3, `docs/product-vs-benchmark.md`). The same three songs:
  15 of 15 stems within 0.02 dB of the benchmark, all 3 absent instruments flagged, none
  flagged wrongly, about 4.5 times faster than with the old defaults.
- **The figures that follow** (P10.4): 69 s of separation for a 4-minute song on this
  Mac's GPU, where the old defaults took 230 s.

What the phase found:

- **The provisional defaults were costing 1.4 to 2.5 dB on drums and bass** on both test
  lists, and running four models to do it.
- **The untuned blends are worse than the single model.** On the tuning tracks no blend
  in the config gains more than 0.03 dB on it and four of five lose 0.9 dB or more. So
  every quality setting now gives the single model (BLOCKED.md D14), until the tuner says
  otherwise.
- **Two tests had been leaning on the config.** One handwritten Jev reply was malformed
  and its test passed through the old quality mapping; one live recording went stale and
  its test skipped itself. Both are fixed; the second took one live Jev request.

Not verified, and why:

- **That this model is the best on music it has not seen.** Which songs each checkpoint
  was trained on is not known. The choice rests on one dataset's tuning tracks and is
  confirmed on two test lists; if the model met those datasets in training, all three
  flatter it.
- **Vocals.** The chosen model is 0.27 dB behind the old vocals model on the MoisesDB
  test list and 0.34 dB ahead on MUSDB18-HQ. On the tuning tracks it led by 0.39 dB.
  That is the rule's answer, and it is a close one.
- **Any blend the tuner may yet find.** P2.12 and P2.13.
- **How it sounds.** SDR chose this model; nobody has listened (B8).
- **More than three songs through the product.** P11.1.

Still open from earlier phases: P2.12, P2.13 and P2.14b (the tuner is still searching)
and P3.11 (optional).

**Next:** Phase 11, added under rule 6.

## Phase 11 summary (2026-11-24)

Done: P11.1 to P11.5.

- **Twenty real songs through the product** (P11.1, `docs/product-vs-benchmark.md`):
  100 of 100 comparable stems within 0.15 dB of the benchmark; all 20 absent instruments
  flagged; 7 of 100 present ones flagged as well.
- **The quality menu** (P11.2) is shown only when the choice changes something.
- **The real-models browser job** (P11.3) passes with the one default model: 35.8 s from
  upload to ready on CPU for the 12-second synthetic signal.
- **The vocal-split model** (P11.4): the committed rule keeps the one in use.
- **P3.11** (P11.5) closed as skipped, with its reason.

What the phase found:

- **The tests could kill a real job.** The sweeper's tests cleaned the real temporary
  directory with a clock set days ahead, and one job of the twenty lost its working
  folder to a test run. The test run now has a temporary directory of its own. Any job
  running during an earlier `make check` on this machine was exposed to the same thing.
- **When the gate flags a stem whose instrument is in the song, the stem is usually
  useless anyway.** Six of the seven such stems scored 1.2 dB or less: the model had
  found next to nothing, mostly for `other`. One was a good piano that is quiet in its
  song, and the note is wrong about that one.
- **`other` is where the product is weakest.** Five of the twenty songs have an `other`
  the model all but missed.

Not verified, and why:

- **The GPU.** Every real job so far ran on CPU. P12.1.
- **A long upload.** The longest song so far was ten minutes; the limit is twenty. P12.2.
- **Songs from anywhere but MoisesDB**, and anyone listening.
- As before: Jev, S3 and Azure (B7), Modal (B4), the listening test (B8).

Still open from earlier phases: P2.12, P2.13 and P2.14b (the tuner is still scoring).

**Next:** Phase 12, added under rule 6.

## Phase 12 summary (2026-11-24)

Done: P12.1 to P12.5.

- **The GPU** (P12.1, `docs/product-on-gpu.md`): 15 of 15 stems within 0.02 dB of the
  benchmark; 88 to 120 s a song where the CPU took 303 to 411 s.
- **The longest upload** (P12.2, `docs/deployment.md`): twenty minutes of audio through a
  worker process in 568 s on the GPU, every stem the right length, 6.46 GB at most.
- **The leftover as `other`** (P12.3, `docs/leftover-other.md`): no clear difference by
  SDR (2.61 against 2.56 dB on 27 tuning tracks; 2.80 against 2.76 on 16 test tracks). Put
  to the owner as D15.
- **Dropouts** (P12.5, `docs/dropouts.md`, added during the phase): on six songs the
  product is silent for 5.7% of the time the true stems sound; 3 of 31 stems have a
  stretch of 5 s or more, and 2 of those 3 carry no note.
- **The README** (P12.4) has these; the cost report did not change.

What the phase found:

- **The two upload limits disagree.** Twenty minutes of 24-bit stereo WAV is 318 MB, over
  the API's 250 MiB limit: the duration limit can be reached only as 16-bit WAV, FLAC or
  MP3. Both limits are the owner's (D2); nothing was changed.
- **A stem can be missing for most of a song and say nothing.** In the sixth listening
  song the bass is delivered for about a quarter of the time it plays (34 of 144 s). The gate looks at a
  stem's level over the whole song, and over the whole song that stem is loud enough.
- **The model's `other` goes silent far more than the one it replaced** (18.0% against
  0.5% of the true stem's time on six songs; 7.6% against 0.5% without the one song whose
  `other` is next to nothing). SDR prefers it all the same. Whether silence or bleed is the better
  fault there is for a listener.
- **Three heavy jobs do not fit this Mac.** I started the listening kit on the GPU beside
  the benchmark and the leftover run; swap filled, all three stalled for most of an hour,
  and the kit was made on the CPU instead. One heavy job beside the benchmark, never two.

Not verified, and why:

- **Nobody has listened.** The owner's test is ready and waits for them (P13.5).
- **Where a dropped instrument goes**: into another stem, or out of the six altogether.
  Not looked at; it bears on D15.
- **Six 20-minute stems in the browser player.** The long job was run through the worker
  only.
- As before: Jev beyond the recorded requests, S3 and Azure (B7), Modal (B4), the panel
  test (B8), and why CI failed once on Python 3.12.

Still open from earlier phases: P2.12, P2.13 and P2.14b (the MUSDB18-HQ tuning stage is
running; see the top of this file).

**Next:** Phase 13 has one task left and it is the owner's. A Phase 14 is added under
rule 6 so that the work does not wait.

## Phase 13 so far (2026-11-24)

Done: P13.1 to P13.4. Open: P13.5, which is the owner's (only they can listen); the phase
is summarised properly when its report exists.

- **The kit** (`runs/listening/owner-v1/`, not in the repository): six songs, 30 steps,
  83 clips of 30 seconds; the true stem, the product now and the product before, blind.
  `make listen` serves it on this machine only.
- **Its first build was wrong in three ways**, found by reading its levels before anyone
  listened: silent clips where the stem does not play in the song's loudest half minute,
  two identical clips where the model did not change, and one true stem too faint to
  judge. All three are fixed in the builder and tested (see P13.4).
- **Not verified:** how any of it sounds. I have no ears; the page was driven in a browser
  and its audio measured, nothing more.

## A correction about CI (2026-11-24)

Several summaries above say that CI "has never run" (Phases 5 and 6) or list CI on Linux
as unverified. That was wrong in one way that matters: I had never seen its results, and
wrote that as if nothing had run. The workflow has run on every push since the repository
got its remote on 2026-11-22.

What is known now, read from GitHub's public API on 2026-11-24 at 19:40Z, after the owner
made the repository public: 88 runs, 84 of them successful, 4 failed. All four jobs pass
on Linux in the successful runs: the Python checks on 3.12 and 3.14, the web checks, and
the browser tests in Chromium. The failures were on commits `17a0ae3` and `844a74d`
(2026-11-22), `96b0304` (2026-11-24, 07:37Z) and `bc53163` (2026-11-24, 17:37Z). In the
last of these the test step failed on Python 3.12 and passed on 3.14, and the next push
passed on both; I have not read its log (logs need a login), so the cause is not known.
The three earlier ones I have not looked into either.

Why it came up: each push runs four jobs, the repository was private, and GitHub told the
owner that 1,804 of the month's 2,000 free minutes were used. About 130 pushes in three
days did that. The owner made the repository public, where the minutes are not metered,
and asked for pushes to go on as before.

Checked the same day, because the whole history became public with that: no key or token
from `.env` is in any commit, `.env` was never committed, and the one string that looks
like a key is a made-up one in a test. The commit author lines carry the owner's name and
email address, as they always did.
