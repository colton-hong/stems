# PROGRESS

Single source of truth for where the build is. Read this first, every iteration.

**Current phase: 8 (the gate on real output, and real jobs end to end), started 2026-11-24.** The owner asked on 2026-11-24 for each phase to follow the last without waiting, and for new phases to be added when none is left; Phase 8 was added that way. Phases 0, 1, 4, 5, 6 and 7 are complete apart from checks that need the owner's accounts (S3 and Azure, B7; Modal, B4; the listening test, B8). Still open from earlier phases: P3.11 (optional); Phase 2's real benchmarks, running in the background (`tools/run_real_benchmarks.sh`; P2.10 and P2.11 are done, the tuning stage on MoisesDB started 2026-11-24T04:53Z; P2.12, P2.13, P2.14b: each later iteration checks `bench status`, commits a report when its stage has ended, and ticks the task).

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
- [ ] P2.12 Tune blend methods and weights on tracks outside the frozen test lists, then score them on the test lists. **Accept:** the tuning tracks are listed and disjoint from the frozen lists. *Running since 2026-11-24T04:53Z (stage 3, run `moisesdb-tuning-v1`, 900 jobs): stages 3 and 4 of `tools/run_real_benchmarks.sh` run `bench tune` on `moisesdb_tuning` and `musdb18hq_train` and write `docs/benchmarks/tuning-moisesdb.md` and `tuning-musdb18hq.md`. Applying what they choose and scoring it on the test lists is done by hand afterwards.*
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
- [ ] P3.11 Optional: MERT probe and Essentia model compared with BEATs on Part A. **Accept:** results added to the same report, or a note saying it was skipped and why. *Waits for P3.10: it adds to that report, and compares against BEATs.*

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
- [ ] P8.2 Run it and report. **Accept:** `docs/gate-on-real-output.md` generated from a finished run, with a verdict for each question (can an empty stem be told from a real one; does the bleed score prefer the better of two outputs).
- [ ] P8.3 Act on P7.1 and P8.2, one signal at a time. **Accept:** `listening.calibrated` becomes one switch per signal, each true only if its held-out check passed; the gate and the web app use exactly the signals that are on (a stem found empty gets a note that says the instrument was not found, not the generic one); with every switch off nothing changes; tests for both states; `make web-e2e` passes.
- [ ] P8.4 Presence in a full mix: thresholds of its own from the scores of run `gate-v1`, or a stated dead end. **Accept:** `docs/gate-calibration.md` gains a section with thresholds chosen on the calibration mixes and checked on the held-out ones against a bar fixed first; `track_features` uses them if they pass, and otherwise reports "unknown" and no instruments instead of guesses.
- [ ] P8.5 A real job through the API and the browser: real models, no stand-ins. **Accept:** one command runs it (the API, the local provider, the real registry, the synthetic fixture) and checks that all six stems arrive and play; run here once, with each model's time and its device stated; `docs/deployment.md` no longer says "not run" for it.
- [ ] P8.6 Dependency audit. **Accept:** `pip-audit` and `npm audit` are run and their dated results are in `docs/security-review.md`, with what was changed and what was left and why.
- [ ] P8.7 The web app at phone width. **Accept:** upload to download, and the axe checks, pass at 375 px wide with no sideways scrolling, inside `make web-e2e`.
- [ ] P8.8 Rate limits behind a reverse proxy (BLOCKED.md D7). **Accept:** a setting names the proxies to trust; the forwarded address is used only when the connection comes from one of them; with the setting empty nothing changes; tests for a trusted proxy, an untrusted one and a forged header; `docs/deployment.md` says how to set it.

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
