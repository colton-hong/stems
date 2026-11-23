# PROGRESS

Single source of truth for where the build is. Read this first, every iteration.

**Current phase: 2 (benchmark harness), finishing on real data; Phase 3 follows.** On 2026-11-22 the owner asked for both datasets to be downloaded for them, and said: start the benchmarks when the downloads finish, then do the next phase. So Phase 3 may start once the real benchmarks are running, without a further go-ahead; the stop-and-wait rule applies again at the end of Phase 3.

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
- [ ] P2.10 Benchmark on MUSDB18-HQ test. **Accept:** `docs/benchmarks/musdb18hq.md` generated from a completed run. *Running (2026-11-22): run `musdb18hq-v1`, started by `tools/run_real_benchmarks.sh`; dataset verified and unpacked, lists frozen in `3cb7b64`. Stage 1 is the default models and blends, stage 5 every other separator; tick when the report is generated from the finished run.*
- [ ] P2.11 Benchmark on the MoisesDB subset, including guitar, piano and lead vs backing. **Accept:** `docs/benchmarks/moisesdb.md` generated from a completed run. *Running (2026-11-22): run `moisesdb-v1`, stage 2 of the same script (100 of 2,050 jobs were done when the script was restarted to put MUSDB18-HQ first; they are kept). Stage 6 adds every other separator.*
- [x] P2.12a Blend tuner, proven on synthetic data. **Accept:** `bench tune` refuses any list that is not a tuning list or that shares a track with its test list; on a tuning list it tries every method and a grid of weights per slot and mode, and what it picks equals a brute-force search in a test. — `bench tune --dataset LIST [--slots] [--out FILE]` (`stems_bench.tune`). Tuning lists and the test lists each must be disjoint from are declared in one table (`synthetic_tuning` / `synthetic`, `musdb18hq_train` / `musdb18hq_test`, `moisesdb_tuning` / `moisesdb_subset`), and the check runs before any model does. It runs the members with their audio kept, then tries, per slot and mode, the four methods and weights in steps of 0.1 (two models) or 0.2 (three), 22 and 44 candidates, scoring each by median SDR over tracks; scores are cached per track, so it resumes. A slot gets one method for both modes and each mode its weights; a candidate must beat an earlier, simpler one by more than 0.001 dB. Output: `tuning.json`, a Markdown table (equal-weight baseline, choice, gain) listing the tuning tracks and the lists they were checked against, and the choice as a config snippet that is **not applied**. Candidates are compared with `metrics.sdr_frames`, the plain SDR definition, tested equal to museval's SDR. Also new: `bench/frozen/synthetic_tuning.json` (4 tracks, seeds 201 to 204), `python -m stems_bench.datasets.moisesdb --freeze-tuning` (30 seeded tracks outside the subset), and `ensemble.Blender`, which computes spectrograms once for many candidate blends. 12 tests. Run once with real models (`synthetic-tuning-v1`, commit `0b357e6`, MPS): 120 of 120 jobs, 16 blends tuned, written to `docs/benchmarks/tuning-synthetic.md`. That run proves the tool end to end and nothing else: on sine tones it mostly learns to drop the models that return silence, and nothing from it is in the config.
- [ ] P2.12 Tune blend methods and weights on tracks outside the frozen test lists, then score them on the test lists. **Accept:** the tuning tracks are listed and disjoint from the frozen lists. *Queued (2026-11-22): stages 3 and 4 of `tools/run_real_benchmarks.sh` run `bench tune` on `moisesdb_tuning` and `musdb18hq_train` and write `docs/benchmarks/tuning-moisesdb.md` and `tuning-musdb18hq.md`. Applying what they choose and scoring it on the test lists is done by hand afterwards.*
- [ ] P2.13 Recommend a default model or ensemble per slot from measured gain against time. **Accept:** `docs/benchmarks/recommendation.md` cites the runs; `config/defaults.yaml` is updated and `provisional` set to false. *Waits for P2.10 to P2.12 to finish.*
- [x] P2.14 Measure how much scores move between devices (CPU, MPS, and CUDA when available) for the same model and tracks. **Accept:** the benchmark report states the per-stem SDR spread across devices on the synthetic set and, once available, on real tracks; if it exceeds run-to-run noise, every comparison in a report uses a single device. — `bench run --trial N` repeats jobs under their own keys and `bench compare` sets the differences side by side (`make bench-device-spread`). On the synthetic set, the provisional default model of each slot (6 models, 8 tracks, 150 scores per comparison), runs of 2026-11-22: **running again on the same device changed nothing** (largest difference 0.000 dB on MPS and on CPU: inference is deterministic on both), **moving from MPS to CPU changed one track's score by up to 0.817 dB** (`bs_roformer_sw_6stem` piano, `synth-03`) and moved a model's median by up to 0.221 dB (`mel_roformer_kim_vocals` vocals, 6.108 to 6.329). The per-output table is `docs/benchmarks/device-spread-synthetic.md`. The device effect exceeds run-to-run noise, so scores from different devices are not compared: a run asks for one device and refuses to be resumed on another, a report is built from one run, and every report now says so. The MDX-Net models run on CPU whatever is asked (no MPS in onnxruntime); the device column shows it. No CUDA here. Real tracks are task P2.14b.
- [ ] P2.14b Device spread on real tracks: the same repeat and cross-device runs on MUSDB18-HQ (and on CUDA through Modal if there is a token). **Accept:** `docs/benchmarks/device-spread-musdb18hq.md` generated by `bench compare`. *MPS and CPU runs wait for the benchmark queue to free the machine (two heavy runs at once do not fit in 24 GB). CUDA is **BLOCKED (B4)**.*
- [x] P2.15 Blend modes in the product pipeline: `separate --mode two_model_blend` and `--mode full_ensemble`. **Accept:** on the fixture with weightless models, each mode runs every listed model once, blends per stem with the configured method and weights, and the manifest names the members and weights; `single_best` is unchanged. — `separate --mode two_model_blend | full_ensemble`. `default_assignment` turns each slot's model list, weights and method from config into a `Blend`; `plan_composition` accepts a model id or a `Blend` per stem and still plans one run per model, however many stems and blends it serves; `compose` blends per stem with `stems_worker.ensemble`. Six-stem `other` in a blend mode is the blended four-stem other minus the blended guitar and piano, and a blend that mixes four-stem and six-stem `other` models is refused. Vocal split and de-reverb are blended the same way. `--model SLOT=ID` replaces that slot's blend with one model. The manifest gives `models`, `weights` and `blend_method` for every blended stem and file; a `single_best` manifest is byte-for-byte the shape it was. Cached single-model outputs are reused by blends. 11 tests with weightless models. Not run with real models yet: which blend is worth its time is what P2.10 to P2.13 measure.

## Phase 3: listening layer and the Clef-omni test

- [ ] P3.1 BEATs wrapper. **Accept:** returns 527 class probabilities per 10-second window on the fixture, on this machine.
- [x] P3.2 AudioSet class-to-stem mapping file. **Accept:** committed as data, with a test that every mapped class exists in the label set. — `models/listening/audioset_stem_map.yaml` (version 1) with AudioSet's own 527-class list beside it (`audioset_class_labels.csv`, hash-pinned in the map). 78 classes are mapped: vocals 13, drums 9, bass 2, guitar 6, piano 2, other 46; a stem's score is the largest probability among its classes. The report's core classes are where it says, widened to their neighbours; unpitched percussion that is not a drum kit goes to `other`, as in MoisesDB's own grouping. Every class from "Music" to "Scratching" is either mapped or listed as deliberately unmapped with a reason (14, such as "Keyboard (musical)"). `stems_worker.listening.audioset.load_stem_map()` validates it (unknown class, class in two stems, wrong label file) and gives `stem_scores` by index or by class id, so a tagger's own class order does not matter. Its SHA-256 is exposed for the Clef-omni pre-registration. 10 tests. Done before P3.1, which is blocked on the BEATs file (B10).
- [x] P3.3 LAION-CLAP wrapper with text prompts and a high-band check. **Accept:** on fixture clips with and without content above 8 kHz the high-band signal differs in the right direction. — `stems_worker.listening.clap.ClapTagger` (`laion_clap_music`, 48 kHz, 10 s windows, 512-dimension embeddings). `embed_audio`, `embed_text`, `similarity`, `score_prompts` (zero-shot: softmax over prompts at the model's own temperature) and `high_band`: one minus the cosine similarity between a clip and the same clip low-passed at 8 kHz. On the fixture with the real checkpoint: **0.047 for the mix against 0.0002 for the low-passed mix; 0.098 for the drums stem against 0.0000004 for the bass stem**, and the prompts name the drums and bass stems correctly (0.99 each). The `laion_clap` package would have downgraded numpy and installed a training stack, so its three model files are taken from the pinned 1.1.7 wheel (`ensure_clap_lib`; hash equals PyPI's) with small stubs for the rest. The checkpoint loads with `weights_only=True` and a three-type numpy allowlist, strictly into the model. The text tower is built from RoBERTa-base's config and the tokenizer from its `tokenizer.json`, both pinned to a commit in the registry, so nothing is fetched from a hub at run time. Runs on CPU and MPS (embeddings agree to 2e-7). New pins: transformers 5.24.0, tokenizers 0.23.3, torchlibrosa 0.1.0. The model's mel front end stops at 14 kHz, so "above 8 kHz" means 8 to 14 kHz. 8 fast tests, 4 with the real weights.
- [x] P3.4 `track_features` builder and per-stem signals (presence, bleed estimate, notes). **Accept:** output validates against the `track_features` part of schema (b) and the `metrics` part of schema (c). *Note before starting (2026-11-22, a look at 3 MoisesDB tracks, one 30 s window each, not a measurement):* with plain prompts ("the sound of drums"), CLAP's cosine similarity singles out isolated drums, bass and guitar stems (0.33 to 0.45 against 0.2 or less for the wrong prompts), but is weak for vocals and piano stems (about 0.15, not separated from other prompts) and weak on full mixes (0.02 to 0.24). So CLAP alone is not a presence detector for mixes; `track_features` needs the AudioSet tagger (BEATs, blocked on B10) as planned, with CLAP as the second opinion and the high-band check. Build the feature builder against a tagger interface, with thresholds in config and marked uncalibrated until measured on MoisesDB. — Done: `stems_worker.listening.features` (`build_track_features`, `stem_signals`) reads any tagger that returns 527 AudioSet probabilities per window; schemas transcribed from the report into `common/stems_common/schemas/` and every output is validated against them; BS.1770 loudness in `listening/loudness.py`, checked against the standard's 48 kHz coefficients and its -3.01 LKFS reference tone; class lists for backing vocals, strings, synth, brass, genres and reverb in `models/listening/audioset_feature_classes.yaml`; thresholds in `config/defaults.yaml` under `listening`, `calibrated: false`. With no tagger the hearing fields are `unknown`/null and levels and loudness still hold. **Not yet shown:** how it behaves with a real tagger. Every tagger-dependent test uses a scripted one, because BEATs is still missing (B10); 39 tests.
- [ ] P3.5 Pre-registration file for the Clef-omni test: thresholds A1-A3, B1-B3, C, the 50 track ids, 30-second windows, class-to-stem mapping. **Accept:** committed before any Clef-omni request; its commit hash is embedded in the test's log header. *Unblocked 2026-11-22: the 50 track ids are frozen in `bench/frozen/moisesdb_subset.json`.*
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
