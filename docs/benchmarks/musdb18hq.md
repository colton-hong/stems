# Benchmark: musdb18hq_test

Generated from the result files of one run. Every figure here is copied from those
files or is a median or count of values in them; `report.json` next to the run lists
each file with its hash.

- Run: `musdb18hq-v1`, state `done`, started 2026-11-23T04:28:44+00:00
- Command: `bench run --dataset musdb18hq_test --run-id musdb18hq-v1 --ensembles --quiet`
- Code: commit `3cb7b649262645875713f632af061bb262768e84` with uncommitted changes
- Results computed on commit(s): `3cb7b649262645875713f632af061bb262768e84` (some with uncommitted changes)
- Protocol version: 1
- Frozen track list: sha256 `8f1c292a931ab44532173f8d613cef3697259935d9c36529a3efe9e179a6584f`, 50 tracks in this run
- Device asked for: `mps`; chunk overlap 0.25; batch size 1
- Packages: torch 2.15.0, numpy 2.5.5, scipy 1.18.1, museval 0.4.1, onnxruntime 1.32.0, demucs 4.1.0
- Jobs (one model on one track): 1000 computed in this run, 0 taken from earlier runs, 0 failed, 1050 not applicable
- Audio: 50 tracks with results, 12471.0 s in total, median 246.9 s per track

## How to read the tables

- Scores are in dB, higher is better. A model's score for a stem is the median over
  tracks of each track's median over 1-second windows (museval BSS Eval v4).
- Every stem is scored as the pair (target, input minus target), so figures are
  comparable between models with different numbers of outputs.
- "tracks" is how many tracks the score rests on, out of those the model ran on. A
  track where the stem is absent is not scored.
- Time is for the whole model on one track, all of its outputs together, on the
  device shown. It does not include loading the model.
- Every job here asked for the device `mps`. The same model scores
  differently on another device (see the `device-spread` report beside this one),
  so do not compare these figures with a report made on a different device.
- "high band" is the estimate's energy at and above 8 kHz relative to the
  reference's, in dB: 0 is a match, negative means lost top end.

## vocals

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 12.30 | 25.94 | 12.33 | 50 of 50 | 69.75 | 17.04 | mps | -0.5 |  |
| `bs_roformer_viperx_vocals` | 12.29 | 27.13 | 12.25 | 50 of 50 | 130.48 | 31.61 | mps | -0.5 |  |
| `mel_roformer_kim_vocals` | 11.95 | 26.09 | 11.82 | 50 of 50 | 52.56 | 12.72 | mps | -0.4 |  |
| `bs_polarformer_vocals` | 11.86 | 25.85 | 11.82 | 50 of 50 | 113.25 | 27.88 | mps | -0.5 |  |
| `scnet_xl_ihf` | 11.45 | 24.34 | 11.49 | 50 of 50 | 80.06 | 19.76 | mps | -0.7 |  |
| `scnet_large` | 10.29 | 21.07 | 10.23 | 50 of 50 | 32.90 | 7.77 | mps | -1.0 |  |
| `htdemucs_ft` | 9.25 | 19.98 | 8.90 | 50 of 50 | 31.93 | 7.74 | mps | -0.5 |  |
| `htdemucs_6s` | 8.91 | 18.88 | 8.79 | 50 of 50 | 11.09 | 2.62 | mps | -0.5 |  |

## drums

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 13.35 | 25.90 | 14.05 | 50 of 50 | 69.75 | 17.04 | mps | -0.3 |  |
| `scnet_xl_ihf` | 11.92 | 23.22 | 12.50 | 50 of 50 | 80.06 | 19.76 | mps | -0.3 |  |
| `scnet_large` | 10.73 | 23.23 | 11.86 | 50 of 50 | 32.90 | 7.77 | mps | -1.5 |  |
| `htdemucs_ft` | 10.28 | 22.24 | 10.97 | 50 of 50 | 31.93 | 7.74 | mps | -0.5 |  |
| `htdemucs_6s` | 9.62 | 20.35 | 10.12 | 50 of 50 | 11.09 | 2.62 | mps | -0.5 |  |

## bass

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 12.47 | 24.35 | 8.63 | 50 of 50 | 69.75 | 17.04 | mps | -13.9 |  |
| `scnet_xl_ihf` | 10.60 | 20.98 | 4.87 | 50 of 50 | 80.06 | 19.76 | mps | 8.5 |  |
| `htdemucs_ft` | 9.95 | 22.62 | 6.75 | 50 of 50 | 31.93 | 7.74 | mps | -4.3 |  |
| `scnet_large` | 9.93 | 20.89 | 4.79 | 50 of 50 | 32.90 | 7.77 | mps | 3.9 |  |
| `htdemucs_6s` | 9.02 | 19.78 | 4.66 | 50 of 50 | 11.09 | 2.62 | mps | -6.0 |  |

## other_4stem

Four-stem other: guitar, piano and everything else that is not vocals, drums or bass.

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 10.42 | 20.47 | 10.31 | 50 of 50 | 69.75 | 17.04 | mps | -3.1 | sum of its guitar, piano and other |
| `scnet_xl_ihf` | 8.79 | 17.81 | 8.32 | 50 of 50 | 80.06 | 19.76 | mps | -3.3 |  |
| `scnet_large` | 8.04 | 14.10 | 8.23 | 50 of 50 | 32.90 | 7.77 | mps | 0.2 |  |
| `htdemucs_ft` | 7.33 | 13.88 | 7.44 | 50 of 50 | 31.93 | 7.74 | mps | -2.5 |  |
| `htdemucs_6s` | 6.32 | 13.65 | 6.48 | 50 of 50 | 11.09 | 2.62 | mps | -3.7 | sum of its guitar, piano and other |

## instrumental

Everything except the vocals.

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_viperx_vocals` | 21.21 | 29.66 | 21.62 | 50 of 50 | 130.48 | 31.61 | mps | -0.1 |  |
| `mel_roformer_kim_vocals` | 20.95 | 29.08 | 21.33 | 50 of 50 | 52.56 | 12.72 | mps | -0.1 |  |
| `bs_polarformer_vocals` | 20.94 | 29.55 | 21.22 | 50 of 50 | 113.25 | 27.88 | mps | -0.2 |  |

## Single best, two-model blend, full ensemble

Each slot of `config/defaults.yaml` in its three modes, on the same tracks. A blend's
time is the sum of the times of every model it needs, as if that stem alone were
asked for. "vs single best" is the blend's median SDR minus the single best's.

### vocals

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| single best | vocals | vocals | 11.95 | 26.09 | 11.82 |  | 50 of 50 | 52.57 | 12.73 | mps | `mel_roformer_kim_vocals` (1.00) |  |
| two-model blend | vocals | vocals | 12.56 | 26.64 | 12.39 | +0.61 | 50 of 50 | 181.66 | 44.19 | mps | `mel_roformer_kim_vocals` (0.50), `bs_roformer_viperx_vocals` (0.50) | waveform |
| full ensemble | vocals | vocals | 12.58 | 26.38 | 12.49 | +0.63 | 50 of 50 | 297.58 | 72.23 | mps | `mel_roformer_kim_vocals` (0.33), `bs_roformer_viperx_vocals` (0.33), `bs_polarformer_vocals` (0.33) | waveform |

### drums

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| single best | drums | drums | 11.92 | 23.22 | 12.50 |  | 50 of 50 | 80.07 | 19.76 | mps | `scnet_xl_ihf` (1.00) |  |
| two-model blend | drums | drums | 11.84 | 22.78 | 12.61 | -0.08 | 50 of 50 | 111.32 | 27.44 | mps | `scnet_xl_ihf` (0.50), `htdemucs_ft` (0.50) | waveform |
| full ensemble | drums | drums | 11.37 | 23.01 | 12.71 | -0.55 | 50 of 50 | 144.73 | 35.51 | mps | `scnet_xl_ihf` (0.33), `scnet_large` (0.33), `htdemucs_ft` (0.33) | waveform |

### bass

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| single best | bass | bass | 9.95 | 22.62 | 6.75 |  | 50 of 50 | 31.94 | 7.74 | mps | `htdemucs_ft` (1.00) |  |
| two-model blend | bass | bass | 11.87 | 23.51 | 8.21 | +1.93 | 50 of 50 | 101.36 | 24.76 | mps | `htdemucs_ft` (0.50), `bs_roformer_sw_6stem` (0.50) | waveform |
| full ensemble | bass | bass | 12.08 | 22.89 | 8.25 | +2.13 | 50 of 50 | 181.29 | 44.42 | mps | `htdemucs_ft` (0.33), `bs_roformer_sw_6stem` (0.33), `scnet_xl_ihf` (0.33) | waveform |

### other

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| single best | other | other_4stem | 8.79 | 17.81 | 8.32 |  | 50 of 50 | 80.07 | 19.76 | mps | `scnet_xl_ihf` (1.00) |  |
| two-model blend | other | other_4stem | 8.69 | 15.65 | 8.95 | -0.10 | 50 of 50 | 112.77 | 27.65 | mps | `scnet_xl_ihf` (0.50), `scnet_large` (0.50) | waveform |
| full ensemble | other | other_4stem | 8.78 | 15.23 | 9.08 | +0.00 | 50 of 50 | 144.73 | 35.51 | mps | `scnet_xl_ihf` (0.33), `scnet_large` (0.33), `htdemucs_ft` (0.33) | waveform |

## Not applicable to this dataset

- `mel_roformer_guitar_becruily`: musdb18hq_test has no reference for any of its outputs (['guitar'])
- `mel_roformer_karaoke_aufr33_viperx`: needs the 'vocals' input, which musdb18hq_test does not have
- `uvr_vr_5hp_karaoke`: needs the 'vocals' input, which musdb18hq_test does not have
- `uvr_mdxnet_kara_2`: needs the 'vocals' input, which musdb18hq_test does not have
- `mel_roformer_dereverb_anvuew`: needs the 'vocals_wet' input, which musdb18hq_test does not have
- `uvr_vr_deecho_dereverb`: needs the 'vocals_wet' input, which musdb18hq_test does not have
- `guitar:single_best`: musdb18hq_test has no guitar reference
- `guitar:two_model_blend`: musdb18hq_test has no guitar reference
- `guitar:full_ensemble`: musdb18hq_test has no guitar reference
- `piano:single_best`: musdb18hq_test has no piano reference
- `piano:two_model_blend`: musdb18hq_test has no piano reference
- `piano:full_ensemble`: musdb18hq_test has no piano reference
- `vocal_split:single_best`: needs the 'vocals' input, which musdb18hq_test does not have
- `vocal_split:two_model_blend`: needs the 'vocals' input, which musdb18hq_test does not have
- `vocal_split:full_ensemble`: needs the 'vocals' input, which musdb18hq_test does not have
- `dereverb:single_best`: needs the 'vocals_wet' input, which musdb18hq_test does not have
- `dereverb:two_model_blend`: needs the 'vocals_wet' input, which musdb18hq_test does not have
- `dereverb:full_ensemble`: needs the 'vocals_wet' input, which musdb18hq_test does not have
- `other_composed:single_best`: musdb18hq_test has no other reference
- `other_composed:two_model_blend`: musdb18hq_test has no other reference
- `other_composed:full_ensemble`: musdb18hq_test has no other reference

