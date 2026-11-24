# Benchmark: moisesdb_subset

Generated from the result files of one run. Every figure here is copied from those
files or is a median or count of values in them; `report.json` next to the run lists
each file with its hash.

- Run: `moisesdb-v1`, state `done`, started 2026-11-23T17:01:23+00:00
- Command: `bench run --dataset moisesdb_subset --run-id moisesdb-v1 --ensembles --quiet`
- Code: commit `0e76c68931e14c11ea904587e899a5a5ca4df72c`
- Results computed on commit(s): `0e76c68931e14c11ea904587e899a5a5ca4df72c`, `5d811ee4a2a281ce4c23ff6db12d3d5dcae023fd`, `a7734623283645bf2bcc0479300eb816e8b33f42`
- Protocol version: 1
- Frozen track list: sha256 `4a5dfd20b191282dc68ffd6d2f24e3f8b80ca16ab913f5d308820267ebdc2035`, 50 tracks in this run
- Device asked for: `mps`; chunk overlap 0.25; batch size 1
- Packages: torch 2.15.0, numpy 2.5.5, scipy 1.18.1, museval 0.4.1, onnxruntime 1.32.0, demucs 4.1.0
- Jobs (one model on one track): 1699 computed in this run, 101 taken from earlier runs, 0 failed, 250 not applicable
- Audio: 50 tracks with results, 11300.5 s in total, median 228.6 s per track

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
| `mel_roformer_kim_vocals` | 10.82 | 23.47 | 10.97 | 49 of 50 | 48.25 | 12.65 | mps | -0.4 |  |
| `bs_roformer_viperx_vocals` | 10.78 | 23.92 | 10.69 | 49 of 50 | 115.07 | 30.16 | mps | -0.4 |  |
| `bs_roformer_sw_6stem` | 10.55 | 23.31 | 10.93 | 49 of 50 | 66.17 | 17.14 | mps | -0.4 |  |
| `bs_polarformer_vocals` | 10.22 | 23.22 | 10.47 | 49 of 50 | 108.26 | 27.64 | mps | -0.4 |  |
| `scnet_xl_ihf` | 9.97 | 22.28 | 10.28 | 49 of 50 | 75.23 | 20.08 | mps | -0.6 |  |
| `scnet_large` | 8.87 | 19.65 | 8.78 | 49 of 50 | 28.04 | 8.09 | mps | -0.7 |  |
| `htdemucs_ft` | 8.56 | 18.98 | 8.88 | 49 of 50 | 29.18 | 7.72 | mps | -0.5 |  |
| `htdemucs_6s` | 8.33 | 18.42 | 8.80 | 49 of 50 | 10.88 | 2.65 | mps | -0.5 |  |

## drums

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 12.95 | 24.29 | 13.44 | 48 of 50 | 66.17 | 17.14 | mps | -0.2 |  |
| `scnet_xl_ihf` | 11.50 | 22.12 | 12.46 | 48 of 50 | 75.23 | 20.08 | mps | -0.2 |  |
| `htdemucs_ft` | 10.51 | 21.74 | 11.32 | 48 of 50 | 29.18 | 7.72 | mps | -0.4 |  |
| `scnet_large` | 10.21 | 22.22 | 11.35 | 48 of 50 | 28.04 | 8.09 | mps | -1.9 |  |
| `htdemucs_6s` | 9.76 | 20.52 | 10.89 | 48 of 50 | 10.88 | 2.65 | mps | -0.4 |  |

## bass

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 15.09 | 29.28 | 13.15 | 48 of 50 | 66.17 | 17.14 | mps | -16.7 |  |
| `htdemucs_ft` | 13.19 | 27.76 | 10.92 | 48 of 50 | 29.18 | 7.72 | mps | -10.9 |  |
| `scnet_xl_ihf` | 12.99 | 27.54 | 11.91 | 48 of 50 | 75.23 | 20.08 | mps | 6.9 |  |
| `htdemucs_6s` | 12.78 | 26.68 | 10.54 | 48 of 50 | 10.88 | 2.65 | mps | -13.0 |  |
| `scnet_large` | 11.85 | 25.23 | 10.44 | 48 of 50 | 28.04 | 8.09 | mps | 2.3 |  |

## guitar

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 9.19 | 19.22 | 8.65 | 45 of 50 | 66.17 | 17.14 | mps | -4.6 |  |
| `mel_roformer_guitar_becruily` | 5.44 | 11.09 | 5.54 | 45 of 50 | 24.74 | 6.40 | mps | -3.6 |  |
| `htdemucs_6s` | 5.17 | 11.23 | 5.21 | 45 of 50 | 10.88 | 2.65 | mps | -4.5 |  |

## piano

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 5.80 | 19.15 | 4.77 | 20 of 50 | 66.17 | 17.14 | mps | -11.1 |  |
| `htdemucs_6s` | 0.23 | 4.74 | 0.05 | 20 of 50 | 10.88 | 2.65 | mps | -7.8 | output is 63% silence |

## other

Six-stem other: what is left without vocals, drums, bass, guitar and piano.

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 2.97 | 13.14 | 2.86 | 33 of 50 | 66.17 | 17.14 | mps | -11.5 |  |
| `htdemucs_6s` | 0.12 | 6.43 | 0.12 | 33 of 50 | 10.88 | 2.65 | mps | -17.1 |  |

## other_4stem

Four-stem other: guitar, piano and everything else that is not vocals, drums or bass.

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 10.33 | 20.75 | 10.01 | 50 of 50 | 66.17 | 17.14 | mps | -5.7 | sum of its guitar, piano and other |
| `scnet_xl_ihf` | 8.88 | 18.29 | 9.00 | 50 of 50 | 75.23 | 20.08 | mps | -4.0 |  |
| `htdemucs_ft` | 7.73 | 15.36 | 7.72 | 50 of 50 | 29.18 | 7.72 | mps | -3.8 |  |
| `scnet_large` | 7.55 | 15.49 | 7.47 | 50 of 50 | 28.04 | 8.09 | mps | -2.0 |  |
| `htdemucs_6s` | 7.25 | 15.32 | 7.08 | 50 of 50 | 10.88 | 2.65 | mps | -6.5 | sum of its guitar, piano and other |

## instrumental

Everything except the vocals.

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_viperx_vocals` | 19.15 | 28.72 | 18.96 | 49 of 50 | 115.07 | 30.16 | mps | -0.2 |  |
| `bs_polarformer_vocals` | 18.92 | 26.86 | 18.78 | 49 of 50 | 108.26 | 27.64 | mps | -0.2 |  |
| `mel_roformer_kim_vocals` | 18.72 | 27.36 | 19.04 | 49 of 50 | 48.25 | 12.65 | mps | -0.2 |  |

## lead_vocals

The model is fed the reference vocals stem.

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `mel_roformer_karaoke_aufr33_viperx` | 10.53 | 17.69 | 11.37 | 26 of 50 | 47.79 | 12.54 | mps | 0.0 |  |
| `uvr_mdxnet_kara_2` | 9.59 | 13.97 | 12.69 | 26 of 50 | 20.22 | 5.41 | cpu | -0.1 |  |
| `uvr_vr_5hp_karaoke` | 8.94 | 15.76 | 10.74 | 26 of 50 | 16.50 | 4.45 | mps | -0.4 |  |

## backing_vocals

The model is fed the reference vocals stem.

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `mel_roformer_karaoke_aufr33_viperx` | 4.28 | 11.90 | 3.58 | 26 of 50 | 47.79 | 12.54 | mps | -1.4 | output is 63% silence |
| `uvr_vr_5hp_karaoke` | 2.24 | 5.81 | 3.11 | 26 of 50 | 16.50 | 4.45 | mps | -1.3 | output is 57% silence |
| `uvr_mdxnet_kara_2` | 0.15 | 7.20 | 0.89 | 26 of 50 | 20.22 | 5.41 | cpu | -1.0 | output is 60% silence |

## Single best, two-model blend, full ensemble

Each slot of `config/defaults.yaml` in its three modes, on the same tracks. A blend's
time is the sum of the times of every model it needs, as if that stem alone were
asked for. "vs single best" is the blend's median SDR minus the single best's.

### vocals

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| single best | vocals | vocals | 10.82 | 23.47 | 10.97 |  | 49 of 50 | 48.27 | 12.65 | mps | `mel_roformer_kim_vocals` (1.00) |  |
| two-model blend | vocals | vocals | 10.99 | 23.61 | 11.24 | +0.18 | 49 of 50 | 163.49 | 42.81 | mps | `mel_roformer_kim_vocals` (0.50), `bs_roformer_viperx_vocals` (0.50) | waveform |
| full ensemble | vocals | vocals | 10.97 | 23.57 | 11.29 | +0.16 | 49 of 50 | 275.05 | 70.52 | mps | `mel_roformer_kim_vocals` (0.33), `bs_roformer_viperx_vocals` (0.33), `bs_polarformer_vocals` (0.33) | waveform |

### drums

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| single best | drums | drums | 11.50 | 22.12 | 12.46 |  | 48 of 50 | 75.24 | 20.08 | mps | `scnet_xl_ihf` (1.00) |  |
| two-model blend | drums | drums | 11.52 | 22.20 | 12.55 | +0.01 | 48 of 50 | 106.94 | 27.76 | mps | `scnet_xl_ihf` (0.50), `htdemucs_ft` (0.50) | waveform |
| full ensemble | drums | drums | 11.38 | 22.45 | 12.54 | -0.12 | 48 of 50 | 138.04 | 36.05 | mps | `scnet_xl_ihf` (0.33), `scnet_large` (0.33), `htdemucs_ft` (0.33) | waveform |

### bass

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| single best | bass | bass | 13.19 | 27.76 | 10.92 |  | 48 of 50 | 29.19 | 7.72 | mps | `htdemucs_ft` (1.00) |  |
| two-model blend | bass | bass | 14.71 | 28.40 | 12.85 | +1.52 | 48 of 50 | 95.10 | 24.81 | mps | `htdemucs_ft` (0.50), `bs_roformer_sw_6stem` (0.50) | waveform |
| full ensemble | bass | bass | 14.72 | 28.23 | 12.97 | +1.53 | 48 of 50 | 174.13 | 45.14 | mps | `htdemucs_ft` (0.33), `bs_roformer_sw_6stem` (0.33), `scnet_xl_ihf` (0.33) | waveform |

### guitar

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| single best | guitar | guitar | 9.19 | 19.22 | 8.65 |  | 45 of 50 | 66.18 | 17.15 | mps | `bs_roformer_sw_6stem` (1.00) |  |
| two-model blend | guitar | guitar | 7.98 | 13.87 | 8.35 | -1.21 | 45 of 50 | 90.72 | 23.59 | mps | `bs_roformer_sw_6stem` (0.50), `mel_roformer_guitar_becruily` (0.50) | waveform |
| full ensemble | guitar | guitar | 7.23 | 13.22 | 7.84 | -1.96 | 45 of 50 | 102.22 | 26.65 | mps | `bs_roformer_sw_6stem` (0.33), `mel_roformer_guitar_becruily` (0.33), `htdemucs_6s` (0.33) | waveform |

### piano

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| single best | piano | piano | 5.80 | 19.15 | 4.77 |  | 20 of 50 | 66.18 | 17.15 | mps | `bs_roformer_sw_6stem` (1.00) |  |
| two-model blend | piano | piano | 3.60 | 12.82 | 4.02 | -2.19 | 20 of 50 | 77.59 | 20.15 | mps | `bs_roformer_sw_6stem` (0.50), `htdemucs_6s` (0.50) | waveform |
| full ensemble | piano | piano | 3.60 | 12.82 | 4.02 | -2.19 | 20 of 50 | 77.59 | 20.15 | mps | `bs_roformer_sw_6stem` (0.50), `htdemucs_6s` (0.50) | waveform |

### other

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| single best | other | other_4stem | 8.88 | 18.29 | 9.00 |  | 50 of 50 | 75.25 | 20.08 | mps | `scnet_xl_ihf` (1.00) |  |
| two-model blend | other | other_4stem | 8.41 | 16.65 | 8.73 | -0.46 | 50 of 50 | 105.07 | 28.35 | mps | `scnet_xl_ihf` (0.50), `scnet_large` (0.50) | waveform |
| full ensemble | other | other_4stem | 8.71 | 16.64 | 9.22 | -0.17 | 50 of 50 | 138.05 | 36.05 | mps | `scnet_xl_ihf` (0.33), `scnet_large` (0.33), `htdemucs_ft` (0.33) | waveform |

### vocal_split

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| single best | backing_vocals | backing_vocals | 4.28 | 11.90 | 3.58 |  | 26 of 50 | 47.81 | 12.55 | mps | `mel_roformer_karaoke_aufr33_viperx` (1.00) |  |
| two-model blend | backing_vocals | backing_vocals | 3.40 | 7.78 | 4.33 | -0.88 | 26 of 50 | 64.50 | 16.99 | mps | `mel_roformer_karaoke_aufr33_viperx` (0.50), `uvr_vr_5hp_karaoke` (0.50) | waveform |
| full ensemble | backing_vocals | backing_vocals | 1.39 | 7.93 | 3.28 | -2.88 | 26 of 50 | 84.82 | 22.38 | cpu+mps | `mel_roformer_karaoke_aufr33_viperx` (0.33), `uvr_vr_5hp_karaoke` (0.33), `uvr_mdxnet_kara_2` (0.33) | waveform |
| single best | lead_vocals | lead_vocals | 10.53 | 17.69 | 11.37 |  | 26 of 50 | 47.81 | 12.55 | mps | `mel_roformer_karaoke_aufr33_viperx` (1.00) |  |
| two-model blend | lead_vocals | lead_vocals | 10.03 | 16.98 | 11.61 | -0.50 | 26 of 50 | 64.50 | 16.99 | mps | `mel_roformer_karaoke_aufr33_viperx` (0.50), `uvr_vr_5hp_karaoke` (0.50) | waveform |
| full ensemble | lead_vocals | lead_vocals | 10.08 | 16.20 | 12.58 | -0.45 | 26 of 50 | 84.82 | 22.38 | cpu+mps | `mel_roformer_karaoke_aufr33_viperx` (0.33), `uvr_vr_5hp_karaoke` (0.33), `uvr_mdxnet_kara_2` (0.33) | waveform |

### other_composed

The four-stem blend with the same mode's guitar and piano taken out: what the pipeline delivers as the six-stem other.

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| single best | other | other | 1.99 | 8.33 | 1.59 |  | 33 of 50 | 143.87 | 37.46 | mps | `scnet_xl_ihf` (1.00), minus guitar: `bs_roformer_sw_6stem`, minus piano: `bs_roformer_sw_6stem` |  |
| two-model blend | other | other | 1.28 | 4.40 | 0.73 | -0.71 | 33 of 50 | 211.35 | 55.23 | mps | `scnet_xl_ihf` (0.50), `scnet_large` (0.50), minus guitar: `bs_roformer_sw_6stem`, minus guitar: `mel_roformer_guitar_becruily`, minus piano: `bs_roformer_sw_6stem`, minus piano: `htdemucs_6s` | waveform |
| full ensemble | other | other | 1.47 | 5.93 | 1.06 | -0.51 | 33 of 50 | 240.39 | 62.93 | mps | `scnet_xl_ihf` (0.33), `scnet_large` (0.33), `htdemucs_ft` (0.33), minus guitar: `bs_roformer_sw_6stem`, minus guitar: `mel_roformer_guitar_becruily`, minus guitar: `htdemucs_6s`, minus piano: `bs_roformer_sw_6stem`, minus piano: `htdemucs_6s` | waveform |

## Not applicable to this dataset

- `mel_roformer_dereverb_anvuew`: needs the 'vocals_wet' input, which moisesdb_subset does not have
- `uvr_vr_deecho_dereverb`: needs the 'vocals_wet' input, which moisesdb_subset does not have
- `dereverb:single_best`: needs the 'vocals_wet' input, which moisesdb_subset does not have
- `dereverb:two_model_blend`: needs the 'vocals_wet' input, which moisesdb_subset does not have
- `dereverb:full_ensemble`: needs the 'vocals_wet' input, which moisesdb_subset does not have

