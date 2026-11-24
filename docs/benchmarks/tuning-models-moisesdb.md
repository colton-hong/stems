# Benchmark: moisesdb_tuning

Generated from the result files of one run. Every figure here is copied from those
files or is a median or count of values in them; `report.json` next to the run lists
each file with its hash.

- Run: `moisesdb-tuning-v1`, state `done`, started 2026-11-24T04:53:13+00:00
- Command: `bench tune --dataset moisesdb_tuning --run-id moisesdb-tuning-v1 --out docs/benchmarks/tuning-moisesdb.md --quiet`
- Code: commit `8df9a85f768497859e93d50e63652794e67537ef`
- Results computed on commit(s): `8df9a85f768497859e93d50e63652794e67537ef`
- Protocol version: 1
- Frozen track list: sha256 `2e47e5d120e505fdde5ece16143adfe2057f554be971dbf3fbc015a30848eab2`, 30 tracks in this run
- Device asked for: `mps`; chunk overlap 0.25; batch size 1
- Packages: torch 2.15.0, numpy 2.5.5, scipy 1.18.1, museval 0.4.1, onnxruntime 1.32.0, demucs 4.1.0
- Jobs (one model on one track): 780 computed in this run, 0 taken from earlier runs, 0 failed, 120 not applicable
- Audio: 30 tracks with results, 6398.2 s in total, median 213.6 s per track

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
| `bs_roformer_sw_6stem` | 11.48 | 25.42 | 11.86 | 30 of 30 | 64.83 | 18.25 | mps | -0.6 |  |
| `bs_polarformer_vocals` | 11.20 | 24.82 | 11.41 | 30 of 30 | 101.14 | 29.97 | mps | -0.6 |  |
| `bs_roformer_viperx_vocals` | 11.15 | 25.33 | 11.55 | 30 of 30 | 109.80 | 29.88 | mps | -0.5 |  |
| `mel_roformer_kim_vocals` | 11.09 | 24.24 | 11.36 | 30 of 30 | 45.70 | 12.51 | mps | -0.6 |  |
| `scnet_xl_ihf` | 11.02 | 23.38 | 11.27 | 30 of 30 | 90.47 | 25.25 | mps | -0.9 |  |
| `scnet_large` | 9.29 | 19.60 | 9.71 | 30 of 30 | 25.99 | 7.83 | mps | -1.0 |  |
| `htdemucs_ft` | 8.78 | 20.20 | 9.10 | 30 of 30 | 29.68 | 8.24 | mps | -0.6 |  |
| `htdemucs_6s` | 8.65 | 19.36 | 9.24 | 30 of 30 | 11.48 | 3.77 | mps | -0.7 |  |

## drums

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 12.14 | 24.56 | 13.01 | 30 of 30 | 64.83 | 18.25 | mps | -0.1 |  |
| `scnet_xl_ihf` | 11.26 | 20.79 | 12.14 | 30 of 30 | 90.47 | 25.25 | mps | -0.1 |  |
| `htdemucs_ft` | 10.57 | 21.93 | 11.86 | 30 of 30 | 29.68 | 8.24 | mps | -0.2 |  |
| `scnet_large` | 10.46 | 21.78 | 11.41 | 30 of 30 | 25.99 | 7.83 | mps | -1.2 |  |
| `htdemucs_6s` | 10.29 | 20.74 | 11.52 | 30 of 30 | 11.48 | 3.77 | mps | -0.1 |  |

## bass

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 15.64 | 28.70 | 13.65 | 30 of 30 | 64.83 | 18.25 | mps | -17.1 |  |
| `scnet_xl_ihf` | 12.89 | 26.70 | 12.35 | 30 of 30 | 90.47 | 25.25 | mps | 7.6 |  |
| `htdemucs_ft` | 12.62 | 26.77 | 11.82 | 30 of 30 | 29.68 | 8.24 | mps | -6.9 |  |
| `htdemucs_6s` | 12.33 | 25.75 | 11.54 | 30 of 30 | 11.48 | 3.77 | mps | -6.5 |  |
| `scnet_large` | 11.58 | 24.37 | 10.53 | 30 of 30 | 25.99 | 7.83 | mps | 5.1 |  |

## guitar

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 7.85 | 18.14 | 7.71 | 27 of 30 | 64.83 | 18.25 | mps | -3.2 |  |
| `mel_roformer_guitar_becruily` | 3.59 | 7.18 | 4.06 | 27 of 30 | 23.71 | 6.54 | mps | -2.7 |  |
| `htdemucs_6s` | 3.27 | 9.40 | 3.75 | 27 of 30 | 11.48 | 3.77 | mps | -4.0 |  |

## piano

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 8.63 | 22.61 | 8.01 | 15 of 30 | 64.83 | 18.25 | mps | -18.1 |  |
| `htdemucs_6s` | 1.97 | 8.27 | 1.06 | 15 of 30 | 11.48 | 3.77 | mps | -4.9 |  |

## other

Six-stem other: what is left without vocals, drums, bass, guitar and piano.

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 2.61 | 15.75 | 1.52 | 27 of 30 | 64.83 | 18.25 | mps | -12.3 |  |
| `htdemucs_6s` | 0.01 | 6.23 | 0.00 | 27 of 30 | 11.48 | 3.77 | mps | -20.8 | output is 57% silence |

## other_4stem

Four-stem other: guitar, piano and everything else that is not vocals, drums or bass.

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 10.17 | 21.23 | 10.00 | 30 of 30 | 64.83 | 18.25 | mps | -5.5 | sum of its guitar, piano and other |
| `scnet_xl_ihf` | 9.19 | 19.35 | 9.49 | 30 of 30 | 90.47 | 25.25 | mps | -5.5 |  |
| `scnet_large` | 7.50 | 16.81 | 7.55 | 30 of 30 | 25.99 | 7.83 | mps | -4.0 |  |
| `htdemucs_ft` | 7.49 | 15.89 | 7.27 | 30 of 30 | 29.68 | 8.24 | mps | -4.9 |  |
| `htdemucs_6s` | 6.28 | 15.06 | 6.20 | 30 of 30 | 11.48 | 3.77 | mps | -7.0 | sum of its guitar, piano and other |

## instrumental

Everything except the vocals.

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_viperx_vocals` | 18.86 | 28.87 | 19.40 | 30 of 30 | 109.80 | 29.88 | mps | -0.2 |  |
| `mel_roformer_kim_vocals` | 18.75 | 27.58 | 19.40 | 30 of 30 | 45.70 | 12.51 | mps | -0.1 |  |
| `bs_polarformer_vocals` | 18.71 | 27.32 | 19.26 | 30 of 30 | 101.14 | 29.97 | mps | -0.2 |  |

## lead_vocals

The model is fed the reference vocals stem.

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `mel_roformer_karaoke_aufr33_viperx` | 12.32 | 18.93 | 13.41 | 14 of 30 | 46.85 | 12.76 | mps | 0.0 |  |
| `uvr_mdxnet_kara_2` | 10.99 | 16.31 | 12.09 | 14 of 30 | 20.32 | 5.79 | cpu | -0.1 |  |
| `uvr_vr_5hp_karaoke` | 10.06 | 17.54 | 11.12 | 14 of 30 | 16.58 | 4.48 | mps | -0.3 |  |

## backing_vocals

The model is fed the reference vocals stem.

| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `mel_roformer_karaoke_aufr33_viperx` | 4.92 | 7.81 | 7.30 | 14 of 30 | 46.85 | 12.76 | mps | -0.5 | output is 68% silence |
| `uvr_vr_5hp_karaoke` | 1.09 | 4.70 | 5.48 | 14 of 30 | 16.58 | 4.48 | mps | -0.4 | output is 60% silence |
| `uvr_mdxnet_kara_2` | 0.84 | 9.39 | 2.00 | 14 of 30 | 20.32 | 5.79 | cpu | -0.5 | output is 64% silence |

## Single best, two-model blend, full ensemble

Each slot of `config/defaults.yaml` in its three modes, on the same tracks. A blend's
time is the sum of the times of every model it needs, as if that stem alone were
asked for. "vs single best" is the blend's median SDR minus the single best's.

### vocals

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| two-model blend | vocals | vocals | 11.38 | 24.88 | 11.88 | n/a | 30 of 30 | 155.56 | 42.44 | mps | `mel_roformer_kim_vocals` (0.50), `bs_roformer_viperx_vocals` (0.50) | waveform |
| full ensemble | vocals | vocals | 11.51 | 24.91 | 11.97 | n/a | 30 of 30 | 260.36 | 72.23 | mps | `mel_roformer_kim_vocals` (0.33), `bs_roformer_viperx_vocals` (0.33), `bs_polarformer_vocals` (0.33) | waveform |

### drums

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| two-model blend | drums | drums | 11.26 | 21.03 | 12.56 | n/a | 30 of 30 | 123.38 | 33.94 | mps | `scnet_xl_ihf` (0.50), `htdemucs_ft` (0.50) | waveform |
| full ensemble | drums | drums | 11.22 | 21.36 | 12.53 | n/a | 30 of 30 | 150.10 | 42.26 | mps | `scnet_xl_ihf` (0.33), `scnet_large` (0.33), `htdemucs_ft` (0.33) | waveform |

### bass

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| two-model blend | bass | bass | 14.73 | 27.72 | 13.58 | n/a | 30 of 30 | 94.31 | 26.81 | mps | `htdemucs_ft` (0.50), `bs_roformer_sw_6stem` (0.50) | waveform |
| full ensemble | bass | bass | 13.77 | 27.38 | 13.66 | n/a | 30 of 30 | 189.99 | 52.05 | mps | `htdemucs_ft` (0.33), `bs_roformer_sw_6stem` (0.33), `scnet_xl_ihf` (0.33) | waveform |

### guitar

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| two-model blend | guitar | guitar | 6.36 | 11.96 | 6.77 | n/a | 27 of 30 | 88.02 | 24.72 | mps | `bs_roformer_sw_6stem` (0.50), `mel_roformer_guitar_becruily` (0.50) | waveform |
| full ensemble | guitar | guitar | 5.68 | 11.02 | 6.17 | n/a | 27 of 30 | 101.19 | 28.39 | mps | `bs_roformer_sw_6stem` (0.33), `mel_roformer_guitar_becruily` (0.33), `htdemucs_6s` (0.33) | waveform |

### piano

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| two-model blend | piano | piano | 5.70 | 13.71 | 5.84 | n/a | 15 of 30 | 77.62 | 21.88 | mps | `bs_roformer_sw_6stem` (0.50), `htdemucs_6s` (0.50) | waveform |
| full ensemble | piano | piano | 5.70 | 13.71 | 5.84 | n/a | 15 of 30 | 77.63 | 21.87 | mps | `bs_roformer_sw_6stem` (0.50), `htdemucs_6s` (0.50) | waveform |

### other

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| two-model blend | other | other_4stem | 8.47 | 17.91 | 8.83 | n/a | 30 of 30 | 119.01 | 33.17 | mps | `scnet_xl_ihf` (0.50), `scnet_large` (0.50) | waveform |
| full ensemble | other | other_4stem | 8.51 | 17.49 | 9.00 | n/a | 30 of 30 | 150.10 | 42.24 | mps | `scnet_xl_ihf` (0.33), `scnet_large` (0.33), `htdemucs_ft` (0.33) | waveform |

### vocal_split

| mode | output | scored against | SDR | SIR | SAR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| two-model blend | backing_vocals | backing_vocals | 3.37 | 6.85 | 6.98 | n/a | 14 of 30 | 62.65 | 17.43 | mps | `mel_roformer_karaoke_aufr33_viperx` (0.50), `uvr_vr_5hp_karaoke` (0.50) | waveform |
| full ensemble | backing_vocals | backing_vocals | 1.78 | 7.00 | 4.70 | n/a | 14 of 30 | 83.33 | 23.28 | cpu+mps | `mel_roformer_karaoke_aufr33_viperx` (0.33), `uvr_vr_5hp_karaoke` (0.33), `uvr_mdxnet_kara_2` (0.33) | waveform |
| two-model blend | lead_vocals | lead_vocals | 11.94 | 18.59 | 12.81 | n/a | 14 of 30 | 62.65 | 17.43 | mps | `mel_roformer_karaoke_aufr33_viperx` (0.50), `uvr_vr_5hp_karaoke` (0.50) | waveform |
| full ensemble | lead_vocals | lead_vocals | 11.83 | 18.15 | 13.22 | n/a | 14 of 30 | 83.33 | 23.28 | cpu+mps | `mel_roformer_karaoke_aufr33_viperx` (0.33), `uvr_vr_5hp_karaoke` (0.33), `uvr_mdxnet_kara_2` (0.33) | waveform |

## Not applicable to this dataset

- `mel_roformer_dereverb_anvuew`: needs the 'vocals_wet' input, which moisesdb_tuning does not have
- `uvr_vr_deecho_dereverb`: needs the 'vocals_wet' input, which moisesdb_tuning does not have
- `dereverb:two_model_blend`: needs the 'vocals_wet' input, which moisesdb_tuning does not have
- `dereverb:full_ensemble`: needs the 'vocals_wet' input, which moisesdb_tuning does not have

