# Benchmark: synthetic

Generated from the result files of one run. Every figure here is copied from those
files or is a median or count of values in them; `report.json` next to the run lists
each file with its hash.

- Run: `synthetic-v1`, state `done`, started 2026-11-22T19:57:35+00:00
- Command: `bench run --dataset synthetic --all --ensembles --run-id synthetic-v1`
- Code: commit `905a0cf382a0964ff2e3a8995e089366e6672b47`
- Results computed on commit(s): `905a0cf382a0964ff2e3a8995e089366e6672b47`
- Protocol version: 1
- Frozen track list: sha256 `21644f45b9dc47d06155f4a4da7c022b2f6286e2e86a8f6e0642554d286c2ccc`, 8 tracks in this run
- Device asked for: `mps`; chunk overlap 0.25; batch size 1
- Packages: torch 2.15.0, numpy 2.5.5, scipy 1.18.1, museval 0.4.1, onnxruntime 1.32.0, demucs 4.1.0
- Jobs (one model on one track): 560 computed in this run, 0 taken from earlier runs, 0 failed, 0 not applicable
- Audio: 8 tracks with results, 96.0 s in total, median 12.0 s per track

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
- **Only SDR is reported for this dataset.** Its tracks are sine tones and
  noise, for which museval's SIR and SAR are numerical noise. The scores show that
  the harness works; they do not show how good a model is on music.

## vocals

| model | SDR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_zfturbo_4stem` | 14.64 | 7 of 8 | 4.57 | 22.83 | mps | -0.7 |  |
| `mdxnet_kuielab_b` | 11.01 | 7 of 8 | 3.51 | 17.54 | cpu | 0.8 |  |
| `scnet_large_zfturbo` | 10.60 | 7 of 8 | 1.42 | 7.11 | mps | -2.0 |  |
| `uvr_mdxnet_inst_hq_3` | 9.42 | 7 of 8 | 2.48 | 12.38 | cpu | 1.1 |  |
| `uvr_mdxnet_inst_hq_2` | 8.14 | 7 of 8 | 2.57 | 12.84 | cpu | -2.4 |  |
| `mel_roformer_kim_vocals` | 6.11 | 7 of 8 | 3.80 | 18.99 | mps | -8.9 |  |
| `mdxnet_kuielab_a` | 5.36 | 7 of 8 | 3.84 | 19.22 | cpu | -1.5 |  |
| `mdx23c_zfturbo_4stem` | 4.83 | 7 of 8 | 1.79 | 8.93 | mps | 8.8 |  |
| `htdemucs_ft` | 4.56 | 7 of 8 | 2.21 | 11.04 | mps | -3.3 |  |
| `htdemucs` | 4.33 | 7 of 8 | 0.57 | 2.84 | mps | -3.3 |  |
| `uvr_mdxnet_inst_hq_1` | 1.97 | 7 of 8 | 2.64 | 13.20 | cpu | -1.9 |  |
| `uvr_mdxnet_main` | 1.11 | 7 of 8 | 2.50 | 12.49 | cpu | -5.0 |  |
| `uvr_mdxnet_inst_hq_5` | 1.02 | 7 of 8 | 2.08 | 10.42 | cpu | 0.1 |  |
| `scnet_xl` | 0.76 | 7 of 8 | 1.86 | 9.29 | mps | -7.7 | output is 63% silence |
| `uvr_mdxnet_voc_ft` | 0.01 | 7 of 8 | 2.47 | 12.37 | cpu | -7.6 |  |
| `htdemucs_6s` | 0.00 | 7 of 8 | 0.70 | 3.51 | mps | -28.8 | output is 100% silence |
| `scnet_small` | 0.00 | 7 of 8 | 0.41 | 2.04 | mps | -2.3 | output is 67% silence |
| `mel_roformer_viperx_vocals` | 0.00 | 7 of 8 | 7.27 | 36.34 | mps | -76.4 | output is 100% silence |
| `bs_roformer_viperx_vocals` | 0.00 | 7 of 8 | 8.99 | 44.96 | mps | -81.0 | output is 100% silence |
| `bs_polarformer_vocals` | 0.00 | 7 of 8 | 9.04 | 45.18 | mps | -85.5 | output is 100% silence |
| `bs_roformer_sw_6stem` | 0.00 | 7 of 8 | 5.50 | 27.52 | mps | -74.7 | output is 100% silence |
| `scnet_masked_xl_ihf` | 0.00 | 7 of 8 | 3.32 | 16.58 | mps | -26.8 | output is 89% silence |
| `scnet_large` | 0.00 | 7 of 8 | 1.43 | 7.17 | mps | -10.5 | output is 98% silence |
| `scnet_xl_ihf` | 0.00 | 7 of 8 | 3.35 | 16.77 | mps | -14.9 | output is 100% silence |
| `mdx23c_vocals_zfturbo` | -0.02 | 7 of 8 | 1.74 | 8.68 | mps | -6.3 |  |
| `uvr_mdx23c_instvoc_hq` | -0.02 | 7 of 8 | 1.73 | 8.66 | mps | -6.3 |  |
| `uvr_mdxnet_inst_hq_4` | -0.07 | 7 of 8 | 2.14 | 10.68 | cpu | -4.6 |  |

## drums

| model | SDR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 15.48 | 8 of 8 | 5.50 | 27.52 | mps | 0.0 |  |
| `htdemucs` | 14.61 | 8 of 8 | 0.57 | 2.84 | mps | 0.0 |  |
| `htdemucs_ft` | 14.06 | 8 of 8 | 2.21 | 11.04 | mps | 0.2 |  |
| `scnet_large_zfturbo` | 13.79 | 8 of 8 | 1.42 | 7.11 | mps | 0.0 |  |
| `htdemucs_6s` | 13.66 | 8 of 8 | 0.70 | 3.51 | mps | 0.0 |  |
| `bs_roformer_zfturbo_4stem` | 11.87 | 8 of 8 | 4.57 | 22.83 | mps | 0.0 |  |
| `scnet_xl` | 11.83 | 8 of 8 | 1.86 | 9.29 | mps | -0.5 |  |
| `scnet_xl_ihf` | 11.77 | 8 of 8 | 3.35 | 16.77 | mps | 0.1 |  |
| `scnet_masked_xl_ihf` | 11.40 | 8 of 8 | 3.32 | 16.58 | mps | 0.0 |  |
| `scnet_large` | 10.42 | 8 of 8 | 1.43 | 7.17 | mps | -0.5 |  |
| `mdx23c_zfturbo_4stem` | 10.20 | 8 of 8 | 1.79 | 8.93 | mps | -3.9 |  |
| `scnet_small` | 9.35 | 8 of 8 | 0.41 | 2.04 | mps | -0.5 |  |
| `mdxnet_kuielab_b` | 8.27 | 8 of 8 | 3.51 | 17.54 | cpu | -18.8 |  |
| `mdxnet_kuielab_a` | 7.70 | 8 of 8 | 3.84 | 19.22 | cpu | -6.2 |  |

## bass

| model | SDR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_sw_6stem` | 19.25 | 8 of 8 | 5.50 | 27.52 | mps | n/a |  |
| `htdemucs` | 18.54 | 8 of 8 | 0.57 | 2.84 | mps | n/a |  |
| `htdemucs_ft` | 18.41 | 8 of 8 | 2.21 | 11.04 | mps | n/a |  |
| `htdemucs_6s` | 18.16 | 8 of 8 | 0.70 | 3.51 | mps | n/a |  |
| `scnet_large_zfturbo` | 17.12 | 8 of 8 | 1.42 | 7.11 | mps | n/a |  |
| `bs_roformer_zfturbo_4stem` | 15.68 | 8 of 8 | 4.57 | 22.83 | mps | n/a |  |
| `scnet_xl` | 14.97 | 8 of 8 | 1.86 | 9.29 | mps | n/a |  |
| `mdxnet_kuielab_a` | 14.65 | 8 of 8 | 3.84 | 19.22 | cpu | n/a |  |
| `scnet_large` | 14.39 | 8 of 8 | 1.43 | 7.17 | mps | n/a |  |
| `scnet_xl_ihf` | 14.27 | 8 of 8 | 3.35 | 16.77 | mps | n/a |  |
| `scnet_masked_xl_ihf` | 14.11 | 8 of 8 | 3.32 | 16.58 | mps | n/a |  |
| `mdx23c_zfturbo_4stem` | 13.90 | 8 of 8 | 1.79 | 8.93 | mps | n/a |  |
| `mdxnet_kuielab_b` | 13.89 | 8 of 8 | 3.51 | 17.54 | cpu | n/a |  |
| `scnet_small` | 13.42 | 8 of 8 | 0.41 | 2.04 | mps | n/a |  |

## guitar

| model | SDR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---|---:|---|
| `htdemucs_6s` | 3.88 | 7 of 8 | 0.70 | 3.51 | mps | n/a |  |
| `mel_roformer_guitar_becruily` | 0.36 | 7 of 8 | 1.69 | 8.47 | mps | n/a |  |
| `bs_roformer_sw_6stem` | 0.00 | 7 of 8 | 5.50 | 27.52 | mps | n/a | output is 100% silence |

## piano

| model | SDR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---|---:|---|
| `htdemucs_6s` | 0.00 | 7 of 8 | 0.70 | 3.51 | mps | -23.4 | output is 100% silence |
| `bs_roformer_sw_6stem` | 0.00 | 7 of 8 | 5.50 | 27.52 | mps | -78.2 | output is 100% silence |

## other

Six-stem other: what is left without vocals, drums, bass, guitar and piano.

| model | SDR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---|---:|---|
| `htdemucs_6s` | -5.92 | 8 of 8 | 0.70 | 3.51 | mps | n/a |  |
| `bs_roformer_sw_6stem` | -8.38 | 8 of 8 | 5.50 | 27.52 | mps | n/a |  |

## other_4stem

Four-stem other: guitar, piano and everything else that is not vocals, drums or bass.

| model | SDR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---|---:|---|
| `bs_roformer_zfturbo_4stem` | 12.29 | 8 of 8 | 4.57 | 22.83 | mps | 1.0 |  |
| `mdxnet_kuielab_b` | 12.02 | 8 of 8 | 3.51 | 17.54 | cpu | 20.4 |  |
| `mdxnet_kuielab_a` | 11.77 | 8 of 8 | 3.84 | 19.22 | cpu | -0.5 |  |
| `scnet_large_zfturbo` | 10.47 | 8 of 8 | 1.42 | 7.11 | mps | 3.3 |  |
| `mdx23c_zfturbo_4stem` | 7.61 | 8 of 8 | 1.79 | 8.93 | mps | 22.9 |  |
| `htdemucs_ft` | 4.94 | 8 of 8 | 2.21 | 11.04 | mps | 4.1 |  |
| `htdemucs` | 4.13 | 8 of 8 | 0.57 | 2.84 | mps | 4.3 |  |
| `scnet_masked_xl_ihf` | 1.19 | 8 of 8 | 3.32 | 16.58 | mps | 5.1 |  |
| `scnet_xl` | 0.15 | 8 of 8 | 1.86 | 9.29 | mps | 13.9 |  |
| `htdemucs_6s` | -0.02 | 8 of 8 | 0.70 | 3.51 | mps | 1.8 | sum of its guitar, piano and other |
| `scnet_small` | -0.13 | 8 of 8 | 0.41 | 2.04 | mps | 3.8 |  |
| `scnet_large` | -0.19 | 8 of 8 | 1.43 | 7.17 | mps | 6.8 |  |
| `bs_roformer_sw_6stem` | -0.34 | 8 of 8 | 5.50 | 27.52 | mps | 10.1 | sum of its guitar, piano and other |
| `scnet_xl_ihf` | -0.35 | 8 of 8 | 3.35 | 16.77 | mps | 4.8 |  |
| `bs_roformer_viperx_other` | -0.43 | 8 of 8 | 3.79 | 18.96 | mps | 5.2 |  |

## instrumental

Everything except the vocals.

| model | SDR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---|---:|---|
| `uvr_mdxnet_inst_hq_3` | 13.48 | 7 of 8 | 2.48 | 12.38 | cpu | 0.1 |  |
| `uvr_mdxnet_inst_hq_2` | 13.22 | 7 of 8 | 2.57 | 12.84 | cpu | 0.1 |  |
| `mel_roformer_kim_vocals` | 8.27 | 7 of 8 | 3.80 | 18.99 | mps | 0.0 |  |
| `uvr_mdxnet_inst_hq_1` | 7.29 | 7 of 8 | 2.64 | 13.20 | cpu | 0.1 |  |
| `uvr_mdxnet_main` | 6.22 | 7 of 8 | 2.50 | 12.49 | cpu | 0.0 |  |
| `uvr_mdxnet_inst_hq_5` | 5.88 | 7 of 8 | 2.08 | 10.42 | cpu | -0.1 |  |
| `uvr_mdxnet_voc_ft` | 4.60 | 7 of 8 | 2.47 | 12.37 | cpu | 0.0 |  |
| `mdx23c_vocals_zfturbo` | 4.43 | 7 of 8 | 1.74 | 8.68 | mps | 0.0 |  |
| `uvr_mdx23c_instvoc_hq` | 4.43 | 7 of 8 | 1.73 | 8.66 | mps | 0.0 |  |
| `bs_roformer_viperx_vocals` | 4.21 | 7 of 8 | 8.99 | 44.96 | mps | 0.0 |  |
| `mel_roformer_viperx_vocals` | 4.21 | 7 of 8 | 7.27 | 36.34 | mps | 0.0 |  |
| `bs_polarformer_vocals` | 4.21 | 7 of 8 | 9.04 | 45.18 | mps | 0.0 |  |
| `uvr_mdxnet_inst_hq_4` | 4.17 | 7 of 8 | 2.14 | 10.68 | cpu | 0.1 |  |

## lead_vocals

The model is fed the reference vocals stem.

| model | SDR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---|---:|---|
| `uvr_vr_5hp_karaoke` | 18.12 | 7 of 8 | 2.05 | 10.24 | mps | -11.2 |  |
| `uvr_vr_6hp_karaoke` | 15.49 | 7 of 8 | 2.06 | 10.29 | mps | -6.5 |  |
| `uvr_mdxnet_kara_2` | 4.09 | 7 of 8 | 1.81 | 9.04 | cpu | -1.7 |  |
| `uvr_mdxnet_kara` | 0.65 | 7 of 8 | 1.00 | 5.02 | cpu | -0.3 | output is 50% silence |
| `bs_roformer_karaoke_anvuew` | 0.00 | 7 of 8 | 5.76 | 28.80 | mps | -99.3 | output is 100% silence |
| `bs_roformer_karaoke_frazer_becruily` | 0.00 | 7 of 8 | 8.40 | 42.00 | mps | -96.1 | output is 100% silence |
| `mel_roformer_karaoke_becruily` | 0.00 | 7 of 8 | 3.64 | 18.19 | mps | -108.7 | output is 100% silence |
| `mel_roformer_karaoke_aufr33_viperx` | 0.00 | 7 of 8 | 3.67 | 18.33 | mps | -106.9 | output is 100% silence |

## backing_vocals

The model is fed the reference vocals stem.

| model | SDR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---|---:|---|
| `uvr_vr_5hp_karaoke` | 15.83 | 7 of 8 | 2.05 | 10.24 | mps | n/a |  |
| `uvr_vr_6hp_karaoke` | 14.36 | 7 of 8 | 2.06 | 10.29 | mps | n/a |  |
| `uvr_mdxnet_kara_2` | 7.52 | 7 of 8 | 1.81 | 9.04 | cpu | n/a |  |
| `uvr_mdxnet_kara` | -0.89 | 7 of 8 | 1.00 | 5.02 | cpu | n/a |  |
| `mel_roformer_karaoke_becruily` | -2.24 | 7 of 8 | 3.64 | 18.19 | mps | n/a |  |
| `bs_roformer_karaoke_anvuew` | -2.25 | 7 of 8 | 5.76 | 28.80 | mps | n/a |  |
| `bs_roformer_karaoke_frazer_becruily` | -2.25 | 7 of 8 | 8.40 | 42.00 | mps | n/a |  |
| `mel_roformer_karaoke_aufr33_viperx` | -2.25 | 7 of 8 | 3.67 | 18.33 | mps | n/a |  |

## dry

The vocals without reverb; the model is fed the vocals with reverb.

| model | SDR | tracks | s per track | s per min of audio | device | high band | note |
|---|---:|---:|---:|---:|---|---:|---|
| `uvr_vr_deecho_dereverb` | 12.96 | 7 of 8 | 2.06 | 10.29 | mps | 0.2 |  |
| `uvr_vr_deecho_aggressive` | 12.60 | 7 of 8 | 1.38 | 6.89 | mps | -0.2 |  |
| `uvr_vr_deecho_normal` | 12.26 | 7 of 8 | 1.39 | 6.97 | mps | 0.1 |  |
| `uvr_vr_denoise` | 12.02 | 7 of 8 | 1.41 | 7.04 | mps | 0.2 |  |
| `mel_roformer_dereverb_anvuew` | 8.45 | 7 of 8 | 3.83 | 19.14 | mps | 0.2 |  |
| `mel_roformer_dereverb_echo_sucial` | 0.00 | 7 of 8 | 3.91 | 19.57 | mps | -85.9 | output is 99% silence |

## Single best, two-model blend, full ensemble

Each slot of `config/defaults.yaml` in its three modes, on the same tracks. A blend's
time is the sum of the times of every model it needs, as if that stem alone were
asked for. "vs single best" is the blend's median SDR minus the single best's.

### vocals

| mode | output | scored against | SDR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| single best | vocals | vocals | 6.11 |  | 7 of 8 | 3.80 | 19.02 | mps | `mel_roformer_kim_vocals` (1.00) |  |
| two-model blend | vocals | vocals | 3.32 | -2.78 | 7 of 8 | 12.80 | 64.00 | mps | `mel_roformer_kim_vocals` (0.50), `bs_roformer_viperx_vocals` (0.50) | waveform |
| full ensemble | vocals | vocals | 2.19 | -3.92 | 7 of 8 | 21.83 | 109.13 | mps | `mel_roformer_kim_vocals` (0.33), `bs_roformer_viperx_vocals` (0.33), `bs_polarformer_vocals` (0.33) | waveform |

### drums

| mode | output | scored against | SDR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| single best | drums | drums | 11.77 |  | 8 of 8 | 3.36 | 16.78 | mps | `scnet_xl_ihf` (1.00) |  |
| two-model blend | drums | drums | 13.15 | +1.38 | 8 of 8 | 5.57 | 27.83 | mps | `scnet_xl_ihf` (0.50), `htdemucs_ft` (0.50) | waveform |
| full ensemble | drums | drums | 12.76 | +0.99 | 8 of 8 | 7.00 | 35.02 | mps | `scnet_xl_ihf` (0.33), `scnet_large` (0.33), `htdemucs_ft` (0.33) | waveform |

### bass

| mode | output | scored against | SDR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| single best | bass | bass | 18.41 |  | 8 of 8 | 2.21 | 11.05 | mps | `htdemucs_ft` (1.00) |  |
| two-model blend | bass | bass | 19.14 | +0.73 | 8 of 8 | 7.72 | 38.61 | mps | `htdemucs_ft` (0.50), `bs_roformer_sw_6stem` (0.50) | waveform |
| full ensemble | bass | bass | 17.43 | -0.98 | 8 of 8 | 11.06 | 55.32 | mps | `htdemucs_ft` (0.33), `bs_roformer_sw_6stem` (0.33), `scnet_xl_ihf` (0.33) | waveform |

### guitar

| mode | output | scored against | SDR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| single best | guitar | guitar | 0.00 |  | 7 of 8 | 5.51 | 27.53 | mps | `bs_roformer_sw_6stem` (1.00) |  |
| two-model blend | guitar | guitar | 0.20 | +0.20 | 7 of 8 | 7.20 | 36.02 | mps | `bs_roformer_sw_6stem` (0.50), `mel_roformer_guitar_becruily` (0.50) | waveform |
| full ensemble | guitar | guitar | 2.80 | +2.80 | 7 of 8 | 7.90 | 39.51 | mps | `bs_roformer_sw_6stem` (0.33), `mel_roformer_guitar_becruily` (0.33), `htdemucs_6s` (0.33) | waveform |

### piano

| mode | output | scored against | SDR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| single best | piano | piano | 0.00 |  | 7 of 8 | 5.51 | 27.53 | mps | `bs_roformer_sw_6stem` (1.00) |  |
| two-model blend | piano | piano | 0.00 | +0.00 | 7 of 8 | 6.21 | 31.03 | mps | `bs_roformer_sw_6stem` (0.50), `htdemucs_6s` (0.50) | waveform |
| full ensemble | piano | piano | 0.00 | +0.00 | 7 of 8 | 6.20 | 31.02 | mps | `bs_roformer_sw_6stem` (0.50), `htdemucs_6s` (0.50) | waveform |

### other

| mode | output | scored against | SDR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| single best | other | other_4stem | -0.35 |  | 8 of 8 | 3.36 | 16.78 | mps | `scnet_xl_ihf` (1.00) |  |
| two-model blend | other | other_4stem | -0.12 | +0.23 | 8 of 8 | 4.79 | 23.97 | mps | `scnet_xl_ihf` (0.50), `scnet_large` (0.50) | waveform |
| full ensemble | other | other_4stem | 1.76 | +2.11 | 8 of 8 | 7.00 | 35.02 | mps | `scnet_xl_ihf` (0.33), `scnet_large` (0.33), `htdemucs_ft` (0.33) | waveform |

### vocal_split

| mode | output | scored against | SDR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| single best | backing_vocals | backing_vocals | -2.25 |  | 7 of 8 | 3.67 | 18.35 | mps | `mel_roformer_karaoke_aufr33_viperx` (1.00) |  |
| two-model blend | backing_vocals | backing_vocals | 2.89 | +5.13 | 7 of 8 | 5.72 | 28.60 | mps | `mel_roformer_karaoke_aufr33_viperx` (0.50), `uvr_vr_5hp_karaoke` (0.50) | waveform |
| full ensemble | backing_vocals | backing_vocals | 5.27 | +7.52 | 7 of 8 | 7.53 | 37.64 | cpu+mps | `mel_roformer_karaoke_aufr33_viperx` (0.33), `uvr_vr_5hp_karaoke` (0.33), `uvr_mdxnet_kara_2` (0.33) | waveform |
| single best | lead_vocals | lead_vocals | 0.00 |  | 7 of 8 | 3.67 | 18.35 | mps | `mel_roformer_karaoke_aufr33_viperx` (1.00) |  |
| two-model blend | lead_vocals | lead_vocals | 5.45 | +5.45 | 7 of 8 | 5.72 | 28.60 | mps | `mel_roformer_karaoke_aufr33_viperx` (0.50), `uvr_vr_5hp_karaoke` (0.50) | waveform |
| full ensemble | lead_vocals | lead_vocals | 5.55 | +5.55 | 7 of 8 | 7.53 | 37.64 | cpu+mps | `mel_roformer_karaoke_aufr33_viperx` (0.33), `uvr_vr_5hp_karaoke` (0.33), `uvr_mdxnet_kara_2` (0.33) | waveform |

### dereverb

| mode | output | scored against | SDR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| single best | dry | dry | 8.45 |  | 7 of 8 | 3.83 | 19.15 | mps | `mel_roformer_dereverb_anvuew` (1.00) |  |
| two-model blend | dry | dry | 11.04 | +2.59 | 7 of 8 | 5.88 | 29.41 | mps | `mel_roformer_dereverb_anvuew` (0.50), `uvr_vr_deecho_dereverb` (0.50) | waveform |
| full ensemble | dry | dry | 11.04 | +2.59 | 7 of 8 | 5.88 | 29.41 | mps | `mel_roformer_dereverb_anvuew` (0.50), `uvr_vr_deecho_dereverb` (0.50) | waveform |

### other_composed

The four-stem blend with the same mode's guitar and piano taken out: what the pipeline delivers as the six-stem other.

| mode | output | scored against | SDR | vs single best | tracks | s per track | s per min of audio | device | models (weight) | method |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| single best | other | other | -8.38 |  | 8 of 8 | 8.86 | 44.28 | mps | `scnet_xl_ihf` (1.00), minus guitar: `bs_roformer_sw_6stem`, minus piano: `bs_roformer_sw_6stem` |  |
| two-model blend | other | other | -8.31 | +0.07 | 8 of 8 | 12.70 | 63.48 | mps | `scnet_xl_ihf` (0.50), `scnet_large` (0.50), minus guitar: `bs_roformer_sw_6stem`, minus guitar: `mel_roformer_guitar_becruily`, minus piano: `bs_roformer_sw_6stem`, minus piano: `htdemucs_6s` | waveform |
| full ensemble | other | other | -5.99 | +2.39 | 8 of 8 | 14.90 | 74.51 | mps | `scnet_xl_ihf` (0.33), `scnet_large` (0.33), `htdemucs_ft` (0.33), minus guitar: `bs_roformer_sw_6stem`, minus guitar: `mel_roformer_guitar_becruily`, minus guitar: `htdemucs_6s`, minus piano: `bs_roformer_sw_6stem`, minus piano: `htdemucs_6s` | waveform |

