# Score differences between runs: synthetic

The same models on the same tracks, run more than once. Each figure is a difference
between SDR values in the result files of the runs named here; nothing else is
measured. A track's SDR is its median over 1-second windows.

- Frozen track list: sha256 `21644f45b9dc47d06155f4a4da7c022b2f6286e2e86a8f6e0642554d286c2ccc`
- Protocol version: 1; chunk overlap 0.25
- `synthetic-v1` (mps): `bench run --dataset synthetic --all --ensembles --run-id synthetic-v1`, commit `905a0cf382a0964ff2e3a8995e089366e6672b47`
- `synthetic-mps-again` (mps, trial 1): `bench run --dataset synthetic --models mel_roformer_kim_vocals,scnet_xl_ihf,htdemucs_ft,bs_roformer_sw_6stem,mel_roformer_karaoke_aufr33_viperx,mel_roformer_dereverb_anvuew --trial 1 --run-id synthetic-mps-again`, commit `17a0ae39f7a33a1c5e9fb29317b381c563e33b9d`
- `synthetic-cpu` (cpu): `bench run --dataset synthetic --models mel_roformer_kim_vocals,scnet_xl_ihf,htdemucs_ft,bs_roformer_sw_6stem,mel_roformer_karaoke_aufr33_viperx,mel_roformer_dereverb_anvuew --device cpu --run-id synthetic-cpu`, commit `a9ebf8e9602be7e1454a30546ba16fd77a121a16`
- `synthetic-cpu-again` (cpu, trial 1): `bench run --dataset synthetic --models mel_roformer_kim_vocals,scnet_xl_ihf,htdemucs_ft,bs_roformer_sw_6stem,mel_roformer_karaoke_aufr33_viperx,mel_roformer_dereverb_anvuew --device cpu --trial 1 --run-id synthetic-cpu-again`, commit `0b357e6f7b8149c7d1659368d87d77106b062e21` with uncommitted changes

## Summary

| second run | against | kind | scores compared | median difference | largest |
|---|---|---|---:|---:|---:|
| `synthetic-mps-again` (mps, trial 1) | `synthetic-v1` (mps) | same device, run again | 150 | 0.000 | 0.000 |
| `synthetic-cpu-again` (cpu, trial 1) | `synthetic-cpu` (cpu) | same device, run again | 150 | 0.000 | 0.000 |
| `synthetic-cpu` (cpu) | `synthetic-v1` (mps) | another device | 150 | 0.000 | 0.817 |

Differences are absolute values in dB, one per model output per track.

## What follows from it

Moving to another device changes a score by up to 0.817 dB; running again on the same device changes it by up to 0.000 dB.
The device effect is larger than run-to-run noise, so scores from different
devices must not be compared with each other. Every benchmark report is built
from one run, and a run asks for one device.

## `synthetic-mps-again` (mps, trial 1) against `synthetic-v1` (mps)

| model | output | tracks | SDR first | SDR second | second minus first | largest on one track | track |
|---|---|---:|---:|---:|---:|---:|---|
| `bs_roformer_sw_6stem` | bass | 8 | 19.252 | 19.252 | +0.000 | 0.000 | `synth-01` |
| `bs_roformer_sw_6stem` | drums | 8 | 15.476 | 15.476 | +0.000 | 0.000 | `synth-01` |
| `bs_roformer_sw_6stem` | guitar | 7 | 0.000 | 0.000 | +0.000 | 0.000 | `synth-01` |
| `bs_roformer_sw_6stem` | other | 8 | -8.376 | -8.376 | +0.000 | 0.000 | `synth-01` |
| `bs_roformer_sw_6stem` | other_4stem | 8 | -0.336 | -0.336 | +0.000 | 0.000 | `synth-01` |
| `bs_roformer_sw_6stem` | piano | 7 | 0.000 | 0.000 | +0.000 | 0.000 | `synth-01` |
| `bs_roformer_sw_6stem` | vocals | 7 | 0.000 | 0.000 | +0.000 | 0.000 | `synth-01` |
| `htdemucs_ft` | bass | 8 | 18.405 | 18.405 | +0.000 | 0.000 | `synth-01` |
| `htdemucs_ft` | drums | 8 | 14.060 | 14.060 | +0.000 | 0.000 | `synth-01` |
| `htdemucs_ft` | other | 8 | 4.942 | 4.942 | +0.000 | 0.000 | `synth-01` |
| `htdemucs_ft` | vocals | 7 | 4.560 | 4.560 | +0.000 | 0.000 | `synth-01` |
| `mel_roformer_dereverb_anvuew` | dry | 7 | 8.455 | 8.455 | +0.000 | 0.000 | `synth-01` |
| `mel_roformer_karaoke_aufr33_viperx` | backing_vocals | 7 | -2.246 | -2.246 | +0.000 | 0.000 | `synth-01` |
| `mel_roformer_karaoke_aufr33_viperx` | lead_vocals | 7 | 0.000 | 0.000 | +0.000 | 0.000 | `synth-01` |
| `mel_roformer_kim_vocals` | instrumental | 7 | 8.272 | 8.272 | +0.000 | 0.000 | `synth-01` |
| `mel_roformer_kim_vocals` | vocals | 7 | 6.108 | 6.108 | +0.000 | 0.000 | `synth-01` |
| `scnet_xl_ihf` | bass | 8 | 14.268 | 14.268 | +0.000 | 0.000 | `synth-01` |
| `scnet_xl_ihf` | drums | 8 | 11.769 | 11.769 | +0.000 | 0.000 | `synth-01` |
| `scnet_xl_ihf` | other | 8 | -0.352 | -0.352 | +0.000 | 0.000 | `synth-01` |
| `scnet_xl_ihf` | vocals | 7 | -0.002 | -0.002 | +0.000 | 0.000 | `synth-01` |

## `synthetic-cpu-again` (cpu, trial 1) against `synthetic-cpu` (cpu)

| model | output | tracks | SDR first | SDR second | second minus first | largest on one track | track |
|---|---|---:|---:|---:|---:|---:|---|
| `bs_roformer_sw_6stem` | bass | 8 | 19.204 | 19.204 | +0.000 | 0.000 | `synth-01` |
| `bs_roformer_sw_6stem` | drums | 8 | 15.439 | 15.439 | +0.000 | 0.000 | `synth-01` |
| `bs_roformer_sw_6stem` | guitar | 7 | 0.000 | 0.000 | +0.000 | 0.000 | `synth-01` |
| `bs_roformer_sw_6stem` | other | 8 | -8.376 | -8.376 | +0.000 | 0.000 | `synth-01` |
| `bs_roformer_sw_6stem` | other_4stem | 8 | -0.338 | -0.338 | +0.000 | 0.000 | `synth-01` |
| `bs_roformer_sw_6stem` | piano | 7 | 0.000 | 0.000 | +0.000 | 0.000 | `synth-01` |
| `bs_roformer_sw_6stem` | vocals | 7 | 0.000 | 0.000 | +0.000 | 0.000 | `synth-01` |
| `htdemucs_ft` | bass | 8 | 18.405 | 18.405 | +0.000 | 0.000 | `synth-01` |
| `htdemucs_ft` | drums | 8 | 14.060 | 14.060 | +0.000 | 0.000 | `synth-01` |
| `htdemucs_ft` | other | 8 | 4.942 | 4.942 | +0.000 | 0.000 | `synth-01` |
| `htdemucs_ft` | vocals | 7 | 4.560 | 4.560 | +0.000 | 0.000 | `synth-01` |
| `mel_roformer_dereverb_anvuew` | dry | 7 | 8.655 | 8.655 | +0.000 | 0.000 | `synth-01` |
| `mel_roformer_karaoke_aufr33_viperx` | backing_vocals | 7 | -2.246 | -2.246 | +0.000 | 0.000 | `synth-01` |
| `mel_roformer_karaoke_aufr33_viperx` | lead_vocals | 7 | 0.000 | 0.000 | +0.000 | 0.000 | `synth-01` |
| `mel_roformer_kim_vocals` | instrumental | 7 | 8.415 | 8.415 | +0.000 | 0.000 | `synth-01` |
| `mel_roformer_kim_vocals` | vocals | 7 | 6.329 | 6.329 | +0.000 | 0.000 | `synth-01` |
| `scnet_xl_ihf` | bass | 8 | 14.268 | 14.268 | +0.000 | 0.000 | `synth-01` |
| `scnet_xl_ihf` | drums | 8 | 11.769 | 11.769 | +0.000 | 0.000 | `synth-01` |
| `scnet_xl_ihf` | other | 8 | -0.352 | -0.352 | +0.000 | 0.000 | `synth-01` |
| `scnet_xl_ihf` | vocals | 7 | -0.002 | -0.002 | +0.000 | 0.000 | `synth-01` |

## `synthetic-cpu` (cpu) against `synthetic-v1` (mps)

| model | output | tracks | SDR first | SDR second | second minus first | largest on one track | track |
|---|---|---:|---:|---:|---:|---:|---|
| `bs_roformer_sw_6stem` | piano | 7 | 0.000 | 0.000 | +0.000 | 0.817 | `synth-03` |
| `mel_roformer_kim_vocals` | vocals | 7 | 6.108 | 6.329 | +0.221 | 0.527 | `synth-03` |
| `mel_roformer_karaoke_aufr33_viperx` | backing_vocals | 7 | -2.246 | -2.246 | +0.000 | 0.268 | `synth-01` |
| `mel_roformer_kim_vocals` | instrumental | 7 | 8.272 | 8.415 | +0.143 | 0.215 | `synth-03` |
| `mel_roformer_dereverb_anvuew` | dry | 7 | 8.455 | 8.655 | +0.201 | 0.201 | `synth-07` |
| `bs_roformer_sw_6stem` | bass | 8 | 19.252 | 19.204 | -0.048 | 0.137 | `synth-07` |
| `bs_roformer_sw_6stem` | drums | 8 | 15.476 | 15.439 | -0.037 | 0.081 | `synth-07` |
| `mel_roformer_karaoke_aufr33_viperx` | lead_vocals | 7 | 0.000 | 0.000 | +0.000 | 0.056 | `synth-01` |
| `bs_roformer_sw_6stem` | other_4stem | 8 | -0.336 | -0.338 | -0.001 | 0.009 | `synth-08` |
| `bs_roformer_sw_6stem` | other | 8 | -8.376 | -8.376 | +0.000 | 0.005 | `synth-02` |
| `scnet_xl_ihf` | other | 8 | -0.352 | -0.352 | +0.000 | 0.002 | `synth-07` |
| `scnet_xl_ihf` | bass | 8 | 14.268 | 14.268 | +0.000 | 0.000 | `synth-02` |
| `scnet_xl_ihf` | drums | 8 | 11.769 | 11.769 | +0.000 | 0.000 | `synth-02` |
| `htdemucs_ft` | other | 8 | 4.942 | 4.942 | +0.000 | 0.000 | `synth-05` |
| `scnet_xl_ihf` | vocals | 7 | -0.002 | -0.002 | +0.000 | 0.000 | `synth-07` |
| `htdemucs_ft` | vocals | 7 | 4.560 | 4.560 | +0.000 | 0.000 | `synth-05` |
| `bs_roformer_sw_6stem` | guitar | 7 | 0.000 | 0.000 | +0.000 | 0.000 | `synth-03` |
| `htdemucs_ft` | bass | 8 | 18.405 | 18.405 | +0.000 | 0.000 | `synth-07` |
| `htdemucs_ft` | drums | 8 | 14.060 | 14.060 | +0.000 | 0.000 | `synth-02` |
| `bs_roformer_sw_6stem` | vocals | 7 | 0.000 | 0.000 | +0.000 | 0.000 | `synth-03` |

