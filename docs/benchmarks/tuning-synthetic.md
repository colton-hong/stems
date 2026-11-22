# Blend tuning: synthetic_tuning

Which method and weights give each slot's blend the best SDR on the tuning tracks.
Every figure is computed from the separated audio of the run named here. Nothing
here is a test score: what is chosen below is scored on the test lists by an
ordinary benchmark run.

- Run: `synthetic-tuning-v1`, 2026-11-22T22:17:55+00:00
- Command: `bench tune --dataset synthetic_tuning --run-id synthetic-tuning-v1 --out docs/benchmarks/tuning-synthetic.md`
- Code: commit `0b357e6f7b8149c7d1659368d87d77106b062e21`
- Device asked for: `mps`; chunk overlap 0.25
- Tuning list: `synthetic_tuning`, sha256 `b752a6e4490cd860db4d832a61006d463881d936de980c0764ea60345a3140a6`, 4 tracks: `tune-01`, `tune-02`, `tune-03`, `tune-04`
- Shares no track with `synthetic` (8 tracks, sha256 `21644f45b9dc47d06155f4a4da7c022b2f6286e2e86a8f6e0642554d286c2ccc`): checked before the run

A candidate's score is the median over tracks of the track's median SDR on 1-second
windows, averaged over the slot's outputs, in dB. "Equal" is the plain average of the
waveforms with equal weights, which is what the config holds until it is tuned.

| slot | mode | models | equal | method | weights | score | gain | best of any method |
|---|---|---|---:|---|---|---:|---:|---|
| vocals | two-model blend | `mel_roformer_kim_vocals`, `bs_roformer_viperx_vocals` | 2.58 | waveform | 1.00, 0.00 | 4.14 | +1.56 |  |
| vocals | full ensemble | `mel_roformer_kim_vocals`, `bs_roformer_viperx_vocals`, `bs_polarformer_vocals` | 1.85 | waveform | 1.00, 0.00, 0.00 | 4.14 | +2.29 |  |
| drums | two-model blend | `scnet_xl_ihf`, `htdemucs_ft` | 13.31 | stft_max | none | 14.33 | +1.02 |  |
| drums | full ensemble | `scnet_xl_ihf`, `scnet_large`, `htdemucs_ft` | 12.74 | stft_max | none | 14.18 | +1.44 |  |
| bass | two-model blend | `htdemucs_ft`, `bs_roformer_sw_6stem` | 19.47 | waveform | 0.20, 0.80 | 19.85 | +0.38 |  |
| bass | full ensemble | `htdemucs_ft`, `bs_roformer_sw_6stem`, `scnet_xl_ihf` | 17.79 | waveform | 0.20, 0.80, 0.00 | 19.85 | +2.06 |  |
| guitar | two-model blend | `bs_roformer_sw_6stem`, `mel_roformer_guitar_becruily` | 0.13 | stft_max | none | 0.23 | +0.10 | waveform (0.00, 1.00): 0.23 |
| guitar | full ensemble | `bs_roformer_sw_6stem`, `mel_roformer_guitar_becruily`, `htdemucs_6s` | 2.29 | stft_max | none | 3.32 | +1.03 |  |
| piano | two-model blend | `bs_roformer_sw_6stem`, `htdemucs_6s` | 0.00 | waveform | 0.50, 0.50 | 0.00 | +0.00 |  |
| piano | full ensemble | `bs_roformer_sw_6stem`, `htdemucs_6s` | 0.00 | waveform | 0.50, 0.50 | 0.00 | +0.00 |  |
| other | two-model blend | `scnet_xl_ihf`, `scnet_large` | -0.39 | stft_min | none | -0.06 | +0.34 |  |
| other | full ensemble | `scnet_xl_ihf`, `scnet_large`, `htdemucs_ft` | 2.38 | stft_min | none | 5.64 | +3.26 |  |
| vocal_split | two-model blend | `mel_roformer_karaoke_aufr33_viperx`, `uvr_vr_5hp_karaoke` | 5.05 | waveform | 0.00, 1.00 | 23.99 | +18.94 |  |
| vocal_split | full ensemble | `mel_roformer_karaoke_aufr33_viperx`, `uvr_vr_5hp_karaoke`, `uvr_mdxnet_kara_2` | 4.36 | waveform | 0.00, 1.00, 0.00 | 23.99 | +19.63 |  |
| dereverb | two-model blend | `mel_roformer_dereverb_anvuew`, `uvr_vr_deecho_dereverb` | 11.18 | stft_max | none | 14.90 | +3.73 |  |
| dereverb | full ensemble | `mel_roformer_dereverb_anvuew`, `uvr_vr_deecho_dereverb` | 11.18 | stft_max | none | 14.90 | +3.73 |  |

## As config

Not applied. To use it, put these under the matching slots of
`config/defaults.yaml`, then score the result on the test lists.

```yaml
ensembles:
  slots:
    vocals:
      blend_method: waveform
      two_model_weights: [1, 0]
      full_ensemble_weights: [1, 0, 0]
    drums:
      blend_method: stft_max
    bass:
      blend_method: waveform
      two_model_weights: [0.2, 0.8]
      full_ensemble_weights: [0.2, 0.8, 0]
    guitar:
      blend_method: stft_max
    piano:
      blend_method: waveform
      two_model_weights: [0.5, 0.5]
      full_ensemble_weights: [0.5, 0.5]
    other:
      blend_method: stft_min
    vocal_split:
      blend_method: waveform
      two_model_weights: [0, 1]
      full_ensemble_weights: [0, 1, 0]
    dereverb:
      blend_method: stft_max
```

