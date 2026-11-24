# Blend tuning: moisesdb_tuning

Which method and weights give each slot's blend the best SDR on the tuning tracks.
Every figure is computed from the separated audio of the run named here. Nothing
here is a test score: what is chosen below is scored on the test lists by an
ordinary benchmark run.

- Run: `moisesdb-tuning-v1`, 2026-11-24T18:27:54+00:00
- Command: `bench tune --dataset moisesdb_tuning --run-id moisesdb-tuning-v1 --out docs/benchmarks/tuning-moisesdb.md --quiet`
- Code: commit `d0405ae2199e5fe70b42ed493f065f03f00e3e6e` with uncommitted changes
- Device asked for: `mps`; chunk overlap 0.25
- Tuning list: `moisesdb_tuning`, sha256 `2e47e5d120e505fdde5ece16143adfe2057f554be971dbf3fbc015a30848eab2`, 30 tracks: `02ee37da-eea3-42b4-83bf-ab7f243afa13`, `04798708-6915-4dbc-842e-d394d545d4eb`, `125fc63d-9b69-4170-a46a-42c91bc28446`, `22ea41a3-1766-4a76-8071-380b27f1869a`, `25789239-1075-43b9-bfc9-51dff4a29590`, `260c431b-72d4-4ac6-bae5-ee49ce5c0fe4`, `30cfc60a-5a57-4000-a05e-65006c8f6f74`, `378742ba-5ba8-44c2-9cfd-8a609decca57`, `5f04798d-c7be-4b8a-90bd-1fcd9946e875`, `6c70d5e0-5972-444a-86f8-a558dbb92d92`, `6cd44645-ed19-4ecc-a57c-58d400005b29`, `765d5131-afd9-4ad7-8786-8bef5705c1c2`, `87a5da23-f17b-44da-accf-c04832f81a14`, `8e2d0c5c-6764-4d74-a740-391d7931ffd7`, `9653a690-c28c-4e8f-962e-ff7ed18b8ee9`, `97b07e0e-274e-4212-a66b-44210a48724d`, `9c8a5c66-f6d8-4425-8671-6b7aa6a2663b`, `a199697c-3cdc-47c7-a9c7-c1b07dd6c9dd`, `aefc1609-976b-423e-8516-f7d588d64ff7`, `afca84b2-0277-4b1b-8696-5f14543f338c`, `b207da3d-4baf-485a-98e1-657602479b3a`, `b8d6f3eb-f2d6-4342-af90-6d09f5257b6b`, `bdcc429e-ed95-40d3-a1af-bad268d66b25`, `c228818e-eabe-434b-9d60-2fb84a6c5b2a`, `d4262245-3143-4c05-8423-6cbdc6253042`, `dda2c057-6d73-43fa-a130-d7d6562c09ca`, `de22c812-b5c3-4c7d-92aa-f62faf8b63f7`, `f0c565c5-fc73-4da1-b979-0fac0167f671`, `f4d79498-4ba0-4813-9b9e-986374242573`, `faad432d-6ad0-492d-96f1-321eeb9685b5`
- Shares no track with `moisesdb_subset` (50 tracks, sha256 `4a5dfd20b191282dc68ffd6d2f24e3f8b80ca16ab913f5d308820267ebdc2035`): checked before the run

A candidate's score is the median over tracks of the track's median SDR on 1-second
windows, averaged over the slot's outputs, in dB. "Equal" is the plain average of the
waveforms with equal weights, which is what the config holds until it is tuned.

| slot | mode | models | equal | method | weights | score | gain | best of any method |
|---|---|---|---:|---|---|---:|---:|---|
| vocals | two-model blend | `mel_roformer_kim_vocals`, `bs_roformer_viperx_vocals` | 11.38 | waveform | 0.40, 0.60 | 11.40 | +0.01 | stft_magnitude (0.40, 0.60): 11.41 |
| vocals | full ensemble | `mel_roformer_kim_vocals`, `bs_roformer_viperx_vocals`, `bs_polarformer_vocals` | 11.51 | waveform | 0.20, 0.40, 0.40 | 11.55 | +0.04 |  |
| drums | two-model blend | `scnet_xl_ihf`, `htdemucs_ft` | 11.26 | stft_magnitude | 0.80, 0.20 | 11.36 | +0.11 |  |
| drums | full ensemble | `scnet_xl_ihf`, `scnet_large`, `htdemucs_ft` | 11.22 | stft_magnitude | 0.80, 0.00, 0.20 | 11.36 | +0.15 |  |
| bass | two-model blend | `htdemucs_ft`, `bs_roformer_sw_6stem` | 14.73 | waveform | 0.10, 0.90 | 15.67 | +0.94 | stft_magnitude (0.10, 0.90): 15.68 |
| bass | full ensemble | `htdemucs_ft`, `bs_roformer_sw_6stem`, `scnet_xl_ihf` | 13.77 | waveform | 0.00, 1.00, 0.00 | 15.64 | +1.87 |  |
| guitar | two-model blend | `bs_roformer_sw_6stem`, `mel_roformer_guitar_becruily` | 6.36 | waveform | 1.00, 0.00 | 7.85 | +1.49 |  |
| guitar | full ensemble | `bs_roformer_sw_6stem`, `mel_roformer_guitar_becruily`, `htdemucs_6s` | 5.68 | waveform | 1.00, 0.00, 0.00 | 7.85 | +2.17 |  |
| piano | two-model blend | `bs_roformer_sw_6stem`, `htdemucs_6s` | 5.70 | waveform | 1.00, 0.00 | 8.63 | +2.92 |  |
| piano | full ensemble | `bs_roformer_sw_6stem`, `htdemucs_6s` | 5.70 | waveform | 1.00, 0.00 | 8.63 | +2.92 |  |
| other | two-model blend | `scnet_xl_ihf`, `scnet_large` | 8.47 | waveform | 1.00, 0.00 | 9.19 | +0.71 |  |
| other | full ensemble | `scnet_xl_ihf`, `scnet_large`, `htdemucs_ft` | 8.51 | waveform | 0.80, 0.00, 0.20 | 9.24 | +0.73 | stft_magnitude (0.80, 0.00, 0.20): 9.26 |
| vocal_split | two-model blend | `mel_roformer_karaoke_aufr33_viperx`, `uvr_vr_5hp_karaoke` | 13.40 | waveform | 1.00, 0.00 | 19.02 | +5.61 |  |
| vocal_split | full ensemble | `mel_roformer_karaoke_aufr33_viperx`, `uvr_vr_5hp_karaoke`, `uvr_mdxnet_kara_2` | 11.86 | waveform | 1.00, 0.00, 0.00 | 19.02 | +7.16 |  |
| dereverb | two-model blend | mel_roformer_dereverb_anvuew, uvr_vr_deecho_dereverb | n/a | | | | | |
| dereverb | full ensemble | mel_roformer_dereverb_anvuew, uvr_vr_deecho_dereverb | n/a | | | | | |

## As config

Not applied. To use it, put these under the matching slots of
`config/defaults.yaml`, then score the result on the test lists.

```yaml
ensembles:
  slots:
    vocals:
      blend_method: waveform
      two_model_weights: [0.4, 0.6]
      full_ensemble_weights: [0.2, 0.4, 0.4]
    drums:
      blend_method: stft_magnitude
      two_model_weights: [0.8, 0.2]
      full_ensemble_weights: [0.8, 0, 0.2]
    bass:
      blend_method: waveform
      two_model_weights: [0.1, 0.9]
      full_ensemble_weights: [0, 1, 0]
    guitar:
      blend_method: waveform
      two_model_weights: [1, 0]
      full_ensemble_weights: [1, 0, 0]
    piano:
      blend_method: waveform
      two_model_weights: [1, 0]
      full_ensemble_weights: [1, 0]
    other:
      blend_method: waveform
      two_model_weights: [1, 0]
      full_ensemble_weights: [0.8, 0, 0.2]
    vocal_split:
      blend_method: waveform
      two_model_weights: [1, 0]
      full_ensemble_weights: [1, 0, 0]
```

