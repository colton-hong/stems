# Corrections to the research report

Things in `docs/research-report.pdf` that turned out to be wrong, out of date, or resolvable,
each with the source checked. The project proceeds on the corrected fact. Add new entries at
the bottom with the next number.

All checks below were made on 2026-11-22 from this machine.

| # | Report says | What was found | Source | Effect on the build |
|---|---|---|---|---|
| C1 | §4: every Jev answer comes with a probability and a confidence score. | A `noul` answer has only `type` and `noul` (the probability). `choice` answers have `choice`, `confidence`, `probabilities`. `score` answers have `score`, `confidence`, `legend`, `probabilities`. | Response schema embedded in https://docs.typesafe.ai/api | For noul questions confidence is derived as `max(p, 1 − p)`. The gate's `confidence` comes from the score question, which does report one. |
| C2 | §4: `score` "returns numeric score". | The score is the probability-weighted average of the rubric levels and can fall between integers. A score question takes 1 to 10 ordered criteria; a choice question takes 1 to 255 options. | Same page | The gate compares the fractional score with `score_min`. `score_label` is the most probable level. |
| C3 | §1: the 6-stem checkpoint is `BS-Rofo-SW-Fixed.ckpt` (jarredou). | `huggingface.co/jarredou/BS-ROFO-SW-Fixed` returns HTTP 401 (private or removed). The same-sized file `BS-Roformer-SW.ckpt` (699,412,152 bytes) and its config are public in the audio-separator model release. | HEAD requests; https://github.com/nomadkaraoke/python-audio-separator/releases/tag/model-configs | Registry entry `bs_roformer_sw_6stem` uses the mirror. Byte identity with the "Fixed" file is not verified. |
| C4 | §5: LAION-CLAP music checkpoint is "~600 MB on disk". | `music_audioset_epoch_15_esc_90.14.pt` is 2,352,471,003 bytes (2.35 GB). | https://huggingface.co/api/models/lukewys/laion_clap/tree/main | Download and memory budgets use 2.35 GB. |
| C5 | §5: LAION-CLAP checkpoint licence "not separately stated". | The Hugging Face repo that hosts the checkpoint is tagged `cc0-1.0`. | https://huggingface.co/api/models/lukewys/laion_clap | Informational only. |
| C6 | §1: Sucial de-reverb checkpoint is in the "~913 MB class". | `dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt` is 835,997,896 bytes. | https://huggingface.co/api/models/Sucial/Dereverb-Echo_Mel_Band_Roformer/tree/main | None beyond the registry size. |
| C7 | §5: the 780 tokens/minute audio rate and the 1-64 questions limit for Clef-omni rest on secondary sources. | Both are stated on the Workers AI model page: "Audio: about 780 tokens per minute" and `questions` with `minProperties: 1`, `maxProperties: 64`. The request body also needs `"model": "clef-omni"`. | https://developers.cloudflare.com/workers-ai/models/clef-omni/ | Recorded in the `clef_omni` registry entry. Two report items move from unverified to verified. |
| C8 | §6: Modal's GPU rates could not be read from the pricing page. | The page now shows T4 $0.000164/s, L4 $0.000222/s, A10 $0.000306/s, CPU $0.0000131/core/s, memory $0.00000222/GiB/s, and $30/month of free credit on Starter. These match the report's secondary figures. | https://modal.com/pricing | Cost-per-song work in Phase 7 can use these rates, re-checked on the day. |
| C9 | §6: Modal GPU functions are preemptible "with no non-preemptible opt-out". | The pricing page lists "Non-preemptible execution: 3x base prices". Whether that option covers GPU functions was not confirmed. | https://modal.com/pricing | The Modal worker must tolerate preemption regardless (idempotent, resumable jobs). |
| C10 | §5: self-hosted Clef-omni is "~64 GB BF16". | The weight files on Hugging Face total about 70.7 GB (65.9 GiB): 15 shards plus a 218 MB decision head. | https://huggingface.co/api/models/Cloudflare/clef-omni/tree/main | None. We never download them. |
| C11 | §4: Python SDK version 0.7.5 is "secondary, unverified". | `typesafe-sdk` 0.7.5 is the current release on PyPI. | `pip index versions typesafe-sdk` | None yet. Phase 4 decides between the SDK and plain HTTP. |
| C12 | Brief and §9: MUSDB18-HQ needs an access request. | The Zenodo record is open: `musdb18hq.zip`, 22,656,664,047 bytes, MD5 `12d4f2ecd55245a4688754dd76363103`, downloads without an account. Its licence is still "educational purposes only", no commercial use. | https://zenodo.org/records/3338373, read 2026-11-22 |
| C13 | §9: MoisesDB download size "not verified"; access by request. | `moisesdb.zip` is 88,822,120,667 bytes. The form at music.ai/research (name, e-mail, organization, justification) returns a download link at once, valid for 7 days; no approval step was seen. The dataset's README gives SHA-256 `4cde33ce416ac7c868cffcb60eb31f5c741ab7ae5601cbb9d99ed498b72c48c1`. | Form submitted for the owner on 2026-11-22; size from the storage server's `Content-Range`; hash from github.com/moises-ai/moises-db README at commit 1fbc5c78 |

## Report items that are still unverified

These stay unverified until the phase that touches them.

- Jev rate limits (the docs now say only "set per organization", shown in the console) and
  whether signup is open. Phase 4, needs your account.
- Jev latency (vendor figure 70-500 ms). Phase 4.
- Whether the hosted Clef-omni endpoint resamples to 16 kHz mono itself, its real rate
  limit, and what "16 MiB decoded" means. Phase 3.
- Whether weights exist yet for SubRoFormer or Sony MIMO BS-RoFormer. Re-check at the start
  of Phase 2.
- That the BEATs OneDrive link still serves the checkpoint. Phase 1 or 3.
- Everything listed under `unverified:` on individual registry entries.
