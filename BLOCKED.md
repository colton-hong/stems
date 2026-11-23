# BLOCKED

Things only you can provide, in the order the build will need them. **B2 and B3 (the two
datasets) are resolved** (2026-11-22, downloaded at the owner's request into `~/Datasets/`,
both verified against their published hashes, unpacked, and their track lists frozen and
committed). When you resolve an item, delete it or mark it done, and the next loop iteration
picks up the tasks it unblocks.

Rule for loop iterations: if a task needs something on this list, skip it and take the next
unblocked task.

## Needed from you

| # | What I need | Exactly | Unblocks | First needed |
|---|---|---|---|---|
| B4 | Modal account and token | Create the account, add a payment method, then set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in `.env`. | GPU timings in Phase 2 (without it every timing is Apple MPS), P5.12 live check, P7.3 cost per song | Phase 2, optional until Phase 5 |
| B5 | Cloudflare account id and Workers AI token | Set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_AUTH_TOKEN` in `.env`. The report estimates the whole test at under $0.02, inside the free daily allowance. | P3.9 (running the Clef-omni test). The harness itself is built and tested without it. | Phase 3 |
| B6 | TypeSafe API key with prepaid credit | Create a key at https://console.typesafe.ai, add credit (the report found no free starter credit), set `TYPESAFE_API_KEY` in `.env`. | P4.9 (live Jev check and recording real responses), P7.1 calibration against real Jev answers. Everything else runs on the fallback and the recorded mock. | Phase 4 |
| B7 | Bucket credentials, only if you want live storage checks | S3-compatible: `S3_ENDPOINT_URL`, `S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`. Azure: `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_STORAGE_CONTAINER`. | Live verification of P5.2 and P5.3. Without them those backends are tested against stubbed clients only, and PROGRESS.md will say so. | Phase 5, optional |
| B8 | Listening panel and reference purchases | At least 5 listeners including one who is not you; minimum LALAL.AI Pro and Moises Pro plans for the reference stems (report §9). | Running the listening test. The kit (P7.5) is built without them. | Phase 7 |
| B9 | Azure quota requests | Yours to file if you want Azure (report §6 has paste-ready texts). The build does not wait for them. | Filling in the Azure Container Apps worker | Whenever |

## Decisions I made that you can overrule

I picked a default for each so the work does not stop. Say so if you want a different answer.

| # | Question | Default I am building | Changing it later costs |
|---|---|---|---|
| D1 | How is "private to the uploader" enforced with no sign-up? | No accounts. A random secret in an HttpOnly cookie identifies the browser; only its hash is stored. Clearing cookies loses access to the files, which still auto-delete. | Moderate once Phase 5 is built (adds an auth system) |
| D2 | Upload limits | 250 MiB and 20 minutes per file, in `config/defaults.yaml` | One config line |
| D3 | Output sample rate | Stems are delivered at 44.1 kHz stereo even if the upload was 48 kHz or mono | Small (one resample step) |
| D4 | Weights location and volume | All 48 loadable models, 16.66 GB, into `models/weights/` (git-ignored), downloaded in the background in Phase 1. Set `STEMS_MODELS_DIR` to put them elsewhere. | None |
| D5 | What happens to a stem the gate cannot verify? | It is delivered with a warning at once, without retries. That covers a stem nothing listened to (no tagger) and, while `listening.calibrated` is false, any verdict that rests on the tagger: such verdicts carry `gate.rules.uncalibrated_confidence` (0.5), under the gate's 0.7. Reason: the measured BEATs half of the Clef-omni test showed the uncalibrated mapping is not reliable enough to act on, and a retry would produce a stem that is just as unverifiable. A silent stem is a level measurement and is still retried. The warning says only that the stem was not checked; what the tagger suspects stays in the log. Jev's confident answers are acted on as decided. | One config value (`uncalibrated_confidence`), or `calibrated: true` after P7.1 |
| D6 | May the planner add de-reverb the user did not ask for? | No. De-reverb is planned only on request; Jev only picks which model. The gate can still add it as a retry on a failing vocals stem. | Small (one condition in `decision/plan.py`) |

## Resolved

- **B1 Node.js** (2026-11-22): the owner asked for a project-local install. Node v24.22.3 LTS
  is in `.tools/node` (git-ignored), installed by `tools/install_node.sh`, which verifies the
  nodejs.org checksum. `make` targets pick it up automatically.
