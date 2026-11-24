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
| D7 | Rate limits | Per browser: 20 uploads an hour, 20 request-parses a minute, 3 jobs in progress at once. Per IP address: 40 uploads an hour, 40 parses a minute. An upload counts when it is attempted, accepted or not. Counted in memory per API process; the address is the connection's, so behind a proxy every user shares one. All in `config/defaults.yaml` under `rate_limits`. | One config value each; a shared store and proxy headers if the API is ever run as several processes or behind a proxy |
| D8 | What happens to a job whose request text names no stem? | It ends as `failed` with the message "No stem could be recognised..." and the parsed request attached. The job table has no review state; the web app is meant to parse and confirm before it uploads. | Small (a status value and one branch in `stems_api.sync`) |
| D9 | What is kept after a job is deleted or expires? | The row stays as a marker (status, dates, sizes). The file name, analysis, plan, stems, events and decision log are removed. Timing rows (model, device, two durations) are kept for the cost figures. `DELETE /me` removes every row. The owner cookie lasts 400 days. | Small (`stems_api.db.store.scrub`) |
| D10 | Does the web app ever upload a request as free text? | No. Words typed on the upload page are parsed first and fill in the stem ticks, format and options, which the person sees and can correct; the job is then sent with ticked stems. If the parse is unsure the page says so and will not upload until a stem is ticked. The API still accepts text for other clients. | Small (one form field) |
| D11 | How does the player keep stems in time? | It decodes every stem in full and starts them all at one instant on one audio clock. That is exact, and costs memory: about 21 MB per minute per stem by arithmetic. Tracks over 10 minutes load only when asked. | Large (a streaming player with drift correction is a different design) |

## Resolved

- **B1 Node.js** (2026-11-22): the owner asked for a project-local install. Node v24.22.3 LTS
  is in `.tools/node` (git-ignored), installed by `tools/install_node.sh`, which verifies the
  nodejs.org checksum. `make` targets pick it up automatically.
- **B5 Cloudflare account id and Workers AI token** (2026-11-24): in `.env`, written there by
  the owner's Muse assistant after the owner signed in. Checked without printing them:
  Cloudflare's token-verify endpoint answers `active`, and a Workers AI read on the account
  succeeds. The Clef-omni half of P3.9 can run.
- **B6 TypeSafe API key** (2026-11-24): in `.env`, same route. Three real requests succeeded
  (P4.9). Whether the account has prepaid credit or a free allowance was not looked at: no
  purchase was made from here.
