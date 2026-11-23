# BLOCKED

Things only you can provide, in the order the build will need them. **B2 and B3 (the two
datasets) are being downloaded** (started 2026-11-22 at the owner's request, into
`~/Datasets/`): MUSDB18-HQ is open on Zenodo after all, and MoisesDB's form returned a link
at once. When both are unpacked and verified the rows below are removed. When you resolve an item, delete it or mark it done, and the next loop iteration
picks up the tasks it unblocks.

Rule for loop iterations: if a task needs something on this list, skip it and take the next
unblocked task.

## Needed from you

| # | What I need | Exactly | Unblocks | First needed |
|---|---|---|---|---|
| B2 | MUSDB18-HQ on local disk | Request access on Zenodo (https://zenodo.org/records/3338373), download, unpack, and set `MUSDB18HQ_PATH` in `.env` to the folder that contains `train/` and `test/`. About 23 GB. | P2.10 (real MUSDB18-HQ benchmark), P2.12 (blend tuning on its train split), P2.13, P2.14b. Once the path is set I freeze the track lists, you or I commit them, and the runs start in the background | Phase 2 |
| B3 | MoisesDB on local disk | Request access through the Moises / Music.AI research site (https://music.ai/research/), download, unpack, and set `MOISESDB_PATH` in `.env` to the folder that contains `moisesdb_v0.1/`. | P2.5b (freezing the 50-track subset: I run `python -m stems_bench.datasets.moisesdb --freeze` once the path is set), P2.11, all of the Clef-omni test (P3.5, P3.9), P7.1 calibration | Phase 2 |
| B4 | Modal account and token | Create the account, add a payment method, then set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in `.env`. | GPU timings in Phase 2 (without it every timing is Apple MPS), P5.12 live check, P7.3 cost per song | Phase 2, optional until Phase 5 |
| B5 | Cloudflare account id and Workers AI token | Set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_AUTH_TOKEN` in `.env`. The report estimates the whole test at under $0.02, inside the free daily allowance. | P3.9 (running the Clef-omni test). The harness itself is built and tested without it. | Phase 3 |
| B6 | TypeSafe API key with prepaid credit | Create a key at https://console.typesafe.ai, add credit (the report found no free starter credit), set `TYPESAFE_API_KEY` in `.env`. | P4.9 (live Jev check and recording real responses), P7.1 calibration against real Jev answers. Everything else runs on the fallback and the recorded mock. | Phase 4 |
| B7 | Bucket credentials, only if you want live storage checks | S3-compatible: `S3_ENDPOINT_URL`, `S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`. Azure: `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_STORAGE_CONTAINER`. | Live verification of P5.2 and P5.3. Without them those backends are tested against stubbed clients only, and PROGRESS.md will say so. | Phase 5, optional |
| B8 | Listening panel and reference purchases | At least 5 listeners including one who is not you; minimum LALAL.AI Pro and Moises Pro plans for the reference stems (report §9). | Running the listening test. The kit (P7.5) is built without them. | Phase 7 |
| B9 | Azure quota requests | Yours to file if you want Azure (report §6 has paste-ready texts). The build does not wait for them. | Filling in the Azure Container Apps worker | Whenever |
| B10 | The BEATs checkpoint file | The only official source is a OneDrive share link (https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea) that now serves a sign-in page to scripts, and its API form returns 401. Either open it in a browser, download the file, and save it as `models/weights/beats_iter3plus_as2m_cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` (the next download run hashes and pins it), or tell me to look for a mirror and I will list candidates with their hashes for you to choose from. | P3.1 and everything in Phase 3 that uses BEATs. Nothing in Phase 1 or 2. | Phase 3 |

## Decisions I made that you can overrule

I picked a default for each so the work does not stop. Say so if you want a different answer.

| # | Question | Default I am building | Changing it later costs |
|---|---|---|---|
| D1 | How is "private to the uploader" enforced with no sign-up? | No accounts. A random secret in an HttpOnly cookie identifies the browser; only its hash is stored. Clearing cookies loses access to the files, which still auto-delete. | Moderate once Phase 5 is built (adds an auth system) |
| D2 | Upload limits | 250 MiB and 20 minutes per file, in `config/defaults.yaml` | One config line |
| D3 | Output sample rate | Stems are delivered at 44.1 kHz stereo even if the upload was 48 kHz or mono | Small (one resample step) |
| D4 | Weights location and volume | All 48 loadable models, 16.66 GB, into `models/weights/` (git-ignored), downloaded in the background in Phase 1. Set `STEMS_MODELS_DIR` to put them elsewhere. | None |

## Resolved

- **B1 Node.js** (2026-11-22): the owner asked for a project-local install. Node v24.22.3 LTS
  is in `.tools/node` (git-ignored), installed by `tools/install_node.sh`, which verifies the
  nodejs.org checksum. `make` targets pick it up automatically.
