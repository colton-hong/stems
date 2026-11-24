# Deployment guide

How to run Stems, from one laptop to rented GPUs and a bucket. Every step below is marked:

- **Run here** means it was carried out on the development machine (an Apple-silicon Mac)
  on the date given, and worked.
- **Not run** means the code exists and its tests pass, but nobody has done this for real.
  Expect to fix small things the first time.

Only the local setup has ever run end to end: every test with stand-in models (frequency
bands, not music), and one job with the real models on a synthetic test signal (below).
No real song has been through it, and no deployment beyond one machine has been tried.

## What there is to deploy

| Part | What it is | Where it runs |
|---|---|---|
| API | `python -m stems_api` (FastAPI). Owns the database, takes uploads, hands out links. Never loads a model. | Anywhere with Python |
| Worker | `python -m stems_worker.jobs JOB_ID`. Does one job: reads the upload from storage, separates, writes the stems back. | Started by the API through a *provider*: this machine, or Modal |
| Storage | Where uploads and stems live for 7 days. | A folder, an S3-compatible bucket, or Azure Blob |
| Web app | `web/` (Next.js). | Anywhere with Node.js |

The API and the worker never talk to each other. They share only storage, so the worker
can be on a rented GPU that cannot reach your laptop.

## 1. Everything on one machine

| Step | Command | Status |
|---|---|---|
| Install Python packages | `make setup` | Run here, 2026-11-22 and since |
| Install Node.js if you have none | `make node-setup` | Run here, 2026-11-22 |
| Install web packages | `make web-setup` | Run here, 2026-11-22 and 2026-11-24 |
| Check everything | `make check`, `make web-check`, `make web-e2e` | Run here, 2026-11-24 |
| Start the API | `make api`, which runs `python -m stems_api` (port 8000) | Run here, 2026-11-24, as `python -m stems_api` |
| Start the web app for development | `cd web && npm run dev` (port 3000) | Run here, 2026-11-24 |
| Build and serve the web app | `cd web && npm run build && npm run start` | Run here, 2026-11-24, by the browser tests |
| Upload a song in the browser and get stems, **stand-in models** | set `STEMS_WORKER_FAKE_MODELS=1` before `make api` | Run here, 2026-11-24 |
| Upload in the browser and get stems, **real models** | `make web-e2e-real` (starts its own API and web app; weights must be on disk) | Run here, 2026-11-24, on the synthetic fixture |

Nothing has to be configured. With no `.env` the API uses a SQLite file in `runs/dev/`,
keeps files in `runs/dev/storage/`, makes its own link-signing secret there, runs jobs on
this machine one at a time, and decides with rules instead of Jev.

### The one job with real models

`make web-e2e-real` starts the API with the real registry and no stand-ins, builds and
serves the web app, and in Chrome uploads a 12-second synthetic signal (tones and noise
made by code, not music), asks for all six stems, waits, plays them and downloads one.
Run on 2026-11-24 with `STEMS_DEVICE=cpu`, because the benchmark had this Mac's GPU; the
figures are from the job's own result file (`web/.e2e-real/real-job.json`, not in the
repository).

- Upload to ready: 103.3 seconds, on a machine that was also running a benchmark.
- Six stems arrived, started on one clock in the player, and one downloaded as audio.
- No key was used, so the rules decided, and the tagger ran but is not acted on: four
  stems carry the "could not be quality-checked" note.
- Guitar and piano came back more than 19.8 dB under the mix, so the gate tried the
  fallback model and then a blend for each, and delivered them with the "silent or very
  quiet" note. That is the rule working as built; whether it is right here cannot be
  said, since the signal's "guitar" and "piano" are tones no model was trained on.

| Model | For | Seconds for 12 s of audio, CPU |
|---|---|---:|
| `mel_roformer_kim_vocals` | vocals | 12.7 |
| `scnet_xl_ihf` | drums, other | 49.2 |
| `htdemucs_ft` | bass | 3.9 |
| `bs_roformer_sw_6stem` | guitar, piano | 17.5 |
| `mel_roformer_guitar_becruily` | guitar, retry | 5.3 |
| `htdemucs_6s` | piano, retry | 3.8 |

Each time includes whatever the benchmark was taking from the machine at that moment and
is one run; they are not speed measurements (those are in `docs/benchmarks/`). The
blends of the second retry reused the outputs above and ran no model again.

Run again the same day at 14:34 UTC, after two changes to the defaults: the gate no longer
tries an empty stem again, and the tagger is off. Four models ran and nothing else
(126.4 seconds from upload to ready, on a busier machine); guitar and piano were delivered
at once with the "silent or very quiet" note.

And once more at 15:58 UTC with the default model chosen that afternoon
(`docs/benchmarks/recommendation.md`): one model, `bs_roformer_sw_6stem`, for all six
stems, 28.5 seconds for the 12 seconds of audio on CPU and 35.8 seconds from upload to
ready. Vocals, guitar and piano carried the "silent or very quiet" note: on this signal,
which is tones and noise, the model puts almost nothing in those stems.

What this does not show: a real song in the browser, a long one, the GPU path through the
API, or a first run that has to download its checkpoints (about 16.7 GB for all of them). On a
24 GB Mac do not run a real job on the GPU while a benchmark is running.

### The longest upload

`python -m stems_bench.long_upload --device mps` makes a synthetic signal exactly as long
as the limit allows (`uploads.max_duration_seconds`, 20 minutes), puts it in storage as an
upload and starts the worker the way the API does, as a process of its own, with the real
default model. Run on 2026-11-24 at 18:39 UTC on this Mac's GPU, while a benchmark was
using the same GPU; the figures are from `runs/long-upload/long-upload-v1/result.json`
(not in the repository).

- The job finished: 568 seconds from start to result for 1,200 seconds of audio, 526 of
  them in the model.
- All six stems came back at exactly the upload's length.
- The worker process held at most 6.46 GB, as the operating system counts it. Memory the
  GPU holds on the worker's behalf may not all be in that figure.
- One job at this length is what was tried. The local provider runs one job at a time,
  so two such jobs do not overlap on one machine.

One thing this turned up: as a 24-bit WAV, twenty minutes of stereo is 318 MB, which is
over the API's size limit (`uploads.max_bytes`, 250 MiB). The API would refuse that file
before any of this ran; the test put it straight into storage. A twenty-minute upload
gets through the API as 16-bit WAV (212 MB), FLAC or MP3. The two limits are both in
`config/defaults.yaml`.

## 2. Settings

Read from the environment first, then from `.env` in the repository root (never
committed). `.env.example` lists them all.

| Variable | What for | Default |
|---|---|---|
| `STEMS_DATABASE_URL` | The database | `sqlite:///runs/dev/stems.db` |
| `STEMS_STORAGE_BACKEND` | `local`, `s3` or `azure_blob` | `local` |
| `STEMS_STORAGE_LOCAL_DIR` | Folder for local storage | `runs/dev/storage` |
| `STEMS_SIGNING_SECRET` | Signs local download links | made once, kept in the storage folder |
| `STEMS_WORKER_PROVIDER` | `local` or `modal` | `local` |
| `STEMS_WEB_ORIGIN` | The address the web app is served from; the only origin the API accepts | `http://localhost:3000` |
| `STEMS_TRUSTED_PROXIES` | Addresses or networks of the reverse proxies in front of the API, comma-separated. Forwarded headers are believed only from them | unset: believed from nobody |
| `NEXT_PUBLIC_API_URL` | Where the web app finds the API (set when building the web app) | `http://localhost:8000` |
| `TYPESAFE_API_KEY` | Jev, for decisions. Without it the rules decide. | unset |
| `S3_BUCKET`, `S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | The S3 backend | unset |
| `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_STORAGE_CONTAINER` | The Azure Blob backend | unset |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | The Modal provider | unset |
| `STEMS_MODELS_DIR` | Where checkpoints are kept | `models/weights` |

Limits, timers and thresholds are not environment variables: they are in
`config/defaults.yaml`.

## 3. Putting it on the internet

**Not run.** None of this has been done; these are the things the code assumes.

- **Serve both over HTTPS.** The owner cookie is marked Secure only when the API sees an
  HTTPS request. If a proxy terminates TLS, it must tell the API so, in the
  `X-Forwarded-Proto` header, and the API must be told to believe that proxy.
- **Name the proxy.** Start the API with `--trusted-proxies 10.0.0.5` (or set
  `STEMS_TRUSTED_PROXIES`), giving the address the proxy connects from, or its network
  (`10.0.0.0/24`), several separated by commas. From those addresses, and no others, the
  API takes the visitor's address from `X-Forwarded-For` (so each visitor has their own
  rate limit, where otherwise all would share the proxy's) and HTTPS from
  `X-Forwarded-Proto`. With nothing named the headers are ignored, including from this
  machine. `*` is refused: it would let any visitor choose the address they are counted
  under. The proxy must append the address it saw to `X-Forwarded-For`, as nginx's
  `$proxy_add_x_forwarded_for` does; the last entry not written by a trusted proxy is the
  one used. *Tested with uvicorn's own middleware (`tests/test_api_proxy.py`,
  2026-11-24); not run behind a real proxy.*
- **Same site.** The cookie is SameSite=Lax, so the web app and the API must share a
  registrable domain (`app.example.org` and `api.example.org` are fine; two unrelated
  domains are not). Set `STEMS_WEB_ORIGIN` to the web app's exact origin and build the web
  app with `NEXT_PUBLIC_API_URL` set to the API's.
- **One API process.** Rate limits and the local job queue live in the API's memory. A
  second process would double the limits and not see the first one's queue.
- **The database.** SQLite is what every test uses. The SQL is meant to work on Postgres
  and has never run there.
- **Terms and privacy.** The pages in the web app are placeholders that say so. Offering
  this to other people needs real ones, and the models' licences (several are
  non-commercial) checked against the use.
- Read `docs/security-review.md` for what was reviewed and what was left.

## 4. Storage in a bucket

**Not run** against any real bucket (BLOCKED.md B7). The backends pass their tests
against the real client libraries with the network replaced.

S3 or anything that speaks its protocol (Cloudflare R2, Backblaze B2, MinIO):

```
STEMS_STORAGE_BACKEND=s3
S3_BUCKET=<bucket>
S3_ENDPOINT_URL=<endpoint>      # leave unset for AWS itself
AWS_ACCESS_KEY_ID=<key id>
AWS_SECRET_ACCESS_KEY=<secret>
AWS_REGION=<region>             # AWS only
```

Azure Blob:

```
STEMS_STORAGE_BACKEND=azure_blob
AZURE_STORAGE_CONNECTION_STRING=<a connection string that contains AccountKey>
AZURE_STORAGE_CONTAINER=<container>
```

Things to do on the bucket, none of which this code does for you:

- Keep it private. Downloads use presigned URLs (S3) or SAS tokens (Azure) that last 15
  minutes.
- Allow the web app's origin to `GET` from the bucket (a CORS rule). The player fetches
  each stem through its signed link from the browser, and without the rule it cannot.
  Plain downloads work without it.
- Optionally add a lifecycle rule that deletes objects after 8 days, as a safety net
  under the API's own 7-day sweeper.

The first real run should be: set the variables, start the API with
`STEMS_WORKER_FAKE_MODELS=1`, upload a short file, play it, download it, delete it, and
look in the bucket to see that the job's prefix is gone.

## 5. Workers on Modal

**Not run** (BLOCKED.md B4: needs a Modal account). The provider is tested against a
stand-in for Modal's client; `worker/stems_worker/modal_app.py` has never been deployed
and its container image has never been built.

Modal needs storage it can reach, so section 4 comes first: a Modal container cannot
read a folder on your machine.

1. Install the client: `pip install -e ".[modal]"`, then `modal token new` (opens a
   browser; this is your account).
2. Create a Modal Secret named `stems` holding the storage variables from section 4,
   plus `TYPESAFE_API_KEY` if you use Jev.
3. Deploy the worker: `modal deploy worker/stems_worker/modal_app.py`. It asks for a T4,
   mounts a Volume named `stems-weights` for checkpoints, and allows an hour per job.
4. Tell the API: `STEMS_WORKER_PROVIDER=modal`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`,
   and the same storage settings.
5. Upload a short file. The first job downloads checkpoints into the Volume.

What to check on the first run, because nobody has: that the image builds; that the
function finds `config/defaults.yaml` and the registry under `/app`; how long a song
takes on a T4 (this is the number `docs/cost-per-song.md` is missing); and that
cancelling a job stops the call.

## 6. Azure Container Apps

**Not built**, on purpose (project decision 9: do not depend on Azure GPU quota).
`api/stems_api/providers/azure_aca.py` is a class with two methods left to write,
`_start_job` and `_execution_state`, and its docstring says what each must do. Selecting
it today stops the API at startup with a message that says so. The quota requests it
would need are yours to file (BLOCKED.md B9).

## 7. Keeping it running

- **Deletion.** The API deletes a job's files 7 days after upload (`retention.days`), at
  startup and every 10 minutes. If the API is not running, nothing is deleted until it
  starts again.
- **Restarts.** Jobs that were waiting or running when the API stopped are handed back
  to the provider when it starts. A finished job is never redone.
- **Logs.** The API logs to its terminal. A local worker logs to `runs/jobs/<id>.log`,
  removed with its job.
- **Backups.** There are none, and little to back up: files are meant to disappear, and
  losing the database loses the list of jobs, not the ability to run new ones. Files
  whose rows were lost would no longer be swept; a bucket lifecycle rule covers that.
- **Upgrades.** `python -m stems_api.db.migrate` brings the database to the current
  schema; the API also does it at startup.
