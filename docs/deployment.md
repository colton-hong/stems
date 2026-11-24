# Deployment guide

How to run Stems, from one laptop to rented GPUs and a bucket. Every step below is marked:

- **Run here** means it was carried out on the development machine (an Apple-silicon Mac)
  on the date given, and worked.
- **Not run** means the code exists and its tests pass, but nobody has done this for real.
  Expect to fix small things the first time.

Only the local setup has ever run end to end, and only with stand-in models (frequency
bands, not music). No deployment beyond one machine has been tried.

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
| Upload a song in the browser and get stems, **real models** | nothing extra: weights download on first use | **Not run** |

Nothing has to be configured. With no `.env` the API uses a SQLite file in `runs/dev/`,
keeps files in `runs/dev/storage/`, makes its own link-signing secret there, runs jobs on
this machine one at a time, and decides with rules instead of Jev.

The last row is the gap that matters. Real checkpoints have run through the command
line (`separate`, Phase 1) and through the benchmark, but a job with real models has
never gone through the API. The first one downloads the checkpoints its plan names
(about 16.7 GB for all of them) and needs the memory they need; on a 24 GB Mac do not run
it while a benchmark is running.

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
  HTTPS request. If a proxy terminates TLS, it must tell the API so. `python -m
  stems_api` has no option for that; running the app through uvicorn directly
  (`uvicorn stems_api.app:create_app --factory --proxy-headers --forwarded-allow-ips ...`)
  should do it, and would also give the rate limiter each visitor's own address. This has
  not been set up or tested.
- **Same site.** The cookie is SameSite=Lax, so the web app and the API must share a
  registrable domain (`app.example.org` and `api.example.org` are fine; two unrelated
  domains are not). Set `STEMS_WEB_ORIGIN` to the web app's exact origin and build the web
  app with `NEXT_PUBLIC_API_URL` set to the API's.
- **One API process.** Rate limits and the local job queue live in the API's memory. A
  second process would double the limits and not see the first one's queue. Behind a
  proxy the rate limiter sees the proxy's address for everyone (BLOCKED.md D7).
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
