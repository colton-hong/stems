# Security review: uploads, downloads, and what is kept

Done on 2026-11-24 for task P7.6, by reading the code of the API (`api/stems_api`), the
worker's job entry point (`worker/stems_worker/jobs.py`) and the web app (`web/`), with the
question "what can a stranger's upload do to the server, and what can anyone other than
the uploader get at?". It is a review by the author of the code, not an independent audit
and not a penetration test. Nothing was attacked from outside; no scanner was run.

Each finding below names the test that now holds it. "Left as it is" lists what was seen
and not changed, with the reason.

## Found and fixed

| # | What was wrong | What it could lead to | Fix | Held by |
|---|---|---|---|---|
| 1 | The worker kept separated audio in a shared cache (`runs/cache/separation`), keyed by a hash of the upload. The sweeper never looked there. | Stems of an upload outliving its 7-day deletion, and its "Delete now". | The API's worker caches only inside the job's own temporary folder, which is removed when the job ends. The shared cache is now used by the command-line tools and the benchmark only. | `test_nothing_made_from_the_upload_is_left_on_disk` |
| 2 | A worker stopped with SIGTERM (a deleted job, a shutdown) died without removing its temporary folder. | A copy of the upload left in the system's temp directory. | The worker turns SIGTERM into a normal exit, so the folder is removed on the way out. Folders left by a process that was killed outright are removed by the sweeper after 48 hours. | `test_a_worker_told_to_stop_cleans_up_and_writes_no_result`, `test_temporary_folders_left_by_a_dead_process_are_removed` |
| 3 | A job's worker log (`runs/jobs/<id>.log`) was never deleted. | Tracebacks and file names from a job outliving the job. | Deleting or expiring a job removes its log. | `test_a_jobs_log_goes_when_the_job_goes` |
| 4 | A deleted or expired job kept its decision log rows, whose answers describe the request and the audio. | Details of a deleted upload staying in the database. | Deletion and expiry remove them. (Found in Phase 5.) | `test_delete_removes_every_object_and_the_links_stop_working`, `test_a_job_past_retention_loses_its_objects_and_is_marked_expired` |
| 5 | The upload check ran the media decoders (libsndfile, FFmpeg through PyAV) inside the API process. | A file that crashes or hangs a decoder taking the API down, or tying up its threads. | The check runs in a child process with a 20-second limit. No answer, or a late one, refuses the upload. | `test_a_decoder_that_hangs_or_crashes_refuses_the_upload_and_nothing_else`, `test_the_time_limit_is_real` |
| 6 | Any sample rate and channel count was accepted. | A 20-minute file at an extreme rate decoding to many gigabytes in the worker. | Uploads outside 8 kHz to 192 kHz, or with more than 8 channels, are refused. | `test_extreme_sample_rates_and_channel_counts_are_refused` |
| 7 | A failed job showed the browser the exception's own message. | Server paths and other internals shown to a user. | The browser is told the kind of error only; the message goes to the server's log. A problem with the audio itself is still said in full. | `test_a_failure_tells_the_user_its_kind_and_keeps_the_details_in_the_log` |
| 8 | Jobs were limited per browser and per address, with no limit overall. | Many browsers filling the queue of a machine that runs one job at a time. | At most 12 jobs queued or running in all; beyond that an upload gets 503 with `Retry-After`. | `test_uploads_are_turned_away_when_the_whole_server_is_busy` |
| 9 | The local storage folder was created with the default permissions. | Other users of the same machine reading uploads off the disk. | A folder this code creates is made owner-only (0700). One that already existed is left alone. | `test_a_storage_folder_this_code_creates_is_private` |
| 10 | Replies carried no instructions about caching or content types; pages could be put in a frame. | Job data kept in a shared cache; a page framed by another site. | The API sends `Cache-Control: no-store`, `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer` on every reply. The web app sends the last two plus `X-Frame-Options: DENY`, a `frame-ancestors 'none'` policy and a permissions policy with no camera, microphone or location. | `test_every_reply_says_how_it_may_be_handled`, the browser test of page headers |
| 11 | A download link's token verified in more than one spelling. | Nothing gained by an attacker (same file, same expiry), but the token was not what it claimed to be. (Found in Phase 5.) | Only the one spelling the server writes is accepted. | `test_a_token_has_exactly_one_spelling` |
| 12 | The test suites read the owner's `.env`. | A test calling a paid service with a real key. (Found in Phase 6.) | Tests, and every process they start, are pointed at an `.env` that does not exist. | `test_another_dotenv_file_can_be_named_and_the_tests_name_one_that_is_absent` |

## Looked at and found sound

- **Who can reach a job.** Every job route goes through `JobStore`, which is bound to the
  owner the cookie identifies; another browser gets exactly the 404 a missing job gets
  (`test_every_job_route_answers_404_to_another_browser`, and the same in a real browser).
- **The owner cookie.** 256 random bits, HttpOnly, SameSite=Lax, Secure over HTTPS, and
  only its SHA-256 in the database.
- **Requests from other sites.** The cookie's SameSite rule keeps it off cross-site
  POSTs; state-changing requests that name another origin are refused as well; CORS
  names only the configured web origin.
- **Object keys.** Built only from server-made UUIDs and a fixed list of stem names,
  checked before any backend sees them; the uploaded file's name never reaches a key or
  a path. `..`, absolute paths and symbolic links out of the storage folder are refused.
- **Download links.** An HMAC over the key and the expiry, compared in constant time;
  they expire in 15 minutes and are made only for stems and the zip, never for the
  upload itself. The download name is stripped of quotes, slashes and control characters.
- **Upload size.** Enforced while the body is read, so a missing or false
  `Content-Length` does not get around it.
- **What a user types.** Shown as text everywhere in the web app; nothing is inserted as
  HTML. Database access is through the ORM with bound parameters.
- **Secrets.** Keys come from the environment or `.env`, which git ignores; the Jev and
  Clef-omni clients scrub them from logs; the committed recordings were checked for them.

## Dependency audit (2026-11-24, task P8.6)

Both package sets were checked against the public advisory databases. Neither tool is
part of `make check` or CI; these are two runs on one day.

| What | Command | When (UTC) | Result |
|---|---|---|---|
| Python packages installed in `.venv` | `pip-audit --path .venv/lib/python3.14/site-packages` (pip-audit 2.10.1, installed in a separate environment so that auditing changed nothing in `.venv`) | 2026-11-24 11:32 | 137 packages listed. 136 checked: no known vulnerability. 1 skipped: this project itself, which is not on PyPI. |
| npm packages of the web app | `npm audit` in `web/`, which reads `package-lock.json` | 2026-11-24 11:09 | 402 packages (19 runtime, 345 development, 60 optional). No known vulnerability at any severity. |

**Changed:** nothing. There was nothing to fix.

**Left, and why:**

- **A clean result on one day.** An advisory published tomorrow is not in it. Nothing
  reruns the audit; to repeat it, run the two commands above (pip-audit from outside
  `.venv`).
- **Known advisories only.** No package's code was read, and a flaw nobody has reported
  is not found this way. The media decoders (FFmpeg inside the PyAV wheel, libsndfile
  inside soundfile) are the packages most exposed to hostile files; they are covered as
  packages, and their decoding of an upload is confined as finding 5 describes.
- **Model files are not packages** and neither tool looks at them. They are downloaded
  from the URLs in `models/registry.yaml`; the SHA-256 of each is recorded in
  `models/checksums.lock.json` the first time it is fetched, and a file that later
  differs is refused. Every `torch.load` in the worker passes `weights_only=True`, so
  loading a checkpoint does not run code stored in it. That the files were what their
  authors published when first fetched is taken on trust.
- **Python, Node.js and the operating system** are outside both audits.

## Left as it is, and why

- **A download link is all anyone needs.** Whoever has a link can fetch that file until
  it expires, cookie or not. That is how signed links work and is what PLAN.md asks for;
  the 15-minute life is the protection. The links do appear in the server's access log.
- **Rate limits live in one process and count the connection's address.** Behind a
  reverse proxy every user shares one address, and several API processes each allow the
  full limit. Fine for one process on one machine; a wider deployment needs the proxy's
  forwarded address and a shared counter (BLOCKED.md D7).
- **Event streams are not limited.** A browser can hold many `/events` connections open.
  Each costs little while idle; there is no cap.
- **The database file has default permissions.** It holds cookie hashes, file names and
  job details, not audio. It sits in `runs/dev/`, which other tools also use, so it was
  not locked down here.
- **No full Content-Security-Policy for scripts.** Next.js needs inline scripts unless
  each is given a nonce; the policy sent covers framing, plugins, base and form targets
  only.
- **The decoders themselves.** The child process contains a crash or a hang during the
  upload check. The worker then decodes the whole file, in its own process, with the
  same libraries: a file that passes the first second and breaks a decoder later fails
  that job, not the API.
- **API documentation is served** at `/api/docs`. It describes the routes and nothing
  private.
- **Dependencies** were checked for known advisories on 2026-11-24 (next section) and
  nothing else: their code was not read.
- **Cloud storage.** The S3 and Azure backends have never run against a real bucket
  (BLOCKED.md B7). Bucket policy, encryption at rest and CORS there are unreviewed.
