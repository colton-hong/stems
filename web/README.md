# web

The Stems web app: Next.js 16 (App Router, TypeScript), no UI library. It talks to the API
in `api/` and to nothing else.

```
make node-setup   # only if you have no Node.js >= 22.13; installs v24 LTS into .tools/node
make web-setup    # npm ci
make web-check    # lint, typecheck, unit tests, build
make web-e2e      # browser tests (see below)
```

To use it by hand, start the API and the app in two terminals and open
http://localhost:3000:

```
make api
```

```
cd web && npm run dev
```

`NEXT_PUBLIC_API_URL` says where the API is (default `http://localhost:8000`); the API's
`STEMS_WEB_ORIGIN` must name the address the app is served from (default
`http://localhost:3000`). Use `localhost` for both, not `127.0.0.1` for one of them: the
owner cookie is tied to the host name.

## Where things are

| Path | What |
|---|---|
| `app/` | Pages: upload (`/`), a job (`/jobs/[id]`), this browser's jobs (`/jobs`), Terms and Privacy placeholders |
| `components/` | `UploadForm`, `JobView`, `StemPlayer`, `Waveform`, `JobsList`, `ConfirmButton`, `Notice` |
| `lib/api.ts` | The typed API client and its error type |
| `lib/player.ts` | The player engine: every stem started at one time on one audio clock |
| `lib/useJob.ts` | A job kept current from the API's server-sent events, with polling as fallback |
| `lib/*.test.ts` | Unit tests, run by Node's own test runner (`npm test`) |
| `e2e/` | Browser tests (Playwright, with axe for accessibility), at desktop and phone width |
| `e2e-real/` | One browser test with the real models (`make web-e2e-real`) |

## Browser tests

`make web-e2e` starts the real API on port 8787 with `STEMS_WORKER_FAKE_MODELS=1` (stand-in
models that need no weights, so the stems are frequency bands, not music), builds and
serves the app on port 3939, and drives Chrome through: upload, the job reaching Ready by
itself, play, solo, mute, seek, download, delete, and the pages around them. The run
uses its own database and storage under `web/.e2e/` and no `.env`, so it never touches
real jobs or keys.

Locally it uses the Chrome already installed. Elsewhere, install Playwright's browser
(`npx playwright install chromium`) and set `STEMS_E2E_CHANNEL=chromium`.

The "Speed or quality" menu is shown only when the choice can change what a job does. The
API says what each setting gives (`quality_gives` in `/api/config`); when the rules plan
the job and every setting gives the same thing, the page shows a line saying so instead.

`e2e/mobile.spec.ts` repeats the flow and the axe checks as Chrome emulates a phone held
upright (375 by 812, touch): nothing may run off the side of the screen, and no button,
menu or link-button may be smaller than 24 by 24 pixels.

`make web-e2e-real` is separate and slow: the same servers with the real models (the
weights must be on disk), one upload of a 12-second synthetic signal asking for all six
stems, played and downloaded in Chrome. It writes what ran, and how long each model took,
to `web/.e2e-real/real-job.json`. Set `STEMS_DEVICE=cpu` to keep it off the GPU.

What these tests do not show: how real separated music sounds in the player, any browser
other than Chrome, a real phone (the phone tests are desktop Chrome at a phone's size),
or a real screen reader. The accessibility checks are axe, a keyboard-only test, and
roles and names read from the accessibility tree.

## The player and memory

The player decodes every stem in full so that they can share one clock. By arithmetic that is
about 21 MB per minute per stem (32-bit float, stereo, 44.1 kHz); it has not been measured. Tracks up to 10 minutes load
by themselves; longer ones wait for a click on "Load the player".

## Version pins that are deliberate

- **TypeScript `~6.0.3`, not 7.** `typescript-eslint` 8 (pulled in by `eslint-config-next`)
  refuses TypeScript 7 and accepts only `<6.1.0`.
- **ESLint 9, not 10.** `eslint-plugin-import`, `eslint-plugin-jsx-a11y` and
  `eslint-plugin-react` (also from `eslint-config-next`) declare peer support up to ESLint 9.
  npm marks ESLint 9 as no longer supported, so move to 10 as soon as those plugins allow it.

Next.js telemetry is switched off in the npm scripts. `AGENTS.md` in this folder is
written by `next dev` itself.
