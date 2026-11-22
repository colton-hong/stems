# web

Next.js 16 (App Router, TypeScript) front end. The real UI is built in Phase 6; today this is
a placeholder page that installs, lints, type-checks and builds.

```
make node-setup   # only if you have no Node.js >= 22.13; installs v24 LTS into .tools/node
make web-setup    # npm ci
make web-check    # lint, typecheck, build
```

## Version pins that are deliberate

- **TypeScript `~6.0.3`, not 7.** `typescript-eslint` 8 (pulled in by `eslint-config-next`)
  refuses TypeScript 7 and accepts only `<6.1.0`.
- **ESLint 9, not 10.** `eslint-plugin-import`, `eslint-plugin-jsx-a11y` and
  `eslint-plugin-react` (also from `eslint-config-next`) declare peer support up to ESLint 9.
  npm marks ESLint 9 as no longer supported, so move to 10 as soon as those plugins allow it.

Next.js telemetry is switched off in the npm scripts.
