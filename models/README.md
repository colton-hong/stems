# models

`registry.yaml` lists every checkpoint the pipeline can load. Code reads it only through
`stems_models.registry.load_registry()`.

```
python -m stems_models                          # validate and print a summary table
python -m stems_models.check_sources            # HEAD every URL, write runs/registry/source_check.json
python -m stems_models.check_sources --write    # also refresh size_bytes in registry.yaml
```

## Downloading

```
python -m stems_models.download mel_roformer_kim_vocals      # one model, by id
python -m stems_models.download htdemucs_ft --stem bass      # only the files one stem needs
python -m stems_models.download --tier candidate --background
python -m stems_models.download --status                     # or: make models-status
```

Files go to `models/weights/<id>/` (or under `STEMS_MODELS_DIR`) and are never committed.
Downloads resume from a `.part` file, skip anything already complete, and check the byte
count against the registry.

`checksums.lock.json` is committed. The first download of a URL records its SHA-256 there;
every later download, and every load, must match. If a file changes upstream the download
is refused: check the source, then delete that URL's entry from the lock file to accept the
new content.

Progress for each model is in `runs/downloads/<id>/status.json`, and background runs log to
`runs/downloads/background.log`. In code, call `stems_models.download.ensure_model(id)`,
which returns the verified file paths.

## Entry fields

| Field | Meaning |
|---|---|
| `id` | Stable identifier the pipeline, config and benchmark use. Lowercase, digits, underscores. |
| `name` | Human-readable name. |
| `kind` | `separator`, `tagger` or `decision_model`. |
| `architecture` | Model family, for grouping in benchmark reports. |
| `loader` | Which adapter loads it: `msst`, `demucs`, `uvr_mdx_onnx`, `uvr_vr`, `beats`, `laion_clap`, `panns`, `hf_transformers`, `essentia_tf`, `workers_ai`, or `none`. |
| `stems` | Outputs the checkpoint emits: the six product stems plus `instrumental`, `lead_vocals`, `backing_vocals`, `dry`. |
| `candidate_for` | Slots it is benchmarked for: a product stem, `vocal_split`, `dereverb`, `tagging`, `listening_test`. |
| `tier` | `candidate` (named in the brief), `baseline`, `extra` (benchmarked later), `watch` (no weights yet). |
| `origin` | `report_s1`, `report_s5`, or `added` (found while verifying sources). |
| `status` | `available` (downloadable), `hosted` (API only), `unavailable` (no public weights). |
| `sample_rate_hz` | Rate the model expects. |
| `source_page` | Where a person can read about it. |
| `files` | Each with `role` (`weights`, `config`, `config_index`), `filename`, `url`, `size_bytes`, and optional `stem` for one-file-per-stem bags. `size_bytes` must be the line after `url`. |
| `config` | What the loader needs beyond the files. Each adapter validates its own keys. |
| `licence_note` | One line, for reference. It gates nothing. |
| `notes` | Anything a reader should know. |
| `unverified` | Facts to confirm when the model is first loaded. Clear them as they are confirmed. |

## Two meanings of "other"

`other` in a 4-stem model still contains guitar and piano. `other` in a 6-stem model does
not. `ModelEntry.other_contains_guitar_piano` tells them apart. Two-stem vocal models emit
`instrumental`, never `other`.

## Adding a model

Add an entry, run `python -m stems_models.check_sources --write`, run `make check`. To make
it a default, edit `config/defaults.yaml`, and only with a benchmark run that supports it.
