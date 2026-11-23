#!/bin/sh
# The benchmarks on real music, staged so the first reports arrive early.
#
#   sh tools/run_real_benchmarks.sh            # from the repository root
#
# Stage 1 and 2 score the config's default models and blends on each dataset; stage 3 and
# 4 add every other separator. Each stage ends by rewriting that dataset's report. Jobs
# that already have a result are skipped, so this is safe to stop and run again.
#
# Needs MUSDB18HQ_PATH and MOISESDB_PATH in .env and both frozen lists committed. A
# dataset that is not ready is skipped with its reason and the rest still runs.
set -u
PY=.venv/bin/python

stage() {
  dataset=$1; run_id=$2; report=$3; shift 3
  echo "=== $(date -u +%FT%TZ) $dataset: $*"
  $PY -m stems_bench run --dataset "$dataset" --run-id "$run_id" "$@"
  code=$?
  if [ $code -eq 2 ]; then
    echo "=== $dataset skipped (see the message above)"
    return
  fi
  $PY -m stems_bench report "$run_id" --out "$report"
}

stage musdb18hq_test  musdb18hq-v1 docs/benchmarks/musdb18hq.md --ensembles --quiet
stage moisesdb_subset moisesdb-v1  docs/benchmarks/moisesdb.md  --ensembles --quiet
stage musdb18hq_test  musdb18hq-v1 docs/benchmarks/musdb18hq.md --all --ensembles --quiet
stage moisesdb_subset moisesdb-v1  docs/benchmarks/moisesdb.md  --all --ensembles --quiet
echo "=== $(date -u +%FT%TZ) done"
