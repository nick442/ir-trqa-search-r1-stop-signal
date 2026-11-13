#!/usr/bin/env bash
# Convenience wrapper around sbatch. Run from the project root on cnlogin22.
#
#   ./submit.sh smoke_notice                 both notice variants on 3 wiki1 queries each (one 2-GPU job)
#   ./submit.sh search_r1 <wiki1|ecommerce> [num_shards]                         baseline
#   ./submit.sh search_r1_notice <wiki1|ecommerce> [num_shards]                  baseline + notice
#   ./submit.sh search_r1_novel <wiki1|ecommerce> [num_shards] [unit] [depth]    memory
#   ./submit.sh search_r1_novel_notice <wiki1|ecommerce> [num_shards] [unit] [depth]   memory + notice
#                                            unit (passage|document) and depth default to config.env
#   ./submit.sh status                       your queued/running jobs
#
# SPLIT, RUN_NAME, SEED (and the novelty settings) are read from config.env NOW and frozen into the job:
# editing config.env afterwards does not change jobs that are already queued.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source ./config.env
mkdir -p logs

SB=(sbatch --parsable -A "$ACCOUNT")
[ -n "${MAIL_USER:-}" ] && SB+=(--mail-user="$MAIL_USER" --mail-type=END,FAIL)

default_shards() { case "$1" in wiki1) echo 1 ;; ecommerce) echo 2 ;; esac; }

# Settings frozen into every generation job at submit time (see jobs/common.sh)
FROZEN="SUBMIT_SPLIT=$SPLIT,SUBMIT_RUN_NAME=$RUN_NAME,SUBMIT_SEED=$SEED"

guard_run_name() {  # the Phase 1 results in results/baseline are final: nothing may be added there by accident
  if [ "$RUN_NAME" = baseline ] && [ "${ALLOW_BASELINE_WRITE:-0}" != 1 ]; then
    echo "ERROR: config.env has RUN_NAME=baseline. Phase 2 runs go to RUN_NAME=phase2 (never results/baseline)." >&2
    echo "       Change RUN_NAME in config.env, or prefix the command with ALLOW_BASELINE_WRITE=1 if you really mean it." >&2
    exit 1
  fi
}

novelty_env() {  # novelty_env [unit] [depth] -> sets NV_UNIT NV_DEPTH NV_ENV (defaults from config.env)
  NV_UNIT="${1:-${NOVELTY_UNIT:-}}"; NV_DEPTH="${2:-${NOVELTY_DEPTH:-}}"
  case "$NV_UNIT" in passage|document) ;; *) echo "ERROR: novelty unit must be passage or document, got '$NV_UNIT' (set NOVELTY_UNIT in config.env)" >&2; exit 1 ;; esac
  [[ "$NV_DEPTH" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: novelty depth must be a positive integer, got '$NV_DEPTH' (set NOVELTY_DEPTH in config.env)" >&2; exit 1; }
  NV_ENV="SUBMIT_NOVELTY_UNIT=$NV_UNIT,SUBMIT_NOVELTY_DEPTH=$NV_DEPTH"
}

gen() {  # gen <system> <dataset> [num_shards] [extra env]
  local system="$1" ds="$2" n="${3:-$(default_shards "$2")}" extra="${4:-}"
  guard_run_name
  echo "submitting $system | $ds $SPLIT | $n shard(s) | seed $SEED${NV_UNIT:+ | novelty $NV_UNIT d$NV_DEPTH} -> results/$RUN_NAME/${ds}_${SPLIT}/" >&2
  "${SB[@]}" --job-name="trqa-${system}-${ds}" --array=0-$((n - 1)) \
    --export=ALL,SYSTEM="$system",DATASET="$ds",NUM_SHARDS="$n","$FROZEN"${extra:+,$extra} jobs/20_generate.sbatch
}

cmd="${1:-help}"
case "$cmd" in
  smoke_notice)
    guard_run_name; novelty_env "" ""
    echo "submitting smoke: search_r1_notice + search_r1_novel_notice ($NV_UNIT, depth ${SMOKE_NOTICE_DEPTH:-6}) | wiki1 $SPLIT | 3 queries each | seed $SEED -> results/$RUN_NAME/wiki1_${SPLIT}/*_notice_smoke" >&2
    "${SB[@]}" --array=0 --export=ALL,"$FROZEN","SUBMIT_NOVELTY_UNIT=$NV_UNIT" jobs/21_smoke_notice.sbatch ;;
  search_r1|search_r1_notice)
    [ $# -ge 2 ] || { echo "usage: $0 $cmd <wiki1|ecommerce> [num_shards]"; exit 1; }
    gen "$cmd" "$2" "${3:-}" ;;
  search_r1_novel|search_r1_novel_notice)
    [ $# -ge 2 ] || { echo "usage: $0 $cmd <wiki1|ecommerce> [num_shards] [passage|document] [depth]"; exit 1; }
    novelty_env "${4:-}" "${5:-}"
    gen "$cmd" "$2" "${3:-}" "$NV_ENV" ;;
  status) squeue -u "$USER" -o "%.10i %.28j %.8T %.10M %.10l %R" ;;
  *) sed -n '2,12p' "$0" ;;
esac
