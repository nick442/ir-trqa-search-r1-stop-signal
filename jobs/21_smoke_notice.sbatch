#!/usr/bin/env bash
#SBATCH --job-name=trqa-smoke-notice
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
# 2x2 experiment: ONE smoke job for both notice variants, one after the other on the same 2 GPUs
# (the model is loaded twice, ~2 min each). Submit via ./submit.sh smoke_notice.
#   part 1: search_r1_notice        (baseline + notice)
#   part 2: search_r1_novel_notice  (memory + notice) with a small depth (SMOKE_NOTICE_DEPTH, default 6), so the
#           top D is used up after two searches of a query and the notice shows up within a few questions
# Both: first LIMIT (3) questions of wiki1, TRQA_DEBUG=1, folders *_smoke.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export DATASET=wiki1 NUM_SHARDS=1 LIMIT="${LIMIT:-3}" EXP_TAG=_smoke TRQA_DEBUG=1
echo "######## smoke part 1 of 2: search_r1_notice (baseline + notice) ########"
GEN_JOB="${GEN_JOB:-jobs/20_generate.sbatch}"   # the local test harness points this at a copy without the GPU checks
SYSTEM=search_r1_notice bash "$GEN_JOB"
echo "######## smoke part 2 of 2: search_r1_novel_notice (memory + notice), depth ${SMOKE_NOTICE_DEPTH:-6} ########"
SYSTEM=search_r1_novel_notice SUBMIT_NOVELTY_DEPTH="${SMOKE_NOTICE_DEPTH:-6}" bash "$GEN_JOB"
echo "######## smoke_notice finished ########"
