# shellcheck shell=bash
# Sourced by every job script. Sets paths, activates the environment and
# maps a dataset name (wiki1 | ecommerce) to its queries/qrels/index/corpus.
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f "$PROJECT_DIR/config.env" ]; then
  echo "ERROR: run sbatch from the project root (the folder containing config.env)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$PROJECT_DIR/config.env"

# Settings frozen at submit time (submit.sh, v9+). A job waiting in the queue keeps the SPLIT / RUN_NAME /
# SEED / novelty settings it was submitted with, even if config.env is edited in the meantime. Jobs
# submitted without these variables (by hand, or by an older submit.sh) use config.env as it is when
# they start.
SPLIT="${SUBMIT_SPLIT:-$SPLIT}"
RUN_NAME="${SUBMIT_RUN_NAME:-$RUN_NAME}"
SEED="${SUBMIT_SEED:-$SEED}"
NOVELTY_UNIT="${SUBMIT_NOVELTY_UNIT:-${NOVELTY_UNIT:-}}"
NOVELTY_DEPTH="${SUBMIT_NOVELTY_DEPTH:-${NOVELTY_DEPTH:-}}"

TRQA_DIR="$PROJECT_DIR/total-recall-qa"
RESULTS_ROOT="$DATA_ROOT/results/$RUN_NAME"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR" "$RESULTS_ROOT"

# Keep every cache off the 5 GB home directory
export HF_HOME="$DATA_ROOT/hf_home"
export HF_DATASETS_CACHE="$DATA_ROOT/hf_home/datasets"
export PIP_CACHE_DIR="$DATA_ROOT/pip_cache"
export TMPDIR="${TMPDIR:-$DATA_ROOT/tmp}"
mkdir -p "$TMPDIR"
export JAVA_HOME="$DATA_ROOT/jdk"
export PATH="$JAVA_HOME/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# pyserini creates an OpenAI client at import time and crashes without a key (not used for BM25)
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-used}"

if [ ! -f "$DATA_ROOT/venv/bin/activate" ]; then
  echo "ERROR: no venv at $DATA_ROOT/venv" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$DATA_ROOT/venv/bin/activate"

case "$SPLIT" in
  test) SPLIT_DIR=test; SPLIT_SHORT=test ;;
  val)  SPLIT_DIR=validation; SPLIT_SHORT=val ;;
  *) echo "ERROR: SPLIT must be test or val" >&2; exit 1 ;;
esac

# set_dataset wiki1|ecommerce  -> TRQA_NAME QUERIES QRELS INDEX_DIR CORPUS
set_dataset() {
  DATASET="$1"
  case "$DATASET" in
    wiki1)
      TRQA_NAME=wikidata   # name the TRQA code uses for the Wiki1 subset
      QUERIES="$TRQA_DIR/dataset/TRQA/Wiki1/$SPLIT_DIR/queries_wiki1_${SPLIT_SHORT}.jsonl"
      QRELS="$TRQA_DIR/dataset/TRQA/Wiki1/$SPLIT_DIR/trec_qrels_wiki1_${SPLIT_SHORT}.txt"
      INDEX_DIR="$DATA_ROOT/indices/wiki_partial"
      CORPUS="${WIKI_PARTIAL_CORPUS:-$DATA_ROOT/corpus/data/shared/corpus_wikipedia_en-20251001_rewritten_partial.jsonl}"
      SHARED_INDEX="${WIKI_PARTIAL_BM25_INDEX:-}"
      ;;
    ecommerce)
      TRQA_NAME=synthetic_ecommerce
      QUERIES="$TRQA_DIR/dataset/TRQA/E-Commerce/$SPLIT_DIR/queries_ecommerce_${SPLIT_SHORT}.jsonl"
      QRELS="$TRQA_DIR/dataset/TRQA/E-Commerce/$SPLIT_DIR/trec_qrels_ecommerce_${SPLIT_SHORT}.txt"
      INDEX_DIR="$DATA_ROOT/indices/ecommerce"
      CORPUS="${ECOMMERCE_CORPUS:-$DATA_ROOT/corpus/data/ecommerce/corpus_ecommerce_rewritten.jsonl}"
      SHARED_INDEX="${ECOMMERCE_BM25_INDEX:-}"
      ;;
    *) echo "ERROR: unknown dataset '$DATASET' (use wiki1 or ecommerce)" >&2; exit 1 ;;
  esac
  for f in "$QUERIES" "$QRELS"; do
    [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
  done
  # A prebuilt index from shared/ is used through a symlink, because the TRQA code
  # expects <index_dir>/bm25_index
  if [ -n "$SHARED_INDEX" ]; then
    INDEX_DIR="$DATA_ROOT/indices/${DATASET}_shared"
    mkdir -p "$INDEX_DIR"
    ln -sfn "$SHARED_INDEX" "$INDEX_DIR/bm25_index"
  fi
}

require_index() {
  if ! ls "$INDEX_DIR"/bm25_index/segments_* >/dev/null 2>&1 || [ -d "$INDEX_DIR/bm25_index.tmp" ]; then
    echo "ERROR: BM25 index for $DATASET not ready at $INDEX_DIR/bm25_index" >&2
    exit 1
  fi
}

print_header() {
  echo "=================================================================="
  echo " job ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+ (array task $SLURM_ARRAY_TASK_ID)} on $(hostname) at $(date)"
  echo " $*"
  echo " SPLIT=$SPLIT RUN_NAME=$RUN_NAME SEED=$SEED$([ -n "${SUBMIT_RUN_NAME:-}" ] && echo " (frozen at submit time)")"
  echo " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
  echo "=================================================================="
}
