#!/usr/bin/env bash
#SBATCH --job-name=trqa-gen
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
# Search-R1 with BM25 as its search tool, in one of four variants (only the search tool differs):
#   SYSTEM=search_r1              baseline: the BM25 top 3 for every query
#   SYSTEM=search_r1_notice       baseline + notice: adds "No more new results for this query." when none of
#                                 the 3 passages is new in this question
#   SYSTEM=search_r1_novel        memory: pages through the top NOVELTY_DEPTH passages, never repeats one
#                                 (fills up with old passages when fewer than 3 new ones remain)
#   SYSTEM=search_r1_novel_notice memory + notice: no fill-ups; shows the remaining new passages (possibly
#                                 none) plus the notice when fewer than 3 new ones remain
# The 7B model (~15 GB in fp16) does not fit on one 11 GB RTX 2080 Ti, so it is split
# over 2 GPUs (--device_map even: 14 of the 28 layers per GPU). One array task = one shard of the queries.
#
# Submit via ./submit.sh, or by hand, e.g.:
#   sbatch -A csedui00041 --array=0-1 --export=ALL,SYSTEM=search_r1,DATASET=wiki1,NUM_SHARDS=2 jobs/20_generate.sbatch
# Resuming: if a task hits the 12 h limit, submit the exact same command again;
# finished queries are skipped.

source jobs/common.sh
: "${SYSTEM:?set SYSTEM=search_r1, search_r1_notice, search_r1_novel or search_r1_novel_notice}"
: "${DATASET:?set DATASET=wiki1 or ecommerce}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
LIMIT="${LIMIT:-}"          # e.g. LIMIT=2 for a smoke test
EXP_TAG="${EXP_TAG:-}"      # optional suffix for the experiment folder

set_dataset "$DATASET"
require_index

case "$SYSTEM" in
  search_r1)
    EXP="search_r1_bm25${EXP_TAG}"
    SYS_ARGS=(--generation_method deep_research --deep_research_model search_r1 --retrieval_topk "$DRA_TOPK" --max_iter "$MAX_ITER")
    MODEL_TO_CHECK="$SEARCH_R1_MODEL" ;;
  search_r1_novel)
    # Same model, prompts, seed, temperature, MAX_ITER and DRA_TOPK as search_r1; only the search tool differs.
    case "${NOVELTY_UNIT:-}" in passage|document) ;; *) echo "ERROR: NOVELTY_UNIT must be passage or document (config.env), got '${NOVELTY_UNIT:-}'" >&2; exit 1 ;; esac
    [[ "${NOVELTY_DEPTH:-}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: NOVELTY_DEPTH must be a positive integer (config.env), got '${NOVELTY_DEPTH:-}'" >&2; exit 1; }
    [ "$RUN_NAME" != baseline ] || { echo "ERROR: novelty runs must not write to results/baseline; set RUN_NAME=phase2 in config.env" >&2; exit 1; }
    EXP="search_r1_bm25_novel_${NOVELTY_UNIT}_d${NOVELTY_DEPTH}${EXP_TAG}"
    SYS_ARGS=(--generation_method deep_research --deep_research_model search_r1 --retrieval_topk "$DRA_TOPK" --max_iter "$MAX_ITER"
              --novelty_unit "$NOVELTY_UNIT" --novelty_depth "$NOVELTY_DEPTH")
    MODEL_TO_CHECK="$SEARCH_R1_MODEL" ;;
  search_r1_notice)
    # Baseline + notice. Same as search_r1 except for the notice line.
    [ "$RUN_NAME" != baseline ] || { echo "ERROR: notice runs must not write to results/baseline; set RUN_NAME=phase2 in config.env" >&2; exit 1; }
    EXP="search_r1_bm25_notice${EXP_TAG}"
    SYS_ARGS=(--generation_method deep_research --deep_research_model search_r1 --retrieval_topk "$DRA_TOPK" --max_iter "$MAX_ITER"
              --stop_notice)
    MODEL_TO_CHECK="$SEARCH_R1_MODEL" ;;
  search_r1_novel_notice)
    # Memory + notice. Same as search_r1_novel except: no fill-ups, and the notice line.
    case "${NOVELTY_UNIT:-}" in passage|document) ;; *) echo "ERROR: NOVELTY_UNIT must be passage or document (config.env), got '${NOVELTY_UNIT:-}'" >&2; exit 1 ;; esac
    [[ "${NOVELTY_DEPTH:-}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: NOVELTY_DEPTH must be a positive integer (config.env), got '${NOVELTY_DEPTH:-}'" >&2; exit 1; }
    [ "$RUN_NAME" != baseline ] || { echo "ERROR: notice runs must not write to results/baseline; set RUN_NAME=phase2 in config.env" >&2; exit 1; }
    EXP="search_r1_bm25_novel_${NOVELTY_UNIT}_d${NOVELTY_DEPTH}_nofill_notice${EXP_TAG}"
    SYS_ARGS=(--generation_method deep_research --deep_research_model search_r1 --retrieval_topk "$DRA_TOPK" --max_iter "$MAX_ITER"
              --novelty_unit "$NOVELTY_UNIT" --novelty_depth "$NOVELTY_DEPTH" --stop_notice)
    MODEL_TO_CHECK="$SEARCH_R1_MODEL" ;;
  *) echo "ERROR: SYSTEM must be search_r1, search_r1_notice, search_r1_novel or search_r1_novel_notice" >&2; exit 1 ;;
esac

EXP_DIR="$RESULTS_ROOT/${DATASET}_${SPLIT}/$EXP"
SHARD_FILE="$(python tools/make_shards.py "$QUERIES" "$EXP_DIR/_shards" "$NUM_SHARDS" | sed -n "$((SHARD + 1))p")"
OUT="$EXP_DIR/shard_${SHARD}of${NUM_SHARDS}"
print_header "$SYSTEM | $DATASET $SPLIT | shard $SHARD/$NUM_SHARDS ($(wc -l < "$SHARD_FILE") queries) -> $OUT"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, '| GPUs:', torch.cuda.device_count())"

# Models were downloaded by the prepare job; stay offline so a flaky network can't kill a 12 h job
if python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_TO_CHECK', local_files_only=True)" 2>/dev/null; then
  export HF_HUB_OFFLINE=1
else
  echo "WARNING: $MODEL_TO_CHECK not in cache yet, it will be downloaded now"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation on the 11 GB cards
cd "$TRQA_DIR"
# shellcheck disable=SC2086
python c5_task_evaluation/run_evalution.py \
  --pipeline generation \
  --dataset "$TRQA_NAME" --subset "$SPLIT" \
  --dataset_file "$SHARD_FILE" --qrel_file "$QRELS" \
  --retriever bm25 --index_dir "$INDEX_DIR" --corpus_path "$CORPUS" \
  --retrieval_eval_ks $RETRIEVAL_EVAL_KS \
  "${SYS_ARGS[@]}" \
  --model_dtype "$MODEL_DTYPE" --device_map even --devices 0 --prefill_chunk_size 2048 \
  --seed "$SEED" --run "$RUN_NAME" --output_dir "$OUT" \
  ${LIMIT:+--limit "$LIMIT"}

cd "$PROJECT_DIR"
echo; echo "---- merged results so far ($EXP_DIR) ----"
python tools/merge_results.py "$EXP_DIR" "$QUERIES" || true
