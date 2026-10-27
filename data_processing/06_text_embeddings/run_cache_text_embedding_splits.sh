#!/usr/bin/env bash
set -euo pipefail
: "${ROBODREAM_DATA_DIR:?source shell/export.sh first}"

# Launch 8 disjoint caching jobs, pinned 1 job per GPU via CUDA_VISIBLE_DEVICES,
# then merge all split pickle shards into the final OUTPUT file and delete shards.
#
# Usage:
#   source shell/export.sh
#   bash data_processing/06_text_embeddings/run_cache_text_embedding_splits.sh
#
# Optional overrides (env vars):
#   ANNOTATIONS_JSON=${ROBODREAM_DATA_DIR}/droid_meta/droid_language_annotations.json
#   TEXT_ENCODER_PATH=${ROBODREAM_CKPT_DIR}/google-t5/t5-11b
#   OUTPUT=${ROBODREAM_DATA_DIR}/droid_t5_text_embeddings.pkl.gz
#   TOTAL_SPLITS=8                         (number of GPUs, one split per GPU)
#   BATCH_SIZE=32
#   MAX_LENGTH=512
#   UPLOAD_S3=s3://.../Droid_traj2demo/    (leave empty to disable upload)
#   PYTHON=python
#   MERGE_SHARDS=1                         (set 0 to skip merging)
#   UPLOAD_SHARDS=0                        (set 1 to upload each shard too)

ANNOTATIONS_JSON="${ANNOTATIONS_JSON:-${ROBODREAM_DATA_DIR}/droid_meta/droid_language_annotations.json}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-${ROBODREAM_CKPT_DIR}/google-t5/t5-11b}"
OUTPUT="${OUTPUT:-${ROBODREAM_DATA_DIR}/droid_t5_text_embeddings.pkl.gz}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
UPLOAD_S3="${UPLOAD_S3:-}"
PYTHON="${PYTHON:-python}"
MERGE_SHARDS="${MERGE_SHARDS:-1}"
UPLOAD_SHARDS="${UPLOAD_SHARDS:-0}"

TOTAL_SPLITS="${TOTAL_SPLITS:-8}"
LOG_DIR="${LOG_DIR:-$(dirname "$OUTPUT")/logs_cache_text_embedding}"
mkdir -p "$LOG_DIR"

echo "Launching ${TOTAL_SPLITS} splits"
echo "  ANNOTATIONS_JSON=$ANNOTATIONS_JSON"
echo "  TEXT_ENCODER_PATH=$TEXT_ENCODER_PATH"
echo "  OUTPUT=$OUTPUT"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  MAX_LENGTH=$MAX_LENGTH"
echo "  UPLOAD_S3=${UPLOAD_S3:-<disabled>}"
echo "  MERGE_SHARDS=$MERGE_SHARDS"
echo "  UPLOAD_SHARDS=$UPLOAD_SHARDS"
echo "  LOG_DIR=$LOG_DIR"
echo

pids=()
for split_idx in $(seq 0 $((TOTAL_SPLITS - 1))); do
  gpu="${split_idx}"
  log_file="${LOG_DIR}/split${split_idx}.log"

  echo "Starting split ${split_idx}/${TOTAL_SPLITS} on GPU ${gpu} -> ${log_file}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    set -x
    cmd=(
      "$PYTHON" data_processing/06_text_embeddings/cache_text_embedding.py
      --annotations_json "$ANNOTATIONS_JSON"
      --text_encoder_path "$TEXT_ENCODER_PATH"
      --output "$OUTPUT"
      --batch_size "$BATCH_SIZE"
      --max_length "$MAX_LENGTH"
      --total_splits "$TOTAL_SPLITS"
      --split_idx "$split_idx"
    )
    # By default, avoid uploading shards; upload only the merged final file.
    if [[ -n "$UPLOAD_S3" && "$UPLOAD_SHARDS" == "1" ]]; then
      cmd+=( --upload_s3 "$UPLOAD_S3" )
    else
      cmd+=( --upload_s3 "" )
    fi
    "${cmd[@]}"
  ) >"$log_file" 2>&1 &

  pids+=( "$!" )
done

echo
echo "Launched PIDs: ${pids[*]}"
echo "Waiting for all splits to finish..."

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done

if [[ "$rc" -ne 0 ]]; then
  echo "One or more splits failed. Check logs in: $LOG_DIR" >&2
  exit "$rc"
fi

echo "All splits finished successfully."

if [[ "$MERGE_SHARDS" != "1" ]]; then
  echo "MERGE_SHARDS!=1; skipping merge."
  exit 0
fi

echo
echo "Merging shard pickle files into: $OUTPUT"

# Reconstruct shard filenames matching cache_text_embedding.py's naming scheme.
shard_paths=()
if [[ "$OUTPUT" == *.pkl.gz ]]; then
  base="${OUTPUT%.pkl.gz}"
  for i in $(seq 0 $((TOTAL_SPLITS - 1))); do
    shard_paths+=( "${base}.part${i}of${TOTAL_SPLITS}.pkl.gz" )
  done
else
  ext="${OUTPUT##*.}"
  base="${OUTPUT%.*}"
  for i in $(seq 0 $((TOTAL_SPLITS - 1))); do
    shard_paths+=( "${base}.part${i}of${TOTAL_SPLITS}.${ext}" )
  done
fi

for sp in "${shard_paths[@]}"; do
  if [[ ! -f "$sp" ]]; then
    echo "Missing shard: $sp" >&2
    exit 2
  fi
done

"$PYTHON" - "$OUTPUT" "${shard_paths[@]}" <<'PY'
import gzip
import os
import pickle
import sys

out_path = sys.argv[1]
shards = sys.argv[2:]

merged = {}
for sp in shards:
    opener = gzip.open if sp.endswith(".gz") else open
    with opener(sp, "rb") as f:
        part = pickle.load(f)
    merged.update(part)

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
opener_out = gzip.open if out_path.endswith(".gz") else open
with opener_out(out_path, "wb") as f:
    pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Merged {len(shards)} shards -> {out_path} ({len(merged)} entries)")
PY

echo "Removing shard files..."
rm -f "${shard_paths[@]}"

if [[ -n "$UPLOAD_S3" ]]; then
  if [[ "$UPLOAD_S3" == s3://* ]]; then
    if [[ "$UPLOAD_S3" == */ ]]; then
      dst="${UPLOAD_S3}$(basename "$OUTPUT")"
    else
      dst="$UPLOAD_S3"
    fi
    echo "Uploading merged file to S3: $dst"
    aws s3 cp "$OUTPUT" "$dst"
  else
    echo "UPLOAD_S3 is set but does not look like an s3:// URI: $UPLOAD_S3" >&2
    exit 3
  fi
fi

echo "Done."

