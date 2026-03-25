#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash test_batch.sh
#   bash test_batch.sh /mnt/data2/experiments/3dgs/output_Stylized

ROOT_DIR="${1:-/mnt/data2/experiments/3dgs/output_Stylized}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python"
IMG_SIZE="256"
BACKBONE="raft"
DISTANCE="cosine"
PAIRING="adjacent"
SHORT_STEP="1"
LONG_STEP="5"
FORCE_RECALC="${FORCE_RECALC:-0}"

if [[ ! -d "$ROOT_DIR" ]]; then
	echo "[ERROR] Directory not found: $ROOT_DIR" >&2
	exit 1
fi

OUT_FILE="$SCRIPT_DIR/met3r_batch_results_$(date +%Y%m%d_%H%M%S).tsv"

# Reuse successful rows from the latest previous run to avoid recomputation.
LATEST_FILE=""
if [[ "$FORCE_RECALC" != "1" ]]; then
	LATEST_FILE="$(ls -1t "$SCRIPT_DIR"/met3r_batch_results_*.tsv 2>/dev/null | head -n 1 || true)"
	if [[ "$LATEST_FILE" == "$OUT_FILE" ]]; then
		LATEST_FILE=""
	fi
fi

echo -e "dataset\tshort_step\tshort_pairs\tshort_mean_score\tshort_status\tlong_step\tlong_pairs\tlong_mean_score\tlong_status\trenders_dir" > "$OUT_FILE"
echo "[INFO] Root directory: $ROOT_DIR"
echo "[INFO] Output file: $OUT_FILE"
echo "[INFO] Consistency setup: short_step=$SHORT_STEP, long_step=$LONG_STEP, pairing=$PAIRING"
if [[ -n "$LATEST_FILE" ]]; then
	echo "[INFO] Reusing successful rows from: $LATEST_FILE"
fi

get_cached_line() {
	local render_dir="$1"
	if [[ -z "$LATEST_FILE" || ! -f "$LATEST_FILE" ]]; then
		return 1
	fi
	awk -F '\t' -v rd="$render_dir" 'NR>1 && $10==rd && $5=="success" && $9=="success" {line=$0} END {if (line!="") print line}' "$LATEST_FILE"
}

mapfile -t RENDER_DIRS < <(find "$ROOT_DIR" -type d -path '*/train/ours_*/renders' | sort)

if [[ ${#RENDER_DIRS[@]} -eq 0 ]]; then
	echo "[WARN] No directories matched pattern */train/ours_*/renders under: $ROOT_DIR"
	exit 0
fi

echo "[INFO] Found ${#RENDER_DIRS[@]} render directories"

run_eval() {
	local render_dir="$1"
	local frame_step="$2"

	set +e
	local cmd_output
	cmd_output="$($PYTHON_BIN "$SCRIPT_DIR/mytest.py" \
		--input-dir "$render_dir" \
		--img-size "$IMG_SIZE" \
		--backbone "$BACKBONE" \
		--distance "$DISTANCE" \
		--pairing "$PAIRING" \
		--frame-step "$frame_step" 2>&1)"
	local exit_code=$?
	set -e

	if [[ $exit_code -ne 0 ]]; then
		echo "failed|NA|NA|$cmd_output"
		return
	fi

	local summary_line
	summary_line="$(echo "$cmd_output" | grep -E '^RESULT pairs=[0-9]+ mean_score=[0-9eE+.-]+' | tail -n 1 || true)"
	if [[ -z "$summary_line" ]]; then
		echo "parse_failed|NA|NA|$cmd_output"
		return
	fi

	local pairs mean_score
	pairs="$(echo "$summary_line" | sed -E 's/^RESULT pairs=([0-9]+) mean_score=.*/\1/')"
	mean_score="$(echo "$summary_line" | sed -E 's/^RESULT pairs=[0-9]+ mean_score=([0-9eE+.-]+).*/\1/')"
	echo "success|$pairs|$mean_score|"
}

for render_dir in "${RENDER_DIRS[@]}"; do
	dataset="$(basename "$(dirname "$(dirname "$(dirname "$render_dir")")")")"

	cached_line="$(get_cached_line "$render_dir" || true)"
	if [[ -n "$cached_line" ]]; then
		echo "[SKIP] $dataset -> reuse cached success"
		echo -e "$cached_line" >> "$OUT_FILE"
		continue
	fi

	echo "[RUN] $dataset -> $render_dir"

	short_result="$(run_eval "$render_dir" "$SHORT_STEP")"
	short_status="${short_result%%|*}"
	short_rest="${short_result#*|}"
	short_pairs="${short_rest%%|*}"
	short_rest="${short_rest#*|}"
	short_mean_score="${short_rest%%|*}"
	short_msg="${short_rest#*|}"

	long_result="$(run_eval "$render_dir" "$LONG_STEP")"
	long_status="${long_result%%|*}"
	long_rest="${long_result#*|}"
	long_pairs="${long_rest%%|*}"
	long_rest="${long_rest#*|}"
	long_mean_score="${long_rest%%|*}"
	long_msg="${long_rest#*|}"

	if [[ "$short_status" != "success" ]]; then
		echo "[WARN] short failed for $dataset (step=$SHORT_STEP)"
		echo "$short_msg"
	fi

	if [[ "$long_status" != "success" ]]; then
		echo "[WARN] long failed for $dataset (step=$LONG_STEP)"
		echo "$long_msg"
	fi

	echo "[OK] $dataset short(step=$SHORT_STEP): $short_status pairs=$short_pairs mean=$short_mean_score | long(step=$LONG_STEP): $long_status pairs=$long_pairs mean=$long_mean_score"
	echo -e "${dataset}\t${SHORT_STEP}\t${short_pairs}\t${short_mean_score}\t${short_status}\t${LONG_STEP}\t${long_pairs}\t${long_mean_score}\t${long_status}\t${render_dir}" >> "$OUT_FILE"
done

echo "[DONE] Batch evaluation finished."
echo "[DONE] Results saved to: $OUT_FILE"
if [[ "$FORCE_RECALC" != "1" ]]; then
	echo "[DONE] Tip: set FORCE_RECALC=1 to recompute everything."
fi
