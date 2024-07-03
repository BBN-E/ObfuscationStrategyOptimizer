#!/usr/bin/env bash
# End-to-end reproduction of Table 1 for one dataset.
#
#   bash scripts/run_pipeline.sh data/amt-10 work/amt-10
#
# Stage 1 generates the candidates that ship with this repository (LLM Rewriting
# and Recursive LLM Rewriting). The Machine Translation and STEER candidates
# need models released elsewhere -- see obfuscation_systems/mt and
# obfuscation_systems/steer. To include them, write their JSONL output into
# $WORK_DIR/obf/ and add them to $WORK_DIR/systems.json before stage 2.
set -euo pipefail

DATA_DIR="${1:?usage: run_pipeline.sh <data dir> <work dir>}"
WORK_DIR="${2:?usage: run_pipeline.sh <data dir> <work dir>}"

TRAIN_JSONL="$DATA_DIR/X_train.jsonl"
TEST_JSONL="$DATA_DIR/X_test.jsonl"
LLAMA_MODEL="${LLAMA_MODEL:-TheBloke/Llama-2-7B-GPTQ}"

mkdir -p "$WORK_DIR"/{obf,metrics,results}

# --- 1. Generate obfuscation candidates --------------------------------------
echo "== Generating LLM candidates =="
python -m obfuscation_systems.llm_rewriting.generate_rephrasing_gptq \
    --input_jsonl "$TEST_JSONL" \
    --output_path "$WORK_DIR/obf/llama2" \
    --llama_model_path "$LLAMA_MODEL" \
    --num_passes 2 \
    --remove_short_sentences \
    --check_length

if [ ! -f "$WORK_DIR/systems.json" ]; then
    cat > "$WORK_DIR/systems.json" <<EOF
{
  "Llama-2 7B": "$WORK_DIR/obf/llama2_pass_1.jsonl",
  "Recursive Llama-2 7B": "$WORK_DIR/obf/llama2_pass_2.jsonl"
}
EOF
fi

# --- 2. Score every candidate -------------------------------------------------
echo "== Scoring candidates =="
python scripts/score_all.py \
    --orig_jsonl "$TEST_JSONL" \
    --systems "$WORK_DIR/systems.json" \
    --metrics_dir "$WORK_DIR/metrics" \
    --out_score_files "$WORK_DIR/scores.json"

# --- 3. Select the best system per author (Eq. 5) -----------------------------
echo "== Running OSO =="
python -m oso.selector \
    --orig_jsonl "$TEST_JSONL" \
    --systems "$WORK_DIR/systems.json" \
    --score_files "$WORK_DIR/scores.json" \
    --obf_weights configs/weights_paper.json \
    --out_jsonl "$WORK_DIR/results/oso/final_obfuscated.jsonl"

# --- 4. Adversarial evaluation and final table --------------------------------
echo "== Evaluating against the attribution system =="
python -c "
import json, sys
systems = json.load(open('$WORK_DIR/systems.json'))
systems['OSO'] = '$WORK_DIR/results/oso/final_obfuscated.jsonl'
json.dump(systems, open('$WORK_DIR/systems_with_oso.json', 'w'), indent=2)
"

python -m oso.evaluation.acc_drop \
    --train_jsonl "$TRAIN_JSONL" \
    --test_jsonl "$TEST_JSONL" \
    --systems "$WORK_DIR/systems_with_oso.json" \
    --classifier "$WORK_DIR/results/attributor.pkl" \
    --out_csv "$WORK_DIR/results/acc_drop.csv"

python -m oso.evaluation.report \
    --avg_metrics "$WORK_DIR/results/oso/avg_system_metrics.csv" \
    --privacy "$WORK_DIR/results/acc_drop.csv" \
    --out_csv "$WORK_DIR/results/table1.csv"

echo "== Done: $WORK_DIR/results/table1.csv =="
