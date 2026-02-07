#!/usr/bin/env bash
# ============================================================
# Run ALL ablation experiments: 6 models × 9 subjects = 54 runs
# ============================================================
#
# Usage:
#   bash scripts/run_all_ablations.sh [CONFIG_PATH]
#
# Default config: configs/default.yaml
# Results: results/logs/*.json, results/checkpoints/*.pt, results/figures/*.png
# ============================================================

set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
SCRIPT="scripts/run_single.py"

echo "=============================================="
echo " EEG Hybrid CNN + Attention — Ablation Sweep"
echo "=============================================="
echo "Config: $CONFIG"
echo "Start:  $(date)"
echo ""

# Ensure data exists
DATA_DIR="./data/processed"
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "Preprocessing data..."
    python -m data.download_bciciv2a --output_dir ./data/raw --processed_dir "$DATA_DIR"
    echo ""
fi

# GPU check
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠  No GPU detected — running on CPU (will be slow)"
    echo ""
fi

# ── Define ablation grid ──
# Format: "architecture attention_type"
EXPERIMENTS=(
    "eegnet none"
    "deep_convnet none"
    "hybrid_cnn none"
    "hybrid_cnn se"
    "hybrid_cnn cbam"
    "hybrid_cnn mha"
)

SUBJECTS=(1 2 3 4 5 6 7 8 9)

TOTAL=$((${#EXPERIMENTS[@]} * ${#SUBJECTS[@]}))
COUNT=0
FAILED=0

echo "Running $TOTAL experiments (${#EXPERIMENTS[@]} models × ${#SUBJECTS[@]} subjects)"
echo "----------------------------------------------"

for EXP in "${EXPERIMENTS[@]}"; do
    read -r ARCH ATTN <<< "$EXP"
    
    for SUB in "${SUBJECTS[@]}"; do
        COUNT=$((COUNT + 1))
        EXP_NAME="${ARCH}_${ATTN}_subject$(printf '%02d' $SUB)"
        
        echo ""
        echo "[$COUNT/$TOTAL] $EXP_NAME"
        echo "  Architecture: $ARCH | Attention: $ATTN | Subject: $SUB"
        
        if python "$SCRIPT" \
            --config "$CONFIG" \
            --architecture "$ARCH" \
            --attention_type "$ATTN" \
            --subject "$SUB" 2>&1 | tail -5; then
            echo "  ✓ $EXP_NAME completed"
        else
            echo "  ✗ $EXP_NAME FAILED"
            FAILED=$((FAILED + 1))
        fi
    done
done

echo ""
echo "=============================================="
echo " Ablation Sweep Complete"
echo "=============================================="
echo "Total:    $TOTAL"
echo "Success:  $((TOTAL - FAILED))"
echo "Failed:   $FAILED"
echo "End:      $(date)"
echo ""

# ── Post-hoc summary ──
echo "Generating summary..."
python -c "
import sys
sys.path.insert(0, '.')
from visualizations.summary_plots import generate_results_table
from utils.statistics import compare_models

# Results table
generate_results_table('results/logs', save_path='results/results_table.txt')

# Statistical comparisons
comparisons = [
    ('hybrid_cnn_se', 'eegnet_none', 'SE vs EEGNet'),
    ('hybrid_cnn_cbam', 'eegnet_none', 'CBAM vs EEGNet'),
    ('hybrid_cnn_mha', 'eegnet_none', 'MHA vs EEGNet'),
    ('hybrid_cnn_se', 'hybrid_cnn_none', 'SE vs CNN-only'),
    ('hybrid_cnn_cbam', 'hybrid_cnn_none', 'CBAM vs CNN-only'),
    ('hybrid_cnn_mha', 'hybrid_cnn_none', 'MHA vs CNN-only'),
]

print('\n' + '='*60)
print('Statistical Comparisons (Wilcoxon signed-rank)')
print('='*60)

for a, b, label in comparisons:
    result = compare_models('results/logs', a, b)
    if 'error' not in result:
        sig = '***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else 'n.s.'
        print(f'{label:>25s}: p={result[\"p_value\"]:.4f} {sig}  Δ={result[\"mean_diff\"]:+.4f}')
    else:
        print(f'{label:>25s}: {result[\"error\"]}')
"

echo ""
echo "Results saved to: results/"
echo "  results/logs/            — per-experiment JSON logs"
echo "  results/checkpoints/     — model checkpoints"
echo "  results/figures/         — attention & saliency maps"
echo "  results/results_table.txt — summary table"
