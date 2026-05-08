#!/usr/bin/env bash
# Full pipeline for 9NOR sweet-taste receptor molecule generation.
#
# Usage:
#   bash scripts/run_9nor_inference.sh
#
# Optional flags:
#   --fart      path to training data CSV           (default: ../Datasets/.../tamgen_trainingdata.csv)
#   --datadir   path to binarized 9NOR pocket data  (default: data/9nor_processed)
#   --ckpt      path to model checkpoint            (default: checkpoints/crossdock_pdb_A10/checkpoint_best.pt)
#   --out       output folder                       (default: results_9nor)
#   --beam      beam size                           (default: 20)
#   --seed-mol  seed SMILES for refinement mode     (default: none — pure de novo)
#               e.g. --seed-mol "O=C1NS(=O)(=O)c2ccccc21"  (saccharin)
#
# Run from the TamGen-main directory with the TamGen conda env active:
#   conda activate TamGen
#   bash scripts/run_9nor_inference.sh
#   bash scripts/run_9nor_inference.sh --seed-mol "O=C1NS(=O)(=O)c2ccccc21"

set -euo pipefail

# ---------- defaults ----------
DATADIR="data/9nor_processed"
CKPT="checkpoints/crossdock_pdb_A10/checkpoint_best.pt"
RESULTS="results_9nor"
BEAM=20
TESTSET="test"
FART="../Datasets/datasets-clean/datasets-clean-csv/tamgen_trainingdata.csv"
SEED_MOL=""

# ---------- parse flags ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fart)     FART="$2";     shift 2 ;;
        --datadir)  DATADIR="$2";  shift 2 ;;
        --ckpt)     CKPT="$2";     shift 2 ;;
        --out)      RESULTS="$2";  shift 2 ;;
        --beam)     BEAM="$2";     shift 2 ;;
        --seed-mol) SEED_MOL="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ ! -f "$FART" ]]; then
    echo "ERROR: training data file not found: $FART"
    exit 1
fi

mkdir -p "$RESULTS"

# ---------- conditioned generation (VAE) ----------
echo "=== [1/4] Conditioned (VAE) generation ==="
python generate_multiseed.py \
    "$DATADIR" \
    -s tg -t m1 \
    --task translation_coord \
    --path "$CKPT" \
    --gen-subset "$TESTSET" \
    --beam $BEAM --nbest $BEAM --max-tokens 1024 \
    --seed 1 --sample-beta 1.0 \
    --use-src-coord \
    ${SEED_MOL:+--prefix-string "$SEED_MOL"} \
    --gen-vae | tee "$RESULTS/raw_vae.txt"

echo "=== [2/4] Formatting VAE output ==="
python scripts/format_output.py "$RESULTS/raw_vae.txt" "$RESULTS/vae_candidates.csv"
echo "Candidates: $RESULTS/vae_candidates.csv"

# ---------- unconditioned generation ----------
echo "=== [3/4] Unconditioned generation ==="
python generate_multiseed.py \
    "$DATADIR" \
    -s tg -t m1 \
    --task translation_coord \
    --path "$CKPT" \
    --gen-subset "$TESTSET" \
    --beam $BEAM --nbest $BEAM --max-tokens 1024 \
    --seed 1 --sample-beta 1.0 \
    --use-src-coord \
    ${SEED_MOL:+--prefix-string "$SEED_MOL"} | tee "$RESULTS/raw_nonvae.txt"

echo "=== [4/4] Formatting non-VAE output ==="
python scripts/format_output.py "$RESULTS/raw_nonvae.txt" "$RESULTS/nonvae_candidates.csv"
echo "Candidates: $RESULTS/nonvae_candidates.csv"

# ---------- sweetness filtering — two LogP sets per Linda's recommendation ----------
# Set 1 (polar):      0 <= LogP <= 3  — moderate polarity, synthetic sweeteners
# Set 2 (sugar-like): LogP < 0        — highly polar, natural sugars (sucrose LogP ~ -3.7)
echo "=== [5/6] Sweetness filtering — polar set (0 ≤ LogP ≤ 3) ==="
python scripts/sweetness_filter.py \
    --generated "$RESULTS/nonvae_candidates.csv" \
    --training-data "$FART" \
    --out "$RESULTS/nonvae_ranked_polar.csv" \
    --logp-min 0 --logp-max 3

echo "=== [6/6] Sweetness filtering — sugar set (LogP < 0) ==="
python scripts/sweetness_filter.py \
    --generated "$RESULTS/nonvae_candidates.csv" \
    --training-data "$FART" \
    --out "$RESULTS/nonvae_ranked_sugar.csv" \
    --logp-max 0

echo ""
echo "=== Done ==="
echo "Results saved to $RESULTS/"
echo "  Polar set  (0 ≤ LogP ≤ 3):  $RESULTS/nonvae_ranked_polar.csv"
echo "  Sugar set  (LogP < 0):       $RESULTS/nonvae_ranked_sugar.csv"
