#!/usr/bin/env bash
# Full pipeline for 9NOV (holo, sucralose-bound) sweet-taste receptor molecule generation.
#
# 9NOV is the human TAS1R2/TAS1R3 Venus Flytrap heterodimer with sucralose bound
# (cryo-EM, 3.30 Å). Unlike 9NOR (apo), this gives the VAE a real ligand reference,
# so VAE-conditioned generation should actually produce useful candidates here.
#
# Usage:
#   bash scripts/run_9nov_inference.sh
#
# Optional flags:
#   --training-data  path to sweetness training CSV (default: ../Datasets/.../tamgen_trainingdata.csv)
#                    Alias: --fart (legacy — the variable used to be FART-only;
#                    now it's the combined FART + FlavorDB + COCONUT set.)
#   --datadir   path to binarized 9NOV pocket data  (default: data/9nov_processed)
#   --ckpt      path to model checkpoint            (default: checkpoints/crossdock_pdb_A10/checkpoint_best.pt)
#   --out       output folder                       (default: output/run_<today>_9nov_holo)
#   --beam      beam size                           (default: 20)
#   --seed-mol  optional beam-prefix override for VAE   (default: empty)
#               Sucralose is already baked into the binarized data via the scaffold
#               prep (see inputs/9nov_scaffold.txt), so the VAE encoder reads it
#               automatically. Only pass --seed-mol to override with a different
#               molecule, e.g. --seed-mol "COC(=O)C(Cc1ccccc1)NC(=O)C(N)CC(=O)O"
#
# Run from the TamGen-main directory with the TamGen conda env active:
#   conda activate TamGen
#   bash scripts/run_9nov_inference.sh

set -euo pipefail

# ---------- defaults ----------
DATADIR="data/9nov_processed"
CKPT="checkpoints/crossdock_pdb_A10/checkpoint_best.pt"
RESULTS="output/run_$(date +%Y-%m-%d)_9nov_holo"
BEAM=20
TESTSET="gen_9nov"
TRAINING_DATA="../Datasets/datasets-clean/datasets-clean-csv/tamgen_trainingdata.csv"
# Empty by default — sucralose is already in the binarized target data, so the VAE
# encoder picks it up without needing a forced beam prefix. Set this if you want
# to override the bound-ligand reference with a different seed.
SEED_MOL=""

# ---------- parse flags ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --training-data) TRAINING_DATA="$2"; shift 2 ;;
        --fart)          TRAINING_DATA="$2"; shift 2 ;;  # backward-compat alias
        --datadir)  DATADIR="$2";  shift 2 ;;
        --ckpt)     CKPT="$2";     shift 2 ;;
        --out)      RESULTS="$2";  shift 2 ;;
        --beam)     BEAM="$2";     shift 2 ;;
        --seed-mol) SEED_MOL="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ ! -f "$TRAINING_DATA" ]]; then
    echo "ERROR: training data file not found: $TRAINING_DATA"
    exit 1
fi

mkdir -p "$RESULTS"

# ---------- conditioned generation (VAE) ----------
echo "=== [1/8] Conditioned (VAE) generation ==="
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

echo "=== [2/8] Formatting VAE output ==="
python scripts/format_output.py "$RESULTS/raw_vae.txt" "$RESULTS/vae_candidates.csv"
echo "Candidates: $RESULTS/vae_candidates.csv"

# ---------- unconditioned generation ----------
# NOTE: --prefix-string deliberately NOT passed here. The non-VAE branch should be
# unbiased de novo generation on the holo pocket — directly comparable to the 9NOR
# non-VAE baseline. Passing the seed here causes every output to start with the seed
# SMILES, producing sucralose+fragment mixtures that fail the flavor filter.
echo "=== [3/8] Unconditioned generation (de novo, no seed) ==="
python generate_multiseed.py \
    "$DATADIR" \
    -s tg -t m1 \
    --task translation_coord \
    --path "$CKPT" \
    --gen-subset "$TESTSET" \
    --beam $BEAM --nbest $BEAM --max-tokens 1024 \
    --seed 1 --sample-beta 1.0 \
    --use-src-coord | tee "$RESULTS/raw_nonvae.txt"

echo "=== [4/8] Formatting non-VAE output ==="
python scripts/format_output.py "$RESULTS/raw_nonvae.txt" "$RESULTS/nonvae_candidates.csv"
echo "Candidates: $RESULTS/nonvae_candidates.csv"

# ---------- sweetness filtering — two LogP sets per Linda's recommendation ----------
# Set 1 (polar):      0 <= LogP <= 3  — moderate polarity, synthetic sweeteners
# Set 2 (sugar-like): LogP < 0        — highly polar, natural sugars (sucrose LogP ~ -3.7)
echo "=== [5/8] Sweetness filtering — polar set (0 ≤ LogP ≤ 3) ==="
python scripts/sweetness_filter.py \
    --generated "$RESULTS/nonvae_candidates.csv" \
    --training-data "$TRAINING_DATA" \
    --out "$RESULTS/nonvae_ranked_polar.csv" \
    --logp-min 0 --logp-max 3

echo "=== [6/8] Sweetness filtering (non-VAE) — sugar set (LogP < 0) ==="
python scripts/sweetness_filter.py \
    --generated "$RESULTS/nonvae_candidates.csv" \
    --training-data "$TRAINING_DATA" \
    --out "$RESULTS/nonvae_ranked_sugar.csv" \
    --logp-max 0

echo "=== [7/8] Sweetness filtering (VAE) — polar set (0 ≤ LogP ≤ 3) ==="
python scripts/sweetness_filter.py \
    --generated "$RESULTS/vae_candidates.csv" \
    --training-data "$TRAINING_DATA" \
    --out "$RESULTS/vae_ranked_polar.csv" \
    --logp-min 0 --logp-max 3

echo "=== [8/8] Sweetness filtering (VAE) — sugar set (LogP < 0) ==="
python scripts/sweetness_filter.py \
    --generated "$RESULTS/vae_candidates.csv" \
    --training-data "$TRAINING_DATA" \
    --out "$RESULTS/vae_ranked_sugar.csv" \
    --logp-max 0

echo ""
echo "=== Done ==="
echo "Results saved to $RESULTS/"
echo "  Non-VAE polar  (0 ≤ LogP ≤ 3):  $RESULTS/nonvae_ranked_polar.csv"
echo "  Non-VAE sugar  (LogP < 0):       $RESULTS/nonvae_ranked_sugar.csv"
echo "  VAE polar      (0 ≤ LogP ≤ 3):  $RESULTS/vae_ranked_polar.csv"
echo "  VAE sugar      (LogP < 0):       $RESULTS/vae_ranked_sugar.csv"
