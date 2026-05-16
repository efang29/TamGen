# TamGen — Sweet Molecule Project

We're using TamGen (a 3D protein-pocket → SMILES model) to design new sweet-tasting molecules, then ranking them with a sweetness classifier we trained on a combined FART + FlavorDB + COCONUT dataset (~22k molecules, balanced 50/50 sweet/non-sweet).

We have two targets:
- **9NOR** — apo (empty) sweet-taste receptor TAS1R2/TAS1R3. Good for non-VAE de novo generation.
- **9NOV** — same receptor but with sucralose bound (holo, cryo-EM 3.30 Å). Both VAE and non-VAE work here. Recommended default.

For why VAE doesn't work on apo targets but does on holo, see [docs/apoenzyme_targeting.md](docs/apoenzyme_targeting.md).

## How to run it

Recommended (holo, both VAE + non-VAE work):
```bash
conda activate TamGen
bash scripts/run_9nov_inference.sh
```

Or the apo baseline (non-VAE only):
```bash
bash scripts/run_9nor_inference.sh
```

Outputs land in `output/run_<today>_9nov_holo/` (or `_inference/` for 9NOR). The useful files are `nonvae_ranked_polar.csv`, `nonvae_ranked_sugar.csv`, plus `vae_ranked_polar.csv` and `vae_ranked_sugar.csv` if you ran 9NOV.

You can override anything with flags:
```bash
bash scripts/run_9nov_inference.sh --out output/my_run --beam 40
```
Flags are documented at the top of each `scripts/run_9no*_inference.sh`.

## What's where

- `inputs/` — pocket info per target (`9nor_*.csv`, `9nov_center.csv`, `9nov_scaffold.txt`)
- `output/` — every run gets its own dated subfolder, so nothing gets overwritten
- `scripts/` — the stuff we actually wrote and edit
- `docs/` — project notes

Everything else (`fairseq/`, `model/`, `checkpoints/`, `data/`, `generate.py`, `train.py`, etc.) is the original TamGen library. Don't touch it unless you really know what you're doing.

## Where to change stuff

Almost all knobs live in two files:

**[scripts/run_9nov_inference.sh](scripts/run_9nov_inference.sh)** (or its `_9nor_` twin) — the pipeline driver. Defaults are at the top:
- `BEAM=20` — beam size, more = more candidates but slower
- `SEED_MOL=""` — VAE beam-prefix override; usually leave empty (for 9NOV the bound ligand is already baked into the data; for 9NOR the seed mechanism doesn't really help)
- `RESULTS=...` — output folder
- `TRAINING_DATA=...` — path to the sweetness training CSV (the combined dataset). CLI flag is `--training-data`, with `--fart` kept as a legacy alias.
- The sweetness filtering steps near the bottom pick the LogP buckets (polar 0-3, sugar <0). Edit those `--logp-min/--logp-max` flags to change MW/LogP ranges.

**[scripts/sweetness_filter.py](scripts/sweetness_filter.py)** — the sweetness classifier and flavor filter:
- Lines 18-29 — allowed atoms, halogen cap, drug-like scaffolds to reject
- Lines 138-143 — default MW/LogP cutoffs (but the run script overrides these anyway)

## The pipeline

The 9NOV script runs 8 steps (the 9NOR one is 6 — same minus the VAE filtering):
1. VAE generation → `raw_vae.txt`
2. Parse → `vae_candidates.csv`
3. Non-VAE generation → `raw_nonvae.txt`
4. Parse → `nonvae_candidates.csv`
5. Sweetness filter on non-VAE, polar set (LogP 0-3) → `nonvae_ranked_polar.csv`
6. Sweetness filter on non-VAE, sugar set (LogP < 0) → `nonvae_ranked_sugar.csv`
7. Sweetness filter on VAE, polar set → `vae_ranked_polar.csv` (9NOV only)
8. Sweetness filter on VAE, sugar set → `vae_ranked_sugar.csv` (9NOV only)

Ranked CSV columns: `test_id, smiles, nlogP, sweet_prob, mw, logp, passes_atom_filter, passes_flavor_filter, fart_val_acc`. Sorted so molecules that pass the filter and look most sweet are on top.

## Heads-up

- **9NOR vs 9NOV non-VAE are different chemistries, not better/worse.** Only ~27% overlap between the two non-VAE candidate pools. 9NOR leans toward natural-product scaffolds; 9NOV leans toward sulfonamides and amino-acid-like compounds. Worth running both and taking the union.
- **VAE only works on 9NOV.** On 9NOR (apo) the VAE has nothing to encode and produces metal-cation adducts. On 9NOV the bound sucralose is baked into the prep data via `inputs/9nov_scaffold.txt`, so the VAE encoder reads it automatically. Output: 85 sugar-class candidates including the model rediscovering sucrose itself.
- **It's CPU-only here**, so a full run takes a few minutes per branch. Bigger beam = longer.
