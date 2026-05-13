# TamGen — 9NOR Sweet Molecule Project

We're using TamGen (a 3D protein-pocket → SMILES model) to design new sweet-tasting molecules for the 9NOR sweet-taste receptor, then ranking them with a sweetness classifier we trained.

For why VAE-conditioned generation doesn't really work here, see [docs/apoenzyme_targeting.md](docs/apoenzyme_targeting.md). Short version: 9NOR has no ligand bound, so the VAE has nothing to encode and the model collapses to garbage.

## How to run it

```bash
conda activate TamGen
bash scripts/run_9nor_inference.sh
```

Outputs land in `output/run_<today>_inference/`. The useful files are `nonvae_ranked_polar.csv` and `nonvae_ranked_sugar.csv`.

You can also pass flags:
```bash
bash scripts/run_9nor_inference.sh --out output/my_run --seed-mol "O=C1NS(=O)(=O)c2ccccc21"
```
Flags are documented at the top of [scripts/run_9nor_inference.sh](scripts/run_9nor_inference.sh).

## What's where

- `inputs/` — 9NOR pocket info you might edit (`9nor_center.csv`, `9nor_input.csv`)
- `output/` — every run gets its own dated subfolder, so nothing gets overwritten
- `scripts/` — the stuff we actually wrote and edit
- `docs/` — project notes

Everything else (`fairseq/`, `model/`, `checkpoints/`, `data/`, `generate.py`, `train.py`, etc.) is the original TamGen library. Don't touch it unless you really know what you're doing.

## Where to change stuff

Almost all knobs live in two files:

**[scripts/run_9nor_inference.sh](scripts/run_9nor_inference.sh)** — the pipeline driver. Defaults are at the top (lines 24-30):
- `BEAM=20` — beam size, more = more candidates but slower
- `SEED_MOL=""` — pass a SMILES to force outputs to start with it (only useful with VAE, and VAE barely works on 9NOR, so usually leave empty)
- `RESULTS=...` — output folder
- `FART=...` — path to the sweetness training CSV
- Lines 87-102 pick the LogP buckets (polar 0-3, sugar <0). Edit them to change MW/LogP ranges.

**[scripts/sweetness_filter.py](scripts/sweetness_filter.py)** — the sweetness classifier and flavor filter:
- Lines 18-29 — allowed atoms, halogen cap, drug-like scaffolds to reject
- Lines 138-143 — default MW/LogP cutoffs (but the run script overrides these anyway)

## The pipeline (6 steps)

The bash script runs these in order:
1. VAE generation → `raw_vae.txt`
2. Parse → `vae_candidates.csv`
3. Non-VAE generation → `raw_nonvae.txt`
4. Parse → `nonvae_candidates.csv`
5. Sweetness filter, polar set (LogP 0-3) → `nonvae_ranked_polar.csv`
6. Sweetness filter, sugar set (LogP < 0) → `nonvae_ranked_sugar.csv`

Ranked CSV columns: `test_id, smiles, nlogP, sweet_prob, mw, logp, passes_atom_filter, passes_flavor_filter, fart_val_acc`. Sorted so the molecules that pass the filter and look most sweet are on top.

## Heads-up

- **The VAE branch is mostly broken on 9NOR.** It runs but produces trivial fragments or seed-plus-metal-ion garbage. The real candidates come from the non-VAE branch. We've tried saccharin and aspartame as VAE seeds — didn't help. Notes are in [docs/apoenzyme_targeting.md](docs/apoenzyme_targeting.md) and the `output/run_2026-05-13_vae_seeded/` folder.
- **It's CPU-only here**, so a full run takes a few minutes per branch. Bigger beam = longer.
