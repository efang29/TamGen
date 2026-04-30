# Apoenzyme Targeting Investigation

## What is an Apoenzyme?

An **apoenzyme** is a protein in its ligand-free (apo) state — the receptor with nothing bound to it.
A **holoenzyme** is the same receptor with a ligand (drug/molecule) already docked.

TamGen was trained primarily on **holo** structures from the CrossDocked dataset, where the binding pocket shape is defined by the presence of a co-crystallized ligand. This gives the model a clean, well-defined pocket to target.

**9NOR is an apo structure** — there is no co-crystallized ligand in the deposited PDB file.
This is why the VAE-conditioned generation failed: there was no reference molecule to encode.

---

## Why This Matters for Our Case

Sweet taste receptors (T1R2/T1R3, which 9NOR represents) are challenging because:
1. They are GPCRs (G-protein coupled receptors) — large, flexible, membrane-embedded
2. The orthosteric binding site for sweeteners is in the Venus Flytrap (VFT) domain
3. Most available structures are apo or partially occupied
4. The pocket geometry changes significantly upon ligand binding (induced fit)

---

## Training Methods for Apoenzyme Structures

### Option 1: Use the non-VAE mode (already implemented)
The unconditioned TamGen generation does not require a reference ligand — it works directly from the pocket coordinates. This is what we are already doing and what produced our 410 candidates.

### Option 2: Prepare pocket using predicted binding site center
Instead of letting TamGen auto-detect a pocket from a ligand (which fails for apo structures), provide explicit center coordinates from the known VFT binding site literature.

Our `9nor_center.csv` already does this:
```
pdb_id, center_x, center_y, center_z
9nor, 180.001, 183.451, 160.87
```
This is the correct approach for apo structures.

### Option 3: Fine-tune on apo-structure data
Fine-tune the pre-trained TamGen checkpoint on a dataset of apo sweet-taste receptor structures paired with known sweetener SMILES (e.g., from the FART dataset or published sweetener-T1R2/T1R3 binding studies).

**Steps to implement:**
1. Collect known sweetener SMILES (from FART, FlavorDB, literature)
2. Use the 9NOR apo structure as the pocket template
3. Create training pairs: (9NOR pocket coords, sweetener SMILES)
4. Fine-tune using `train.py` with a very low learning rate on these pairs
5. Use the fine-tuned checkpoint for generation

This is the most involved option but would produce the most targeted results.

### Option 4: AlphaFold2 + Pocket Prediction
Use AlphaFold2-predicted structures for sweet taste receptor conformations paired with RoseTTAFold or P2Rank for pocket prediction. This gives cleaner, more consistent pocket definitions than experimental apo structures.

---

## Recommended Path Forward

For the current project scope:
1. **Continue with non-VAE generation on 9NOR center coordinates** (already working, 410 candidates)
2. **When time permits**, collect 10–20 known sweet molecule SMILES and set up a small fine-tuning run using Option 3
3. For the final deliverable, note that apo-structure generation is an open research challenge and our non-VAE approach is the correct workaround

---

## Relevant TamGen Code Entry Points

- `scripts/build_data/prepare_pdb_ids_center.py` — pocket prep from center coords (what we use)
- `train.py` — fine-tuning entry point (use `--restore-file <checkpoint>` to start from pre-trained weights)
- `generate_multiseed.py` — generation without reference ligand (non-VAE mode)
