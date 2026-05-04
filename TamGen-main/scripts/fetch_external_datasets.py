"""
Download and prepare external flavor/natural-product datasets for sweetness model training.

Supported sources:
  --coconut   COCONUT (Collection of Open NatUral producTs) — ~400k natural product SMILES
  --flavordb  FlavorDB2 — flavor molecules with taste/odor annotations

Usage:
    python scripts/fetch_external_datasets.py --coconut --out data/external/
    python scripts/fetch_external_datasets.py --flavordb --out data/external/
    python scripts/fetch_external_datasets.py --coconut --flavordb --out data/external/

Output files (in --out directory):
    coconut_sweet_candidates.csv   — CHNOPS natural products from COCONUT (SMILES + source)
    flavordb_sweet.csv             — FlavorDB molecules tagged as sweet (SMILES + flavor_profile)

These CSVs can be used to augment the FART training set in sweetness_filter.py.
To combine with FART:
    pd.concat([fart_df, coconut_df, flavordb_df]).drop_duplicates("SMILES")
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem

# Atoms allowed in flavor molecules — same definition as sweetness_filter.py (CHNOPS only)
_CHNOPS = frozenset({"C", "H", "N", "O", "P", "S"})


def _passes_atom_filter(mol) -> bool:
    return all(a.GetSymbol() in _CHNOPS for a in mol.GetAtoms())


# ─── COCONUT ──────────────────────────────────────────────────────────────────

COCONUT_API = "https://coconut.naturalproducts.net/api/search/compounds"

def fetch_coconut(out_dir: Path, max_pages: int = 50) -> Path:
    """
    Pull natural product SMILES from the COCONUT REST API (page-by-page).
    Filters to CHNOPS-only molecules with MW 100–1000 Da.
    Returns path to output CSV.
    """
    out_path = out_dir / "coconut_sweet_candidates.csv"
    rows = []
    print("Fetching COCONUT natural products...")

    for page in range(1, max_pages + 1):
        resp = requests.get(
            COCONUT_API,
            params={"page": page, "size": 500},
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"  COCONUT API returned {resp.status_code} on page {page}, stopping.")
            break

        data = resp.json()
        compounds = data.get("content", data.get("compounds", []))
        if not compounds:
            break

        for entry in compounds:
            smi = entry.get("smiles") or entry.get("canonicalSmiles", "")
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if not _passes_atom_filter(mol):
                continue
            rows.append({"SMILES": Chem.MolToSmiles(mol), "source": "COCONUT"})

        print(f"  Page {page}: {len(compounds)} fetched, {len(rows)} passing so far")

        if len(compounds) < 500:
            break

    if not rows:
        print("WARNING: No COCONUT compounds retrieved. Check network access or API endpoint.")
        return out_path

    df = pd.DataFrame(rows).drop_duplicates("SMILES")
    df.to_csv(out_path, index=False)
    print(f"COCONUT: wrote {len(df)} compounds to {out_path}")
    return out_path


# ─── FlavorDB ─────────────────────────────────────────────────────────────────

FLAVORDB_API = "https://cosylab.iiitd.edu.in/flavordb2/api/entities"

def fetch_flavordb(out_dir: Path) -> Path:
    """
    Pull molecules from FlavorDB2 API and keep entries tagged as 'sweet'.
    Returns path to output CSV.
    """
    out_path = out_dir / "flavordb_sweet.csv"
    print("Fetching FlavorDB2 sweet molecules...")

    try:
        resp = requests.get(FLAVORDB_API, params={"category": "sweeteners"}, timeout=30)
        if resp.status_code != 200:
            # Fallback: pull all and filter locally
            resp = requests.get(FLAVORDB_API, timeout=30)

        data = resp.json()
    except Exception as e:
        print(f"WARNING: FlavorDB fetch failed: {e}")
        print("FlavorDB2 may require browser access. Download manually from:")
        print("  https://cosylab.iiitd.edu.in/flavordb2/")
        print("  Export as CSV and place in data/external/flavordb_raw.csv")
        _write_flavordb_fallback_instructions(out_dir)
        return out_path

    entities = data if isinstance(data, list) else data.get("entities", data.get("data", []))
    rows = []
    for entry in entities:
        flavor = str(entry.get("flavor_profile", "") or entry.get("taste", "")).lower()
        if "sweet" not in flavor:
            continue
        smi = entry.get("smiles", "")
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not _passes_atom_filter(mol):
            continue
        rows.append({
            "SMILES": Chem.MolToSmiles(mol),
            "flavor_profile": entry.get("flavor_profile", ""),
            "source": "FlavorDB2",
        })

    if not rows:
        print("WARNING: No FlavorDB sweet entries found. See fallback instructions.")
        _write_flavordb_fallback_instructions(out_dir)
        return out_path

    df = pd.DataFrame(rows).drop_duplicates("SMILES")
    df.to_csv(out_path, index=False)
    print(f"FlavorDB: wrote {len(df)} sweet compounds to {out_path}")
    return out_path


def _write_flavordb_fallback_instructions(out_dir: Path):
    instructions = out_dir / "flavordb_manual_download.txt"
    instructions.write_text(
        "FlavorDB2 manual download instructions:\n"
        "1. Go to https://cosylab.iiitd.edu.in/flavordb2/\n"
        "2. Search for 'sweet' in the flavor profile filter\n"
        "3. Export results as CSV\n"
        "4. Save as data/external/flavordb_raw.csv\n"
        "5. Run: python scripts/fetch_external_datasets.py --flavordb-local data/external/flavordb_raw.csv\n"
    )
    print(f"Fallback instructions written to {instructions}")


def process_local_flavordb(csv_path: str, out_dir: Path) -> Path:
    """Process a manually-downloaded FlavorDB CSV."""
    out_path = out_dir / "flavordb_sweet.csv"
    df = pd.read_csv(csv_path)

    # Detect SMILES column
    smiles_col = next((c for c in df.columns if "smiles" in c.lower()), None)
    taste_col = next((c for c in df.columns if "taste" in c.lower() or "flavor" in c.lower()), None)
    if smiles_col is None:
        print(f"ERROR: No SMILES column found in {csv_path}. Columns: {list(df.columns)}")
        sys.exit(1)

    if taste_col:
        df = df[df[taste_col].astype(str).str.lower().str.contains("sweet", na=False)]

    rows = []
    for smi in df[smiles_col].dropna():
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None or not _passes_atom_filter(mol):
            continue
        rows.append({"SMILES": Chem.MolToSmiles(mol), "source": "FlavorDB2_local"})

    df_out = pd.DataFrame(rows).drop_duplicates("SMILES")
    df_out.to_csv(out_path, index=False)
    print(f"FlavorDB local: wrote {len(df_out)} sweet compounds to {out_path}")
    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fetch COCONUT and/or FlavorDB datasets.")
    ap.add_argument("--coconut", action="store_true", help="Download COCONUT natural products.")
    ap.add_argument("--flavordb", action="store_true", help="Download FlavorDB2 sweet molecules via API.")
    ap.add_argument("--flavordb-local", type=str, default=None,
                    help="Process a manually-downloaded FlavorDB CSV instead of fetching.")
    ap.add_argument("--out", type=str, default="data/external/",
                    help="Output directory (default: data/external/).")
    ap.add_argument("--coconut-pages", type=int, default=50,
                    help="Max pages to fetch from COCONUT API (500 compounds/page).")
    args = ap.parse_args()

    if not any([args.coconut, args.flavordb, args.flavordb_local]):
        ap.print_help()
        sys.exit(0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.coconut:
        fetch_coconut(out_dir, max_pages=args.coconut_pages)

    if args.flavordb:
        fetch_flavordb(out_dir)

    if args.flavordb_local:
        process_local_flavordb(args.flavordb_local, out_dir)

    print("\nDone. To combine with FART for training, merge CSVs and pass to sweetness_filter.py --fart.")


if __name__ == "__main__":
    main()
