import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Atoms allowed in flavor/sweetener molecules.
# CHNOPS covers natural sweeteners. Halogens (F, Cl, Br, I) are allowed up to _MAX_HALOGENS.
# Cap raised to 3 so sucralose-class candidates (3 Cl) survive when conditioning on 9NOV.
# All metals and other heavy atoms are rejected.
_CHNOPS = frozenset({"C", "H", "N", "O", "P", "S"})
_HALOGENS = frozenset({"F", "Cl", "Br", "I"})
_ALLOWED_ATOMS = _CHNOPS | _HALOGENS
_MAX_HALOGENS = 3

# Reject molecules that look pharmaceutical rather than food-grade.
# Piperazine + haloarene is a common drug scaffold (antidepressants, antipsychotics).
_DRUG_SMARTS = [
    Chem.MolFromSmarts("N1CCNCC1"),          # piperazine
    Chem.MolFromSmarts("N1CCCC1"),            # pyrrolidine attached to arene
    Chem.MolFromSmarts("[#6]-[F,Cl,Br,I]"),  # direct C-halogen bond (aryl or alkyl halide)
]


def _passes_atom_filter(mol) -> bool:
    """Reject molecules with atoms outside CHNOPS+halogens, excessive halogens,
    or pharmaceutical scaffolds (piperazine/haloarene drug-like patterns)."""
    halogen_count = 0
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in _ALLOWED_ATOMS:
            return False
        if sym in _HALOGENS:
            halogen_count += 1
    if halogen_count > _MAX_HALOGENS:
        return False
    # If halogens present, reject known pharma scaffolds
    if halogen_count > 0:
        for smarts in _DRUG_SMARTS:
            if smarts is not None and mol.HasSubstructMatch(smarts):
                return False
    return True


@dataclass(frozen=True)
class FlavorConstraints:
    mw_min: float = 100.0
    mw_max: float = 1000.0
    logp_min: float = float("-inf")
    logp_max: float = 3.0


def is_good_flavor_candidate(smiles: str, c: FlavorConstraints) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    return (
        c.mw_min < mw < c.mw_max
        and c.logp_min <= logp <= c.logp_max
        and _passes_atom_filter(mol)
    )


def featurize_smiles(smiles_list: list[str], *, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        X[i, :] = arr
    return X


def train_sweetness_model(fart_csv: str, *, seed: int = 0):
    df = pd.read_csv(fart_csv)
    # Support multiple teammate dataset schemas.
    # - FART (Ananya cleaned): Canonicalized SMILES, Canonicalized Taste (e.g. "sweet")
    # - Other: SMILES, Sweetness_Label (0/1)
    if {"Canonicalized SMILES", "Canonicalized Taste"}.issubset(df.columns):
        df = df.rename(columns={"Canonicalized SMILES": "SMILES", "Canonicalized Taste": "Taste"})
        df["Sweetness_Label"] = (df["Taste"].astype(str).str.lower() == "sweet").astype(int)
    elif {"SMILES", "Sweetness_Label"}.issubset(df.columns):
        pass
    else:
        raise ValueError(
            f"{fart_csv} must contain either SMILES,Sweetness_Label or Canonicalized SMILES,Canonicalized Taste"
        )

    df = df.dropna(subset=["SMILES", "Sweetness_Label"]).copy()
    df["Sweetness_Label"] = df["Sweetness_Label"].astype(int)

    valid = df["SMILES"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    df = df[valid]
    if df.empty:
        raise ValueError("No valid SMILES found in training dataset after filtering.")

    X = featurize_smiles(df["SMILES"].astype(str).tolist())
    y = df["Sweetness_Label"].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=1)),
        ]
    )
    model.fit(X_train, y_train)
    val_acc = float(model.score(X_val, y_val)) if len(np.unique(y_val)) > 1 else float("nan")
    return model, val_acc


def main():
    ap = argparse.ArgumentParser(
        description="Score TamGen-generated SMILES for sweetness + filter for flavor constraints."
    )
    ap.add_argument("--generated", required=True, help="TamGen output CSV (must have smiles column).")
    ap.add_argument("--training-data", dest="training_data", default=None,
                    help="Combined training CSV (SMILES, Sweetness_Label). Use tamgen_trainingdata.csv.")
    ap.add_argument("--fart", default=None, help="Alias for --training-data (legacy).")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mw-min", type=float, default=100.0)
    ap.add_argument("--mw-max", type=float, default=1000.0)
    ap.add_argument("--logp-min", type=float, default=float("-inf"),
                    help="Minimum LogP (default: no lower bound). Use 0 for the polar set.")
    ap.add_argument("--logp-max", type=float, default=3.0,
                    help="Maximum LogP (default 3.0). Use 0 for the sugar/highly-polar set.")
    ap.add_argument("--top-n", type=int, default=None,
                    help="Keep only the top N rows after ranking (default: keep all).")
    args = ap.parse_args()

    training_csv = args.training_data or args.fart
    if training_csv is None:
        ap.error("one of --training-data or --fart is required")

    constraints = FlavorConstraints(
        mw_min=args.mw_min,
        mw_max=args.mw_max,
        logp_min=args.logp_min,
        logp_max=args.logp_max,
    )

    model, val_acc = train_sweetness_model(training_csv, seed=args.seed)

    gen = pd.read_csv(args.generated)
    if "smiles" not in gen.columns:
        raise ValueError("--generated must contain a 'smiles' column")

    smiles = gen["smiles"].astype(str).tolist()
    Xg = featurize_smiles(smiles)
    sweet_prob = model.predict_proba(Xg)[:, 1]

    rows = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mw = np.nan
            logp = np.nan
            passes = False
            passes_atom = False
        else:
            mw = float(Descriptors.MolWt(mol))
            logp = float(Descriptors.MolLogP(mol))
            passes_atom = _passes_atom_filter(mol)
            passes = is_good_flavor_candidate(smi, constraints)
        rows.append(
            {
                "smiles": smi,
                "sweet_prob": float(sweet_prob[i]),
                "mw": mw,
                "logp": logp,
                "passes_atom_filter": bool(passes_atom),
                "passes_flavor_filter": bool(passes),
            }
        )

    out = gen.merge(pd.DataFrame(rows), on="smiles", how="left")
    out["fart_val_acc"] = val_acc

    sort_cols = ["passes_flavor_filter", "sweet_prob"]
    if "nlogP" in out.columns:
        sort_cols.append("nlogP")
    out = out.sort_values(
        by=sort_cols,
        ascending=[False] * len(sort_cols),
        kind="mergesort",
    )

    if args.top_n is not None:
        out = out.head(args.top_n)

    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")
    print(f"FART val accuracy (quick sanity check): {val_acc:.3f}" if np.isfinite(val_acc) else "FART val accuracy: n/a")
    print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
