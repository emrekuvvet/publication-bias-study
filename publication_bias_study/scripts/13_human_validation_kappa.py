"""
13_human_validation_kappa.py
-----------------------------
Computes Cohen's kappa between human codes and LLM codes for the 40-paper
validation sample once the human coding sheet is complete.

Usage
------
    python scripts/13_human_validation_kappa.py

Prerequisites
--------------
    validation/human_coding_sheet_40.xlsx  — filled in by authors
        Column 'human_direction' : 1, -1, or 0  (direction code)
        Column 'human_inscope'   : 'yes' or 'no'

Output
------
    Prints Cohen's kappa (linear-weighted) for:
      - Direction coding (3 categories: +1, 0, -1)
      - Binary directional coding (directional vs. null)
      - In-scope classification
    Saves results to output/tables/human_validation_kappa.csv
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

ROOT  = pathlib.Path(__file__).parent.parent.resolve()
SHEET = ROOT / "validation" / "human_coding_sheet_40.xlsx"
CODES = ROOT / "validation" / "llm_codes_hidden.csv"
OUT   = ROOT / "output" / "tables" / "human_validation_kappa.csv"


def linear_weights(n):
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            W[i, j] = 1 - abs(i - j) / (n - 1)
    return W


def kappa_linear(y_true, y_pred, labels):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    n  = len(labels)
    W  = linear_weights(n)
    n_obs = cm.sum()
    exp   = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n_obs
    po    = (W * cm).sum() / n_obs
    pe    = (W * exp).sum() / n_obs
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else 1.0


def main():
    sheet = pd.read_excel(SHEET)
    codes = pd.read_csv(CODES)

    merged = sheet.merge(codes, on="paper_num")

    # Check completeness
    missing = merged["human_direction"].isna() | (merged["human_direction"] == "")
    if missing.any():
        print(f"WARNING: {missing.sum()} papers have no human direction code — excluding.")
        merged = merged[~missing]

    human_dir = merged["human_direction"].astype(int).values
    llm_dir   = merged["llm_direction"].astype(int).values
    labels_3  = [-1, 0, 1]

    k3  = kappa_linear(human_dir, llm_dir, labels_3)
    k2  = cohen_kappa_score(
        (np.abs(human_dir) == 1).astype(int),
        (np.abs(llm_dir)   == 1).astype(int)
    )

    print(f"\nN = {len(merged)} papers coded")
    print(f"Direction kappa (linear-weighted, 3-cat): κ = {k3:.3f}")
    print(f"Binary directional kappa:                 κ = {k2:.3f}")

    # Agreement table
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(human_dir, llm_dir, labels=labels_3)
    df_cm = pd.DataFrame(cm, index=["Human −1","Human 0","Human +1"],
                              columns=["LLM −1","LLM 0","LLM +1"])
    print("\nConfusion matrix (rows = human, cols = LLM):")
    print(df_cm.to_string())

    # In-scope kappa if available
    if "human_inscope" in merged.columns and merged["human_inscope"].notna().any():
        inscope_map = {"yes": 1, "no": 0, "Yes": 1, "No": 0}
        hi = merged["human_inscope"].map(inscope_map)
        valid = hi.notna()
        if valid.sum() >= 10:
            ki = cohen_kappa_score(hi[valid].astype(int),
                                   np.ones(valid.sum(), dtype=int))  # LLM says all in-scope
            print(f"\nIn-scope kappa (human vs LLM in-scope=1): κ = {ki:.3f}")

    results = pd.DataFrame([{
        "Metric": "Direction (linear-weighted, 3-cat)",
        "N": len(merged),
        "Kappa": round(k3, 4),
    }, {
        "Metric": "Binary directional",
        "N": len(merged),
        "Kappa": round(k2, 4),
    }])
    results.to_csv(OUT, index=False)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
