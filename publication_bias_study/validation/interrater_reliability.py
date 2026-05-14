"""
interrater_reliability.py
--------------------------
Compute inter-rater agreement between two RAs and between each RA
and the LLM coding.  Generates output/tables/table_irf.csv.

Inputs:
    validation/human_coding_sheet.xlsx  (columns: paper_id, ra1_code, ra2_code)
    data/classified/coded_direction.parquet

Usage:
    python validation/interrater_reliability.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, classification_report

ROOT     = pathlib.Path(__file__).parent.parent.resolve()
SHEET    = ROOT / "validation" / "human_coding_sheet.xlsx"
LLM_PATH = ROOT / "data" / "classified" / "coded_direction.parquet"
OUT_DIR  = ROOT / "output" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def weighted_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """Linear-weighted Cohen's kappa for ordinal codes {-1, 0, +1}."""
    labels = [-1, 0, 1]
    return cohen_kappa_score(y1, y2, labels=labels, weights="linear")


def percent_agreement(y1: np.ndarray, y2: np.ndarray) -> float:
    return float((y1 == y2).mean())


def confusion_matrix_df(y_true: np.ndarray, y_pred: np.ndarray,
                        labels=(-1, 0, 1)) -> pd.DataFrame:
    from sklearn.metrics import confusion_matrix
    mat = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(mat,
                        index=[f"True {l}" for l in labels],
                        columns=[f"Pred {l}" for l in labels])


def main() -> None:
    # ---------------------------------------------------------------- #
    # Load human coding sheet                                           #
    # ---------------------------------------------------------------- #
    if not SHEET.exists():
        print(f"Human coding sheet not found: {SHEET}")
        print("Create it with columns: paper_id, ra1_code, ra2_code")
        print("Codes: +1 (positive), -1 (negative), 0 (null/mixed)")
        sys.exit(0)

    humans = pd.read_excel(SHEET, dtype={"paper_id": str,
                                          "ra1_code": int,
                                          "ra2_code": int})
    humans = humans.dropna(subset=["ra1_code","ra2_code"])
    ra1 = humans["ra1_code"].values
    ra2 = humans["ra2_code"].values

    print(f"Human coding sample: {len(humans):,} papers")

    # ---------------------------------------------------------------- #
    # RA1 vs RA2                                                        #
    # ---------------------------------------------------------------- #
    kappa_ra = weighted_kappa(ra1, ra2)
    pct_ra   = percent_agreement(ra1, ra2)
    print(f"\nRA1 vs RA2:")
    print(f"  Linear-weighted kappa : {kappa_ra:.3f}")
    print(f"  Percent agreement     : {pct_ra*100:.1f}%")
    print(confusion_matrix_df(ra1, ra2).to_string())

    # ---------------------------------------------------------------- #
    # LLM vs human consensus                                            #
    # ---------------------------------------------------------------- #
    if not LLM_PATH.exists():
        print("\nLLM coding not yet available — skip LLM vs human comparison.")
        kappa_llm_ra1 = kappa_llm_ra2 = np.nan
    else:
        llm_df = pd.read_parquet(LLM_PATH)[["paper_id","direction"]]
        merged = humans.merge(llm_df.rename(columns={"direction":"llm_code"}),
                              on="paper_id", how="inner")
        if len(merged) == 0:
            print("\nNo overlap between human sample and LLM codes.")
            kappa_llm_ra1 = kappa_llm_ra2 = np.nan
        else:
            llm  = merged["llm_code"].values
            h_ra1= merged["ra1_code"].values
            h_ra2= merged["ra2_code"].values
            # Consensus = mode of RA1, RA2
            consensus = np.where(h_ra1 == h_ra2, h_ra1,
                                  np.where(h_ra1 == 0, h_ra2,
                                           np.where(h_ra2 == 0, h_ra1, 0)))
            kappa_llm_ra1 = weighted_kappa(h_ra1, llm)
            kappa_llm_ra2 = weighted_kappa(h_ra2, llm)
            kappa_llm_con = weighted_kappa(consensus, llm)
            pct_llm_con   = percent_agreement(consensus, llm)

            print(f"\nLLM vs Human ({len(merged):,} papers):")
            print(f"  LLM vs RA1 kappa    : {kappa_llm_ra1:.3f}")
            print(f"  LLM vs RA2 kappa    : {kappa_llm_ra2:.3f}")
            print(f"  LLM vs Consensus κ  : {kappa_llm_con:.3f}")
            print(f"  LLM vs Consensus %  : {pct_llm_con*100:.1f}%")
            print("\n" + classification_report(
                consensus, llm, labels=[-1,0,1],
                target_names=["Negative","Null","Positive"]
            ))

    # ---------------------------------------------------------------- #
    # Save summary table                                                #
    # ---------------------------------------------------------------- #
    summary = pd.DataFrame([{
        "Comparison":           "RA1 vs RA2",
        "Linear kappa":         round(kappa_ra, 3),
        "Percent agreement (%)":round(pct_ra*100, 1),
        "N":                    len(humans),
        "Meets threshold (κ≥0.70)": "YES" if kappa_ra >= 0.70 else "NO",
    }])
    out_path = OUT_DIR / "table_interrater_reliability.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    if kappa_ra < 0.70:
        print("\n*** WARNING: kappa < 0.70 target ***")
        print("Review coding rubric and resolve disagreements before proceeding.")
    else:
        print(f"\n[OK] kappa={kappa_ra:.3f} meets the ≥ 0.70 threshold.")


if __name__ == "__main__":
    main()
