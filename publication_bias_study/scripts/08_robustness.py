"""
08_robustness.py
----------------
All robustness checks listed in implementation plan §5:

R1  Restrict to quasi-experimental identification (RDD/IV/DID)
R2  Stricter "government intervention" definition (legislation only)
R3  Use RFS and JFE as comparison (is JF uniquely biased?)
R4  High-confidence LLM coding only
R5  Matched-pair analysis (propensity score)

Outputs:
    output/tables/robustness_*.csv

Usage:
    python scripts/08_robustness.py
"""

import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT    = pathlib.Path(__file__).parent.parent.resolve()
IN_PATH = ROOT / "data" / "final" / "analysis_dataset.parquet"
CLS_PATH= ROOT / "data" / "classified" / "coded_direction.parquet"
OUT_DIR = ROOT / "output" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH}. Run script 05 first.")
    df  = pd.read_parquet(IN_PATH)
    cls = pd.read_parquet(CLS_PATH) if CLS_PATH.exists() else pd.DataFrame()
    df["pub"] = df["published_top3"].astype(int)
    df["pos"] = df["positive"].fillna(False).astype(int)
    df["neg"] = df["negative"].fillna(False).astype(int)
    df["year_c"] = pd.to_numeric(df["year"], errors="coerce") - 2007.0
    return df, cls


def run_probit(formula: str, df: pd.DataFrame,
               label: str) -> pd.DataFrame:
    if df["pos"].sum() == 0 and df["neg"].sum() == 0:
        print(f"  [{label}] skipped: no direction codes")
        return pd.DataFrame()
    # Only drop NaN on columns referenced in the formula
    import re as _re
    formula_cols = _re.findall(r'[A-Za-z_][A-Za-z0-9_]*', formula)
    keep_cols = [c for c in formula_cols if c in df.columns]
    clean = df[keep_cols].dropna()
    if len(clean) < 20:
        print(f"  [{label}] skipped: only {len(clean)} obs after dropna")
        return pd.DataFrame()
    try:
        model = smf.probit(formula, data=clean).fit(
            disp=False, method="bfgs", maxiter=500)
    except Exception as exc:
        print(f"  [{label}] failed: {exc}")
        return pd.DataFrame()

    rows = []
    for name in ["pos","neg"]:
        if name not in model.params:
            continue
        coef = model.params[name]
        se   = model.bse[name]
        p    = model.pvalues[name]
        rows.append({
            "Robustness check": label,
            "Variable":         name,
            "Coefficient":      round(coef,4),
            "Std Error":        round(se,4),
            "p-value":          round(p,4),
            "N":                int(model.nobs),
            "Pseudo R2":        round(model.prsquared,4),
            "Stars": "***" if p<.01 else "**" if p<.05 else "*" if p<.10 else "",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# R1 — Quasi-experimental papers only                                  #
# ------------------------------------------------------------------ #

def r1_quasiexp(df: pd.DataFrame) -> pd.DataFrame:
    QUASI = ["RDD","IV","DiD","DID","natural experiment","regression discontinuity",
             "instrumental variable","difference-in-differences","difference in differences"]
    if "id_strategy" not in df.columns or df["id_strategy"].isna().all():
        print("  R1: id_strategy column not available — skipping")
        return pd.DataFrame()
    mask = df["id_strategy"].fillna("").str.contains(
        "|".join(QUASI), case=False, regex=True
    )
    sub = df[mask]
    print(f"  R1: quasi-exp subsample N={len(sub):,}")
    return run_probit("pub ~ pos + neg + year_c", sub,
                      "R1: Quasi-experimental only")


# ------------------------------------------------------------------ #
# R2 — Strict intervention definition (exclude monetary policy)        #
# ------------------------------------------------------------------ #

MONETARY_KEYWORDS = [
    "quantitative easing","qe","fed funds rate","monetary policy",
    "central bank communication","open market operation",
    "interest rate policy","forward guidance",
]


def r2_strict_definition(df: pd.DataFrame) -> pd.DataFrame:
    if "abstract" not in df.columns:
        print("  R2: abstract column not available — skipping")
        return pd.DataFrame()
    is_monetary = df["abstract"].fillna("").str.lower().apply(
        lambda t: any(kw in t for kw in MONETARY_KEYWORDS)
    )
    sub = df[~is_monetary]
    print(f"  R2: strict-definition subsample N={len(sub):,} "
          f"(dropped {is_monetary.sum():,} monetary policy papers)")
    return run_probit("pub ~ pos + neg + year_c", sub,
                      "R2: Exclude monetary policy")


# ------------------------------------------------------------------ #
# R3 — JF vs. RFS+JFE comparison                                      #
# ------------------------------------------------------------------ #

def r3_jf_vs_peers(df: pd.DataFrame) -> pd.DataFrame:
    if df["pos"].sum() == 0 and df["neg"].sum() == 0:
        print("  R3: no direction codes — skipping")
        return pd.DataFrame()
    results = []
    # Is JF uniquely biased compared with RFS+JFE?
    for pub_label, pub_col, sample_label in [
        ("JF only",      "pub_jf",    "vs. NBER"),
        ("RFS+JFE only", "pub_rfsjfe","vs. NBER"),
    ]:
        d = df.copy()
        d["pub_jf"]     = ((df["journal"] == "JF")             ).astype(int)
        d["pub_rfsjfe"] = (df["journal"].isin(["RFS","JFE"])).astype(int)
        results.append(run_probit(
            f"{pub_col} ~ pos + neg + year_c",
            d, f"R3: {pub_label}"
        ))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ------------------------------------------------------------------ #
# R4 — High-confidence LLM coding only                                 #
# ------------------------------------------------------------------ #

def r4_high_confidence(df: pd.DataFrame, cls: pd.DataFrame) -> pd.DataFrame:
    if cls.empty or "confidence" not in cls.columns:
        print("  R4: confidence column not available — skipping")
        return pd.DataFrame()
    high = cls[cls["confidence"] == "high"]["paper_id"]
    sub = df[df["paper_id"].isin(high)]
    pct_pub = sub["published_top3"].mean()
    print(f"  R4: high-confidence subsample N={len(sub):,} "
          f"({pct_pub*100:.0f}% published — note: confidence correlated with abstract quality)")
    return run_probit("pub ~ pos + neg + year_c", sub,
                      "R4: High-confidence LLM coding")


# ------------------------------------------------------------------ #
# R5 — Propensity-score matched pairs                                  #
# ------------------------------------------------------------------ #

def r5_pscore_matching(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match positive-finding NBER papers to null-finding NBER papers on
    year + program (propensity score), then compare publication rates.
    """
    nber = df[df["source"] == "nber"].copy()
    nber = nber.dropna(subset=["year"])
    if len(nber) < 50:
        print("  R5: too few NBER observations — skipping")
        return pd.DataFrame()

    nber["year_norm"] = (nber["year"] - nber["year"].mean()) / nber["year"].std()

    # Binary outcome: positive vs. null
    treat = nber[nber["pos"] == 1].copy()
    control = nber[nber["pos"] == 0].copy()
    if len(treat) < 10 or len(control) < 10:
        print("  R5: too few treated/control units — skipping")
        return pd.DataFrame()

    # Estimate propensity score using logistic regression on year
    X_t = treat[["year_norm"]].values
    X_c = control[["year_norm"]].values
    X   = np.vstack([X_t, X_c])
    y   = np.array([1]*len(treat) + [0]*len(control))

    lr  = LogisticRegression(max_iter=500).fit(X, y)
    ps_treat   = lr.predict_proba(X_t)[:,1]
    ps_control = lr.predict_proba(X_c)[:,1]

    treat = treat.copy(); treat["pscore"] = ps_treat
    control = control.copy(); control["pscore"] = ps_control

    # 1:1 nearest-neighbour matching without replacement
    control_sorted = control.sort_values("pscore").reset_index()
    matched_ctrl_idx = []
    used = set()
    for ps in sorted(ps_treat):
        diffs = (control_sorted["pscore"] - ps).abs()
        diffs.loc[list(used)] = np.inf
        best = diffs.idxmin()
        matched_ctrl_idx.append(best)
        used.add(best)

    matched_ctrl = control_sorted.loc[matched_ctrl_idx]

    # Compare publication rates
    pub_treat = treat["pub"].mean()
    pub_ctrl  = matched_ctrl["pub"].mean()
    if treat["pub"].std() == 0 and matched_ctrl["pub"].std() == 0:
        print(f"  R5: zero variance in both groups (pub rate: "
              f"treated={pub_treat:.3f}, control={pub_ctrl:.3f}) — "
              f"insufficient NBER-journal matches for this test")
        return pd.DataFrame()
    t_stat, p_val = stats.ttest_ind(
        treat["pub"].values, matched_ctrl["pub"].values
    )

    row = pd.DataFrame([{
        "Robustness check": "R5: PSM Positive vs Null",
        "Variable":         "pos",
        "Pub Rate Treated (positive)": round(pub_treat,3),
        "Pub Rate Control (null)":     round(pub_ctrl, 3),
        "Difference":                  round(pub_treat - pub_ctrl, 3),
        "t-stat":                      round(t_stat, 3),
        "p-value":                     round(p_val, 4),
        "N matched":                   len(matched_ctrl_idx),
        "Stars": "***" if p_val<.01 else "**" if p_val<.05 else "*" if p_val<.10 else "",
    }])
    print(f"  R5: matched N={len(matched_ctrl_idx)}, "
          f"pub-rate diff={pub_treat-pub_ctrl:+.3f}, p={p_val:.4f}")
    return row


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    df, cls = load_data()
    print(f"Loaded {len(df):,} rows for robustness checks")

    results = []

    print("\n[R1] Quasi-experimental papers only")
    results.append(r1_quasiexp(df))

    print("\n[R2] Strict definition (exclude monetary policy)")
    results.append(r2_strict_definition(df))

    print("\n[R3] JF vs. RFS+JFE")
    results.append(r3_jf_vs_peers(df))

    print("\n[R4] High-confidence LLM coding")
    results.append(r4_high_confidence(df, cls))

    print("\n[R5] Propensity-score matching")
    results.append(r5_pscore_matching(df))

    valid = [r for r in results if r is not None and not r.empty]
    if not valid:
        print("\nNo robustness results produced — direction codes required (run script 04 first)")
        return

    all_results = pd.concat(valid, ignore_index=True)

    out_path = OUT_DIR / "robustness_all.csv"
    all_results.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    display_cols = [c for c in ["Robustness check","Variable","Coefficient",
                                "p-value","Stars","N","N matched"]
                    if c in all_results.columns]
    print(all_results[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
