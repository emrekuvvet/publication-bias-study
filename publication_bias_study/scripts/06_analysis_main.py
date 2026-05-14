"""
06_analysis_main.py
-------------------
Primary probit regressions testing directional publication bias.

Model (from implementation plan §4a):
    Published_in_top3(i) = α + β₁ Positive(i) + β₂ Negative(i) + γ X(i) + ε(i)

Outputs saved to output/tables/:
    table1_summary_stats.csv
    table2_publication_rates.csv
    table3_probit_main.csv       ← primary result
    table4_probit_by_journal.csv
    appendix_time_trends.csv

Usage:
    python scripts/06_analysis_main.py
    python scripts/06_analysis_main.py --output-format latex
"""

import argparse
import pathlib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT     = pathlib.Path(__file__).parent.parent.resolve()
IN_PATH  = ROOT / "data" / "final" / "analysis_dataset.parquet"
OUT_DIR  = ROOT / "output" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
# Load data                                                            #
# ------------------------------------------------------------------ #

def load_data() -> pd.DataFrame:
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"Missing {IN_PATH}. Run script 05 first."
        )
    df = pd.read_parquet(IN_PATH)
    # Cast types
    df["published_top3"] = df["published_top3"].astype(bool)
    df["positive"] = df["positive"].fillna(False).astype(bool)
    df["negative"] = df["negative"].fillna(False).astype(bool)
    df["year"]     = pd.to_numeric(df["year"], errors="coerce")
    df["direction"]= pd.to_numeric(df["direction"], errors="coerce")
    return df


# ------------------------------------------------------------------ #
# Table 1 — Summary statistics                                         #
# ------------------------------------------------------------------ #

def table_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, mask in [
        ("All papers",               [True]*len(df)),
        ("NBER working papers",      df["source"] == "nber"),
        ("Published (JF/RFS/JFE)",   df["published_top3"]),
        ("Published — JF",           df["journal"] == "JF"),
        ("Published — RFS",          df["journal"] == "RFS"),
        ("Published — JFE",          df["journal"] == "JFE"),
    ]:
        sub = df[mask]
        rows.append({
            "Sample":         label,
            "N":              len(sub),
            "Positive (%)":   round(sub["positive"].mean()*100, 1),
            "Negative (%)":   round(sub["negative"].mean()*100, 1),
            "Null/Mixed (%)": round((~sub["positive"] & ~sub["negative"]).mean()*100, 1),
            "Year mean":      round(sub["year"].mean(), 1),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Table 2 — Raw publication rates by direction                         #
# ------------------------------------------------------------------ #

def table_pub_rates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dir_val, label in [(1, "Positive (+1)"),
                           (-1, "Negative (-1)"),
                           (0, "Null/Mixed (0)")]:
        sub = df[df["direction"] == dir_val]
        if len(sub) == 0:
            continue
        pub_rate = sub["published_top3"].mean()
        n_pub    = sub["published_top3"].sum()
        ci_lo, ci_hi = proportion_ci(n_pub, len(sub))
        rows.append({
            "Direction":       label,
            "N":               len(sub),
            "N Published":     int(n_pub),
            "Pub Rate":        round(pub_rate*100, 2),
            "95% CI lower":    round(ci_lo*100, 2),
            "95% CI upper":    round(ci_hi*100, 2),
        })
    if not rows:
        print("  No direction-coded papers available — skipping chi2 test")
        return pd.DataFrame()
    # Chi-square test of homogeneity
    freq = df[df["direction"].isin([1,-1,0])].groupby(
        ["direction","published_top3"]
    ).size().unstack(fill_value=0)
    if freq.size > 0 and freq.values.sum() > 0:
        chi2, p, _, _ = stats.chi2_contingency(freq.values)
        print(f"  Chi2 test of pub rate homogeneity: χ²={chi2:.3f}, p={p:.4f}")
    return pd.DataFrame(rows)


def proportion_ci(k: int, n: int, alpha: float = 0.05):
    """Wilson score confidence interval."""
    if n == 0:
        return 0.0, 1.0
    z = stats.norm.ppf(1 - alpha/2)
    p_hat = k / n
    centre = (p_hat + z**2/(2*n)) / (1 + z**2/n)
    margin = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / (1 + z**2/n)
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ------------------------------------------------------------------ #
# Probit regression helpers                                            #
# ------------------------------------------------------------------ #

def run_probit(formula: str, df: pd.DataFrame,
               label: str) -> pd.DataFrame:
    """Fit a probit model and return a tidy coefficient table."""
    try:
        model = smf.probit(formula, data=df).fit(
            disp=False, method="bfgs", maxiter=500
        )
    except Exception as exc:
        print(f"  Warning [{label}]: {exc}")
        return pd.DataFrame()

    result_rows = []
    for name in model.params.index:
        coef = model.params[name]
        se   = model.bse[name]
        z    = model.tvalues[name]
        p    = model.pvalues[name]
        ci_lo, ci_hi = model.conf_int().loc[name]
        stars = ("***" if p < 0.01 else
                 "**"  if p < 0.05 else
                 "*"   if p < 0.10 else "")
        result_rows.append({
            "Specification": label,
            "Variable":      name,
            "Coefficient":   round(coef, 4),
            "Std Error":     round(se,   4),
            "z-stat":        round(z,    3),
            "p-value":       round(p,    4),
            "CI 2.5%":       round(ci_lo,4),
            "CI 97.5%":      round(ci_hi,4),
            "Significance":  stars,
            "N":             int(model.nobs),
            "Pseudo R2":     round(model.prsquared, 4),
        })
    return pd.DataFrame(result_rows)


# ------------------------------------------------------------------ #
# Table 3 — Main probit results                                        #
# ------------------------------------------------------------------ #

def run_lpm(formula: str, df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Fit a linear probability model (OLS) and return tidy coefficient table."""
    import statsmodels.formula.api as smf_ols
    try:
        model = smf_ols.ols(formula, data=df).fit(cov_type='HC3')
    except Exception as exc:
        print(f"  Warning LPM [{label}]: {exc}")
        return pd.DataFrame()
    result_rows = []
    for name in model.params.index:
        p = model.pvalues[name]
        stars = ("***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "")
        result_rows.append({
            "Specification": label,
            "Variable":      name,
            "Coefficient":   round(model.params[name], 4),
            "Std Error":     round(model.bse[name],    4),
            "t-stat":        round(model.tvalues[name],3),
            "p-value":       round(p, 4),
            "Significance":  stars,
            "N":             int(model.nobs),
            "R2":            round(model.rsquared, 4),
        })
    return pd.DataFrame(result_rows)


def table_probit_main(df: pd.DataFrame) -> pd.DataFrame:
    if df["direction"].isna().all():
        print("  No direction codes available — skipping probit (run script 04 first)")
        return pd.DataFrame()
    df = df.copy()
    df["pub"]     = df["published_top3"].astype(int)
    df["pos"]     = df["positive"].astype(int)
    df["neg"]     = df["negative"].astype(int)
    df["year_c"]  = df["year"] - df["year"].mean()
    df["decade"]  = (df["year"] // 10) * 10
    df["log_nauth"] = np.log1p(df["n_authors"].fillna(df["n_authors"].median()))
    df["log_ablen"] = np.log1p(df["abstract_len"].fillna(df["abstract_len"].median()))

    all_results = []

    # Spec 1: bivariate
    all_results.append(run_probit("pub ~ pos + neg", df, "Spec 1: Bivariate"))
    # Spec 2: + year trend
    all_results.append(run_probit("pub ~ pos + neg + year_c", df, "Spec 2: + Year"))
    # Spec 3: + decade FE
    all_results.append(run_probit("pub ~ pos + neg + C(decade)", df, "Spec 3: + Decade FE"))
    # Spec 4: + quality controls (n_authors, abstract length)
    all_results.append(run_probit(
        "pub ~ pos + neg + year_c + log_nauth + log_ablen",
        df, "Spec 4: + Quality Controls"
    ))

    return pd.concat(all_results, ignore_index=True)


# ------------------------------------------------------------------ #
# Table 4 — By journal                                                 #
# ------------------------------------------------------------------ #

def table_probit_by_journal(df: pd.DataFrame) -> pd.DataFrame:
    if df["direction"].isna().all():
        print("  No direction codes available — skipping by-journal probit")
        return pd.DataFrame()
    results = []
    df = df.copy()
    df["pub"] = df["published_top3"].astype(int)
    df["pos"] = df["positive"].astype(int)
    df["neg"] = df["negative"].astype(int)
    df["year_c"] = df["year"] - df["year"].mean()

    for journal in ["JF", "RFS", "JFE"]:
        jdf = df[(df["journal"] == journal) | (~df["published_top3"])].copy()
        jdf["pub_j"] = (df["journal"] == journal).astype(int)
        results.append(run_probit(
            "pub_j ~ pos + neg + year_c",
            jdf, f"Journal = {journal}"
        ))
    return pd.concat(results, ignore_index=True)


# ------------------------------------------------------------------ #
# Appendix — time trends                                               #
# ------------------------------------------------------------------ #

def table_time_trends(df: pd.DataFrame) -> pd.DataFrame:
    if df["direction"].isna().all():
        print("  No direction codes available — skipping time trends")
        return pd.DataFrame()
    periods = [
        ("2000-2007", (2000, 2007)),
        ("2008-2015", (2008, 2015)),
        ("2016-2024", (2016, 2024)),
    ]
    results = []
    df = df.copy()
    df["pub"] = df["published_top3"].astype(int)
    df["pos"] = df["positive"].astype(int)
    df["neg"] = df["negative"].astype(int)

    for label, (y0, y1) in periods:
        sub = df[(df["year"] >= y0) & (df["year"] <= y1)]
        if len(sub) < 20:
            print(f"  Skipping {label}: only {len(sub)} obs")
            continue
        results.append(run_probit(
            "pub ~ pos + neg",
            sub, f"Period: {label}"
        ))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ------------------------------------------------------------------ #
# Output                                                               #
# ------------------------------------------------------------------ #

def save(df: pd.DataFrame, name: str, fmt: str) -> None:
    if df.empty:
        print(f"  Skipping {name}: empty table")
        return
    path_csv = OUT_DIR / f"{name}.csv"
    df.to_csv(path_csv, index=False)
    print(f"  Saved → {path_csv}")
    if fmt == "latex":
        path_tex = OUT_DIR / f"{name}.tex"
        with open(path_tex, "w") as f:
            f.write(df.to_latex(index=False, float_format="{:.4f}".format,
                                escape=False))
        print(f"  Saved → {path_tex}")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def table_lpm(df: pd.DataFrame) -> pd.DataFrame:
    """Linear probability model as robustness check (avoids probit separation issues)."""
    df = df.copy()
    df["pub"]    = df["published_top3"].astype(int)
    df["pos"]    = df["positive"].astype(int)
    df["neg"]    = df["negative"].astype(int)
    df["year_c"] = df["year"] - df["year"].mean()
    results = []
    results.append(run_lpm("pub ~ pos + neg", df, "LPM: Bivariate"))
    results.append(run_lpm("pub ~ pos + neg + year_c", df, "LPM: + Year"))
    return pd.concat(results, ignore_index=True)


def table_binary_directional(df: pd.DataFrame) -> pd.DataFrame:
    """Binary directional vs null: collapses pos/neg into single indicator."""
    df = df.copy()
    df["pub"]        = df["published_top3"].astype(int)
    df["directional"]= (df["direction"] != 0).astype(int)
    df["year_c"]     = df["year"] - df["year"].mean()
    df["log_nauth"]  = np.log1p(df["n_authors"].fillna(df["n_authors"].median()))
    df["log_ablen"]  = np.log1p(df["abstract_len"].fillna(df["abstract_len"].median()))
    results = []
    results.append(run_probit("pub ~ directional", df, "Binary: Bivariate"))
    results.append(run_probit("pub ~ directional + year_c", df, "Binary: + Year"))
    results.append(run_probit(
        "pub ~ directional + year_c + log_nauth + log_ablen",
        df, "Binary: + Quality Controls"))
    return pd.concat(results, ignore_index=True)


def table_validation_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute LLM direction-coding confidence by published vs NBER status."""
    coded = pd.read_parquet(
        pathlib.Path(__file__).parent.parent / "data" / "classified" / "coded_direction.parquet"
    )
    merged = df.merge(coded[["paper_id","confidence"]], on="paper_id", how="left")
    rows = []
    for label, mask in [("Published", df["published_top3"]),
                         ("NBER-only",  ~df["published_top3"])]:
        sub = merged[mask]
        for conf_level in ["high","medium","low"]:
            n = (sub["confidence"] == conf_level).sum()
            rows.append({"Group": label, "Confidence": conf_level,
                         "N": n, "Share (%)": round(n/len(sub)*100, 1)})
    return pd.DataFrame(rows)


def main(output_format: str) -> None:
    df = load_data()
    print(f"Loaded analysis dataset: {len(df):,} rows")
    print(f"  Published: {df['published_top3'].sum():,} / {len(df):,}")

    print("\n[Table 1] Summary statistics")
    save(table_summary(df), "table1_summary_stats", output_format)

    print("\n[Table 2] Raw publication rates by direction")
    save(table_pub_rates(df), "table2_publication_rates", output_format)

    print("\n[Table 3] Primary probit regressions (incl. quality controls)")
    save(table_probit_main(df), "table3_probit_main", output_format)

    print("\n[Table 4] Probit by journal")
    save(table_probit_by_journal(df), "table4_probit_by_journal", output_format)

    print("\n[Appendix] Time-trend regressions (all sub-periods)")
    save(table_time_trends(df), "appendix_time_trends", output_format)

    print("\n[Appendix] Linear probability model")
    save(table_lpm(df), "appendix_lpm", output_format)

    print("\n[Appendix] Binary directional vs null")
    save(table_binary_directional(df), "appendix_binary_directional", output_format)

    print("\n[Appendix] LLM confidence by publication status")
    save(table_validation_stats(df), "appendix_llm_confidence", output_format)

    print("\nAll tables written to output/tables/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-format", choices=["csv","latex"],
                        default="csv")
    args = parser.parse_args()
    main(args.output_format)
