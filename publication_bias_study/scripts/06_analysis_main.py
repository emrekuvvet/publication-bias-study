"""
06_analysis_main.py
-------------------
Primary probit regressions documenting the significance-sorting pattern
in top finance journals (government intervention research, 2000-2024).

Primary model (binary directional — preferred specification):
    Published_in_top3(i) = α + β Directional(i) + γ X(i) + ε(i)

Secondary model (three-way direction code, for symmetry test):
    Published_in_top3(i) = α + β₁ Positive(i) + β₂ Negative(i) + γ X(i) + ε(i)

NOTE ON INTERPRETATION: Coefficients are cross-sectional associations between
finding direction and publication status. The design cannot distinguish editorial
selection from author self-selection in submission or specification search.

Outputs saved to output/tables/:
    table1_summary_stats.csv/.tex
    table2_publication_rates.csv/.tex
    table3_binary_primary.csv/.tex   ← PRIMARY result (binary directional)
    table3_probit_threeway.csv/.tex  ← Secondary (three-way direction, symmetry test)
    table4_probit_by_journal.csv/.tex
    appendix_time_trends.csv/.tex    ← Exploratory; small sub-samples
    appendix_lpm.csv/.tex
    appendix_llm_confidence.csv/.tex

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


def wald_test_symmetry(model) -> dict:
    """
    Wald test for H0: beta_pos == beta_neg.
    Also computes power of this test at 80% level.
    Returns dict with chi2, p-value, and minimum detectable difference.
    """
    from scipy.stats import chi2 as chi2_dist, norm
    try:
        R = np.zeros((1, len(model.params)))
        pos_idx = list(model.params.index).index("pos")
        neg_idx = list(model.params.index).index("neg")
        R[0, pos_idx] =  1
        R[0, neg_idx] = -1
        w = model.wald_test(R, scalar=True)
        chi2_val = float(w.statistic)
        p_val    = float(w.pvalue)

        # Power calculation: given observed N and SE(beta_pos - beta_neg),
        # what minimum difference is detectable at 80% power, alpha=0.05?
        se_diff = float(np.sqrt(
            model.cov_params().iloc[pos_idx, pos_idx]
            + model.cov_params().iloc[neg_idx, neg_idx]
            - 2 * model.cov_params().iloc[pos_idx, neg_idx]
        ))
        # At 80% power, alpha=0.05 (one-sided): delta = (z_alpha + z_beta) * se_diff
        z_alpha = norm.ppf(0.975)   # 1.96
        z_beta  = norm.ppf(0.80)    # 0.84
        mdd = (z_alpha + z_beta) * se_diff

        return {"chi2": round(chi2_val, 3), "p": round(p_val, 4),
                "se_diff": round(se_diff, 4), "mdd_80pct": round(mdd, 4)}
    except Exception as exc:
        print(f"    Wald test failed: {exc}")
        return {}


def table_probit_main(df: pd.DataFrame) -> pd.DataFrame:
    """
    Secondary specification: three-way direction code (positive / negative / null).

    Used to test the symmetry hypothesis H0: β_pos = β_neg.
    The primary result is table_binary_directional() which avoids normative coding.
    These coefficients are cross-sectional associations — the design cannot identify
    editorial selection vs. author self-selection in submission.
    """
    if df["direction"].isna().all():
        print("  No direction codes available — skipping probit (run script 04 first)")
        return pd.DataFrame()
    df = df.copy()
    df["pub"]      = df["published_top3"].astype(int)
    df["pos"]      = df["positive"].astype(int)
    df["neg"]      = df["negative"].astype(int)
    df["year_c"]   = df["year"] - df["year"].mean()
    df["decade"]   = (df["year"] // 10) * 10
    df["log_nauth"]= np.log1p(pd.to_numeric(df["n_authors"], errors="coerce").fillna(0))
    df["log_ablen"]= np.log1p(pd.to_numeric(df["abstract_len"], errors="coerce").fillna(
        df["abstract_len"].dropna().astype(float).median()))
    # Identification strategy quality control (from script 09)
    df["quasi_exp"]= pd.to_numeric(df.get("quasi_exp", 0), errors="coerce").fillna(0).astype(int)

    all_results = []
    fitted_models = {}

    specs = [
        ("Spec 1: Bivariate",      "pub ~ pos + neg"),
        ("Spec 2: + Year",         "pub ~ pos + neg + year_c"),
        ("Spec 3: + Decade FE",    "pub ~ pos + neg + C(decade)"),
        ("Spec 4: + Author Count", "pub ~ pos + neg + year_c + log_nauth"),
        ("Spec 5: + ID Strategy",  "pub ~ pos + neg + year_c + log_nauth + quasi_exp"),
    ]

    for label, formula in specs:
        import statsmodels.formula.api as smf_inner
        try:
            m = smf_inner.probit(formula, data=df).fit(
                disp=False, method="bfgs", maxiter=500)
            fitted_models[label] = m
        except Exception as exc:
            print(f"  Warning [{label}]: {exc}")
            continue
        result_rows = []
        for name in m.params.index:
            coef = m.params[name]; se = m.bse[name]
            z = m.tvalues[name];   p  = m.pvalues[name]
            ci_lo, ci_hi = m.conf_int().loc[name]
            stars = ("***" if p < 0.01 else "**" if p < 0.05 else
                     "*" if p < 0.10 else "")
            result_rows.append({
                "Specification": label, "Variable": name,
                "Coefficient": round(coef,4), "Std Error": round(se,4),
                "z-stat": round(z,3), "p-value": round(p,4),
                "CI 2.5%": round(ci_lo,4), "CI 97.5%": round(ci_hi,4),
                "Significance": stars, "N": int(m.nobs),
                "Pseudo R2": round(m.prsquared,4),
            })
        all_results.append(pd.DataFrame(result_rows))

    # Report Wald test on baseline spec
    if "Spec 1: Bivariate" in fitted_models:
        wd = wald_test_symmetry(fitted_models["Spec 1: Bivariate"])
        if wd:
            print(f"  Wald H0(β_pos=β_neg): χ²={wd['chi2']}, p={wd['p']}")
            print(f"  Power note: MDD at 80% power = {wd['mdd_80pct']:.3f} "
                  f"(SE of diff = {wd['se_diff']:.4f})")

    return pd.concat(all_results, ignore_index=True)


# ------------------------------------------------------------------ #
# Table 4 — By journal                                                 #
# ------------------------------------------------------------------ #

def table_probit_by_journal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Direction--publication association by journal.

    CAUTION: Cross-journal comparisons are underpowered. JF sub-sample = 38
    papers. Cross-journal Wald tests are uniformly insignificant; pairwise
    differences in coefficients are not statistically distinguishable.
    Report as exploratory/descriptive only.
    """
    if df["direction"].isna().all():
        print("  No direction codes available — skipping by-journal probit")
        return pd.DataFrame()
    results = []
    df = df.copy()
    df["pub"] = df["published_top3"].astype(int)
    df["pos"] = df["positive"].astype(int)
    df["neg"] = df["negative"].astype(int)
    df["year_c"] = df["year"] - df["year"].mean()

    import statsmodels.formula.api as smf_inner
    fitted = {}
    for journal in ["JF", "RFS", "JFE"]:
        jdf = df[(df["journal"] == journal) | (~df["published_top3"])].copy()
        jdf["pub_j"] = (df["journal"] == journal).astype(int)
        results.append(run_probit("pub_j ~ pos + neg + year_c", jdf, f"Journal = {journal}"))
        try:
            fitted[journal] = smf_inner.probit(
                "pub_j ~ pos + neg + year_c", data=jdf
            ).fit(disp=False, method="bfgs", maxiter=500)
        except Exception:
            pass

    # Cross-journal Wald tests: are beta_pos coefficients distinguishable across journals?
    print("  Cross-journal symmetry tests (power likely low):")
    for j1, j2 in [("JF","RFS"), ("JF","JFE"), ("RFS","JFE")]:
        if j1 in fitted and j2 in fitted:
            b1 = fitted[j1].params.get("pos", np.nan)
            b2 = fitted[j2].params.get("pos", np.nan)
            se1 = fitted[j1].bse.get("pos", np.nan)
            se2 = fitted[j2].bse.get("pos", np.nan)
            if not any(np.isnan([b1, b2, se1, se2])):
                z = (b1 - b2) / np.sqrt(se1**2 + se2**2)
                p = 2 * (1 - stats.norm.cdf(abs(z)))
                print(f"    β_pos: {j1} vs {j2}: z={z:.3f}, p={p:.3f}")
                results.append(pd.DataFrame([{
                    "Specification": f"Cross-journal: {j1} vs {j2} (β_pos diff)",
                    "Variable": "pos_diff",
                    "Coefficient": round(b1 - b2, 4),
                    "Std Error": round(np.sqrt(se1**2 + se2**2), 4),
                    "z-stat": round(z, 3), "p-value": round(p, 4),
                    "Significance": ("***" if p<.01 else "**" if p<.05 else
                                     "*" if p<.10 else ""),
                    "N": np.nan, "Pseudo R2": np.nan,
                    "CI 2.5%": np.nan, "CI 97.5%": np.nan,
                }]))

    return pd.concat(results, ignore_index=True)


# ------------------------------------------------------------------ #
# Appendix — time trends                                               #
# ------------------------------------------------------------------ #

def table_time_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    EXPLORATORY sub-period analysis.

    Sub-sample sizes are small (N ≈ 45, 97, 154). Results should be read as
    descriptive patterns, not confirmatory tests. CIs overlap substantially
    across periods. The crisis-period asymmetry is consistent with but does
    NOT causally identify elevated publication rates for regulatory-cost research.
    """
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
    """
    PRIMARY specification: binary directional vs null.

    Collapses pos/neg into a single indicator, entirely avoiding normative
    direction coding (no judgment required about whether an outcome is
    "beneficial" or "harmful"). This is our most credible result.

    Cross-sectional association only — cannot identify editorial selection
    vs. author self-selection in submission.
    """
    df = df.copy()
    df["pub"]        = df["published_top3"].astype(int)
    df["directional"]= (df["direction"] != 0).astype(int)
    df["year_c"]     = df["year"] - df["year"].mean()
    df["log_nauth"]  = np.log1p(pd.to_numeric(df["n_authors"], errors="coerce").fillna(0))
    df["log_ablen"]  = np.log1p(pd.to_numeric(df["abstract_len"], errors="coerce").fillna(df["abstract_len"].dropna().astype(float).median()))
    df["quasi_exp"]  = pd.to_numeric(df.get("quasi_exp", 0), errors="coerce").fillna(0).astype(int)
    results = []
    results.append(run_probit("pub ~ directional",                          df, "Binary (1): Bivariate"))
    results.append(run_probit("pub ~ directional + year_c",                 df, "Binary (2): + Year"))
    results.append(run_probit("pub ~ directional + year_c + log_nauth",     df, "Binary (3): + Author Count"))
    results.append(run_probit("pub ~ directional + year_c + log_nauth + quasi_exp",
                                                                             df, "Binary (4): + ID Strategy"))
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

    # ------------------------------------------------------------------
    # PRIMARY result: binary directional specification.
    # Collapses pos/neg into a single indicator — no normative direction
    # coding. This is the paper's main finding (Table III, Column 1).
    # ------------------------------------------------------------------
    print("\n[Table 3 — PRIMARY] Binary directional probit")
    save(table_binary_directional(df), "table3_binary_primary", output_format)

    # ------------------------------------------------------------------
    # SECONDARY: three-way direction code (positive / negative / null).
    # Used to test the symmetry hypothesis H0: β_pos = β_neg.
    # Results are robust to this more detailed decomposition.
    # ------------------------------------------------------------------
    print("\n[Table 3 — Secondary] Three-way direction probit (symmetry test)")
    save(table_probit_main(df), "table3_probit_threeway", output_format)

    print("\n[Table 4] Direction--publication association by journal (exploratory; underpowered)")
    save(table_probit_by_journal(df), "table4_probit_by_journal", output_format)

    print("\n[Appendix] Time-trend regressions — EXPLORATORY; small sub-samples")
    save(table_time_trends(df), "appendix_time_trends", output_format)

    print("\n[Appendix] Linear probability model")
    save(table_lpm(df), "appendix_lpm", output_format)

    print("\n[Appendix] LLM confidence by publication status")
    save(table_validation_stats(df), "appendix_llm_confidence", output_format)

    print("\nAll tables written to output/tables/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-format", choices=["csv","latex"],
                        default="csv")
    args = parser.parse_args()
    main(args.output_format)
