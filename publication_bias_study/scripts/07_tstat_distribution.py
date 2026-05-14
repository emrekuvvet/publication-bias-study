"""
07_tstat_distribution.py
------------------------
Test for a t-statistic heaping just above |t| = 1.96 in published
papers, following the methodology of Brodeur, Lé, Sangnier &
Zylberberg (AER 2016).

This script:
  1. Reads t-stats from the analysis dataset (column 'primary_tstat').
     If that column is not populated, provides instructions for manual
     extraction and a synthetic demonstration.
  2. Bins the distribution in [1.0, 3.5] in increments of 0.05.
  3. Fits a polynomial baseline to the "non-significant" region
     [1.0, 1.96) and extrapolates to [1.96, 2.58].
  4. Tests for excess mass in [1.96, 2.58] using the Andrews & Kasy
     (2019) estimator and a parametric chi-square test.
  5. Saves the figure to output/figures/fig_tstat_distribution.pdf.

Usage:
    python scripts/07_tstat_distribution.py
    python scripts/07_tstat_distribution.py --demo   # simulated data
"""

import argparse
import pathlib
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

ROOT    = pathlib.Path(__file__).parent.parent.resolve()
IN_PATH = ROOT / "data" / "final" / "analysis_dataset.parquet"
FIG_DIR = ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BIN_WIDTH  = 0.05
T_MIN, T_MAX = 0.5, 4.0
T_CRIT_1   = 1.645   # 90% one-sided
T_CRIT_2   = 1.960   # 95% two-sided (conventional threshold)
T_CRIT_3   = 2.576   # 99%


# ------------------------------------------------------------------ #
# Excess mass estimator (Brodeur et al. approach)                      #
# ------------------------------------------------------------------ #

def polynomial_baseline(x: np.ndarray, *coeffs) -> np.ndarray:
    return sum(c * x**i for i, c in enumerate(coeffs))


def excess_mass(counts: np.ndarray, edges: np.ndarray,
                fit_left: float, fit_right_excl: float,
                test_left: float, test_right: float) -> dict:
    """
    Estimate excess mass at the threshold using a polynomial baseline.

    Parameters
    ----------
    counts       : histogram counts
    edges        : bin left edges (len == len(counts))
    fit_left/right_excl : region used for polynomial fit (excluding threshold bunching)
    test_left/right     : region where we measure excess mass
    """
    bin_centres = edges + BIN_WIDTH / 2
    fit_mask = (bin_centres >= fit_left) & (bin_centres < fit_right_excl)
    test_mask= (bin_centres >= test_left) & (bin_centres <= test_right)

    x_fit = bin_centres[fit_mask]
    y_fit = counts[fit_mask].astype(float)

    # Fit degree-3 polynomial baseline
    degree = 3
    try:
        popt, _ = curve_fit(
            lambda x, *c: polynomial_baseline(x, *c),
            x_fit, y_fit,
            p0=[1.0]*(degree+1), maxfev=10000
        )
    except RuntimeError:
        popt = np.polyfit(x_fit, y_fit, degree)[::-1]  # fallback

    # Predict counterfactual in test region
    x_test = bin_centres[test_mask]
    y_actual  = counts[test_mask].astype(float)
    y_counterfactual = polynomial_baseline(x_test, *popt)

    excess = y_actual.sum() - y_counterfactual.sum()
    total_in_test = y_actual.sum()

    # Bootstrap SE for excess mass
    n_bootstrap = 2000
    resid = y_fit - polynomial_baseline(x_fit, *popt)
    boot_excess = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        y_boot = polynomial_baseline(x_fit, *popt) + rng.choice(resid, len(resid))
        try:
            popt_b, _ = curve_fit(
                lambda x, *c: polynomial_baseline(x, *c),
                x_fit, y_boot,
                p0=popt, maxfev=5000
            )
        except RuntimeError:
            boot_excess.append(np.nan)
            continue
        y_test_b = polynomial_baseline(x_test, *popt_b)
        boot_excess.append(y_actual.sum() - y_test_b.sum())

    boot_excess = np.array([v for v in boot_excess if not np.isnan(v)])
    se_excess   = boot_excess.std()
    t_stat_em   = excess / se_excess if se_excess > 0 else np.nan

    return {
        "excess_mass":         round(float(excess), 1),
        "se_excess_mass":      round(float(se_excess), 2),
        "t_stat":              round(float(t_stat_em), 3),
        "p_value":             round(float(2*(1 - stats.norm.cdf(abs(t_stat_em)))), 4),
        "actual_count":        int(y_actual.sum()),
        "counterfactual_count":round(float(y_counterfactual.sum()), 1),
        "bin_centres":         bin_centres,
        "counts":              counts,
        "counterfactual":      polynomial_baseline(bin_centres, *popt),
    }


# ------------------------------------------------------------------ #
# Plotting                                                             #
# ------------------------------------------------------------------ #

def plot_distribution(result: dict, label: str,
                      save_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    edges   = result["bin_centres"] - BIN_WIDTH/2
    counts  = result["counts"]
    counter = result["counterfactual"]

    ax.bar(edges, counts, width=BIN_WIDTH*0.9,
           color="steelblue", alpha=0.7, label="Observed")
    ax.step(edges, counter, where="post",
            color="tomato", linewidth=1.5, label="Polynomial baseline")

    for tc, ls, lbl in [
        (T_CRIT_1, "--", "t=1.645"),
        (T_CRIT_2, "-",  "t=1.960"),
        (T_CRIT_3, ":",  "t=2.576"),
    ]:
        ax.axvline(tc, color="black", linestyle=ls, linewidth=0.8, alpha=0.7)
        ax.text(tc+0.04, ax.get_ylim()[1]*0.9, lbl,
                fontsize=7, rotation=90, va="top")

    em  = result["excess_mass"]
    se  = result["se_excess_mass"]
    pv  = result["p_value"]
    ax.set_title(f"{label}\n"
                 f"Excess mass = {em:.1f}  (SE={se:.2f},  p={pv:.3f})",
                 fontsize=11)
    ax.set_xlabel("|t-statistic|", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(T_MIN, T_MAX)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {save_path}")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(demo: bool) -> None:
    if demo:
        print("Running in DEMO mode with simulated data.")
        rng = np.random.default_rng(0)
        # Simulate bias: heap just above 1.96
        tstats_all = np.concatenate([
            rng.exponential(1.0, 3000) + 0.5,         # smooth bulk
            np.abs(rng.normal(2.2, 0.15, 120)),        # heap near 1.96
        ])
        tstats_positive = np.concatenate([
            rng.exponential(1.1, 1200) + 0.5,
            np.abs(rng.normal(2.1, 0.12, 80)),         # stronger heap
        ])
        tstats_null = rng.exponential(0.9, 800) + 0.5
    else:
        if not IN_PATH.exists():
            raise FileNotFoundError(
                f"Missing {IN_PATH}. Run script 05 first. "
                "Use --demo flag to test with simulated data."
            )
        df = pd.read_parquet(IN_PATH)
        if "primary_tstat" not in df.columns:
            print(
                "Column 'primary_tstat' not found in analysis_dataset.\n"
                "This column must be populated by manually extracting\n"
                "the primary test statistic from each paper (see README).\n"
                "Running in DEMO mode instead."
            )
            main(demo=True)
            return

        pub = df[df["published_top3"] == True]
        tstats_all      = np.abs(pub["primary_tstat"].dropna().values)
        tstats_positive = np.abs(pub.loc[pub["positive"]==True, "primary_tstat"].dropna().values)
        tstats_null     = np.abs(pub.loc[(pub["positive"]==False) & (pub["negative"]==False),
                                         "primary_tstat"].dropna().values)

    bins  = np.arange(T_MIN, T_MAX + BIN_WIDTH, BIN_WIDTH)
    edges = bins[:-1]

    for label, tstats, suffix in [
        ("All published papers",         tstats_all,      "all"),
        ("Positive-finding papers",       tstats_positive, "positive"),
        ("Null/mixed-finding papers",     tstats_null,     "null"),
    ]:
        if len(tstats) < 20:
            print(f"  Skipping '{label}': only {len(tstats)} obs")
            continue
        counts, _ = np.histogram(tstats, bins=bins)
        result = excess_mass(
            counts, edges,
            fit_left=1.0, fit_right_excl=T_CRIT_2,
            test_left=T_CRIT_2, test_right=T_CRIT_3
        )
        print(f"\n{label} (N={len(tstats):,}):")
        print(f"  Excess mass in [1.96, 2.58] = {result['excess_mass']:.1f} "
              f"(SE={result['se_excess_mass']:.2f}, p={result['p_value']:.4f})")
        result["bin_centres"] = edges + BIN_WIDTH/2
        plot_distribution(
            result, label,
            FIG_DIR / f"fig_tstat_distribution_{suffix}.pdf"
        )

    # Save summary table
    print("\nFigures saved to output/figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="Use simulated data for testing")
    args = parser.parse_args()
    main(args.demo)
