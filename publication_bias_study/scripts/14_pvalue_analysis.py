"""
Script 14: Brodeur-style p-value distribution analysis.
- Plots density of |z| statistics
- Caliper test: excess mass below 1.96 vs above
- Density discontinuity test at z=1.96 (p=0.05 threshold)
- Separate by finding direction (positive/negative/null papers)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from scipy.stats import norm
from pathlib import Path

matplotlib.rcParams['font.family'] = 'serif'
OUT_DIR = Path('output/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def caliper_test(z_abs, low=1.645, threshold=1.96, high=2.275):
    """
    Brodeur-style caliper test.
    Compares density in (low, threshold) vs (threshold, high).
    Under the null (no bunching), these should be equal.
    Returns: ratio, chi2, p-value
    """
    below = ((z_abs >= low) & (z_abs < threshold)).sum()
    above = ((z_abs >= threshold) & (z_abs < high)).sum()
    total = below + above
    if total == 0:
        return None, None, None
    expected = total / 2
    chi2 = ((below - expected)**2 + (above - expected)**2) / expected
    p = 1 - stats.chi2.cdf(chi2, df=1)
    ratio = below / above if above > 0 else np.inf
    return ratio, chi2, p

def density_discontinuity_test(z_abs, threshold=1.96, bandwidth=0.5):
    """
    Simple density discontinuity test at z=threshold.
    Estimates density just below and just above the threshold using kernel density.
    """
    below_mask = (z_abs >= threshold - bandwidth) & (z_abs < threshold)
    above_mask = (z_abs >= threshold) & (z_abs < threshold + bandwidth)
    n_below = below_mask.sum()
    n_above = above_mask.sum()
    n = len(z_abs)
    # Proportion test
    if n_below + n_above == 0:
        return None, None
    p_below = n_below / n
    p_above = n_above / n
    se = np.sqrt((p_below * (1 - p_below) + p_above * (1 - p_above)) / n)
    z_stat = (p_below - p_above) / se if se > 0 else 0
    p_val = 2 * (1 - norm.cdf(abs(z_stat)))
    return z_stat, p_val

def plot_zdist(z_abs, title, filename, threshold=1.96):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Caliper and discontinuity computed on the full cleaned sample [0, 50]
    z_full = z_abs[(z_abs >= 0) & (z_abs <= 50)]
    ratio, chi2, p = caliper_test(z_full)
    z_disc, p_disc = density_discontinuity_test(z_full)

    # Histogram trimmed to [0, 6] for display only
    z_hist = z_full[z_full <= 6]
    ax.hist(z_hist, bins=np.arange(0, 6.1, 0.2), density=True,
            color='steelblue', edgecolor='white', linewidth=0.5, alpha=0.8,
            label=f'Observed (N={len(z_full):,}; histogram trimmed to |z|≤6)')

    # Expected chi2(1) density as reference
    x = np.linspace(0, 6, 300)
    chi2_density = stats.chi2.pdf(x**2, df=1) * 2 * x
    ax.plot(x, chi2_density, 'r--', linewidth=1.5, alpha=0.7, label='χ²(1) reference')

    # Threshold lines
    ax.axvline(1.645, color='orange', linestyle=':', linewidth=1.5, label='p=0.10 (z=1.645)')
    ax.axvline(1.960, color='red',    linestyle='--', linewidth=2,   label='p=0.05 (z=1.96)')
    ax.axvline(2.576, color='darkred',linestyle=':',  linewidth=1.5, label='p=0.01 (z=2.576)')

    # Shade the caliper window
    ax.axvspan(1.645, 1.96, alpha=0.1, color='orange', label='Caliper window (below)')
    ax.axvspan(1.96,  2.275, alpha=0.1, color='red',   label='Caliper window (above)')

    textstr = (f'Caliper ratio: {ratio:.2f}\n'
               f'χ²={chi2:.2f}, p={p:.3f}\n'
               f'Disc. p={p_disc:.3f}' if ratio else '')
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('|z| statistic', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(0, 6)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  Saved → output/figures/{filename}")
    return ratio, chi2, p, z_disc, p_disc

def main():
    path = Path('data/final/tstats_all.parquet')
    if not path.exists():
        print("ERROR: data/final/tstats_all.parquet not found. Run script 13 first.")
        return

    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} test statistics from {df['paper_title'].nunique()} papers")

    # Clean: use z_abs column, filter obviously wrong values
    df['z'] = pd.to_numeric(df['z_abs'], errors='coerce').abs()
    df = df[df['z'].notna() & (df['z'] >= 0) & (df['z'] <= 50)]
    print(f"After cleaning: {len(df):,} statistics")

    # ── Full sample ──────────────────────────────────────────────────────────
    print("\n=== Full Sample ===")
    r, c2, p, zd, pd_ = plot_zdist(
        df['z'], 'Distribution of |z| statistics — JF/RFS/JFE (2000–2024)',
        'fig1_zdist_full.pdf'
    )
    print(f"  N={len(df):,} | Caliper ratio={r:.2f} | χ²={c2:.2f} p={p:.3f} | Disc z={zd:.2f} p={pd_:.3f}")

    # ── By direction ─────────────────────────────────────────────────────────
    for direction, label, fname in [
        (1,  'Positive-finding papers',  'fig2a_zdist_positive.pdf'),
        (-1, 'Negative-finding papers',  'fig2b_zdist_negative.pdf'),
        (0,  'Null-finding papers',      'fig2c_zdist_null.pdf'),
    ]:
        sub = df[df['direction'] == direction]
        if len(sub) < 20:
            print(f"  Skipping {label}: only {len(sub)} stats")
            continue
        print(f"\n=== {label} (N={len(sub):,}) ===")
        r, c2, p, zd, pd_ = plot_zdist(
            sub['z'], f'{label} — |z| distribution', fname
        )
        print(f"  Caliper ratio={r:.2f} | χ²={c2:.2f} p={p:.3f} | Disc z={zd:.2f} p={pd_:.3f}")

    # ── By journal ────────────────────────────────────────────────────────────
    for j in ['JF', 'RFS', 'JFE']:
        sub = df[df['journal'] == j]
        print(f"\n=== {j} (N={len(sub):,}) ===")
        if len(sub) < 30:
            continue
        r, c2, p, zd, pd_ = plot_zdist(
            sub['z'], f'{j} — |z| distribution', f'fig3_{j}_zdist.pdf'
        )
        print(f"  Caliper ratio={r:.2f} | χ²={c2:.2f} p={p:.3f} | Disc z={zd:.2f} p={pd_:.3f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    results = []
    subsets = [
        ('Full sample', df),
        ('JF', df[df['journal']=='JF']),
        ('RFS', df[df['journal']=='RFS']),
        ('JFE', df[df['journal']=='JFE']),
        ('Positive-finding', df[df['direction']==1]),
        ('Negative-finding', df[df['direction']==-1]),
        ('Null-finding', df[df['direction']==0]),
    ]
    for label, sub in subsets:
        if len(sub) < 10:
            continue
        z = sub['z'].dropna()
        r, c2, p, zd, pd_ = caliper_test(z)[0], caliper_test(z)[1], caliper_test(z)[2], \
                             density_discontinuity_test(z)[0], density_discontinuity_test(z)[1]
        pct_sig = (z >= 1.96).mean() * 100
        results.append({
            'Subset': label, 'N': len(z),
            'Pct significant (|z|≥1.96)': round(pct_sig, 1),
            'Caliper ratio': round(r, 2) if r else None,
            'Caliper χ²': round(c2, 2) if c2 else None,
            'Caliper p': round(p, 3) if p else None,
            'Disc z-stat': round(zd, 2) if zd else None,
            'Disc p': round(pd_, 3) if pd_ else None,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('output/tables/pvalue_analysis.csv', index=False)
    print("\n" + results_df.to_string(index=False))
    print("\nSaved → output/tables/pvalue_analysis.csv")

if __name__ == '__main__':
    main()
