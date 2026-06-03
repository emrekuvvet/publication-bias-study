"""
Script 19: Compare p-value distributions between NBER working papers and published papers.
Produces caliper table and z-distribution figures for NBER vs. published comparison.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────────
pub  = pd.read_parquet('data/final/tstats_all.parquet')
nber = pd.read_parquet('data/final/nber_tstats_all.parquet')

def clean(df, z_col='z_abs'):
    s = df[z_col].dropna()
    return s[(s >= 0) & (s <= 50)]

pub_z  = clean(pub)
nber_z = clean(nber)

print(f'Published: {len(pub_z):,} z-stats from {pub["paper_title"].nunique()} papers')
print(f'NBER:      {len(nber_z):,} z-stats from {nber["wp_number"].nunique()} papers')

# ── Caliper test ────────────────────────────────────────────────────────────
def caliper(z, lo=1.645, mid=1.96, hi=2.275):
    z = pd.Series(z).dropna()
    z = z[(z >= 0) & (z <= 50)]
    below = int(((z >= lo) & (z < mid)).sum())
    above = int(((z >= mid) & (z < hi)).sum())
    ratio = round(below / above, 3) if above > 0 else np.nan
    chi2, p = stats.chisquare([below, above])
    n_sig = int((z >= 1.96).sum())
    pct_sig = round(100 * n_sig / len(z), 1) if len(z) > 0 else np.nan
    return {
        'N': len(z), 'pct_sig': pct_sig,
        'below': below, 'above': above,
        'caliper_ratio': ratio,
        'caliper_chi2': round(chi2, 2),
        'caliper_p': round(p, 3)
    }

rows = []

# Full sample
for label, z, sample in [('Published', pub_z, 'published'), ('NBER', nber_z, 'nber')]:
    r = caliper(z)
    r.update({'sample': sample, 'subset': 'Full sample'})
    rows.append(r)

# By direction
for dir_label, dir_val in [('Positive', 1), ('Negative', -1), ('Null', 0)]:
    for label, df_, z_col, sample in [
        ('Published', pub, 'z_abs', 'published'),
        ('NBER',      nber, 'z_abs', 'nber')
    ]:
        sub = df_[df_['direction'] == dir_val][z_col]
        r = caliper(sub)
        r.update({'sample': sample, 'subset': f'{dir_label}-finding'})
        rows.append(r)

result = pd.DataFrame(rows)[['sample','subset','N','pct_sig','below','above',
                               'caliper_ratio','caliper_chi2','caliper_p']]
result.to_csv('output/tables/nber_vs_published_caliper.csv', index=False)
print('\n=== NBER vs Published Caliper Comparison ===')
print(result.to_string(index=False))

# ── Figures ─────────────────────────────────────────────────────────────────
out_dir = Path('output/figures')
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False
})

def plot_zdist(z_pub, z_nber, title, fname, bins=None):
    if bins is None:
        bins = np.arange(0, 6.05, 0.1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, z, label, color in zip(
        axes,
        [z_pub, z_nber],
        ['Published papers', 'NBER working papers'],
        ['#2166ac', '#d6604d']
    ):
        z_trim = z[(z >= 0) & (z <= 6)]
        counts, edges = np.histogram(z_trim, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        widths  = edges[1:] - edges[:-1]

        # Color bars: grey < 1.645, blue in caliper, red ≥ 1.96
        bar_colors = []
        for c in centers:
            if c < 1.645:
                bar_colors.append('#bbbbbb')
            elif c < 1.96:
                bar_colors.append('#fdbf6f')
            elif c < 2.275:
                bar_colors.append('#ff7f00')
            else:
                bar_colors.append('#bbbbbb')

        ax.bar(centers, counts, width=widths * 0.9, color=bar_colors, alpha=0.85)
        ax.axvline(1.645, color='#888888', lw=1, ls='--', alpha=0.7)
        ax.axvline(1.960, color='black',   lw=1.5, ls='-')
        ax.axvline(2.275, color='#888888', lw=1, ls='--', alpha=0.7)

        r = caliper(z)
        ax.set_title(f'{label}\nCaliper ratio = {r["caliper_ratio"]:.2f}  (p = {r["caliper_p"]:.3f})',
                     fontsize=11)
        ax.set_xlabel('|z|-statistic', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_xlim(0, 6)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


# Fig A: full sample side-by-side
plot_zdist(pub_z, nber_z,
           'Z-statistic distribution: Published vs. NBER working papers',
           'fig_nber_vs_pub_full.pdf')

# Fig B: negative-finding papers
pub_neg  = clean(pub[pub['direction'] == -1])
nber_neg = clean(nber[nber['direction'] == -1])
plot_zdist(pub_neg, nber_neg,
           'Negative-finding papers: Published vs. NBER working papers',
           'fig_nber_vs_pub_negative.pdf')

# Fig C: positive-finding papers
pub_pos  = clean(pub[pub['direction'] == 1])
nber_pos = clean(nber[nber['direction'] == 1])
plot_zdist(pub_pos, nber_pos,
           'Positive-finding papers: Published vs. NBER working papers',
           'fig_nber_vs_pub_positive.pdf')

print('\nDone. Tables and figures saved.')
