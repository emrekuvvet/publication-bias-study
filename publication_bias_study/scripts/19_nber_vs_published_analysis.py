"""
Script 19: Compare p-value distributions between NBER working papers and published papers.

Key design choices
------------------
* Naive caliper uses chi-square on raw below/above counts (Brodeur 2016).
* Clustered caliper resamples papers (not statistics) to account for within-paper
  correlation.  Three clustered p-values are reported:
    - p_boot: cluster bootstrap, 10,000 draws with replacement at the paper level
    - p_perm: permutation test at the paper level (10,000 label shuffles)
    - p_ttest: one-sample t-test on per-paper caliper ratios vs. H0: ratio = 1
* Caliper window robustness: three window widths are tested (T1-D).
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────────
pub_raw  = pd.read_parquet('data/final/tstats_all.parquet')
nber_raw = pd.read_parquet('data/final/nber_tstats_all.parquet')
analysis = pd.read_parquet('data/final/analysis_dataset.parquet')

# Add paper_id to published stats via join on paper_title
ana_pub = (analysis[analysis['source'] != 'nber']
           [['paper_id', 'title']]
           .rename(columns={'title': 'paper_title'}))
pub = pub_raw.merge(ana_pub, on='paper_title', how='left')
assert pub['paper_id'].notna().all(), "Some pub rows failed to join with analysis"

# NBER: use wp_number as paper_id
nber = nber_raw.copy()
nber['paper_id'] = nber['wp_number']

print(f'Published: {len(pub):,} z-stats from {pub["paper_id"].nunique()} papers')
print(f'NBER:      {len(nber):,} z-stats from {nber["paper_id"].nunique()} papers')


# ── Caliper functions ──────────────────────────────────────────────────────

def caliper_naive(z, lo=1.645, mid=1.96, hi=2.275):
    """Brodeur-style chi-square caliper (treats stats as independent)."""
    z = pd.Series(z, dtype=float).dropna()
    z = z[(z >= 0) & (z <= 50)]
    below = int(((z >= lo) & (z < mid)).sum())
    above = int(((z >= mid) & (z < hi)).sum())
    ratio = below / above if above > 0 else np.nan
    chi2, p = stats.chisquare([below, above]) if above > 0 else (np.nan, np.nan)
    n_sig = int((z >= 1.96).sum())
    pct_sig = round(100 * n_sig / len(z), 1) if len(z) > 0 else np.nan
    return {
        'N': len(z), 'pct_sig': pct_sig,
        'below': below, 'above': above,
        'caliper_ratio': round(ratio, 3) if not np.isnan(ratio) else np.nan,
        'caliper_chi2':  round(chi2, 2)  if not np.isnan(chi2)  else np.nan,
        'caliper_p':     round(p, 3)     if not np.isnan(p)     else np.nan,
    }


def caliper_clustered(z_series, paper_ids, lo=1.645, mid=1.96, hi=2.275,
                      n_iter=10_000, seed=42):
    """
    Caliper test with paper-level clustering.

    Returns the naive caliper result plus three clustered p-values:
      p_boot  — cluster bootstrap (resample papers with replacement)
      p_perm  — permutation under H0: expected below == above (shuffle within papers)
      p_ttest — one-sample t-test on per-paper ratios vs. H0=1
    """
    z   = pd.Series(z_series, dtype=float).reset_index(drop=True)
    pid = pd.Series(paper_ids).reset_index(drop=True)

    mask = z.notna() & (z >= 0) & (z <= 50)
    z, pid = z[mask].reset_index(drop=True), pid[mask].reset_index(drop=True)
    if len(z) == 0:
        return {}

    below_mask = ((z >= lo) & (z < mid)).astype(int)
    above_mask = ((z >= mid) & (z < hi)).astype(int)

    n_below = int(below_mask.sum())
    n_above = int(above_mask.sum())

    if n_above == 0:
        return caliper_naive(z, lo, mid, hi)

    ratio_obs = n_below / n_above
    chi2_obs, p_naive = stats.chisquare([n_below, n_above])

    # Per-paper aggregation
    tmp = pd.DataFrame({'pid': pid, 'b': below_mask, 'a': above_mask})
    by_paper = (tmp.groupby('pid')
                   .agg(nb=('b', 'sum'), na=('a', 'sum'))
                   .reset_index())
    by_paper = by_paper[(by_paper['nb'] + by_paper['na']) > 0]
    nb = by_paper['nb'].values.astype(float)
    na = by_paper['na'].values.astype(float)
    n_papers = len(by_paper)

    rng = np.random.default_rng(seed)

    # --- Cluster bootstrap ---
    boot = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n_papers, size=n_papers)
        b, a = nb[idx].sum(), na[idx].sum()
        boot[i] = b / a if a > 0 else np.nan
    boot = boot[~np.isnan(boot)]
    # Two-sided: centre boot distribution under H0 (ratio=1)
    centred = boot - np.mean(boot) + 1.0
    p_boot = float((centred >= ratio_obs).mean())
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    # --- Permutation test (permute in-window label: below <-> above) ---
    # Under H0 each in-window stat is equally likely to fall below or above
    in_window = below_mask + above_mask       # 0 or 1 per stat
    perm_ratios = np.empty(n_iter)
    for i in range(n_iter):
        # Independently shuffle each stat's below/above assignment within window
        perm_below = rng.binomial(in_window.values.astype(int), 0.5)
        perm_above = in_window.values.astype(int) - perm_below
        b, a = perm_below.sum(), perm_above.sum()
        perm_ratios[i] = b / a if a > 0 else np.nan
    perm_ratios = perm_ratios[~np.isnan(perm_ratios)]
    p_perm = float((perm_ratios >= ratio_obs).mean())

    # --- Paper-level t-test ---
    per_ratio = np.where(na > 0, nb / na, np.nan)
    valid_ratios = per_ratio[~np.isnan(per_ratio)]
    if len(valid_ratios) >= 3:
        _, p_ttest = stats.ttest_1samp(valid_ratios, popmean=1.0)
    else:
        p_ttest = np.nan

    n_sig  = int((z >= 1.96).sum())
    pct_sig = round(100 * n_sig / len(z), 1)

    return {
        'N':              len(z),
        'N_papers':       n_papers,
        'pct_sig':        pct_sig,
        'below':          n_below,
        'above':          n_above,
        'caliper_ratio':  round(ratio_obs, 3),
        'caliper_chi2':   round(chi2_obs, 2),
        'caliper_p':      round(p_naive, 3),
        'p_boot':         round(p_boot, 3),
        'p_perm':         round(p_perm, 3),
        'p_ttest':        round(p_ttest, 3) if not np.isnan(p_ttest) else np.nan,
        'boot_ci_lo':     round(ci_lo, 3),
        'boot_ci_hi':     round(ci_hi, 3),
    }


# ── Cross-group permutation (paper-level label shuffle) ───────────────────

def cross_group_perm(pub_df, nber_df, lo=1.645, mid=1.96, hi=2.275,
                     direction=None, n_iter=10_000, seed=99):
    """
    Permutation test for H0: no difference in caliper ratio between published
    and NBER papers.

    Shuffles the pub/NBER label across all papers; computes the caliper ratio
    for the permuted 'published' group each time; one-sided p-value =
    P(perm_ratio_pub >= observed_ratio_pub).
    """
    if direction is not None:
        pub_df  = pub_df[pub_df['direction']  == direction]
        nber_df = nber_df[nber_df['direction'] == direction]

    def paper_caliper(df):
        z   = pd.Series(df['z_abs'].values, dtype=float)
        pid = df['paper_id'].values
        z = z.fillna(-1)
        mask = (z >= 0) & (z <= 50)
        z, pid = z[mask], pid[mask]
        below = ((z >= lo) & (z < mid)).astype(int)
        above = ((z >= mid) & (z < hi)).astype(int)
        tmp = pd.DataFrame({'pid': pid, 'b': below, 'a': above})
        by_paper = tmp.groupby('pid').agg(nb=('b','sum'), na=('a','sum')).reset_index()
        return by_paper[['pid','nb','na']].values

    pub_arr  = paper_caliper(pub_df)   # shape (n_pub, 3): pid, nb, na
    nber_arr = paper_caliper(nber_df)  # shape (n_nber, 3)

    if len(pub_arr) == 0 or len(nber_arr) == 0:
        return np.nan

    n_pub  = len(pub_arr)
    n_nber = len(nber_arr)
    all_arr = np.vstack([pub_arr, nber_arr])   # pool all papers
    nb_all  = all_arr[:, 1].astype(float)
    na_all  = all_arr[:, 2].astype(float)
    n_total = n_pub + n_nber

    # Observed ratio for published group
    obs_pub_b = nb_all[:n_pub].sum()
    obs_pub_a = na_all[:n_pub].sum()
    if obs_pub_a == 0:
        return np.nan
    ratio_obs = obs_pub_b / obs_pub_a

    rng = np.random.default_rng(seed)
    perm_ratios = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.permutation(n_total)
        perm_pub = idx[:n_pub]
        b = nb_all[perm_pub].sum()
        a = na_all[perm_pub].sum()
        perm_ratios[i] = b / a if a > 0 else np.nan

    valid = perm_ratios[~np.isnan(perm_ratios)]
    p_cross = float((valid >= ratio_obs).mean())
    return round(p_cross, 3)


# ── Main caliper table ──────────────────────────────────────────────────────
rows = []
for label, df_, sample in [('Published', pub, 'published'), ('NBER', nber, 'nber')]:
    # Full sample
    r = caliper_clustered(df_['z_abs'], df_['paper_id'])
    r.update({'sample': sample, 'subset': 'Full sample'})
    rows.append(r)

    # By direction
    for dir_label, dir_val in [('Positive', 1), ('Negative', -1), ('Null', 0)]:
        sub = df_[df_['direction'] == dir_val]
        r = caliper_clustered(sub['z_abs'], sub['paper_id'])
        r.update({'sample': sample, 'subset': f'{dir_label}-finding'})
        rows.append(r)

result = pd.DataFrame(rows)

# Add cross-group permutation p-values for published rows
print('\nRunning cross-group permutation tests...')
cross_ps = {}
for dir_label, dir_val in [(None, None), ('Positive', 1), ('Negative', -1), ('Null', 0)]:
    subset_key = 'Full sample' if dir_val is None else f'{dir_label}-finding'
    p = cross_group_perm(pub, nber, direction=dir_val)
    cross_ps[subset_key] = p
    print(f'  {subset_key}: p_cross={p}')

result['p_cross'] = result.apply(
    lambda row: cross_ps.get(row['subset']) if row['sample'] == 'published' else np.nan,
    axis=1
)

col_order = ['sample', 'subset', 'N', 'N_papers', 'pct_sig',
             'below', 'above', 'caliper_ratio',
             'caliper_chi2', 'caliper_p',
             'p_boot', 'p_perm', 'p_cross', 'p_ttest',
             'boot_ci_lo', 'boot_ci_hi']
result = result[[c for c in col_order if c in result.columns]]
result.to_csv('output/tables/nber_vs_published_caliper.csv', index=False)
print('\n=== NBER vs Published Caliper Comparison (with paper-level clustering) ===')
print(result.to_string(index=False))


# ── Caliper window robustness (T1-D) ───────────────────────────────────────
WINDOWS = [
    ('Standard (1.645–2.275)', 1.645, 1.96, 2.275),
    ('Narrow   (1.70–2.22)',   1.70,  1.96, 2.22),
    ('Wide     (1.50–2.42)',   1.50,  1.96, 2.42),
    ('Brodeur (1.66–2.26)',    1.66,  1.96, 2.26),
]

window_rows = []
for win_label, lo, mid, hi in WINDOWS:
    for label, df_, sample in [('Published', pub, 'published'), ('NBER', nber, 'nber')]:
        r = caliper_clustered(df_['z_abs'], df_['paper_id'], lo=lo, mid=mid, hi=hi)
        r.update({'window': win_label, 'sample': sample})
        window_rows.append(r)

window_result = pd.DataFrame(window_rows)
wcols = ['window', 'sample', 'N', 'below', 'above',
         'caliper_ratio', 'caliper_p', 'p_boot', 'p_perm']
window_result[[c for c in wcols if c in window_result.columns]].to_csv(
    'output/tables/caliper_window_robustness.csv', index=False)
print('\n=== Caliper Window Robustness ===')
print(window_result[[c for c in wcols if c in window_result.columns]].to_string(index=False))


# ── Figures ─────────────────────────────────────────────────────────────────
out_dir = Path('output/figures')
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False
})


def plot_zdist(z_pub_full, z_nber_full, title, fname, bins=None,
               pid_pub=None, pid_nber=None):
    """Plot z-distribution side-by-side.

    Histogram is trimmed to [0,6] for display; caliper statistics are
    computed on the full [0,50] cleaned sample and labelled with the
    full N so figure and table annotations are consistent.
    """
    if bins is None:
        bins = np.arange(0, 6.05, 0.1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    for ax, z_full, pid, label, color in zip(
        axes,
        [z_pub_full, z_nber_full],
        [pid_pub, pid_nber],
        ['Published papers', 'NBER working papers'],
        ['#2166ac', '#d6604d']
    ):
        # Caliper on full cleaned sample
        if pid is not None:
            r = caliper_clustered(z_full, pid)
        else:
            r = caliper_naive(z_full)

        # Histogram trimmed to [0,6]
        z_trim = pd.Series(z_full, dtype=float)
        z_trim = z_trim[z_trim.notna() & (z_trim >= 0) & (z_trim <= 50)]
        z_hist = z_trim[z_trim <= 6]
        counts, edges = np.histogram(z_hist, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        widths  = edges[1:] - edges[:-1]

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

        # Show full N in title; note histogram is capped at 6
        p_show = r.get('p_boot', r.get('caliper_p'))
        p_label = 'p₀ₒₒₜ' if 'p_boot' in r else 'p'
        ax.set_title(
            f'{label}\n'
            f'N={r["N"]:,} stats, {r.get("N_papers","?")} papers\n'
            f'Caliper ratio = {r["caliper_ratio"]:.2f}  '
            f'({p_label} = {p_show:.3f}, clustered)',
            fontsize=12
        )
        ax.set_xlabel('|z|-statistic', fontsize=13)
        ax.set_ylabel('Count', fontsize=13)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlim(0, 6)

    fig.suptitle(title + '\n(histogram trimmed to |z| ≤ 6; caliper uses full sample)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=250, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


# Full sample
z_pub_all  = pub['z_abs']
z_nber_all = nber['z_abs']

plot_zdist(z_pub_all, z_nber_all,
           'Z-statistic distribution: Published vs. NBER working papers',
           'fig_nber_vs_pub_full.pdf',
           pid_pub=pub['paper_id'], pid_nber=nber['paper_id'])

# Negative-finding
pub_neg  = pub[pub['direction'] == -1]
nber_neg = nber[nber['direction'] == -1]
plot_zdist(pub_neg['z_abs'], nber_neg['z_abs'],
           'Negative-finding papers: Published vs. NBER working papers',
           'fig_nber_vs_pub_negative.pdf',
           pid_pub=pub_neg['paper_id'], pid_nber=nber_neg['paper_id'])

# Positive-finding
pub_pos  = pub[pub['direction'] == 1]
nber_pos = nber[nber['direction'] == 1]
plot_zdist(pub_pos['z_abs'], nber_pos['z_abs'],
           'Positive-finding papers: Published vs. NBER working papers',
           'fig_nber_vs_pub_positive.pdf',
           pid_pub=pub_pos['paper_id'], pid_nber=nber_pos['paper_id'])

print('\nDone. Tables and figures saved.')
