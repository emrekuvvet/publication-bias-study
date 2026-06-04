"""
Script 26: Extraction differential quality analysis.

Referee point 3 asks whether differential extraction quality across the two
corpora (typeset publisher PDFs vs. NBER draft PDFs) could itself generate
the published-vs-NBER caliper gap.

This script:
1. Compares coverage rates (papers with ≥1 extracted stat) for published vs NBER
2. Tests whether zero-stat (dropped) published papers differ by journal/vintage
3. Reports internal consistency (|coef/se| vs extracted z_abs) separately per corpus
4. Tests whether the caliper result holds when restricted to high-coverage papers

Outputs:
    output/tables/extraction_differential.csv
    output/tables/extraction_coverage_by_journal.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency, mannwhitneyu

# ── Load ───────────────────────────────────────────────────────────────────
pub_raw  = pd.read_parquet('data/final/tstats_all.parquet')
nber_raw = pd.read_parquet('data/final/nber_tstats_all.parquet')
analysis = pd.read_parquet('data/final/analysis_dataset.parquet')
zero     = pd.read_csv('output/tables/extraction_zero_stat_papers.csv')

# ── 1. Coverage by journal ─────────────────────────────────────────────────
pub_ana = analysis[analysis['source'] != 'nber'].copy()
journal_coverage = []
for jnl in ['jf', 'rfs', 'jfe']:
    total = (pub_ana['source'] == jnl).sum()
    with_stats = pub_raw[pub_raw['paper_title'].isin(
        pub_ana[pub_ana['source'] == jnl]['title'])]['paper_title'].nunique()
    dropped = total - with_stats
    journal_coverage.append({
        'journal': jnl.upper(),
        'total_papers': total,
        'papers_with_stats': with_stats,
        'dropped': dropped,
        'coverage_pct': round(100 * with_stats / total, 1)
    })
jnl_df = pd.DataFrame(journal_coverage)
print('=== Coverage by Journal ===')
print(jnl_df.to_string(index=False))
jnl_df.to_csv('output/tables/extraction_coverage_by_journal.csv', index=False)

# Chi-square: are dropped papers distributed uniformly across journals?
obs = jnl_df['dropped'].values
exp = np.full(len(obs), obs.mean())
chi2_dropped = ((obs - exp)**2 / exp).sum()
from scipy.stats import chi2 as chi2_dist
p_uniform = 1 - chi2_dist.cdf(chi2_dropped, df=len(obs)-1)
print(f'\nChi-square for uniform distribution of dropped papers across journals: '
      f'χ²={chi2_dropped:.2f}, p={p_uniform:.3f}')

# ── 2. Vintage of dropped vs. kept published papers ────────────────────────
pub_ana_yr = pub_ana[['title','year']].copy()
pub_ana_yr['year_num'] = pd.to_numeric(pub_ana_yr['year'], errors='coerce')
covered_titles = pub_raw['paper_title'].unique()
pub_ana_yr['has_stats'] = pub_ana_yr['title'].isin(covered_titles)

yr_kept   = pub_ana_yr[pub_ana_yr['has_stats']]['year_num'].dropna()
yr_dropped= pub_ana_yr[~pub_ana_yr['has_stats']]['year_num'].dropna()

stat_mw, p_mw = mannwhitneyu(yr_kept, yr_dropped, alternative='two-sided')
print(f'\nVintage comparison (kept vs dropped published papers):')
print(f'  Kept   — median year: {yr_kept.median():.0f}  N={len(yr_kept)}')
print(f'  Dropped — median year: {yr_dropped.median():.0f}  N={len(yr_dropped)}')
print(f'  Mann-Whitney U={stat_mw:.0f}, p={p_mw:.3f}')

# ── 3. Internal consistency per corpus ────────────────────────────────────
def internal_consistency(df, label):
    coef = pd.to_numeric(df['coef'], errors='coerce')
    se   = pd.to_numeric(df['se'],   errors='coerce')
    z_ex = pd.to_numeric(df['z_abs'],errors='coerce')
    has_both = coef.notna() & se.notna() & (se > 0) & z_ex.notna() & (z_ex > 0)
    computed = (coef / se).abs()
    diff = (computed - z_ex).abs()
    n_consistent = (diff[has_both] < 0.1).sum()
    n_has_both   = has_both.sum()
    n_z_only     = z_ex.notna().sum() - n_has_both
    return {
        'corpus': label,
        'total_rows': len(df),
        'rows_with_coef_and_se': int(n_has_both),
        'internally_consistent': int(n_consistent),
        'consistency_pct': round(100 * n_consistent / n_has_both, 1) if n_has_both > 0 else np.nan,
        'z_stat_only_rows': int(n_z_only),
        'z_stat_only_pct': round(100 * n_z_only / z_ex.notna().sum(), 1) if z_ex.notna().sum() > 0 else np.nan,
    }

cons_pub  = internal_consistency(pub_raw,  'Published')
cons_nber = internal_consistency(nber_raw, 'NBER')
cons_df = pd.DataFrame([cons_pub, cons_nber])
print('\n=== Internal Consistency ===')
print(cons_df.to_string(index=False))

# ── 4. Caliper restricted to high-coverage papers (≥20 stats) ─────────────
# If differential coverage is not generating the result, restricting to
# papers with many extracted stats should leave the caliper ratio qualitatively similar
pub_counts  = pub_raw.groupby('paper_title').size().reset_index(name='n')
nber_counts = nber_raw.groupby('paper_title').size().reset_index(name='n')

pub_high  = pub_raw[pub_raw['paper_title'].isin(
    pub_counts[pub_counts['n'] >= 20]['paper_title'])]
nber_high = nber_raw[nber_raw['paper_title'].isin(
    nber_counts[nber_counts['n'] >= 20]['paper_title'])]

def simple_caliper(z_series, lo=1.645, mid=1.96, hi=2.275):
    """Returns caliper_ratio = below/above (Brodeur convention)."""
    z = pd.to_numeric(z_series, errors='coerce')
    w = z[(z > lo) & (z < hi)]
    n_below = (w < mid).sum(); n_above = (w >= mid).sum()
    if n_above == 0: return np.nan, int(n_below), int(n_above)
    ratio = n_below / n_above
    return round(ratio, 3), int(n_below), int(n_above)

for label, df_h, df_all in [('Published', pub_high, pub_raw),
                              ('NBER',      nber_high, nber_raw)]:
    r_all, na_all, nb_all = simple_caliper(df_all['z_abs'])
    r_hi,  na_hi,  nb_hi  = simple_caliper(df_h['z_abs'])
    print(f'\n{label} caliper — All papers: ratio={r_all} ({na_all}↔{nb_all}), '
          f'High-coverage (≥20 stats): ratio={r_hi} ({na_hi}↔{nb_hi})')

# ── Summary table ──────────────────────────────────────────────────────────
summary = pd.DataFrame({
    'Metric': [
        'Total papers in scope',
        'Papers with ≥1 extracted stat',
        'Coverage rate',
        'Rows with coef+SE (internal check)',
        'Internally consistent (|coef/SE|≈z_abs)',
        'z-stat-only rows (no coef+SE)',
    ],
    'Published': [
        len(pub_ana),
        len(pub_raw['paper_title'].unique()),
        f"{100*len(pub_raw['paper_title'].unique())/len(pub_ana):.1f}%",
        cons_pub['rows_with_coef_and_se'],
        f"{cons_pub['consistency_pct']}%",
        f"{cons_pub['z_stat_only_pct']}% of extracted stats",
    ],
    'NBER': [
        len(analysis[analysis['source'] == 'nber']),
        len(nber_raw['paper_title'].unique()),
        f"{100*len(nber_raw['paper_title'].unique())/len(analysis[analysis['source']=='nber']):.1f}%",
        cons_nber['rows_with_coef_and_se'],
        f"{cons_nber['consistency_pct']}%",
        f"{cons_nber['z_stat_only_pct']}% of extracted stats",
    ],
})
summary.to_csv('output/tables/extraction_differential.csv', index=False)
print('\n=== Extraction Quality Summary ===')
print(summary.to_string(index=False))
print('\nDone.')
