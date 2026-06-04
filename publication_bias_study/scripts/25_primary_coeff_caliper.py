"""
Script 25: Caliper test restricted to non-control (treatment/main-result) coefficients.

Referee point 4 asks to justify pooling all table statistics and to show robustness
to restricting to main-result coefficients.

Approach: exclude rows whose variable name matches a broad list of common
control-variable patterns (size, leverage, age, GDP, ROA, ROE, volatility,
year FE, intercept, etc.). The remaining 'non-control' rows predominantly
represent treatment indicators and main-result coefficients.

Outputs:
    output/tables/caliper_noncontrol.csv             — non-control subset
    output/tables/caliper_inclusion_sensitivity.csv  — all vs non-control side-by-side
"""

import re
import pandas as pd
import numpy as np
from scipy import stats


# ── Caliper functions (verbatim from script 19) ────────────────────────────

def caliper_naive(z, lo=1.645, mid=1.96, hi=2.275):
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

    tmp = pd.DataFrame({'pid': pid, 'b': below_mask, 'a': above_mask})
    by_paper = (tmp.groupby('pid').agg(nb=('b', 'sum'), na=('a', 'sum')).reset_index())
    by_paper = by_paper[(by_paper['nb'] + by_paper['na']) > 0]
    nb = by_paper['nb'].values.astype(float)
    na = by_paper['na'].values.astype(float)
    n_papers = len(by_paper)
    rng = np.random.default_rng(seed)

    # Cluster bootstrap
    boot = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n_papers, size=n_papers)
        b, a = nb[idx].sum(), na[idx].sum()
        boot[i] = b / a if a > 0 else np.nan
    boot = boot[~np.isnan(boot)]
    centred = boot - np.mean(boot) + 1.0
    p_boot = float((centred >= ratio_obs).mean())
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    # Within-paper permutation
    in_window = below_mask + above_mask
    perm_ratios = np.empty(n_iter)
    for i in range(n_iter):
        perm_below = rng.binomial(in_window.values.astype(int), 0.5)
        perm_above = in_window.values.astype(int) - perm_below
        b, a = perm_below.sum(), perm_above.sum()
        perm_ratios[i] = b / a if a > 0 else np.nan
    perm_ratios = perm_ratios[~np.isnan(perm_ratios)]
    p_perm = float((perm_ratios >= ratio_obs).mean())

    # Paper-level t-test
    per_ratio = np.where(na > 0, nb / na, np.nan)
    valid_ratios = per_ratio[~np.isnan(per_ratio)]
    p_ttest = stats.ttest_1samp(valid_ratios, popmean=1.0)[1] if len(valid_ratios) >= 3 else np.nan

    n_sig = int((z >= 1.96).sum())
    pct_sig = round(100 * n_sig / len(z), 1)

    return {
        'N': len(z), 'N_papers': n_papers, 'pct_sig': pct_sig,
        'below': n_below, 'above': n_above,
        'caliper_ratio': round(ratio_obs, 3),
        'caliper_chi2':  round(chi2_obs, 2),
        'caliper_p':     round(p_naive, 3),
        'p_boot':        round(p_boot, 3),
        'p_perm':        round(p_perm, 3),
        'p_ttest':       round(p_ttest, 3) if not np.isnan(p_ttest) else np.nan,
        'boot_ci_lo':    round(ci_lo, 3),
        'boot_ci_hi':    round(ci_hi, 3),
    }


# ── Load data ──────────────────────────────────────────────────────────────
pub_raw  = pd.read_parquet('data/final/tstats_all.parquet')
nber_raw = pd.read_parquet('data/final/nber_tstats_all.parquet')
analysis = pd.read_parquet('data/final/analysis_dataset.parquet')

ana_pub = (analysis[analysis['source'] != 'nber'][['paper_id', 'title']]
           .rename(columns={'title': 'paper_title'}))
pub = pub_raw.merge(ana_pub, on='paper_title', how='left')
pub['paper_id'] = pub['paper_id'].fillna(pub['paper_title'])
nber = nber_raw.copy()
nber['paper_id'] = nber['wp_number']

# ── Control-variable exclusion regex ──────────────────────────────────────
CONTROL_RE = re.compile(
    r'(^log\s*[\(\[]?\s*(asset|sales|size|market|revenue|total\s+asset|equity|debt|mv|bv|ta)\b'
    r'|(?<!\w)leverage(?!\s*[\*×]?\s*(treat|post|reform))'
    r'|(?<!\w)(long.term\s+)?debt[/ ]?(to|ratio|level)(?!\w)'
    r'|(?<!\w)size(?!\s*[\*×]?\s*(treat|post|reform|policy))(?!\w)'
    r'|(?<!\w)(firm\s+)?age(?!\s*[\*×]?\s*(treat|post|reform|policy))(?!\w)'
    r'|(?<!\w)roa(?!\w)|(?<!\w)roe(?!\w)|(?<!\w)ebitda(?!\w)'
    r'|(?<!\w)profitab[a-z]*(?!\w)'
    r'|(?<!\w)book.?to.?market(?!\w)|(?<!\w)btm(?!\w)'
    r'|(?<!\w)tobin[\'s ]+q(?!\w)|(?<!\w)q\s+ratio(?!\w)'
    r'|(?<!\w)volatilit[a-z]*(?!\w)|(?<!\w)return\s+volat[a-z]*(?!\w)'
    r'|(?<!\w)momentum(?!\w)|(?<!\w)market\s+beta(?!\w)'
    r'|(?<!\w)cash\s+flow(?!\w)'
    r'|(?<!\w)(log\s+)?market\s+(cap|value)(?!\w)|(?<!\w)mcap(?!\w)'
    r'|(?<!\w)stock\s+return(?!\w)'
    r'|\byear\s*\d{2,4}\b|\byear\s+fe\b'
    r'|(?<!\w)intercept(?!\w)|(?<!\w)_cons(?!\w)|(?<!\w)constant(?!\w)'
    r'|fixed\s+effect[s]?|firm\s+(fe|fixed)|country\s+(fe|fixed)'
    r'|industry\s+(fe|fixed)|time\s+(fe|fixed)'
    r'|\bnumber\s+of\s+(analyst|employ|segment|observation)'
    r'|(?<!\w)(log\s+)?gdp(?!\w)|(?<!\w)inflation(?!\w)'
    r'|(?<!\w)interest\s+rate(?!\w))',
    re.IGNORECASE
)

def is_control(var):
    if pd.isna(var) or str(var).strip() == '':
        return False
    return bool(CONTROL_RE.search(str(var)))

pub['is_control']  = pub['variable'].apply(is_control)
nber['is_control'] = nber['variable'].apply(is_control)

pub_nc  = pub[~pub['is_control']].copy()
nber_nc = nber[~nber['is_control']].copy()

n_ctrl_pub  = pub['is_control'].sum()
n_ctrl_nber = nber['is_control'].sum()
print(f"Published  — total: {len(pub):,} | excluded (controls): {n_ctrl_pub:,} "
      f"({n_ctrl_pub/len(pub):.1%}) | non-control: {len(pub_nc):,}")
print(f"NBER       — total: {len(nber):,} | excluded (controls): {n_ctrl_nber:,} "
      f"({n_ctrl_nber/len(nber):.1%}) | non-control: {len(nber_nc):,}")

# ── Run caliper on both full and non-control subsets ──────────────────────
SUBSETS = {
    'Full sample':      None,
    'Positive-finding': 1,
    'Negative-finding': -1,
    'Null-finding':     0,
}

rows = []
for filter_label, (df_pub, df_nber) in [
        ('all_statistics', (pub,    nber)),
        ('non_control',    (pub_nc, nber_nc))]:
    for corpus_label, df in [('published', df_pub), ('nber', df_nber)]:
        for subset_label, val in SUBSETS.items():
            sub = df if val is None else df[df['direction'] == val]
            if len(sub) < 20:
                continue
            res = caliper_clustered(sub['z_abs'], sub['paper_id'])
            if not res:
                continue
            rows.append({'filter': filter_label, 'sample': corpus_label,
                         'subset': subset_label, **res})

result = pd.DataFrame(rows)
result.to_csv('output/tables/caliper_inclusion_sensitivity.csv', index=False)
result[result['filter'] == 'non_control'].to_csv(
    'output/tables/caliper_noncontrol.csv', index=False)

disp = ['filter','subset','N','N_papers','caliper_ratio','caliper_p','p_perm','p_boot',
        'boot_ci_lo','boot_ci_hi']
print('\n=== Published caliper: all vs non-control ===')
print(result[result['sample'] == 'published'][disp].to_string(index=False))
print('\n=== NBER caliper: all vs non-control ===')
print(result[result['sample'] == 'nber'][disp].to_string(index=False))
print('\nDone.')
