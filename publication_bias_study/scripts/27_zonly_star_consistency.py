"""
Script 27: Star-consistency validation of z-stat-only rows.

Referee comment 2.2 asks whether the caliper bunching in z-stat-only rows
reflects extraction artifacts rather than genuine within-paper specification
search.

This script cross-validates extracted z_abs values against reported
significance stars in z-stat-only rows (rows where se is NaN).

If the extraction is valid, z-stat-only rows in the lower caliper window
(1.645, 1.96) should predominantly carry '*' stars (consistent with
10%-but-not-5% significance), and rows in the upper window (1.96, 2.275)
should carry '**' or '***' stars.

Outputs:
    output/tables/zonly_star_consistency.csv
    output/tables/zonly_window_stars.csv
    output/tables/zonly_table_type.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

pub_raw  = pd.read_parquet('data/final/tstats_all.parquet')
nber_raw = pd.read_parquet('data/final/nber_tstats_all.parquet')

# ── Classify rows ─────────────────────────────────────────────────────────────
for df, name in [(pub_raw, 'pub'), (nber_raw, 'nber')]:
    df['zonly'] = df['se'].isna()
    df['z']     = pd.to_numeric(df['z_abs'], errors='coerce')
    df.loc[df['z'] < 0, 'z'] = np.nan
    df.loc[df['z'] > 50, 'z'] = np.nan
    df['corpus'] = name

all_df = pd.concat([pub_raw, nber_raw], ignore_index=True)

# Standard stars only (exclude footnote markers a/b/c/nnn/nn/n/†/+)
std = {'', '*', '**', '***'}

# ── 1. Star-consistency rate ────────────────────────────────────────────────
# Expected star for a given z_abs (using standard thresholds)
def expected_star(z):
    if pd.isna(z):
        return None
    if z >= 2.576:
        return '***'
    if z >= 1.960:
        return '**'
    if z >= 1.645:
        return '*'
    return ''

rows = []
for corpus_name, df in [('Published', pub_raw), ('NBER', nber_raw)]:
    for row_type, mask in [('z-stat-only', df['zonly']),
                            ('coef+SE',    ~df['zonly'])]:
        sub = df[mask & df['stars'].isin(std)].copy()
        sub['exp_star'] = sub['z'].apply(expected_star)
        sub_valid = sub[sub['exp_star'].notna() & sub['z'].notna()]
        n_total    = len(sub_valid)
        n_exact    = (sub_valid['stars'] == sub_valid['exp_star']).sum()
        # Also allow '***' where we expect '***' OR '**' (some papers use 1% = z>2.326)
        n_close    = 0
        for _, row in sub_valid.iterrows():
            s, e = row['stars'], row['exp_star']
            if s == e:
                n_close += 1
            elif e == '**' and s == '***' and row['z'] >= 2.326:
                n_close += 1  # 1% using z>2.326 threshold
            elif e == '' and s == '*' and row['z'] >= 1.645:
                pass  # paper uses 10% threshold where we said none → not close

        rows.append({
            'Corpus': corpus_name, 'Row type': row_type,
            'N rows (std stars)': n_total,
            'Exact star-z match': int(n_exact),
            'Exact consistency %': round(100 * n_exact / n_total, 1) if n_total else np.nan,
        })

cons_df = pd.DataFrame(rows)
cons_df.to_csv('output/tables/zonly_star_consistency.csv', index=False)
print('=== Star-Consistency Rate ===')
print(cons_df.to_string(index=False))

# ── 2. Star distribution by caliper window ──────────────────────────────────
window_rows = []
for corpus_name, df in [('Published', pub_raw), ('NBER', nber_raw)]:
    zonly = df[df['zonly'] & df['stars'].isin(std) & df['z'].notna()].copy()
    for window_label, lo, hi in [('Below threshold (1.645,1.96)', 1.645, 1.96),
                                  ('Above threshold (1.96,2.275)',  1.96,  2.275),
                                  ('Significant (z>1.96)',           1.96,  50)]:
        sub = zonly[(zonly['z'] > lo) & (zonly['z'] < hi)]
        counts = sub['stars'].value_counts().to_dict()
        total  = len(sub)
        window_rows.append({
            'Corpus': corpus_name,
            'Window': window_label,
            'Total': total,
            'stars=""':  counts.get('', 0),
            'stars=*':   counts.get('*', 0),
            'stars=**':  counts.get('**', 0),
            'stars=***': counts.get('***', 0),
            'Star-consistent (*)': counts.get('*', 0) if 'Below' in window_label else
                                   (counts.get('**', 0) + counts.get('***', 0)),
            'Consistency %': round(100 * (counts.get('*', 0) if 'Below' in window_label
                                          else counts.get('**', 0) + counts.get('***', 0))
                                   / total, 1) if total else np.nan,
        })

win_df = pd.DataFrame(window_rows)
win_df.to_csv('output/tables/zonly_window_stars.csv', index=False)
print('\n=== Star Distribution by Caliper Window (z-stat-only rows) ===')
print(win_df[['Corpus','Window','Total','stars=""','stars=*','stars=**','Consistency %']].to_string(index=False))

# ── 3. Caliper ratio comparison: z-stat-only vs coef+SE ─────────────────────
print('\n=== Caliper Ratios by Row Type ===')
for corpus_name, df in [('Published', pub_raw), ('NBER', nber_raw)]:
    for label, mask in [('z-stat-only', df['zonly']), ('coef+SE', ~df['zonly'])]:
        z = df[mask]['z'].dropna()
        z = z[(z >= 0) & (z <= 50)]
        below = int(((z >= 1.645) & (z < 1.96)).sum())
        above = int(((z >= 1.96) & (z < 2.275)).sum())
        ratio = below / above if above > 0 else np.nan
        print(f'  {corpus_name:12s} {label:12s}: below={below:4d} above={above:4d} ratio={ratio:.3f}')

# ── 4. Table-type breakdown for z-stat-only rows ────────────────────────────
if 'table' in pub_raw.columns:
    pub_zonly = pub_raw[pub_raw['zonly']].copy()
    # Categorize by table name pattern
    def classify_table(tbl):
        if pd.isna(tbl) or str(tbl).strip() == '':
            return 'Unknown/blank'
        t = str(tbl).lower()
        if any(x in t for x in ['robust', 'sensitivity', 'check', 'additional', 'suppl']):
            return 'Robustness/supplementary'
        if any(x in t for x in ['main', 'result', 'baseline', 'primary', 'ols', 'reg']):
            return 'Main results'
        if any(x in t for x in ['summary', 'stat', 'descrip', 'variable']):
            return 'Summary statistics'
        return 'Other'

    pub_zonly['tbl_type'] = pub_zonly['table'].apply(classify_table)
    tbl_df = pub_zonly.groupby('tbl_type').agg(
        n=('z_abs', 'count'),
        in_caliper=('z', lambda x: ((x > 1.645) & (x < 1.96)).sum()),
    ).reset_index()
    tbl_df['pct_total'] = (100 * tbl_df['n'] / tbl_df['n'].sum()).round(1)
    tbl_df.to_csv('output/tables/zonly_table_type.csv', index=False)
    print('\n=== Table-type distribution of z-stat-only rows (Published) ===')
    print(tbl_df.to_string(index=False))

print('\nDone.')
