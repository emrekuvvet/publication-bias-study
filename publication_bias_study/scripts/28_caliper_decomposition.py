"""
Script 28: Caliper decomposition — composition effect vs. within-type effect.

Referee Comment 3.1 asks for a decomposition that separates:
  (a) the composition effect: published corpus has more z-stat-only rows (29.0%)
      than NBER (18.7%), and z-stat-only rows bunch more in both corpora
  (b) the within-type effect: within the same row type, published > NBER ratio

Outputs:
    output/tables/caliper_decomposition.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

pub_raw  = pd.read_parquet('data/final/tstats_all.parquet')
nber_raw = pd.read_parquet('data/final/nber_tstats_all.parquet')

for df, name in [(pub_raw, 'pub'), (nber_raw, 'nber')]:
    df['zonly'] = df['se'].isna()
    df['z'] = pd.to_numeric(df['z_abs'], errors='coerce')
    df.loc[df['z'] < 0, 'z'] = np.nan
    df.loc[df['z'] > 50, 'z'] = np.nan
    df['corpus'] = name

def caliper_ratio(z_series):
    z = z_series.dropna()
    below = ((z >= 1.645) & (z < 1.96)).sum()
    above = ((z >= 1.96) & (z < 2.275)).sum()
    return below / above if above > 0 else np.nan

rows = []

# ── Observed ratios ────────────────────────────────────────────────────────────
for corpus_name, df in [('Published', pub_raw), ('NBER', nber_raw)]:
    z_valid = df['z'].notna()
    n_valid = z_valid.sum()
    for label, mask in [('Overall', pd.Series(True, index=df.index)),
                        ('Coef+SE', ~df['zonly']),
                        ('Z-stat-only', df['zonly'])]:
        r = caliper_ratio(df[mask]['z'])
        # Share relative to z-valid rows (the denominator that matches the caliper N)
        share = (mask & z_valid).sum() / n_valid if label != 'Overall' else 1.0
        rows.append({'Corpus': corpus_name, 'Row type': label,
                     'Caliper ratio': round(r, 3),
                     'Share of rows': round(share, 3)})

obs_df = pd.DataFrame(rows)
print('=== Observed caliper ratios ===')
print(obs_df.to_string(index=False))

# ── Decomposition ──────────────────────────────────────────────────────────────
# Shares computed over z-valid rows to match the caliper analysis denominator
pub_z_valid  = pub_raw['z'].notna()
nber_z_valid = nber_raw['z'].notna()
pub_zonly_share  = (pub_raw['zonly'] & pub_z_valid).sum()  / pub_z_valid.sum()
nber_zonly_share = (nber_raw['zonly'] & nber_z_valid).sum() / nber_z_valid.sum()

pub_zonly_ratio   = caliper_ratio(pub_raw[pub_raw['zonly']]['z'])
pub_coefse_ratio  = caliper_ratio(pub_raw[~pub_raw['zonly']]['z'])
nber_zonly_ratio  = caliper_ratio(nber_raw[nber_raw['zonly']]['z'])
nber_coefse_ratio = caliper_ratio(nber_raw[~nber_raw['zonly']]['z'])

pub_overall_obs   = caliper_ratio(pub_raw['z'])
nber_overall_obs  = caliper_ratio(nber_raw['z'])

# Counterfactual published overall ratio if it had NBER's z-stat-only share
pub_cf_nber_share = (nber_zonly_share * pub_zonly_ratio
                     + (1 - nber_zonly_share) * pub_coefse_ratio)

# Counterfactual NBER overall ratio if it had published's z-stat-only share
nber_cf_pub_share = (pub_zonly_share * nber_zonly_ratio
                     + (1 - pub_zonly_share) * nber_coefse_ratio)

observed_gap        = pub_overall_obs - nber_overall_obs
within_type_gap     = (pub_zonly_ratio - nber_zonly_ratio) * nber_zonly_share \
                    + (pub_coefse_ratio - nber_coefse_ratio) * (1 - nber_zonly_share)
composition_effect  = observed_gap - within_type_gap

print('\n=== Caliper Gap Decomposition ===')
print(f'Observed gap (Published - NBER overall):       {observed_gap:+.3f}')
print(f'  Within-type effect (at NBER composition):    {within_type_gap:+.3f}')
print(f'  Composition effect (different z-only shares):{composition_effect:+.3f}')
print(f'\n  Published z-stat-only share: {pub_zonly_share:.3f}')
print(f'  NBER z-stat-only share:      {nber_zonly_share:.3f}')
print(f'  Counterfactual published (NBER shares):      {pub_cf_nber_share:.3f}')
print(f'  Counterfactual NBER (pub shares):            {nber_cf_pub_share:.3f}')

decomp_df = pd.DataFrame([
    {'Component': 'Published overall ratio (observed)', 'Value': round(pub_overall_obs, 3)},
    {'Component': 'NBER overall ratio (observed)', 'Value': round(nber_overall_obs, 3)},
    {'Component': 'Observed gap', 'Value': round(observed_gap, 3)},
    {'Component': 'Within-type effect (at NBER composition)', 'Value': round(within_type_gap, 3)},
    {'Component': 'Composition effect', 'Value': round(composition_effect, 3)},
    {'Component': 'Published z-stat-only share', 'Value': round(pub_zonly_share, 3)},
    {'Component': 'NBER z-stat-only share', 'Value': round(nber_zonly_share, 3)},
    {'Component': 'Published coef+SE ratio', 'Value': round(pub_coefse_ratio, 3)},
    {'Component': 'NBER coef+SE ratio', 'Value': round(nber_coefse_ratio, 3)},
    {'Component': 'Published z-stat-only ratio', 'Value': round(pub_zonly_ratio, 3)},
    {'Component': 'NBER z-stat-only ratio', 'Value': round(nber_zonly_ratio, 3)},
])
decomp_df.to_csv('output/tables/caliper_decomposition.csv', index=False)
print('\nDone. Output: output/tables/caliper_decomposition.csv')
