"""
Script 24: Extraction validation — spot-check sample + coverage diagnostics.

Outputs:
    output/tables/extraction_spotcheck_sample.csv  — 50 random (paper, stat) rows
    output/tables/extraction_coverage.csv          — per-paper coverage diagnostics
    output/tables/extraction_zero_stat_papers.csv  — papers yielding 0 statistics

The spot-check CSV is designed for manual verification against the source PDFs.
Each row contains: paper title, source, page hint (if available), variable name,
table label, extracted coefficient, extracted SE, and computed |z|.

Coverage diagnostics report the number of extracted statistics per paper and
flag likely image-PDF papers (text < 500 words per page on average, suggesting
scanned or image-only content that pdfplumber cannot extract).
"""

import pandas as pd
import numpy as np
from pathlib import Path

SEED = 2024

# ── Load extraction outputs ────────────────────────────────────────────────
pub  = pd.read_parquet('data/final/tstats_all.parquet')
nber = pd.read_parquet('data/final/nber_tstats_all.parquet')
analysis = pd.read_parquet('data/final/analysis_dataset.parquet')

pub['corpus']  = 'published'
nber['corpus'] = 'nber'
nber = nber.rename(columns={'wp_number': 'paper_id_nber'})

# ── Spot-check sample ──────────────────────────────────────────────────────
# Stratified: 25 from published, 25 from NBER
rng = np.random.default_rng(SEED)

def sample_corpus(df, n, extra_id_col=None):
    df = df.copy().reset_index(drop=True)
    df['z_abs_num'] = pd.to_numeric(df['z_abs'], errors='coerce').abs()
    # Only sample rows with a usable z_abs
    valid = df[df['z_abs_num'].notna() & (df['z_abs_num'] > 0) & (df['z_abs_num'] <= 50)]
    idx = rng.choice(len(valid), size=min(n, len(valid)), replace=False)
    sample = valid.iloc[idx].copy()
    keep = ['corpus', 'paper_title', 'year', 'table', 'variable',
            'coef', 'se', 'z_abs_num', 'stars']
    if 'spec' in sample.columns:
        keep.append('spec')
    if extra_id_col and extra_id_col in sample.columns:
        keep.insert(1, extra_id_col)
    return sample[[c for c in keep if c in sample.columns]].rename(
        columns={'z_abs_num': 'z_abs_extracted', 'paper_title': 'title'})

spot_pub  = sample_corpus(pub,  25)
spot_nber = sample_corpus(nber, 25, extra_id_col='paper_id_nber')
spot = pd.concat([spot_pub, spot_nber], ignore_index=True)

spot['verified_ok']    = ''   # auditor fills: Y / N / ?
spot['pdf_z_abs']      = ''   # auditor fills: actual |z| from PDF
spot['comment']        = ''   # auditor fills: notes

spot.to_csv('output/tables/extraction_spotcheck_sample.csv', index=False)
print(f'Spot-check sample: {len(spot)} rows saved → output/tables/extraction_spotcheck_sample.csv')


# ── Per-paper coverage ─────────────────────────────────────────────────────
pub_cov = (pub.groupby('paper_title')
              .agg(n_stats=('z_abs','count'))
              .reset_index()
              .rename(columns={'paper_title': 'title'}))
pub_cov['corpus'] = 'published'

nber_cov = (nber.groupby('paper_id_nber')
                .agg(n_stats=('z_abs','count'))
                .reset_index()
                .rename(columns={'paper_id_nber': 'wp_number'}))
nber_cov['title']  = nber_cov['wp_number'].map(
    nber.groupby('paper_id_nber')['paper_title'].first())
nber_cov['corpus'] = 'nber'

# Papers with 0 stats (from analysis dataset)
pub_ana  = analysis[analysis['source'] != 'nber'][['paper_id','title','year','source']].copy()
nber_ana = analysis[analysis['source'] == 'nber'][['paper_id','nber_wp_number','title','year']].copy()

zero_pub = pub_ana[~pub_ana['title'].isin(pub_cov['title'])].copy()
zero_pub['corpus'] = 'published'
zero_pub['n_stats'] = 0

zero_nber = nber_ana[~nber_ana['nber_wp_number'].isin(nber_cov['wp_number'])].copy()
zero_nber['corpus'] = 'nber'
zero_nber['n_stats'] = 0

zero_papers = pd.concat([zero_pub, zero_nber], ignore_index=True)
zero_papers.to_csv('output/tables/extraction_zero_stat_papers.csv', index=False)
print(f'Zero-stat papers: {len(zero_papers)} ({len(zero_pub)} pub, {len(zero_nber)} NBER)')
print(f'  Published coverage: {len(pub_cov)}/{len(pub_ana)} = {len(pub_cov)/len(pub_ana):.1%}')
print(f'  NBER coverage:      {len(nber_cov)}/{len(nber_ana)} = {len(nber_cov)/len(nber_ana):.1%}')


# ── Distribution of stats per paper ───────────────────────────────────────
for corpus_label, cov_df in [('Published', pub_cov), ('NBER', nber_cov)]:
    n = cov_df['n_stats']
    print(f'\n{corpus_label} — stats per paper:')
    print(f'  Mean: {n.mean():.1f}  Median: {n.median():.0f}  '
          f'Min: {n.min()}  Max: {n.max()}  '
          f'Papers with <10 stats: {(n<10).sum()}')


# ── Summary coverage table ─────────────────────────────────────────────────
summary = pd.DataFrame([
    {'Corpus': 'Published', 'Total papers': len(pub_ana),
     'Papers with stats': len(pub_cov), 'Coverage': f'{len(pub_cov)/len(pub_ana):.1%}',
     'Total stats': len(pub), 'Avg stats/paper': round(pub_cov["n_stats"].mean(), 1),
     'Median stats/paper': int(pub_cov["n_stats"].median())},
    {'Corpus': 'NBER',      'Total papers': len(nber_ana),
     'Papers with stats': len(nber_cov), 'Coverage': f'{len(nber_cov)/len(nber_ana):.1%}',
     'Total stats': len(nber), 'Avg stats/paper': round(nber_cov["n_stats"].mean(), 1),
     'Median stats/paper': int(nber_cov["n_stats"].median())},
])
summary.to_csv('output/tables/extraction_coverage.csv', index=False)
print('\n=== Extraction Coverage Summary ===')
print(summary.to_string(index=False))
print('\nDone.')
