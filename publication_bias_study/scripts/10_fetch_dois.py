"""
Script 10: Fetch DOIs for the 134 published papers via CrossRef API.
Matches on title + journal + year.
"""

import pandas as pd
import requests
import time
import json
from pathlib import Path

JOURNAL_ISSN = {
    'JF':  '0022-1082',
    'RFS': '0893-9454',
    'JFE': '0304-405X',
}

CROSSREF_URL = 'https://api.crossref.org/works'
HEADERS = {'User-Agent': 'PublicationBiasStudy/1.0 (mailto:emrekuvvet@gmail.com)'}

def query_crossref(title, journal, year):
    params = {
        'query.title': title,
        'filter': f'issn:{JOURNAL_ISSN[journal]},from-pub-date:{year},until-pub-date:{year}',
        'rows': 3,
        'select': 'DOI,title,published',
    }
    try:
        r = requests.get(CROSSREF_URL, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        items = r.json()['message']['items']
        if not items:
            return None
        # Take top result
        return items[0]['DOI']
    except Exception as e:
        print(f"  CrossRef error for '{title[:50]}': {e}")
        return None

def title_similarity(a, b):
    a = a.lower().replace('-', ' ').replace('‐', ' ')
    b = b.lower().replace('-', ' ').replace('‐', ' ')
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return 0
    return len(words_a & words_b) / len(words_a | words_b)

def main():
    ana = pd.read_parquet('data/final/analysis_dataset.parquet')
    pub = ana[ana['published_top3']].copy().reset_index(drop=True)
    print(f"Fetching DOIs for {len(pub)} published papers...")

    cache_path = Path('data/raw/doi_cache.json')
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    dois = []
    for i, row in pub.iterrows():
        title = row['title']
        if title in cache:
            dois.append(cache[title])
            continue

        doi = query_crossref(title, row['journal'], int(row['year']))
        cache[title] = doi
        dois.append(doi)

        status = f"✓ {doi}" if doi else "✗ not found"
        print(f"  [{i+1:3d}/{len(pub)}] {title[:60]:60s} {status}")
        time.sleep(0.12)  # CrossRef rate limit: ~10 req/s politely

    cache_path.write_text(json.dumps(cache, indent=2))

    pub['doi'] = dois
    found = pub['doi'].notna().sum()
    print(f"\nDOIs found: {found}/{len(pub)} ({found/len(pub)*100:.1f}%)")
    print(f"Missing:    {len(pub)-found}")

    pub.to_parquet('data/raw/published_with_dois.parquet', index=False)
    print("Saved → data/raw/published_with_dois.parquet")

    # Show missing
    missing = pub[pub['doi'].isna()][['title','journal','year']]
    if len(missing):
        print("\nMissing DOIs:")
        for _, r in missing.iterrows():
            print(f"  [{r['journal']} {r['year']}] {r['title']}")

if __name__ == '__main__':
    main()
