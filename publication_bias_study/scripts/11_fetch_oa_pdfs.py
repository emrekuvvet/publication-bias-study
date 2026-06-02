"""
Script 11: For each DOI, try Unpaywall for open-access PDF URL.
Falls back to Semantic Scholar for preprint links.
Outputs a manifest of what's available vs. needs manual download.
"""

import pandas as pd
import requests
import time
import json
from pathlib import Path

EMAIL = 'emrekuvvet@gmail.com'
HEADERS = {'User-Agent': f'PublicationBiasStudy/1.0 (mailto:{EMAIL})'}

def unpaywall(doi):
    url = f'https://api.unpaywall.org/v2/{doi}?email={EMAIL}'
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 404:
            return None, None
        r.raise_for_status()
        data = r.json()
        best = data.get('best_oa_location') or {}
        pdf_url = best.get('url_for_pdf') or best.get('url')
        is_oa = data.get('is_oa', False)
        return pdf_url, is_oa
    except Exception as e:
        print(f"    Unpaywall error ({doi}): {e}")
        return None, None

def semantic_scholar(title):
    url = 'https://api.semanticscholar.org/graph/v1/paper/search'
    params = {'query': title, 'fields': 'openAccessPdf,externalIds', 'limit': 3}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        papers = r.json().get('data', [])
        for p in papers:
            pdf = p.get('openAccessPdf', {})
            if pdf and pdf.get('url'):
                return pdf['url']
        return None
    except Exception as e:
        print(f"    S2 error: {e}")
        return None

def main():
    df = pd.read_parquet('data/raw/published_with_dois.parquet')
    print(f"Looking up OA PDFs for {len(df)} papers...")

    cache_path = Path('data/raw/oa_cache.json')
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    pdf_urls = []
    sources = []

    for i, row in df.iterrows():
        doi = row.get('doi')
        title = row['title']
        key = doi or title

        if key in cache:
            entry = cache[key]
            pdf_urls.append(entry.get('url'))
            sources.append(entry.get('source'))
            continue

        url, source = None, None

        if doi:
            pdf_url, is_oa = unpaywall(doi)
            if pdf_url:
                url, source = pdf_url, 'unpaywall'
            time.sleep(0.5)

        if not url:
            s2_url = semantic_scholar(title)
            if s2_url:
                url, source = s2_url, 'semantic_scholar'
            time.sleep(0.3)

        cache[key] = {'url': url, 'source': source}
        pdf_urls.append(url)
        sources.append(source)

        status = f"✓ {source}" if url else "✗"
        print(f"  [{i+1:3d}/{len(df)}] {title[:55]:55s} {status}")

    cache_path.write_text(json.dumps(cache, indent=2))

    df['pdf_url'] = pdf_urls
    df['pdf_source'] = sources

    found = df['pdf_url'].notna().sum()
    print(f"\nOA PDFs found:  {found}/{len(df)} ({found/len(df)*100:.1f}%)")
    print(f"Need manual:    {len(df)-found}")

    df.to_parquet('data/raw/published_with_pdfs.parquet', index=False)
    print("Saved → data/raw/published_with_pdfs.parquet")

    # Print manifest
    manual = df[df['pdf_url'].isna()][['journal','year','title']]
    print("\n=== Papers needing manual download ===")
    for _, r in manual.iterrows():
        print(f"  [{r['journal']} {r['year']}] {r['title']}")

if __name__ == '__main__':
    main()
