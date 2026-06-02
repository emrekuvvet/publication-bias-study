"""
Script 15: Download JF papers via Wiley Text & Data Mining API.
Requires WILEY_TDM_TOKEN in .env
Covers JF only (Wiley publisher). JFE=Elsevier, RFS=Oxford handled separately.
"""

import pandas as pd
import requests
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('WILEY_TDM_TOKEN')
PDF_DIR = Path('data/raw/pdfs')
PDF_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'Wiley-TDM-Client-Token': TOKEN,
    'User-Agent': 'PublicationBiasStudy/1.0 (mailto:emrekuvvet@gmail.com)',
}

def safe_filename(title, journal, year):
    slug = title[:60].lower()
    slug = ''.join(c if c.isalnum() or c in ' -' else '_' for c in slug)
    return f"{journal}_{year}_{slug.strip().replace(' ', '_')}.pdf"

def download_wiley(doi, fpath):
    url = f'https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}'
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        if r.status_code == 401:
            return False, "401 Unauthorized — token invalid or expired"
        if r.status_code == 403:
            return False, "403 Forbidden — no TDM rights for this article"
        if r.status_code == 404:
            return False, "404 Not found"
        r.raise_for_status()
        content = r.content
        if len(content) < 5000:
            return False, f"too small ({len(content)} bytes)"
        if b'%PDF' not in content[:1000]:
            return False, "response is not a PDF"
        fpath.write_bytes(content)
        return True, f"{len(content)//1024} KB"
    except Exception as e:
        return False, str(e)

def main():
    if not TOKEN:
        print("ERROR: WILEY_TDM_TOKEN not set in .env")
        return

    df = pd.read_parquet('data/raw/published_with_pdfs.parquet')
    jf = df[(df['journal'] == 'JF') & df['doi'].notna()].copy()
    missing = jf[jf['local_pdf'].isna()].copy()

    print(f"JF papers total: {len(jf)}")
    print(f"Already have PDF: {jf['local_pdf'].notna().sum()}")
    print(f"To download via Wiley TDM: {len(missing)}")
    print()

    local_paths = {}
    ok = fail = 0

    for i, row in missing.iterrows():
        fname = safe_filename(row['title'], row['journal'], int(row['year']))
        fpath = PDF_DIR / fname

        if fpath.exists() and fpath.stat().st_size > 5000:
            local_paths[row['title']] = str(fpath)
            ok += 1
            print(f"  SKIP (exists)  {fname[:70]}")
            continue

        success, msg = download_wiley(row['doi'], fpath)
        if success:
            local_paths[row['title']] = str(fpath)
            ok += 1
            print(f"  ✓ [{ok+fail:2d}] {fname[:65]} ({msg})")
        else:
            fail += 1
            print(f"  ✗ [{ok+fail:2d}] {row['title'][:60]} — {msg}")

        time.sleep(1.0)

    # Update manifest
    for title, path in local_paths.items():
        df.loc[df['title'] == title, 'local_pdf'] = path

    df.to_parquet('data/raw/published_with_pdfs.parquet', index=False)
    print(f"\nDownloaded: {ok} | Failed: {fail}")
    print(f"Total PDFs in data/raw/pdfs/: {len(list(PDF_DIR.glob('*.pdf')))}")

    still_missing = df[(df['journal']=='JF') & df['local_pdf'].isna()]
    if len(still_missing):
        print(f"\n=== {len(still_missing)} JF papers still missing ===")
        for _, r in still_missing.iterrows():
            print(f"  [{r['year']}] {r['title']}")

if __name__ == '__main__':
    main()
