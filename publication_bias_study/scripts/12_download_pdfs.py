"""
Script 12: Download PDFs from OA URLs into data/raw/pdfs/.
Skips already-downloaded files.
"""

import pandas as pd
import requests
import time
import hashlib
from pathlib import Path

PDF_DIR = Path('data/raw/pdfs')
PDF_DIR.mkdir(parents=True, exist_ok=True)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; PublicationBiasStudy/1.0)',
    'Accept': 'application/pdf,*/*',
}

def safe_filename(title, journal, year):
    slug = title[:60].lower()
    slug = ''.join(c if c.isalnum() or c in ' -' else '_' for c in slug)
    slug = slug.strip().replace(' ', '_')
    return f"{journal}_{year}_{slug}.pdf"

def download(url, path):
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        r.raise_for_status()
        content = r.content
        if len(content) < 5000:
            return False, "too small (likely not a PDF)"
        if not content[:4] == b'%PDF':
            # Try to detect if it's a valid PDF anyway
            if b'%PDF' not in content[:1000]:
                return False, "not a PDF"
        path.write_bytes(content)
        return True, f"{len(content)//1024}KB"
    except Exception as e:
        return False, str(e)

def main():
    df = pd.read_parquet('data/raw/published_with_pdfs.parquet')
    available = df[df['pdf_url'].notna()].copy()
    print(f"Downloading {len(available)} PDFs...")

    local_paths = {}
    ok = 0
    fail = 0

    for i, row in available.iterrows():
        fname = safe_filename(row['title'], row['journal'], int(row['year']))
        fpath = PDF_DIR / fname

        if fpath.exists() and fpath.stat().st_size > 5000:
            local_paths[row['title']] = str(fpath)
            ok += 1
            print(f"  [{ok+fail:3d}] SKIP (exists) {fname[:70]}")
            continue

        success, msg = download(row['pdf_url'], fpath)
        if success:
            local_paths[row['title']] = str(fpath)
            ok += 1
            print(f"  [{ok+fail:3d}] ✓ {fname[:70]} ({msg})")
        else:
            fail += 1
            print(f"  [{ok+fail:3d}] ✗ {row['title'][:60]} — {msg}")

        time.sleep(1.0)

    # Save manifest
    df['local_pdf'] = df['title'].map(local_paths)
    df.to_parquet('data/raw/published_with_pdfs.parquet', index=False)

    print(f"\nDownloaded: {ok} | Failed: {fail}")
    print(f"PDFs in data/raw/pdfs/: {len(list(PDF_DIR.glob('*.pdf')))}")

    missing = df[df['local_pdf'].isna()][['journal','year','title','pdf_url']]
    if len(missing):
        print(f"\n=== {len(missing)} papers without local PDF ===")
        for _, r in missing.iterrows():
            print(f"  [{r['journal']} {r['year']}] {r['title']}")

if __name__ == '__main__':
    main()
