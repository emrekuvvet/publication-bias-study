"""
Script 16: Download full-text PDFs for all 157 in-scope NBER working papers.
PDFs are freely available at https://www.nber.org/papers/wXXXXX.pdf
"""

import pandas as pd
import requests
import time
from pathlib import Path

OUT_DIR = Path('data/raw/nber_pdfs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {'User-Agent': 'Mozilla/5.0 (research purposes; emrekuvvet@gmail.com)'}

def download_pdf(wp_number, out_path):
    url = f'https://www.nber.org/papers/{wp_number}.pdf'
    try:
        r = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=30)
        if r.status_code == 200 and len(r.content) > 10_000:
            out_path.write_bytes(r.content)
            return len(r.content)
        else:
            return None
    except Exception as e:
        print(f'    Error: {e}')
        return None

def main():
    df = pd.read_parquet('data/final/analysis_dataset.parquet')
    nber = df[df['source'] == 'nber'].copy().reset_index(drop=True)
    print(f'Downloading {len(nber)} NBER PDFs...')

    ok, fail = 0, []
    for i, row in nber.iterrows():
        wp = row['nber_wp_number']
        out = OUT_DIR / f'{wp}.pdf'

        if out.exists() and out.stat().st_size > 10_000:
            print(f'  [{i+1:3d}/{len(nber)}] CACHED  {wp}  {out.stat().st_size//1024}KB')
            ok += 1
            continue

        size = download_pdf(wp, out)
        if size:
            print(f'  [{i+1:3d}/{len(nber)}] OK      {wp}  {size//1024}KB  {row["title"][:50]}')
            ok += 1
        else:
            print(f'  [{i+1:3d}/{len(nber)}] FAILED  {wp}  {row["title"][:50]}')
            fail.append(wp)

        time.sleep(0.5)

    print(f'\nDone: {ok} downloaded, {len(fail)} failed')
    if fail:
        print('Failed:', fail)

if __name__ == '__main__':
    main()
