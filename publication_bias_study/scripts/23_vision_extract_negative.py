"""
Script 23: Vision-based extraction for the 7 direction=-1 zero-stat NBER papers.
Sends PDF page images directly to Claude vision API instead of relying on OCR text.
"""

import anthropic
import base64
import json
import re
import time
import io
import pandas as pd
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

TARGETS = ['w12719', 'w15281', 'w16763', 'w17149', 'w7992', 'w9385', 'w11353']

PROMPT = """Look at this page from a finance research paper.
Extract ALL test statistics from any regression or estimation tables you see.
Return a JSON array with keys: table, variable, coef, se, z_abs, stars, spec.
Rules:
- Parenthesized values: if |value| > 1 it is a t-stat (z_abs = |t|); if |value| < 1 it is a SE (z_abs = |coef/se|)
- Bracket values [x]: x is a t-stat, z_abs = |x|
- Stars with no SE: *=1.8, **=2.3, ***=3.0
- ONLY regression/estimation tables, NOT summary statistics tables
- ONLY variable rows, skip intercepts and FE indicator rows
- If no regression table on this page, return []
Return ONLY the JSON array."""


def pages_to_images(pdf_path, dpi=150, max_pages=80):
    try:
        imgs = convert_from_path(str(pdf_path), dpi=dpi,
                                  first_page=1, last_page=max_pages)
        return imgs
    except Exception as e:
        print(f'    pdf2image error: {e}')
        return []


def img_to_b64(img):
    buf = io.BytesIO()
    # Resize if too large (Claude vision limit ~5MB per image, but keep cost down)
    w, h = img.size
    if w > 1400:
        img = img.resize((1400, int(h * 1400 / w)), Image.LANCZOS)
    img.save(buf, format='PNG')
    return base64.standard_b64encode(buf.getvalue()).decode('utf-8')


def call_vision(pages_b64, client):
    """Send up to 4 pages at once to Claude vision."""
    content = []
    for b64 in pages_b64:
        content.append({
            'type': 'image',
            'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64}
        })
    content.append({'type': 'text', 'text': PROMPT})

    try:
        msg = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=6000,
            messages=[{'role': 'user', 'content': content}]
        )
        raw = msg.content[0].text.strip()
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        bi = raw.find('[')
        if bi < 0:
            return []
        try:
            return json.loads(raw[bi:raw.rfind(']') + 1])
        except Exception:
            last = raw.rfind('}')
            if last >= 0:
                try:
                    return json.loads(raw[bi:last + 1] + ']')
                except Exception:
                    pass
        return []
    except Exception as e:
        print(f'    vision API error: {e}')
        return []


def main():
    analysis  = pd.read_parquet('data/final/analysis_dataset.parquet')
    nber_df   = analysis[analysis['source'] == 'nber'].set_index('nber_wp_number')
    cache_dir = Path('data/raw/nber_tstats')
    pdf_dir   = Path('data/raw/nber_pdfs')
    client    = anthropic.Anthropic()

    improved = 0

    for wp in TARGETS:
        cache = cache_dir / f'{wp}.json'
        if cache.exists() and len(json.loads(cache.read_text())) > 0:
            print(f'  SKIP {wp} — already has stats')
            continue

        pdf = pdf_dir / f'{wp}.pdf'
        if not pdf.exists():
            print(f'  NO PDF: {wp}')
            continue

        row   = nber_df.loc[wp]
        title = row['title']
        kb    = pdf.stat().st_size // 1024
        print(f'\n{wp} ({kb}KB): {title[:60]}')

        imgs = pages_to_images(pdf)
        if not imgs:
            print('  → no images')
            continue
        print(f'  {len(imgs)} pages → sending to Claude vision in batches of 4')

        all_stats = []
        seen = set()
        batch_size = 4

        for i in range(0, len(imgs), batch_size):
            batch_imgs = imgs[i:i + batch_size]
            b64s = [img_to_b64(img) for img in batch_imgs]
            stats = call_vision(b64s, client)
            new = 0
            for s in stats:
                key = (s.get('table', ''), s.get('variable', ''),
                       round(s.get('z_abs') or 0, 2))
                if key not in seen:
                    seen.add(key)
                    all_stats.append(s)
                    new += 1
            if new:
                print(f'  pages {i+1}-{i+len(batch_imgs)}: +{new} stats')
            time.sleep(0.5)

        if all_stats:
            for s in all_stats:
                s['wp_number']   = wp
                s['direction']   = int(row['direction'])
                s['paper_title'] = title
                s['year']        = int(row['year'])
                s['stars']       = str(s.get('stars') or '')
            cache.write_text(json.dumps(all_stats, indent=2))
            print(f'  ✓ SAVED {len(all_stats)} stats')
            improved += 1
        else:
            print(f'  ✗ 0 stats found')

    # Rebuild master parquet
    all_data = []
    for _, row in analysis[analysis['source'] == 'nber'].iterrows():
        c = cache_dir / f"{row['nber_wp_number']}.json"
        if c.exists():
            data = json.loads(c.read_text())
            for s in data:
                s.setdefault('wp_number',   row['nber_wp_number'])
                s.setdefault('direction',   int(row['direction']))
                s.setdefault('paper_title', row['title'])
                s.setdefault('year',        int(row['year']))
                s.setdefault('positive',    bool(row['positive']))
                s.setdefault('negative',    bool(row['negative']))
                s['stars'] = str(s.get('stars') or '')
            all_data.extend(data)

    result = pd.DataFrame(all_data)
    result['stars'] = result['stars'].astype(str).replace({'None': '', 'nan': ''})
    result.to_parquet('data/final/nber_tstats_all.parquet', index=False)
    result.to_csv('output/tables/nber_tstats_all.csv', index=False)
    print(f'\nTotal NBER stats: {len(result):,} from {result["wp_number"].nunique()} papers')
    print(f'Papers with 0 stats: {157 - result["wp_number"].nunique()}')
    print(f'Recovered this run: {improved}')


if __name__ == '__main__':
    main()
