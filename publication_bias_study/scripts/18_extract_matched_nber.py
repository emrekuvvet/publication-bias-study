"""
Script 18: Extract test statistics from the 19 NBER working papers that are
matched counterparts of published papers (for before/after comparison).
"""

import pandas as pd
import pdfplumber
import anthropic
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

PROMPT = """Extract ALL test statistics from regression tables in this NBER working paper.
Return a JSON array of objects with keys: table, variable, coef, se, z_abs, stars, spec.
If parenthesized values >1 they are t-stats not SEs. Only regression tables, not summary stats.
Return ONLY the JSON array. If none found return [].

TEXT:
{text}"""

num_pat = re.compile(r'\(\s*[-−]?\d+\.\d{1,4}\s*\)')
sig_pat = re.compile(r'[*∗†]{1,3}')


def get_text(pdf_path, max_chars=25000):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [p.extract_text() or '' for p in pdf.pages]
        scores = []
        for pt in pages:
            lines = pt.split('\n')
            s = 0
            for l in lines:
                p = len(num_pat.findall(l))
                if p >= 2:
                    s += p * 3
                s += p + len(sig_pat.findall(l))
            scores.append(s)
        best = max(range(len(scores)), key=lambda i: scores[i])
        start, end = max(0, best - 4), min(len(pages), best + 8)
        return '\n'.join(pages[start:end])[:max_chars]
    except Exception as e:
        print(f'  PDF error: {e}')
        return None


def main():
    pub_papers = pd.read_parquet('data/raw/published_with_pdfs.parquet')
    matched = pub_papers[pub_papers['nber_wp_number'].notna()][
        ['title', 'nber_wp_number', 'direction']
    ].copy()
    print(f'Extracting stats from {len(matched)} matched NBER working papers...')

    client = anthropic.Anthropic()
    cache_dir = Path('data/raw/nber_tstats')
    all_stats = []

    for _, row in matched.iterrows():
        wp = row['nber_wp_number']
        cache = cache_dir / f'{wp}.json'

        if cache.exists():
            data = json.loads(cache.read_text())
            print(f'  CACHED {wp}: {len(data):3d} stats  {row["title"][:50]}')
            for s in data:
                s.setdefault('wp_number', wp)
                s.setdefault('direction', int(row['direction']))
                s.setdefault('paper_title', row['title'])
            all_stats.extend(data)
            continue

        pdf_path = f'data/raw/nber_pdfs/{wp}.pdf'
        text = get_text(pdf_path)
        if not text:
            print(f'  NO TEXT {wp}')
            cache.write_text('[]')
            continue

        msg = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=8000,
            messages=[{'role': 'user', 'content': PROMPT.format(text=text)}]
        )
        raw = msg.content[0].text.strip()
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        bi = raw.find('[')
        try:
            stats_list = json.loads(raw[bi:raw.rfind(']') + 1]) if bi >= 0 else []
        except Exception:
            stats_list = []

        for s in stats_list:
            s['wp_number'] = wp
            s['direction'] = int(row['direction'])
            s['paper_title'] = row['title']

        cache.write_text(json.dumps(stats_list, indent=2))
        all_stats.extend(stats_list)
        print(f'  OK    {wp}: {len(stats_list):3d} stats  {row["title"][:50]}')
        time.sleep(0.5)

    result = pd.DataFrame(all_stats) if all_stats else pd.DataFrame()
    result.to_parquet('data/final/nber_matched_tstats.parquet', index=False)
    result.to_csv('output/tables/nber_matched_tstats.csv', index=False)
    print(f'\nTotal: {len(result)} stats from {result["wp_number"].nunique() if len(result) else 0} papers')


if __name__ == '__main__':
    main()
