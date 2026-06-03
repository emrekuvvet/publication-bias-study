"""
Script 22: Extract bracket-format [t-stat] test statistics from NBER papers.
Handles papers where results appear as: coef  [t-stat] rather than coef (SE).
"""

import fitz
import anthropic
import json
import re
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

PROMPT = """Extract ALL test statistics from this finance paper.
The table format is: coefficient on one line, then [t-statistic] in brackets on the next line.
Example: Market   0.56   [18.91] means coef=0.56 with t-stat=18.91, so z_abs=18.91.
Use z = abs(t-statistic) directly.
ONLY regression tables, not summary stats or parameterization tables. Skip intercepts.
Return JSON array with fields: table, variable, coef, se, z_abs, stars, spec
Return ONLY the JSON array.

TEXT:
{text}"""

bracket_pat = re.compile(r'\[\s*[-−]?\d+\.\d{1,4}\s*\]')


def main():
    analysis  = pd.read_parquet('data/final/analysis_dataset.parquet')
    nber_df   = analysis[analysis['source'] == 'nber'].set_index('nber_wp_number')
    cache_dir = Path('data/raw/nber_tstats')
    pdf_dir   = Path('data/raw/nber_pdfs')
    client    = anthropic.Anthropic()

    # Find zero-stat NBER papers that have bracket patterns
    targets = []
    for wp in nber_df.index:
        cache = cache_dir / f'{wp}.json'
        if cache.exists() and len(json.loads(cache.read_text())) > 0:
            continue
        pdf = pdf_dir / f'{wp}.pdf'
        if not pdf.exists():
            continue
        try:
            doc = fitz.open(pdf)
            full = ' '.join(doc[i].get_text() for i in range(len(doc)))
            doc.close()
            n_brackets = len(bracket_pat.findall(full))
            if n_brackets >= 10:
                targets.append((wp, n_brackets))
        except Exception:
            pass

    print(f'Papers with bracket t-stats: {len(targets)}')
    for wp, nb in sorted(targets, key=lambda x: -x[1]):
        print(f'  {wp}: {nb} brackets  {nber_df.loc[wp, "title"][:55]}')

    print()
    improved = 0

    for wp, _ in targets:
        row   = nber_df.loc[wp]
        title = row['title']
        pdf   = pdf_dir / f'{wp}.pdf'
        print(f'Processing {wp}: {title[:55]}')

        doc    = fitz.open(pdf)
        pages  = [doc[i].get_text() for i in range(len(doc))]
        doc.close()
        scores = [len(bracket_pat.findall(p)) for p in pages]

        all_stats = []
        seen = set()
        chunk, chunk_score = '', 0

        for i, (pt, sc) in enumerate(zip(pages, scores)):
            if sc > 0:
                chunk += pt + '\n'
                chunk_score += sc
            if chunk_score > 30 or (i == len(pages) - 1 and chunk):
                text = chunk[:28000]
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
                    stats = json.loads(raw[bi:raw.rfind(']') + 1]) if bi >= 0 else []
                except Exception:
                    stats = []
                for s in stats:
                    key = (s.get('table', ''), s.get('variable', ''),
                           round(s.get('z_abs') or 0, 2))
                    if key not in seen:
                        seen.add(key)
                        all_stats.append(s)
                chunk, chunk_score = '', 0
                time.sleep(0.4)

        if all_stats:
            for s in all_stats:
                s.update({
                    'wp_number':   wp,
                    'direction':   int(row['direction']),
                    'paper_title': title,
                    'year':        int(row['year']),
                    'stars':       str(s.get('stars', ''))
                })
            (cache_dir / f'{wp}.json').write_text(json.dumps(all_stats, indent=2))
            print(f'  ✓ SAVED {len(all_stats)} stats')
            improved += 1
        else:
            print(f'  ✗ still 0')

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

    if all_data:
        result = pd.DataFrame(all_data)
        result['stars'] = result['stars'].astype(str).replace({'None': '', 'nan': ''})
        result.to_parquet('data/final/nber_tstats_all.parquet', index=False)
        result.to_csv('output/tables/nber_tstats_all.csv', index=False)
        print(f'\nTotal NBER stats: {len(result):,} from {result["wp_number"].nunique()} papers')
        print(f'Papers with 0 stats: {157 - result["wp_number"].nunique()}')
        print(f'Recovered this run: {improved}')


if __name__ == '__main__':
    main()
