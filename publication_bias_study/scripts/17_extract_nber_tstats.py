"""
Script 17: Extract test statistics from NBER working paper PDFs.
Same LLM pipeline as script 13, but for the 157 in-scope NBER papers.
Output: data/final/nber_tstats_all.parquet
"""

import pandas as pd
import pdfplumber
import anthropic
import json
import time
import re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

EXTRACTION_PROMPT = """You are extracting ALL test statistics from a finance research paper.

Below is the text from a paper titled: "{title}" (NBER Working Paper {wp_number}, {year})

Your task: Find EVERY regression table (Table 1, Table 2, etc.) and extract ALL
coefficient + standard error pairs reported. These appear as:
  - A coefficient value (e.g., 0.032, -0.150, 2.3%)
  - A standard error in parentheses below it (e.g., (0.014)) — values near 0
  - OR a t-statistic in parentheses (e.g., (2.29), (−3.12)) — values typically 1-5
  - OR a t-statistic in brackets [2.29]
  - Stars (*, **, ***) indicating significance
  - Note: if parenthesized values are large (>1), they are t-statistics, not SEs. Compute z=|t-stat| directly.

ONLY include test statistics from regression/estimation tables (not summary statistics tables).
ONLY include coefficient-on-variable rows (not intercepts or fixed effect notes).
For each statistic compute: z = |coefficient / std_error| or use the t-stat directly.

Return a JSON array of objects, one per test statistic:
[
  {{
    "table": "Table 2",
    "variable": "Log(assets)",
    "coef": 0.032,
    "se": 0.014,
    "z_abs": 2.286,
    "stars": "**",
    "spec": "Col (1)"
  }},
  ...
]

If you cannot extract specific SEs but see significance stars, estimate:
  * → z ≈ 1.8, ** → z ≈ 2.3, *** → z ≈ 3.0 (use these only as fallback, mark estimated=true)

Return ONLY the JSON array, no other text. If no regression tables found, return [].

PAPER TEXT (focused on regression tables):
{text}"""

client = anthropic.Anthropic()


def _is_reversed_text(text):
    sample = text[:3000]
    reversed_hits = len(re.findall(r'elbaT|noisserg|tneicifeoC|lavretni', sample))
    forward_hits  = len(re.findall(r'Table|regress|Coefficient|interval', sample))
    return reversed_hits > forward_hits

def _fix_reversed(text):
    return '\n'.join(line[::-1] for line in text.split('\n'))

def extract_pdf_text(pdf_path, max_chars=25000):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = [page.extract_text() or '' for page in pdf.pages]
        full_text = '\n'.join(pages_text)

        if len(full_text) < 500:
            return full_text

        if _is_reversed_text(full_text):
            pages_text = [_fix_reversed(t) for t in pages_text]
            full_text  = '\n'.join(pages_text)

        num_in_paren = re.compile(r'\(\s*[-−]?\d+\.\d{1,4}\s*\)')
        sig_pattern  = re.compile(r'[∗\*†]{1,3}')

        page_scores = []
        for pt in pages_text:
            lines = pt.split('\n')
            score = 0
            for line in lines:
                p = len(num_in_paren.findall(line))
                s = len(sig_pattern.findall(line))
                if p >= 2:
                    score += p * 3
                score += p + s
            page_scores.append(score)

        best_page = max(range(len(page_scores)), key=lambda i: page_scores[i])

        if page_scores[best_page] > 0:
            start = max(0, best_page - 4)
            end   = min(len(pages_text), best_page + 8)
        else:
            block = 8
            best_block_start = 0
            best_block_score = -1
            for i in range(max(1, len(page_scores) - block + 1)):
                s = sum(page_scores[i:i + block])
                if s > best_block_score:
                    best_block_score = s
                    best_block_start = i
            start = max(0, best_block_start - 2)
            end   = min(len(pages_text), best_block_start + block + 2)

        return '\n'.join(pages_text[start:end])[:max_chars]
    except Exception as e:
        print(f'    PDF extract error: {e}')
        return None


def extract_tstats_llm(text, title, wp_number, year):
    prompt = EXTRACTION_PROMPT.format(
        title=title, wp_number=wp_number, year=year, text=text
    )
    try:
        msg = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=8000,
            messages=[{'role': 'user', 'content': prompt}]
        )
        raw = msg.content[0].text.strip()
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        bracket_start = raw.find('[')
        if bracket_start == -1:
            return []
        raw = raw[bracket_start:]
        try:
            bracket_end = raw.rfind(']')
            if bracket_end != -1:
                return json.loads(raw[:bracket_end + 1])
        except json.JSONDecodeError:
            pass
        last_brace = raw.rfind('},')
        if last_brace == -1:
            last_brace = raw.rfind('}')
        if last_brace != -1:
            try:
                return json.loads(raw[:last_brace + 1] + ']')
            except json.JSONDecodeError:
                pass
        return []
    except Exception as e:
        print(f'    LLM error: {e}')
        return []


def main():
    df = pd.read_parquet('data/final/analysis_dataset.parquet')
    nber = df[df['source'] == 'nber'].copy().reset_index(drop=True)
    print(f'Extracting test statistics from {len(nber)} NBER papers...')

    pdf_dir  = Path('data/raw/nber_pdfs')
    cache_dir = Path('data/raw/nber_tstats')
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    zero_stat_papers = []

    for i, row in nber.iterrows():
        wp    = row['nber_wp_number']
        title = row['title']
        pdf_path = pdf_dir / f'{wp}.pdf'
        cache_file = cache_dir / f'{wp}.json'

        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            all_stats.extend(cached)
            print(f'  [{i+1:3d}/{len(nber)}] CACHED  {len(cached):3d} stats  {wp}  {title[:45]}')
            if not cached:
                zero_stat_papers.append(wp)
            continue

        if not pdf_path.exists():
            print(f'  [{i+1:3d}/{len(nber)}] NO PDF  {wp}  {title[:45]}')
            cache_file.write_text('[]')
            zero_stat_papers.append(wp)
            continue

        text = extract_pdf_text(pdf_path)
        if not text:
            print(f'  [{i+1:3d}/{len(nber)}] PDF ERR {wp}  {title[:45]}')
            cache_file.write_text('[]')
            zero_stat_papers.append(wp)
            continue

        stats = extract_tstats_llm(text, title, wp, int(row['year']))

        for s in stats:
            s['paper_title']    = title
            s['wp_number']      = wp
            s['year']           = int(row['year'])
            s['direction']      = int(row['direction'])
            s['positive']       = bool(row['positive'])
            s['negative']       = bool(row['negative'])
            s['published_top3'] = bool(row['published_top3'])

        cache_file.write_text(json.dumps(stats, indent=2))
        all_stats.extend(stats)

        if not stats:
            zero_stat_papers.append(wp)
        print(f'  [{i+1:3d}/{len(nber)}] {"✓" if stats else "0"}  {len(stats):3d} stats  {wp}  {title[:45]}')
        time.sleep(0.5)

    if all_stats:
        result = pd.DataFrame(all_stats)
        result.to_parquet('data/final/nber_tstats_all.parquet', index=False)
        result.to_csv('output/tables/nber_tstats_all.csv', index=False)
        print(f'\nTotal NBER test statistics: {len(result)}')
        print(f'Papers with stats: {result["wp_number"].nunique()}')
        print(f'Papers with 0 stats: {len(zero_stat_papers)} → {zero_stat_papers}')
    else:
        print('No statistics extracted.')

if __name__ == '__main__':
    main()
