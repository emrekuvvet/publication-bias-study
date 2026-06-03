"""
Script 21: OCR-based extraction for NBER papers with 0 stats but empirical content.
Converts PDF pages to images, runs tesseract, then passes text to LLM.
Targets papers with 0 paren-decimals in text extraction but high empirical keyword count.
"""

import pandas as pd
import anthropic
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

PROMPT = """Extract ALL test statistics from regression/estimation tables in this finance paper.
Return a JSON array with keys: table, variable, coef, se, z_abs, stars, spec.
Rules:
- Parenthesized values >1 are t-stats (use z=|t| directly); values <1 are SEs (compute z=|coef/se|)
- ONLY regression/estimation tables, NOT summary statistics or means tables
- ONLY variable rows, not intercepts or FE notes
- If you see stars but no SE: *→1.8, **→2.3, ***→3.0, mark estimated=true
Return ONLY the JSON array. If none found return [].

TEXT (OCR-extracted):
{text}"""

num_pat = re.compile(r'\(\s*[-−]?\d+\.\d{1,4}\s*\)')
emp_kw  = re.compile(r'regression|OLS|coefficient|Table [0-9]|estimate|standard error|t-stat|robust', re.I)
theory_kw = re.compile(r'Proposition|Lemma|Proof\b|Theorem\b', re.I)

# Papers to attempt OCR on: 0 paren-decimals but high empirical keywords
# (identified from script 20 diagnosis)
OCR_TARGETS = [
    'w19018',  # Do Strict Capital Requirements Raise the Cost of Capital — 133 emp
    'w24980',  # Setting with the Sun — 140 emp
    'w26710',  # Countercyclical Capital Buffers — 147 emp
    'w29441',  # Macroprudential Policies and Covid-19 — 82 emp
    'w28852',  # Beyond Health — 61 emp
    'w18821',  # Did Housing Policies Cause the Postwar Boom — 48 emp
    'w19951',  # Trapped Factors and China — 27 emp
    'w27733',  # Payments Crises and Consequences — 97 emp
    'w10475',  # Bank Supervision — 76 emp, 15 parens (partial)
    'w16275',  # Automatic Stabilizers — 37 emp, 3 parens (partial)
    'w18582',  # Fettered Consumers — 24 emp, 4 parens (partial)
    'w15281',  # Means-Tested Mortgage Modification — dir=-1
    'w17149',  # Too-Systemic-To-Fail — dir=-1
    'w19018',  # already listed
    'w9385',   # Paying for the FILP — 95 emp
    'w11353',  # International Borrowing — 19 emp
    'w15177',  # Tax-Loss Carryback — dir=+1
    'w16008',  # Build America Bonds — dir=+1
    'w25319',  # Money Markets — 16 emp, 34 theory
    'w21626',  # Phasing Out the GSEs — 35 emp
    'w24706',  # Government Guarantees — 73 emp
]
OCR_TARGETS = list(dict.fromkeys(OCR_TARGETS))  # deduplicate, preserve order


def ocr_pdf(pdf_path, dpi=200, max_pages=60):
    """Convert PDF pages to images and OCR them. Returns list of page texts."""
    try:
        pages = convert_from_path(str(pdf_path), dpi=dpi, first_page=1,
                                   last_page=max_pages)
        texts = []
        for img in pages:
            # Resize to speed up OCR while keeping legible
            w, h = img.size
            if w > 1800:
                img = img.resize((1800, int(h * 1800 / w)), Image.LANCZOS)
            t = pytesseract.image_to_string(img, config='--psm 6')
            texts.append(t)
        return texts
    except Exception as e:
        print(f'    OCR error: {e}')
        return []


def score_pages(pages_text):
    """Score pages by regression-table density."""
    scores = []
    for pt in pages_text:
        lines = pt.split('\n')
        score = 0
        for l in lines:
            p = len(num_pat.findall(l))
            if p >= 2:
                score += p * 3
            score += p
        scores.append(score)
    return scores


def call_llm(text, client):
    if not text or len(text.strip()) < 100:
        return []
    try:
        msg = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=8000,
            messages=[{'role': 'user', 'content': PROMPT.format(text=text)}]
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
        print(f'    LLM error: {e}')
        return []


def main():
    analysis  = pd.read_parquet('data/final/analysis_dataset.parquet')
    nber_df   = analysis[analysis['source'] == 'nber'].set_index('nber_wp_number')
    cache_dir = Path('data/raw/nber_tstats')
    pdf_dir   = Path('data/raw/nber_pdfs')

    print(f'Running OCR on {len(OCR_TARGETS)} target papers...\n')
    client   = anthropic.Anthropic()
    improved = 0

    for wp in OCR_TARGETS:
        cache = cache_dir / f'{wp}.json'
        if cache.exists() and len(json.loads(cache.read_text())) > 0:
            print(f'  SKIP {wp} — already has stats')
            continue

        pdf = pdf_dir / f'{wp}.pdf'
        if not pdf.exists():
            print(f'  NO PDF {wp}')
            continue

        if wp not in nber_df.index:
            print(f'  NOT IN DATASET {wp}')
            continue

        row   = nber_df.loc[wp]
        title = row['title']
        kb    = pdf.stat().st_size // 1024
        print(f'  OCR {wp} ({kb}KB)  {title[:55]}')

        pages_text = ocr_pdf(pdf)
        if not pages_text:
            print(f'    → OCR produced no output')
            continue

        scores = score_pages(pages_text)
        best   = max(range(len(scores)), key=lambda i: scores[i])
        total_parens = sum(len(num_pat.findall(t)) for t in pages_text)
        print(f'    OCR: {len(pages_text)} pages, {total_parens} parens, best page {best+1} (score={scores[best]})')

        stats = []

        if scores[best] > 0:
            # Focus on the best window
            start = max(0, best - 4)
            end   = min(len(pages_text), best + 10)
            text  = '\n'.join(pages_text[start:end])[:28000]
            stats = call_llm(text, client)
            if stats:
                print(f'    → window [{start+1}:{end}]: {len(stats)} stats')
            time.sleep(0.4)

        # If still empty, try chunked full scan
        if not stats:
            full = '\n'.join(pages_text)
            chunk_size = 25000
            seen = set()
            for i in range(0, max(1, len(full) - 5000), chunk_size - 3000):
                chunk = full[i:i + chunk_size]
                if len(num_pat.findall(chunk)) < 2:
                    continue
                cs = call_llm(chunk, client)
                for s in cs:
                    key = (s.get('table', ''), s.get('variable', ''),
                           round(s.get('z_abs') or 0, 2))
                    if key not in seen:
                        seen.add(key)
                        stats.append(s)
                if cs:
                    print(f'    → chunk: +{len(cs)} stats')
                time.sleep(0.4)

        if stats:
            for s in stats:
                s['wp_number']   = wp
                s['direction']   = int(row['direction'])
                s['paper_title'] = title
                s['year']        = int(row['year'])
                if 'stars' in s:
                    s['stars'] = str(s['stars']) if s['stars'] is not None else ''
            cache.write_text(json.dumps(stats, indent=2))
            print(f'    ✓ SAVED {len(stats)} stats')
            improved += 1
        else:
            print(f'    ✗ still 0 stats after OCR')

    # Rebuild master parquet
    all_stats = []
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
                if 'stars' in s:
                    s['stars'] = str(s['stars']) if s['stars'] is not None else ''
            all_stats.extend(data)

    result = pd.DataFrame(all_stats)
    result['stars'] = result['stars'].astype(str).replace({'None': '', 'nan': ''})
    result.to_parquet('data/final/nber_tstats_all.parquet', index=False)
    result.to_csv('output/tables/nber_tstats_all.csv', index=False)
    print(f'\nTotal NBER stats: {len(result):,} from {result["wp_number"].nunique()} papers')
    print(f'Papers with 0 stats: {157 - result["wp_number"].nunique()}')
    print(f'Papers recovered this run: {improved}')


if __name__ == '__main__':
    main()
