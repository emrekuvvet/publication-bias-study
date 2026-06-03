"""
Script 20: Retry zero-stat NBER papers with multiple fallback strategies.
  - Strategy A: pymupdf (fitz) for pdfplumber-error papers
  - Strategy B: full-document chunked extraction for large/skipped papers
  - Strategy C: aggressive page sweep for papers with empirical keywords but 0 parens
"""

import pandas as pd
import anthropic
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

try:
    import fitz  # pymupdf
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("WARNING: pymupdf not available, install with: pip3 install pymupdf --break-system-packages")

import pdfplumber

PROMPT = """Extract ALL test statistics from regression/estimation tables in this finance paper.
Return a JSON array with keys: table, variable, coef, se, z_abs, stars, spec.
Rules:
- Parenthesized values >1 are t-stats (use z=|t| directly); values <1 are SEs (compute z=|coef/se|)
- ONLY regression tables, NOT summary statistics tables
- ONLY variable rows, not intercepts or FE notes
- If you see stars but no numbers, estimate: *→1.8, **→2.3, ***→3.0 and mark estimated=true
Return ONLY the JSON array. If none found return [].

TEXT:
{text}"""

num_pat = re.compile(r'\(\s*[-−]?\d+\.\d{1,4}\s*\)')


def extract_fitz(pdf_path, max_chars=30000):
    """Extract text using pymupdf — handles more PDF variants than pdfplumber."""
    if not HAS_FITZ:
        return None
    try:
        doc = fitz.open(pdf_path)
        pages_text = [doc[i].get_text() for i in range(len(doc))]
        doc.close()

        # Score pages
        page_scores = []
        for pt in pages_text:
            lines = pt.split('\n')
            score = 0
            for l in lines:
                p = len(num_pat.findall(l))
                if p >= 2:
                    score += p * 3
                score += p
            page_scores.append(score)

        best = max(range(len(page_scores)), key=lambda i: page_scores[i])
        if page_scores[best] > 0:
            start, end = max(0, best - 4), min(len(pages_text), best + 10)
        else:
            start, end = 0, min(len(pages_text), 15)

        return '\n'.join(pages_text[start:end])[:max_chars]
    except Exception as e:
        print(f'    fitz error: {e}')
        return None


def extract_pdfplumber_full(pdf_path, max_chars=30000):
    """Extract full document text with wider window."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = [p.extract_text() or '' for p in pdf.pages]

        page_scores = []
        for pt in pages_text:
            lines = pt.split('\n')
            score = 0
            for l in lines:
                p = len(num_pat.findall(l))
                if p >= 2:
                    score += p * 3
                score += p
            page_scores.append(score)

        best = max(range(len(page_scores)), key=lambda i: page_scores[i])
        start, end = max(0, best - 6), min(len(pages_text), best + 12)
        return '\n'.join(pages_text[start:end])[:max_chars]
    except Exception as e:
        print(f'    pdfplumber error: {e}')
        return None


def extract_chunks(pdf_path, chunk_chars=20000):
    """Split full document into overlapping chunks and return all."""
    try:
        if HAS_FITZ:
            doc = fitz.open(pdf_path)
            full = '\n'.join(doc[i].get_text() for i in range(len(doc)))
            doc.close()
        else:
            with pdfplumber.open(pdf_path) as pdf:
                full = '\n'.join(p.extract_text() or '' for p in pdf.pages)
        # Return chunks with 2000-char overlap
        chunks = []
        step = chunk_chars - 2000
        for i in range(0, max(1, len(full) - 2000), step):
            chunks.append(full[i:i + chunk_chars])
        return chunks
    except Exception as e:
        print(f'    chunk extract error: {e}')
        return []


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
    nber_df   = analysis[analysis['source'] == 'nber'].reset_index(drop=True)
    cache_dir = Path('data/raw/nber_tstats')
    pdf_dir   = Path('data/raw/nber_pdfs')

    # Find all zero-stat papers
    zero_rows = []
    for _, row in nber_df.iterrows():
        cache = cache_dir / f"{row['nber_wp_number']}.json"
        if cache.exists():
            if len(json.loads(cache.read_text())) == 0:
                zero_rows.append(row)
        else:
            zero_rows.append(row)

    print(f"Retrying {len(zero_rows)} zero-stat NBER papers...\n")
    client = anthropic.Anthropic()
    improved = 0

    for row in zero_rows:
        wp    = row['nber_wp_number']
        title = row['title']
        pdf   = pdf_dir / f'{wp}.pdf'
        cache = cache_dir / f'{wp}.json'

        if not pdf.exists():
            print(f'  NO PDF  {wp}  {title[:50]}')
            continue

        kb = pdf.stat().st_size // 1024
        print(f'  {wp} ({kb}KB)  {title[:50]}')

        stats = []

        # Strategy A: try fitz first (handles encoding errors)
        if HAS_FITZ:
            text = extract_fitz(pdf)
            if text and len(num_pat.findall(text)) > 0:
                stats = call_llm(text, client)
                if stats:
                    print(f'    → fitz: {len(stats)} stats')

        # Strategy B: pdfplumber with wider window
        if not stats:
            text = extract_pdfplumber_full(pdf)
            if text and len(num_pat.findall(text)) > 0:
                stats = call_llm(text, client)
                if stats:
                    print(f'    → pdfplumber-wide: {len(stats)} stats')

        # Strategy C: chunked full-document scan (for large PDFs or missed tables)
        if not stats and kb > 200:
            chunks = extract_chunks(pdf)
            for j, chunk in enumerate(chunks):
                if len(num_pat.findall(chunk)) < 2:
                    continue
                chunk_stats = call_llm(chunk, client)
                if chunk_stats:
                    stats.extend(chunk_stats)
                    print(f'    → chunk {j+1}/{len(chunks)}: +{len(chunk_stats)} stats')
                time.sleep(0.3)
            # Deduplicate by (table, variable, z_abs)
            seen = set()
            deduped = []
            for s in stats:
                key = (s.get('table',''), s.get('variable',''), round(s.get('z_abs') or 0, 2))
                if key not in seen:
                    seen.add(key)
                    deduped.append(s)
            stats = deduped

        if stats:
            for s in stats:
                s['wp_number']  = wp
                s['direction']  = int(row['direction'])
                s['paper_title'] = title
                s['year']       = int(row['year'])
            cache.write_text(json.dumps(stats, indent=2))
            print(f'    ✓ SAVED {len(stats)} stats')
            improved += 1
        else:
            print(f'    ✗ still 0 stats')

        time.sleep(0.5)

    # Rebuild master parquet
    all_stats = []
    for _, row in nber_df.iterrows():
        cache = cache_dir / f"{row['nber_wp_number']}.json"
        if cache.exists():
            data = json.loads(cache.read_text())
            for s in data:
                s.setdefault('wp_number', row['nber_wp_number'])
                s.setdefault('direction', int(row['direction']))
                s.setdefault('paper_title', row['title'])
                s.setdefault('year', int(row['year']))
            all_stats.extend(data)

    if all_stats:
        result = pd.DataFrame(all_stats)
        # Normalise mixed-type columns before saving
        result['stars'] = result['stars'].astype(str).replace({'None': '', 'nan': ''})
        result.to_parquet('data/final/nber_tstats_all.parquet', index=False)
        result.to_csv('output/tables/nber_tstats_all.csv', index=False)
        print(f'\nTotal NBER stats: {len(result):,} from {result["wp_number"].nunique()} papers')
        print(f'Papers with 0 stats: {157 - result["wp_number"].nunique()}')
        print(f'Papers improved this run: {improved}')
    else:
        print('No stats at all.')


if __name__ == '__main__':
    main()
