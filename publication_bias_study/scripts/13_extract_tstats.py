"""
Script 13: Extract ALL test statistics from each paper's PDF.
For each regression table in the paper, extract coefficients + SEs → compute |z|.
Uses pdfplumber for text extraction + Claude for structured parsing.
"""

import pandas as pd
import pdfplumber
import anthropic
import json
import time
import re
from pathlib import Path

EXTRACTION_PROMPT = """You are extracting ALL test statistics from a finance research paper.

Below is the text from a paper titled: "{title}" ({journal}, {year})

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

PAPER TEXT (first 12000 characters of main body):
{text}"""

client = anthropic.Anthropic()

def _is_reversed_text(text):
    """Detect PDFs where pdfplumber reads text right-to-left (mirrored lines)."""
    sample = text[:3000]
    # Count reversed table/regression keywords
    reversed_hits = len(re.findall(r'elbaT|noisserg|tneicifeoC|lavretni', sample))
    forward_hits  = len(re.findall(r'Table|regress|Coefficient|interval', sample))
    return reversed_hits > forward_hits

def _fix_reversed(text):
    """Reverse each line to correct mirrored PDF text."""
    return '\n'.join(line[::-1] for line in text.split('\n'))

def extract_pdf_text(pdf_path, max_chars=25000):
    """Extract text focused on regression tables.
    Handles: (1) reversed/mirrored text PDFs, (2) multi-column layout,
    (3) math appendices that score higher than tables.
    Strategy: score pages by lines with ≥2 decimal-parens (= SE/t-stat rows),
    then return a window centred on the best pages."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = [page.extract_text() or '' for page in pdf.pages]
        full_text = '\n'.join(pages_text)

        if len(full_text) < 500:
            return full_text

        # Detect and fix reversed text (some Wiley PDFs)
        if _is_reversed_text(full_text):
            pages_text = [_fix_reversed(t) for t in pages_text]
            full_text  = '\n'.join(pages_text)

        # Require decimal point → excludes year citations like (1994), keeps (2.47), (0.043)
        num_in_paren = re.compile(r'\(\s*[-−]?\d+\.\d{1,4}\s*\)')
        sig_pattern  = re.compile(r'[∗\*†]{1,3}')

        # Score each PAGE by regression-table signals
        page_scores = []
        for pt in pages_text:
            lines = pt.split('\n')
            score = 0
            for line in lines:
                p = len(num_in_paren.findall(line))
                s = len(sig_pattern.findall(line))
                # ≥2 decimal-parens on same line = strong regression row signal
                if p >= 2:
                    score += p * 3
                score += p + s
            page_scores.append(score)

        # Find the single highest-scoring page, then expand around it
        best_page = max(range(len(page_scores)), key=lambda i: page_scores[i])

        # If best page has a non-trivial score, centre window on it;
        # otherwise fall back to a contiguous block scan
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

        extracted = '\n'.join(pages_text[start:end])

        return extracted[:max_chars]
    except Exception as e:
        print(f"    PDF extract error: {e}")
        return None

def extract_tstats_llm(text, title, journal, year):
    prompt = EXTRACTION_PROMPT.format(
        title=title, journal=journal, year=year, text=text
    )
    try:
        msg = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=8000,
            messages=[{'role': 'user', 'content': prompt}]
        )
        raw = msg.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        # Extract just the JSON array
        bracket_start = raw.find('[')
        if bracket_start == -1:
            return []
        raw = raw[bracket_start:]
        # Try clean parse first
        try:
            bracket_end = raw.rfind(']')
            if bracket_end != -1:
                return json.loads(raw[:bracket_end + 1])
        except json.JSONDecodeError:
            pass
        # Recovery: find all complete objects (last complete '}' before truncation)
        last_brace = raw.rfind('},')
        if last_brace == -1:
            last_brace = raw.rfind('}')
        if last_brace != -1:
            try:
                return json.loads(raw[:last_brace + 1] + ']')
            except json.JSONDecodeError:
                pass
        return []
    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}")
        return []
    except Exception as e:
        print(f"    LLM error: {e}")
        return []

def main():
    df = pd.read_parquet('data/raw/published_with_pdfs.parquet')
    has_pdf = df[df['local_pdf'].notna()].copy().reset_index(drop=True)
    print(f"Extracting test statistics from {len(has_pdf)} PDFs...")

    out_dir = Path('data/raw/tstats')
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    failed = []

    for i, row in has_pdf.iterrows():
        title = row['title']
        out_file = out_dir / f"{row['journal']}_{int(row['year'])}_{i:03d}.json"

        if out_file.exists():
            cached = json.loads(out_file.read_text())
            all_stats.extend(cached)
            print(f"  [{i+1:3d}/{len(has_pdf)}] CACHED — {len(cached):3d} stats | {title[:55]}")
            continue

        text = extract_pdf_text(row['local_pdf'])
        if not text:
            failed.append(title)
            print(f"  [{i+1:3d}/{len(has_pdf)}] ✗ PDF read failed | {title[:55]}")
            continue

        stats = extract_tstats_llm(text, title, row['journal'], int(row['year']))

        # Tag each stat with paper metadata
        for s in stats:
            s['paper_title'] = title
            s['journal'] = row['journal']
            s['year'] = int(row['year'])
            s['direction'] = int(row['direction'])
            s['positive'] = bool(row['positive'])
            s['negative'] = bool(row['negative'])

        out_file.write_text(json.dumps(stats, indent=2))
        all_stats.extend(stats)

        print(f"  [{i+1:3d}/{len(has_pdf)}] ✓ {len(stats):3d} stats | {title[:55]}")
        time.sleep(0.5)

    # Save master file
    if all_stats:
        result = pd.DataFrame(all_stats)
        result.to_parquet('data/final/tstats_all.parquet', index=False)
        result.to_csv('output/tables/tstats_all.csv', index=False)
        print(f"\nTotal test statistics: {len(result)}")
        print(f"Papers covered: {result['paper_title'].nunique()}")
        print(f"Failed: {len(failed)}")
        print("Saved → data/final/tstats_all.parquet")
    else:
        print("No statistics extracted.")

if __name__ == '__main__':
    main()
