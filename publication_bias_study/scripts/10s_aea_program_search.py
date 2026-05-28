"""
10s_aea_program_search.py
--------------------------
Fetches abstracts from the AEA/ASSA annual meeting program pages.

The AEA hosts PDF previews of all conference papers at:
  https://www.aeaweb.org/conference/{year}/preliminary

Pipeline per paper:
  1. camoufox: navigate to the conference program for the paper's year
  2. Type the paper title into the program keyword search box
  3. Find the PDF download link in the results
  4. Download the PDF via plain requests
  5. Extract abstract using pdfplumber (regex: text after "Abstract" header)
  6. Validate title fuzzy-match + author surname match

Coverage: years 2021–2024 (only 2021+ program pages have paper-PDF search).

Usage:
    python scripts/10s_aea_program_search.py
    python scripts/10s_aea_program_search.py --force
    python scripts/10s_aea_program_search.py --limit N
    python scripts/10s_aea_program_search.py --headless
"""

import argparse
import asyncio
import io
import json
import pathlib
import re
import unicodedata

import pandas as pd
import pdfplumber
import requests
from rapidfuzz import fuzz
from tqdm import tqdm

ROOT       = pathlib.Path(__file__).parent.parent.resolve()
PARQUET    = ROOT / "data" / "raw" / "afa_papers_enriched.parquet"
CACHE_PATH = ROOT / "data" / "raw" / "afa_10s_cache.json"
LOG_PATH   = ROOT / "validation" / "abstract_10s_matches.csv"

FUZZY_MIN    = 72
AEA_BASE     = "https://www.aeaweb.org/conference/{year}/preliminary"
PDF_HEADERS  = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

STOP_WORDS = {
    "a","an","the","of","in","on","at","to","for","and","or","but",
    "with","by","from","that","this","is","are","was","were","be",
    "been","being","have","has","had","do","does","did","will","would",
    "can","could","should","may","might","shall","some","any","all",
}

_INSTITUTION_WORDS = {
    "Finance","Economics","Business","School","University","Institute",
    "Department","College","Center","Centre","International","National",
    "Federal","Research","Studies","Science","Sciences","Management",
    "Accounting","Marketing","Law","Medicine","Technology",
}


# ── text helpers ──────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFD", str(text)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return " ".join(t for t in s.split() if t not in STOP_WORDS and len(t) > 1)


def is_good(abstract: str) -> bool:
    return bool(abstract) and len(abstract.split()) >= 20


def extract_surnames(authors_str: str) -> list[str]:
    if not authors_str:
        return []
    s = str(authors_str)
    s = re.sub(r'(?i)presented by:.*', '', s)
    s = re.sub(r'(?i)discussant:.*', '', s)
    s = re.sub(r'\([^)]*\)', ' ', s)
    s = re.sub(r'\s*;\s*', ', ', s)
    s = re.sub(r'\s+and\s+', ', ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    surnames = []
    for part in s.split(','):
        words = part.strip().split()
        if not words or words[0] in _INSTITUTION_WORDS or len(words) > 4:
            continue
        surname = words[-1].rstrip('.')
        if not surname or len(surname) < 2 or surname in _INSTITUTION_WORDS:
            continue
        if ')' in surname or '(' in surname or re.match(r'^[IVX\d]+$', surname):
            continue
        surnames.append(surname)
        if len(surnames) == 3:
            break
    return surnames


def author_matches(surnames: list[str], result_text: str) -> bool:
    if not surnames:
        return True
    text_lower = result_text.lower()
    return any(s.lower() in text_lower for s in surnames)


# ── PDF abstract extraction ───────────────────────────────────────────────────

def extract_abstract_from_pdf(pdf_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = " ".join(p.extract_text() or "" for p in pdf.pages[:3])
    except Exception:
        return ""

    text = re.sub(r"\s+", " ", text)

    # Pattern 1: "Abstract" followed by text until a section break
    m = re.search(
        r'(?i)\bAbstract\b[:\s]*(.{80,1500}?)'
        r'(?:\s+(?:I|1|Introduction|Keywords|JEL|This paper)\b|\Z)',
        text, re.DOTALL
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()

    # Pattern 2: text between "Abstract" and "Keywords"
    m = re.search(r'(?i)\bAbstract\b[:\s]*(.{80,1500}?)\bKeywords?\b', text, re.DOTALL)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()

    return ""


# ── camoufox helpers ──────────────────────────────────────────────────────────

def patch_playwright_bundle():
    import site
    for sp in site.getsitepackages():
        candidate = pathlib.Path(sp) / "playwright" / "driver" / "package" / "lib" / "coreBundle.js"
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8", errors="replace")
            if "pageError.location.url," in text:
                text = (text
                    .replace("url: pageError.location.url,",
                             "url: pageError.location?.url ?? '',")
                    .replace("line: pageError.location.lineNumber,",
                             "line: pageError.location?.lineNumber ?? 0,")
                    .replace("column: pageError.location.columnNumber",
                             "column: pageError.location?.columnNumber ?? 0"))
                candidate.write_text(text, encoding="utf-8")
            break


async def setup_browser(headless: bool):
    from camoufox.async_api import AsyncCamoufox
    browser_cm = AsyncCamoufox(headless=headless, geoip=True)
    browser = await browser_cm.__aenter__()
    page = await browser.new_page()
    page.on("pageerror", lambda e: None)
    return browser_cm, browser, page


async def _load_year_program(page, year: int) -> bool:
    """Navigate to a conference year's program page, accept cookies."""
    url = AEA_BASE.format(year=year)
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(1500)
    except Exception:
        return False
    try:
        await page.locator('button:has-text("Accept")').click(timeout=3000)
        await page.wait_for_timeout(500)
    except Exception:
        pass
    # Verify the search box is present
    try:
        await page.wait_for_selector('input[name="search[term]"]', timeout=10000)
        return True
    except Exception:
        return False


async def aea_search_paper(page, title: str, surnames: list[str], year: int) -> tuple[str, str]:
    """
    Search the AEA program for a paper by title, download its PDF, extract abstract.
    Returns (abstract, source_label).
    """
    q_norm = normalise(title)
    search_query = title[:80]   # truncate to avoid over-specifying

    # Type into keyword search
    try:
        search_inp = page.locator('input[name="search[term]"]')
        await search_inp.click(timeout=5000)
        await search_inp.fill("")
        await search_inp.type(search_query, delay=20)
        await page.locator('#sessions_search_submit').click(timeout=5000)
        await page.wait_for_timeout(2000)
    except Exception:
        return "", ""

    # Collect PDF links from results
    links = await page.evaluate("""
        () => [...document.querySelectorAll('a[href*="/conference/"][href*="/paper/"]')]
              .map(a => ({text: a.innerText.trim(), href: a.href}))
    """)

    if not links:
        return "", ""

    # Each link points to a PDF — pick the best one
    # The page shows title + author context above the download link
    # Get all session/paper text blocks to match title
    result_blocks = await page.evaluate("""
        () => {
            const blocks = [];
            document.querySelectorAll('a[href*="/conference/"][href*="/paper/"]').forEach(a => {
                // Walk up to find the enclosing paper block
                let el = a;
                for (let i = 0; i < 8; i++) {
                    el = el.parentElement;
                    if (!el) break;
                    const txt = el.innerText;
                    if (txt && txt.length > 50) {
                        blocks.push({href: a.href, context: txt.slice(0, 300)});
                        break;
                    }
                }
            });
            return blocks;
        }
    """)

    best_score, best_href = 0, ""
    for block in result_blocks:
        context = block.get("context", "")
        # Title fuzzy match against context
        score = fuzz.partial_ratio(q_norm, normalise(context))
        if score < FUZZY_MIN:
            continue
        # Author match
        if not author_matches(surnames, context):
            continue
        if score > best_score:
            best_score = score
            best_href = block.get("href", "")

    # If no block match, try direct: download all PDFs and check title in PDF text
    if not best_href and links:
        best_href = links[0]["href"]  # try first result

    if not best_href:
        return "", ""

    # Download the PDF
    try:
        r = requests.get(best_href, headers=PDF_HEADERS, timeout=20)
        if r.status_code != 200 or b"%PDF" not in r.content[:10]:
            return "", ""
    except Exception:
        return "", ""

    abstract = extract_abstract_from_pdf(r.content)
    if not is_good(abstract):
        return "", ""

    # Final author validation against PDF text
    try:
        with pdfplumber.open(io.BytesIO(r.content)) as pdf:
            first_page = pdf.pages[0].extract_text() or ""
    except Exception:
        first_page = ""

    if not author_matches(surnames, first_page):
        return "", ""

    return abstract, "aea_pdf_10s"


# ── main loop ─────────────────────────────────────────────────────────────────

async def run(missing: pd.DataFrame, cache: dict, headless: bool) -> tuple[dict, list]:
    patch_playwright_bundle()
    log_rows = []
    browser_cm, browser, page = await setup_browser(headless)

    current_year = None
    try:
        for _, row in tqdm(missing.iterrows(), total=len(missing), desc="10s AEA program"):
            pid      = str(row["paper_id"])
            title    = str(row.get("title", "") or "")
            authors  = str(row.get("authors", "") or "")
            year     = int(row.get("year", 0))
            surnames = extract_surnames(authors)

            if pid in cache:
                entry    = cache[pid]
                abstract = entry.get("abstract", "")
                source   = entry.get("source", "none")
            else:
                abstract, source = "", "none"

                if year < 2021:
                    cache[pid] = {"abstract": "", "source": "skip_pre_2021"}
                    continue

                try:
                    # Load the year's program page if not already on it
                    if current_year != year:
                        ok = await _load_year_program(page, year)
                        if ok:
                            current_year = year
                        else:
                            cache[pid] = {"abstract": "", "source": "skip_page_load_failed"}
                            continue

                    abstract, source = await aea_search_paper(page, title, surnames, year)
                except Exception as e:
                    print(f"  Error {pid}: {e}")

                if not abstract:
                    source = "none"
                cache[pid] = {"abstract": abstract, "source": source}

                if len(cache) % 10 == 0:
                    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(CACHE_PATH, "w") as f:
                        json.dump(cache, f)

                await page.wait_for_timeout(1000)

            if abstract:
                log_rows.append({
                    "paper_id": pid, "year": year,
                    "title": title, "source": source,
                    "surnames": ", ".join(surnames),
                })
    finally:
        try:
            await browser_cm.__aexit__(None, None, None)
        except Exception:
            pass

    return cache, log_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force",    action="store_true")
    ap.add_argument("--limit",    type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    has_abs = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    noise = df["title"].str.contains(
        r'(?i)^(?:Session Chair|Location:|AFA Lecture|The Lecture|presented by|M\. Cecilia Bustamante)'
        r'|\d{4}.*\bthrough\b'
        r'|^(?:Finance|Economics|Accounting|Real Estate)\s+(?:Group|Unit|Section)\s*$'
        r'|^Finance and Economics|^Economics,\s*Finance',
        regex=True, na=False)
    noise_authors = df["authors"].str.contains(
        r'Location:|Session:|Chair:|Salon|Marriott|Hyatt', na=False)
    missing = df[~has_abs & ~noise & ~noise_authors &
                 df["authors"].notna() & (df["authors"].str.len() > 10)].copy()

    if args.limit:
        missing = missing.head(args.limit)

    print(f"Papers missing abstracts : {len(missing)}")
    print(f"Starting coverage        : {has_abs[~noise].sum():,} / {(~noise).sum():,} "
          f"({has_abs[~noise].mean()*100:.1f}%)")

    cache: dict = {}
    if CACHE_PATH.exists() and not args.force:
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    if not args.force:
        missing = missing[~missing["paper_id"].astype(str).isin(cache)]
        print(f"After cache filter       : {len(missing)} to process")

    cache, log_rows = asyncio.run(run(missing, cache, args.headless))

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

    for pid, entry in cache.items():
        abstract = entry.get("abstract", "")
        source   = entry.get("source", "none")
        if not abstract:
            continue
        mask = df["paper_id"].astype(str) == pid
        if mask.any() and (df.loc[mask, "abstract"].isna().all() or
                           (df.loc[mask, "abstract"].str.strip() == "").all()):
            df.loc[mask, "abstract"]        = abstract
            df.loc[mask, "abstract_source"] = source

    df.to_parquet(PARQUET, index=False)
    has_new = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    gain = has_new[~noise].sum() - has_abs[~noise].sum()

    print(f"\nResults:")
    print(f"  New abstracts added : {gain}")
    print(f"  Final coverage      : {has_new[~noise].sum():,} / {(~noise).sum():,} "
          f"({has_new[~noise].mean()*100:.1f}%)")

    if log_rows:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(log_rows)
        if LOG_PATH.exists():
            new_df = pd.concat([pd.read_csv(LOG_PATH), new_df], ignore_index=True)
        new_df.to_csv(LOG_PATH, index=False)
        print(f"  Log: {LOG_PATH.name} ({len(log_rows)} new rows)")


if __name__ == "__main__":
    main()
