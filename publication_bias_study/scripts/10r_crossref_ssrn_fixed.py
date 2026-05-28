"""
10r_crossref_ssrn_fixed.py
---------------------------
Fixes the surname-extraction bug that caused 10q to fail for 2013+ papers
where authors are formatted as "Name (University)" or "Name; University".

Pipeline per paper:
  1. extract_surnames() — handles all author formats
  2. Crossref API → SSRN abstract_id → camoufox direct navigation
  3. If no Crossref SSRN DOI → SSRN quoted-title search (10p fallback)

Usage:
    python scripts/10r_crossref_ssrn_fixed.py
    python scripts/10r_crossref_ssrn_fixed.py --force
    python scripts/10r_crossref_ssrn_fixed.py --limit N
    python scripts/10r_crossref_ssrn_fixed.py --headless
"""

import argparse
import asyncio
import json
import pathlib
import re
import time
import unicodedata

import pandas as pd
import requests
from rapidfuzz import fuzz
from tqdm import tqdm

ROOT       = pathlib.Path(__file__).parent.parent.resolve()
PARQUET    = ROOT / "data" / "raw" / "afa_papers_enriched.parquet"
CACHE_PATH = ROOT / "data" / "raw" / "afa_10r_cache.json"
LOG_PATH   = ROOT / "validation" / "abstract_10r_matches.csv"

FUZZY_MIN     = 78
CROSSREF_WAIT = 0.5
SSRN_BASE     = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id={}"
SSRN_SEARCH   = "https://papers.ssrn.com/sol3/results.cfm"

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
    "Accounting","Marketing","Law","Medicine","Technology","Harvard",
    "Stanford","MIT","Yale","Princeton","Columbia","Chicago","NYU",
    "Wharton","INSEAD","HEC","LSE","Cornell","Duke","Penn","Northwestern",
}

CROSSREF_SESSION = requests.Session()
CROSSREF_SESSION.headers.update({"User-Agent": "mailto:emrekuvvet@gmail.com"})


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
    """
    Robust surname extractor handling all AFA author formats:
      - "First Last, University... and First Last, University..."  (pre-2013)
      - "First Last (University), First Last (University)"         (2013)
      - "First Last; University First Last; University"            (2016+)
      - "First Last; UniversityFirst Last; University..."          (no space, 2021+)
    """
    if not authors_str:
        return []
    s = str(authors_str)
    # Strip trailing metadata
    s = re.sub(r'(?i)presented by:.*', '', s)
    s = re.sub(r'(?i)discussant:.*', '', s)
    # Remove parenthetical content "(University of ...)"
    s = re.sub(r'\([^)]*\)', ' ', s)
    # Normalize semicolons and " and " to commas
    s = re.sub(r'\s*;\s*', ', ', s)
    s = re.sub(r'\s+and\s+', ', ', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    parts = [p.strip() for p in s.split(',')]
    surnames = []
    for part in parts:
        words = part.split()
        if not words:
            continue
        if words[0] in _INSTITUTION_WORDS:
            continue
        if len(words) > 4:
            continue
        surname = words[-1].rstrip('.')
        if not surname or len(surname) < 2:
            continue
        if surname in _INSTITUTION_WORDS or ')' in surname or '(' in surname:
            continue
        # Skip pure numbers or single chars
        if re.match(r'^[IVX\d]+$', surname):
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


# ── Crossref ──────────────────────────────────────────────────────────────────

def crossref_lookup(title: str, surnames: list[str]) -> str | None:
    """Return SSRN abstract_id if Crossref finds a matching SSRN DOI, else None."""
    query = f"{title} {' '.join(surnames[:2])}"
    try:
        r = CROSSREF_SESSION.get(
            "https://api.crossref.org/works",
            params={"query.title": query, "rows": 5, "select": "title,DOI"},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        items = r.json()["message"]["items"]
    except Exception:
        return None
    finally:
        time.sleep(CROSSREF_WAIT)

    q_norm = normalise(title)
    best_score, best_id = 0, None
    for item in items:
        raw_titles = item.get("title", [])
        if not raw_titles:
            continue
        score = fuzz.token_set_ratio(q_norm, normalise(raw_titles[0]))
        if score < FUZZY_MIN:
            continue
        doi = item.get("DOI", "")
        m = re.match(r"10\.2139/ssrn\.(\d+)", doi, re.IGNORECASE)
        if not m:
            continue
        if score > best_score:
            best_score, best_id = score, m.group(1)
    return best_id


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
    print("  Opening SSRN ...")
    try:
        await page.goto(SSRN_SEARCH, wait_until="domcontentloaded", timeout=60000)
        await _wait_for_page_load(page, timeout_ms=30000)
        await page.wait_for_selector("input#term", timeout=60000)
        await page.wait_for_timeout(1000)
    except Exception as e:
        raise RuntimeError(f"SSRN did not load: {e}")
    try:
        await page.locator("#onetrust-accept-btn-handler").click(timeout=5000)
        await page.wait_for_timeout(800)
    except Exception:
        await page.evaluate("document.getElementById('onetrust-consent-sdk')?.remove()")
    print(f"  SSRN ready — {await page.title()}")
    return browser_cm, browser, page


async def _wait_for_page_load(page, timeout_ms: int = 25000) -> bool:
    try:
        await page.wait_for_function(
            "() => document.title !== 'Just a moment...'", timeout=timeout_ms)
        return True
    except Exception:
        return False


async def _extract_abstract(page) -> str:
    abstract = await page.evaluate("""
        () => {
            const selectors = [
                '.abstract-text p', '.abstract-text', '#abstractDiv',
                '[class*="abstract"] p', '[class*="abstract"]',
            ];
            for (const sel of selectors) {
                const el = document.querySelector(sel);
                if (el) { const t = el.innerText.trim(); if (t.length > 80) return t; }
            }
            for (const h of document.querySelectorAll('h2,h3,h4,strong,b')) {
                if (/^abstract$/i.test(h.innerText.trim())) {
                    const sib = h.nextElementSibling || h.parentElement?.nextElementSibling;
                    if (sib && sib.innerText.trim().length > 80) return sib.innerText.trim();
                }
            }
            return '';
        }
    """)
    abstract = re.sub(r"\s+", " ", abstract).strip()
    return re.sub(r"^Abstract\s*", "", abstract, flags=re.IGNORECASE).strip()


async def _get_page_authors(page) -> str:
    return await page.evaluate("""
        () => {
            const selectors = ['.authors','#authors','[class*="author"]',
                '.paper-authors','.abstract-authors','.author-paper-download','.author-info'];
            for (const sel of selectors) {
                const el = document.querySelector(sel);
                if (el && el.innerText.trim().length > 2) return el.innerText;
            }
            const links = [...document.querySelectorAll('a[href*="per_id"]')];
            if (links.length) return links.map(a => a.innerText).join(' ');
            return document.title;
        }
    """)


async def ssrn_fetch_direct(page, abstract_id: str, surnames: list[str]) -> tuple[str, str]:
    """Navigate directly to SSRN paper by abstract_id."""
    try:
        await page.goto(SSRN_BASE.format(abstract_id), wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(1500)
        if not await _wait_for_page_load(page, timeout_ms=25000):
            return "", ""
    except Exception:
        return "", ""

    page_authors = await _get_page_authors(page)
    if not author_matches(surnames, page_authors):
        return "", ""

    abstract = await _extract_abstract(page)
    if is_good(abstract):
        return abstract, "ssrn_direct_10r"
    return "", ""


async def _do_search(page, query: str) -> list[dict]:
    try:
        inp = page.locator("input#term").first
        await inp.click(timeout=8000)
        await inp.fill("")
        await inp.type(query, delay=15)
        await inp.press("Enter")
        await page.wait_for_load_state("domcontentloaded", timeout=20000)
        await page.wait_for_timeout(1500)
    except Exception:
        return []
    return await page.evaluate("""
        () => {
            const items = [];
            document.querySelectorAll('.results-list .result-item, .ssrn-search-result').forEach(block => {
                const a = block.querySelector('a[href*="abstract_id"], a[href*="/sol3/papers.cfm"]');
                if (!a) return;
                const auth = block.querySelector('.authors, .result-meta, [class*="author"]');
                items.push({ title: a.innerText.trim(), href: a.href,
                             authors: auth ? auth.innerText.trim() : '' });
            });
            if (items.length === 0) {
                document.querySelectorAll('a[href*="abstract_id"], a[href*="/sol3/papers.cfm"]')
                    .forEach(a => { if (a.innerText.trim().length > 5)
                        items.push({ title: a.innerText.trim(), href: a.href, authors: '' }); });
            }
            return items.slice(0, 12);
        }
    """) or []


async def _nav_to_search(page):
    try:
        await page.goto(SSRN_SEARCH, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_selector("input#term", timeout=60000)
    except Exception:
        pass


async def ssrn_search_fallback(page, title: str, surnames: list[str]) -> tuple[str, str]:
    """Quoted SSRN search — used when Crossref has no DOI."""
    q_norm = normalise(title)
    results = await _do_search(page, f'"{title[:100]}"')
    if not results:
        await _nav_to_search(page)
        results = await _do_search(page, title[:120])

    best_combined, best_href = 0, ""
    for item in results:
        cand = item.get("title", "")
        auth = item.get("authors", "")
        score = fuzz.token_set_ratio(q_norm, normalise(cand))
        if score < FUZZY_MIN:
            continue
        if auth and not author_matches(surnames, cand + " " + auth):
            continue
        exact = fuzz.ratio(q_norm, normalise(cand))
        combined = score * 1000 + exact
        if combined > best_combined:
            best_combined, best_href = combined, item.get("href", "")

    if not best_href:
        await _nav_to_search(page)
        return "", ""

    try:
        await page.goto(best_href, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(1500)
        if not await _wait_for_page_load(page, timeout_ms=25000):
            await _nav_to_search(page)
            return "", ""
    except Exception:
        return "", ""

    page_authors = await _get_page_authors(page)
    if not author_matches(surnames, page_authors):
        await _nav_to_search(page)
        return "", ""

    abstract = await _extract_abstract(page)
    await _nav_to_search(page)
    if is_good(abstract):
        return abstract, "ssrn_search_10r"
    return "", ""


# ── main loop ─────────────────────────────────────────────────────────────────

async def run(missing: pd.DataFrame, cache: dict, headless: bool) -> tuple[dict, list]:
    patch_playwright_bundle()
    log_rows = []
    browser_cm, browser, page = await setup_browser(headless)

    try:
        for _, row in tqdm(missing.iterrows(), total=len(missing), desc="10r Crossref+SSRN"):
            pid      = str(row["paper_id"])
            title    = str(row.get("title", "") or "")
            authors  = str(row.get("authors", "") or "")
            surnames = extract_surnames(authors)

            if pid in cache:
                entry    = cache[pid]
                abstract = entry.get("abstract", "")
                source   = entry.get("source", "none")
            else:
                abstract, source = "", "none"
                try:
                    # Step 1: Crossref → direct SSRN navigation
                    abstract_id = crossref_lookup(title, surnames)
                    if abstract_id:
                        abstract, source = await ssrn_fetch_direct(page, abstract_id, surnames)

                    # Step 2: SSRN search fallback
                    if not abstract:
                        await _nav_to_search(page)
                        abstract, source = await ssrn_search_fallback(page, title, surnames)
                except Exception as e:
                    print(f"  Error {pid}: {e}")
                    await _nav_to_search(page)

                if not abstract:
                    source = "none"
                cache[pid] = {"abstract": abstract, "source": source}

                if len(cache) % 20 == 0:
                    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(CACHE_PATH, "w") as f:
                        json.dump(cache, f)

                await page.wait_for_timeout(1500)

            if abstract:
                log_rows.append({
                    "paper_id": pid, "year": row.get("year"),
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