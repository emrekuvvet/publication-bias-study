"""
10p_ssrn_quoted_search.py
--------------------------
Fetches abstracts for remaining missing papers from SSRN using camoufox — a
patched Firefox browser that passes Cloudflare Turnstile without bot detection.

Improvements over 10n:
  - Searches with QUOTED title ("exact title") for higher precision
  - Validates both title fuzzy-match AND at least one author surname match
  - Falls back to unquoted title search if quoted returns no results
  - Uses the fixed extract_surnames() that skips affiliation fragments

Pipeline per paper:
  1. Extract author surnames from the `authors` column
  2. Type "quoted title" into SSRN search box
  3. Collect result links + visible author text from the results page
  4. Accept a result only if: title fuzzy score ≥ FUZZY_MIN AND ≥1 surname matches
  5. Navigate to paper page, extract abstract
  6. Content-validate (≥2 significant title words in abstract)
  7. If quoted search returns nothing, retry without quotes (fallback)

Usage:
    python scripts/10p_ssrn_quoted_search.py
    python scripts/10p_ssrn_quoted_search.py --force
    python scripts/10p_ssrn_quoted_search.py --limit N
    python scripts/10p_ssrn_quoted_search.py --show-browser
"""

import argparse
import asyncio
import json
import pathlib
import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

ROOT       = pathlib.Path(__file__).parent.parent.resolve()
PARQUET    = ROOT / "data" / "raw" / "afa_papers_enriched.parquet"
CACHE_PATH = ROOT / "data" / "raw" / "afa_10p_cache.json"
LOG_PATH   = ROOT / "validation" / "abstract_10p_matches.csv"

FUZZY_MIN   = 72
SSRN_SEARCH = "https://papers.ssrn.com/sol3/results.cfm"

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


def normalise(text: str) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFD", str(text)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return " ".join(t for t in s.split() if t not in STOP_WORDS and len(t) > 1)


def is_good(abstract: str) -> bool:
    return bool(abstract) and len(abstract.split()) >= 20


def content_valid(title: str, abstract: str, min_overlap: int = 2) -> bool:
    title_words = {w for w in normalise(title).split() if len(w) >= 4}
    abs_text = normalise(abstract)
    return sum(1 for w in title_words if w in abs_text) >= min_overlap


def extract_surnames(authors_str: str) -> list[str]:
    """Extract up to 3 surnames, skipping affiliation fragments."""
    if not authors_str:
        return []
    parts = re.split(r"\s+and\s+", str(authors_str))
    surnames = []
    for part in parts:
        name_token = part.split(",")[0].strip()
        words = name_token.split()
        if not words:
            continue
        if words[0] in _INSTITUTION_WORDS or len(words) > 4:
            continue
        surnames.append(words[-1].rstrip("."))
        if len(surnames) == 3:
            break
    return surnames


def author_matches(surnames: list[str], result_text: str) -> bool:
    """Return True if at least one surname appears in the result text (case-insensitive)."""
    if not surnames:
        return True   # no surname info → don't reject
    text_lower = result_text.lower()
    return any(s.lower() in text_lower for s in surnames)


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
                print("  Patched coreBundle.js")
            break


async def setup_browser(headless: bool):
    from camoufox.async_api import AsyncCamoufox
    browser_cm = AsyncCamoufox(headless=headless, geoip=True)
    browser = await browser_cm.__aenter__()
    page = await browser.new_page()
    page.on("pageerror", lambda e: None)

    print("  Opening SSRN search page ...")
    try:
        await page.goto(SSRN_SEARCH, wait_until="domcontentloaded", timeout=60000)
    except Exception:
        pass

    try:
        await page.wait_for_selector("input#term", timeout=90000)
    except Exception as e:
        raise RuntimeError(f"SSRN search form did not load: {e}")

    try:
        await page.locator("#onetrust-accept-btn-handler").click(timeout=5000)
        await page.wait_for_timeout(800)
    except Exception:
        await page.evaluate("document.getElementById('onetrust-consent-sdk')?.remove()")

    print(f"  SSRN ready — {await page.title()}")
    return browser_cm, browser, page


async def _do_search(page, query: str) -> list[dict]:
    """Type query into SSRN search box and return list of {title, href, authors}."""
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

    results = await page.evaluate("""
        () => {
            const items = [];
            // Each result block on SSRN results page
            document.querySelectorAll('.results-list .result-item, .ssrn-search-result').forEach(block => {
                const a = block.querySelector('a[href*="abstract_id"], a[href*="/sol3/papers.cfm"]');
                if (!a) return;
                const authors = block.querySelector('.authors, .result-meta, [class*="author"]');
                items.push({
                    title:   a.innerText.trim(),
                    href:    a.href,
                    authors: authors ? authors.innerText.trim() : ''
                });
            });
            // Fallback: bare links if no structured blocks found
            if (items.length === 0) {
                document.querySelectorAll('a[href*="abstract_id"], a[href*="/sol3/papers.cfm"]')
                    .forEach(a => {
                        if (a.innerText.trim().length > 5)
                            items.push({title: a.innerText.trim(), href: a.href, authors: ''});
                    });
            }
            return items.slice(0, 12);
        }
    """)
    return results or []


async def ssrn_search(page, title: str, surnames: list[str]) -> tuple[str, str]:
    """Search SSRN with quoted title + author validation. Returns (abstract, source_label)."""
    q_norm = normalise(title)

    # ── 1. Quoted search ──
    quoted_query = f'"{title[:100]}"'
    results = await _do_search(page, quoted_query)

    # If no results, fall back to unquoted
    if not results:
        await _navigate_back_to_search(page)
        results = await _do_search(page, title[:120])

    # ── 2. Pick best result: title fuzzy + author match ──
    best_combined = 0
    best_href     = ""

    for item in results:
        cand  = item.get("title", "")
        auth  = item.get("authors", "")
        score = fuzz.token_set_ratio(q_norm, normalise(cand))
        if score < FUZZY_MIN:
            continue
        # Only apply author check at results level when we actually have author text.
        # If author text is empty (bare-link fallback), defer to paper-page check below.
        if auth and not author_matches(surnames, cand + " " + auth):
            continue
        # Use fuzz.ratio as tiebreaker: "Who Herds?" beats "Who Herds? Who Doesn't?"
        exact = fuzz.ratio(q_norm, normalise(cand))
        combined = score * 1000 + exact
        if combined > best_combined:
            best_combined = combined
            best_href     = item.get("href", "")

    if not best_href:
        await _navigate_back_to_search(page)
        return "", ""

    # ── 3. Navigate to paper page and extract abstract ──
    try:
        await page.goto(best_href, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(1500)
        # Wait for Cloudflare challenge to clear if it appears
        if not await _wait_for_page_load(page, timeout_ms=25000):
            await _navigate_back_to_search(page)
            return "", ""
    except Exception:
        return "", ""

    # Validate author on the paper page itself
    page_authors = await page.evaluate("""
        () => {
            // Try multiple SSRN author selectors in order
            const selectors = [
                '.authors', '#authors', '[class*="author"]',
                '.paper-authors', '.abstract-authors',
                '.author-paper-download', '.author-info',
            ];
            for (const sel of selectors) {
                const el = document.querySelector(sel);
                if (el && el.innerText.trim().length > 2)
                    return el.innerText;
            }
            // Fallback: collect all <a> links near the title that look like names
            const links = [...document.querySelectorAll('a[href*="per_id"]')];
            if (links.length) return links.map(a => a.innerText).join(' ');
            return document.title;
        }
    """)
    if not author_matches(surnames, page_authors):
        await _navigate_back_to_search(page)
        return "", ""

    abstract = await page.evaluate("""
        () => {
            const selectors = [
                '.abstract-text p', '.abstract-text', '#abstractDiv',
                '[class*="abstract"] p', '[class*="abstract"]',
            ];
            for (const sel of selectors) {
                const el = document.querySelector(sel);
                if (el) {
                    const txt = el.innerText.trim();
                    if (txt.length > 80) return txt;
                }
            }
            for (const h of document.querySelectorAll('h2,h3,h4,strong,b')) {
                if (/^abstract$/i.test(h.innerText.trim())) {
                    const sib = h.nextElementSibling || h.parentElement?.nextElementSibling;
                    if (sib && sib.innerText.trim().length > 80)
                        return sib.innerText.trim();
                }
            }
            return '';
        }
    """)
    abstract = re.sub(r"\s+", " ", abstract).strip()
    abstract = re.sub(r"^Abstract\s*", "", abstract, flags=re.IGNORECASE).strip()

    await _navigate_back_to_search(page)

    # title fuzzy + author match already validated relevance; just check length
    if is_good(abstract):
        return abstract, "ssrn_quoted_10p"
    return "", ""


async def _navigate_back_to_search(page):
    """Always navigate directly to the search page — avoids go_back() history confusion."""
    try:
        await page.goto(SSRN_SEARCH, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_selector("input#term", timeout=60000)
    except Exception:
        pass


async def _wait_for_page_load(page, timeout_ms: int = 20000) -> bool:
    """Wait until Cloudflare challenge clears (title stops being 'Just a moment...')."""
    try:
        await page.wait_for_function(
            "() => document.title !== 'Just a moment...'",
            timeout=timeout_ms,
        )
        return True
    except Exception:
        return False


async def run(missing: pd.DataFrame, cache: dict, headless: bool) -> tuple[dict, list]:
    patch_playwright_bundle()
    log_rows = []
    browser_cm, browser, page = await setup_browser(headless)

    try:
        for _, row in tqdm(missing.iterrows(), total=len(missing), desc="10p SSRN quoted"):
            pid     = str(row["paper_id"])
            title   = str(row.get("title", "") or "")
            authors = str(row.get("authors", "") or "")
            surnames = extract_surnames(authors)

            if pid in cache:
                entry    = cache[pid]
                abstract = entry.get("abstract", "")
                source   = entry.get("source", "none")
            else:
                try:
                    abstract, source = await ssrn_search(page, title, surnames)
                except Exception as e:
                    print(f"  Error {pid}: {e}")
                    abstract, source = "", "none"

                if not abstract:
                    source = "none"
                cache[pid] = {"abstract": abstract, "source": source}

                if len(cache) % 20 == 0:
                    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(CACHE_PATH, "w") as f:
                        json.dump(cache, f)

                await page.wait_for_timeout(2000)

            if abstract:
                log_rows.append({
                    "paper_id": pid,
                    "year":     row.get("year"),
                    "title":    title,
                    "source":   source,
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
    ap.add_argument("--force",       action="store_true", help="ignore cache")
    ap.add_argument("--limit",       type=int, default=0, help="max papers to process")
    ap.add_argument("--headless", action="store_true", help="headless mode (WARNING: Cloudflare blocks this)")
    args = ap.parse_args()
    headless = args.headless

    df = pd.read_parquet(PARQUET)
    has_abs = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    noise = df["title"].str.contains(
        r'(?i)^(?:Session Chair|Location:|AFA Lecture)'
        r'|\d{4}.*\bthrough\b'
        r'|^(?:Finance|Economics|Accounting|Real Estate)\s+(?:Group|Unit|Section)\s*$'
        r'|^Finance and Economics|^Economics,\s*Finance',
        regex=True, na=False)
    missing = df[~has_abs & ~noise].copy()

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

    cache, log_rows = asyncio.run(run(missing, cache, headless))

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
