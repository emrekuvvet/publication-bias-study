"""
10n_ssrn_abstracts.py
----------------------
Fetches abstracts for remaining missing papers from SSRN using camoufox — a
patched Firefox browser that passes Cloudflare Turnstile without being detected
as a bot.

Prerequisites:
  pip install camoufox[geoip]
  python -m camoufox fetch
  # Also patch Playwright's coreBundle.js (done automatically on first run):
  #   url: pageError.location?.url ?? '',
  #   line: pageError.location?.lineNumber ?? 0,
  #   column: pageError.location?.columnNumber ?? 0

Pipeline per paper:
  1. Search SSRN by title (default scope: title+abstract+keywords)
  2. Fuzzy-match search results (token_set_ratio ≥ FUZZY_MIN)
  3. Navigate to best-matching paper page
  4. Extract abstract from page
  5. Content-validate (≥2 significant title words in abstract)

Usage:
    python scripts/10n_ssrn_abstracts.py
    python scripts/10n_ssrn_abstracts.py --force
    python scripts/10n_ssrn_abstracts.py --limit N
    python scripts/10n_ssrn_abstracts.py --show-browser   # visible browser
"""

import argparse
import asyncio
import json
import pathlib
import re
import time
import unicodedata

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

ROOT       = pathlib.Path(__file__).parent.parent.resolve()
PARQUET    = ROOT / "data" / "raw" / "afa_papers_enriched.parquet"
CACHE_PATH = ROOT / "data" / "raw" / "afa_10n_cache.json"
LOG_PATH   = ROOT / "validation" / "abstract_10n_matches.csv"

FUZZY_MIN  = 72
SSRN_SEARCH = "https://papers.ssrn.com/sol3/results.cfm"

STOP_WORDS = {
    "a","an","the","of","in","on","at","to","for","and","or","but",
    "with","by","from","that","this","is","are","was","were","be",
    "been","being","have","has","had","do","does","did","will","would",
    "can","could","should","may","might","shall","some","any","all",
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


def patch_playwright_bundle():
    """Patch Playwright's coreBundle.js to handle Firefox JS errors with missing location."""
    bundle = pathlib.Path(__file__).parent.parent
    # Find coreBundle.js
    import site
    for sp in site.getsitepackages():
        candidate = pathlib.Path(sp) / "playwright" / "driver" / "package" / "lib" / "coreBundle.js"
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8", errors="replace")
            if "pageError.location.url," in text:
                text = text.replace(
                    "url: pageError.location.url,",
                    "url: pageError.location?.url ?? '',",
                )
                text = text.replace(
                    "line: pageError.location.lineNumber,",
                    "line: pageError.location?.lineNumber ?? 0,",
                )
                text = text.replace(
                    "column: pageError.location.columnNumber",
                    "column: pageError.location?.columnNumber ?? 0",
                )
                candidate.write_text(text, encoding="utf-8")
                print("  Patched coreBundle.js (Firefox pageError null-safety)")
            break


async def setup_browser(headless: bool):
    """Launch camoufox and return (browser, page) with SSRN search loaded."""
    from camoufox.async_api import AsyncCamoufox

    browser_cm = AsyncCamoufox(headless=headless, geoip=True)
    browser = await browser_cm.__aenter__()
    page = await browser.new_page()
    page.on("pageerror", lambda e: None)  # suppress Firefox JS errors

    print("  Opening SSRN search page ...")
    try:
        await page.goto(SSRN_SEARCH, wait_until="domcontentloaded", timeout=60000)
    except Exception:
        pass

    # Wait for real SSRN search form to appear (up to 90s for Cloudflare to clear)
    try:
        await page.wait_for_selector("input#term", timeout=90000)
    except Exception as e:
        raise RuntimeError(f"SSRN search form did not load: {e}")

    # Dismiss cookie consent banner
    try:
        await page.locator("#onetrust-accept-btn-handler").click(timeout=5000)
        await page.wait_for_timeout(800)
    except Exception:
        await page.evaluate("document.getElementById('onetrust-consent-sdk')?.remove()")

    print(f"  SSRN ready — {await page.title()}")
    return browser_cm, browser, page


async def ssrn_search(page, title: str) -> tuple[str, str]:
    """Search SSRN, navigate to best match, return (abstract, matched_title)."""
    q_norm = normalise(title)

    try:
        inp = page.locator("input#term").first
        await inp.click(timeout=8000)
        await inp.fill("")
        await inp.type(title[:120], delay=15)
        await inp.press("Enter")
        await page.wait_for_load_state("domcontentloaded", timeout=20000)
        await page.wait_for_timeout(1500)
    except Exception:
        return "", ""

    # Collect result links
    results = await page.evaluate("""
        () => [...document.querySelectorAll('a[href*="abstract_id"], a[href*="/sol3/papers.cfm"]')]
            .slice(0, 12)
            .map(a => ({title: a.innerText.trim(), href: a.href}))
            .filter(r => r.title.length > 5)
    """)

    best_score = 0
    best_href  = ""
    best_title = ""
    for item in results:
        cand = item.get("title", "")
        score = fuzz.token_set_ratio(q_norm, normalise(cand))
        if score > best_score:
            best_score = score
            best_href  = item.get("href", "")
            best_title = cand

    if best_score < FUZZY_MIN or not best_href:
        # Try to get back to search form
        try:
            await page.go_back(wait_until="domcontentloaded", timeout=10000)
            await page.wait_for_selector("input#term", timeout=15000)
        except Exception:
            pass
        return "", ""

    # Navigate to the paper page
    try:
        await page.goto(best_href, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(1500)
    except Exception:
        return "", ""

    # Extract abstract
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
            // Fallback: "Abstract" heading → next sibling
            for (const h of document.querySelectorAll('h2,h3,h4,strong,b')) {
                if (/^abstract$/i.test(h.innerText.trim())) {
                    const sib = h.nextElementSibling || h.parentElement?.nextElementSibling;
                    if (sib) {
                        const txt = sib.innerText.trim();
                        if (txt.length > 80) return txt;
                    }
                }
            }
            return '';
        }
    """)

    abstract = re.sub(r"\s+", " ", abstract).strip()

    # Return to search page for next query
    try:
        await page.go_back(wait_until="domcontentloaded", timeout=10000)
        await page.wait_for_timeout(500)
        await page.go_back(wait_until="domcontentloaded", timeout=10000)
        await page.wait_for_selector("input#term", timeout=15000)
    except Exception:
        try:
            await page.goto(SSRN_SEARCH, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_selector("input#term", timeout=30000)
        except Exception:
            pass

    if is_good(abstract) and content_valid(title, abstract):
        return abstract, best_title
    return "", ""


async def run(missing: pd.DataFrame, cache: dict, headless: bool) -> tuple[dict, list]:
    patch_playwright_bundle()

    log_rows = []

    browser_cm, browser, page = await setup_browser(headless)

    try:
        for _, row in tqdm(missing.iterrows(), total=len(missing), desc="10n SSRN"):
            pid   = str(row["paper_id"])
            title = str(row.get("title", "") or "")

            if pid in cache:
                entry    = cache[pid]
                abstract = entry.get("abstract", "")
                source   = entry.get("source", "none")
            else:
                try:
                    abstract, matched_title = await ssrn_search(page, title)
                except Exception as e:
                    print(f"  Error searching {pid}: {e}")
                    abstract = ""
                source = "ssrn_10n" if abstract else "none"
                cache[pid] = {"abstract": abstract, "source": source}

                if len(cache) % 20 == 0:
                    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(CACHE_PATH, "w") as f:
                        json.dump(cache, f)

                await page.wait_for_timeout(2500)

            if abstract:
                log_rows.append({
                    "paper_id": pid,
                    "year":     row.get("year"),
                    "title":    title,
                    "source":   source,
                })
    finally:
        try:
            await browser_cm.__aexit__(None, None, None)
        except Exception:
            pass

    return cache, log_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force",        action="store_true", help="ignore cache")
    ap.add_argument("--limit",        type=int, default=0, help="max papers to process")
    ap.add_argument("--headless", action="store_true", help="run browser headless (WARNING: Cloudflare blocks headless)")
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
    print(f"Starting coverage        : {has_abs.sum():,} / {(~noise).sum():,} ({has_abs[~noise].mean()*100:.1f}%)")

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

    # Write abstracts back to parquet
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
    gain = has_new.sum() - has_abs.sum()

    print(f"\nResults:")
    print(f"  New abstracts added : {gain}")
    print(f"  Final coverage      : {has_new.sum():,} / {(~noise).sum():,} ({has_new[~noise].mean()*100:.1f}%)")

    if log_rows:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(log_rows)
        if LOG_PATH.exists():
            new_df = pd.concat([pd.read_csv(LOG_PATH), new_df], ignore_index=True)
        new_df.to_csv(LOG_PATH, index=False)
        print(f"  Log: {LOG_PATH.name} ({len(log_rows)} new rows)")


if __name__ == "__main__":
    main()
