"""
10q_crossref_ssrn_direct.py
----------------------------
Fetches abstracts for remaining missing papers via:

  Step 1 — Crossref API (no browser, no bot-detection)
      Query title + author surnames → find DOI → extract SSRN abstract_id
      from DOIs of the form 10.2139/ssrn.<id>

  Step 2 — camoufox direct navigation (Cloudflare bypass)
      Navigate straight to https://papers.ssrn.com/sol3/papers.cfm?abstract_id=<id>
      — no SSRN search box needed, so indexing gaps for older papers don't matter.

Why 10p misses papers:
  SSRN's keyword search index is incomplete for pre-2010 working papers.
  Papers that don't appear in SSRN search still have Crossref DOI records,
  so direct navigation via the abstract_id always finds the right page.

Usage:
    python scripts/10q_crossref_ssrn_direct.py
    python scripts/10q_crossref_ssrn_direct.py --force
    python scripts/10q_crossref_ssrn_direct.py --limit N
    python scripts/10q_crossref_ssrn_direct.py --headless   # (Cloudflare may block)
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
CACHE_PATH = ROOT / "data" / "raw" / "afa_10q_cache.json"
LOG_PATH   = ROOT / "validation" / "abstract_10q_matches.csv"

FUZZY_MIN     = 80        # stricter than 10p — Crossref results are title-matched
CROSSREF_WAIT = 0.5       # seconds between Crossref calls (polite pool)
SSRN_BASE     = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id={}"

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
    if not surnames:
        return True
    text_lower = result_text.lower()
    return any(s.lower() in text_lower for s in surnames)


# ── Crossref lookup ───────────────────────────────────────────────────────────

def crossref_lookup(title: str, surnames: list[str]) -> tuple[str | None, str]:
    """
    Query Crossref for the paper. Return (ssrn_abstract_id, matched_title)
    if a SSRN DOI is found with a good title match, else (None, '').
    """
    query = f"{title} {' '.join(surnames[:2])}"
    try:
        r = CROSSREF_SESSION.get(
            "https://api.crossref.org/works",
            params={"query.title": query, "rows": 5,
                    "select": "title,DOI,author"},
            timeout=15,
        )
        if r.status_code != 200:
            return None, ""
        items = r.json()["message"]["items"]
    except Exception:
        return None, ""
    finally:
        time.sleep(CROSSREF_WAIT)

    q_norm = normalise(title)
    best_score, best_id, best_title = 0, None, ""

    for item in items:
        raw_titles = item.get("title", [])
        if not raw_titles:
            continue
        cand_title = raw_titles[0]
        score = fuzz.token_set_ratio(q_norm, normalise(cand_title))
        if score < FUZZY_MIN:
            continue
        doi = item.get("DOI", "")
        # Only accept SSRN DOIs: 10.2139/ssrn.<numeric_id>
        m = re.match(r"10\.2139/ssrn\.(\d+)", doi, re.IGNORECASE)
        if not m:
            continue
        if score > best_score:
            best_score = score
            best_id    = m.group(1)
            best_title = cand_title

    return best_id, best_title


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
                print("  Patched coreBundle.js")
            break


async def setup_browser(headless: bool):
    from camoufox.async_api import AsyncCamoufox
    browser_cm = AsyncCamoufox(headless=headless, geoip=True)
    browser = await browser_cm.__aenter__()
    page = await browser.new_page()
    page.on("pageerror", lambda e: None)
    # Initial SSRN navigation to establish session/cookies
    print("  Opening SSRN ...")
    try:
        await page.goto("https://papers.ssrn.com", wait_until="domcontentloaded", timeout=60000)
        await _wait_for_page_load(page, timeout_ms=30000)
        await page.wait_for_timeout(1500)
    except Exception:
        pass
    print(f"  SSRN ready — {await page.title()}")
    return browser_cm, browser, page


async def _wait_for_page_load(page, timeout_ms: int = 25000) -> bool:
    try:
        await page.wait_for_function(
            "() => document.title !== 'Just a moment...'",
            timeout=timeout_ms,
        )
        return True
    except Exception:
        return False


async def ssrn_fetch_direct(page, abstract_id: str, surnames: list[str]) -> tuple[str, str]:
    """Navigate directly to SSRN paper page and extract abstract."""
    url = SSRN_BASE.format(abstract_id)
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(1500)
        if not await _wait_for_page_load(page, timeout_ms=25000):
            return "", ""
    except Exception:
        return "", ""

    # Author validation on the paper page
    page_authors = await page.evaluate("""
        () => {
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
            const links = [...document.querySelectorAll('a[href*="per_id"]')];
            if (links.length) return links.map(a => a.innerText).join(' ');
            return document.title;
        }
    """)
    if not author_matches(surnames, page_authors):
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

    if is_good(abstract):
        return abstract, "ssrn_direct_10q"
    return "", ""


# ── main loop ─────────────────────────────────────────────────────────────────

async def run(missing: pd.DataFrame, cache: dict, headless: bool) -> tuple[dict, list]:
    patch_playwright_bundle()
    log_rows = []
    browser_cm, browser, page = await setup_browser(headless)

    try:
        for _, row in tqdm(missing.iterrows(), total=len(missing), desc="10q Crossref→SSRN"):
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
                    abstract_id, matched_title = crossref_lookup(title, surnames)
                    if abstract_id:
                        abstract, source = await ssrn_fetch_direct(page, abstract_id, surnames)
                except Exception as e:
                    print(f"  Error {pid}: {e}")

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
    ap.add_argument("--force",    action="store_true", help="ignore cache")
    ap.add_argument("--limit",    type=int, default=0, help="max papers to process")
    ap.add_argument("--headless", action="store_true", help="headless (Cloudflare may block)")
    args = ap.parse_args()

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
