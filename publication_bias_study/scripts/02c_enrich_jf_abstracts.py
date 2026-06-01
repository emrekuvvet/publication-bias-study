"""
02c_enrich_jf_abstracts.py
---------------------------
Fetch missing abstracts for JF papers using two sources:
  1. Semantic Scholar API (by DOI, then by title fallback)
  2. Wiley Online Library scraping (by DOI) for papers S2 can't find

Only targets genuine research papers — admin items (announcements,
AFA programs, prize notices, editor reports) are skipped.

After fetching, updates raw_jf in DuckDB and re-exports jf_papers.parquet.
Also re-runs the keyword filter on newly filled abstracts and reports any
new keyword hits (potential in-scope papers missed in the original pass).

Usage:
    python scripts/02c_enrich_jf_abstracts.py
    python scripts/02c_enrich_jf_abstracts.py --limit 50   # test run
    python scripts/02c_enrich_jf_abstracts.py --skip-wiley # S2 only
"""

import argparse
import json
import pathlib
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rapidfuzz import fuzz
from tqdm import tqdm

ROOT     = pathlib.Path(__file__).parent.parent.resolve()
DB_PATH  = ROOT / "publication_bias.duckdb"
OUT_PATH = ROOT / "data" / "raw" / "jf_papers.parquet"
CACHE_PATH = ROOT / "data" / "raw" / "jf_abstract_cache.json"

load_dotenv(ROOT / ".env")

# Import keyword filter from script 03
sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module
_s03 = import_module("03_classify_inscope")
keyword_hit = _s03.keyword_hit

ADMIN_KEYWORDS = [
    "ANNOUNCEMENT", "AMERICAN FINANCE ASSOCIATION", "PRELIMINARY PROGRAM",
    "REPORT OF THE", "PARTICIPANT SCHEDULE", "BRATTLE", "FISCHER BLACK",
    "SMITH BREEDEN", "PRIZE", "MEMBERSHIP MEETING", "ANNUAL MEETING",
    "EXECUTIVE SECRETARY", "EDITOR OF",
]

S2_BASE   = "https://api.semanticscholar.org/graph/v1/paper"
S2_FIELDS = "abstract,title,externalIds"
S2_DELAY  = 1.1   # free tier: ~1 req/s
WILEY_BASE = "https://onlinelibrary.wiley.com/doi"
HEADERS    = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
FUZZY_MIN  = 78


def is_admin(title: str) -> bool:
    if not title:
        return True
    t = title.upper()
    return any(k in t for k in ADMIN_KEYWORDS)


def clean_abstract(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^Abstract\s*[:.]?\s*", "", text, flags=re.IGNORECASE)
    return text


# ------------------------------------------------------------------ #
# Source 1: Semantic Scholar                                          #
# ------------------------------------------------------------------ #

_s2_lock = threading.Lock()
_s2_last  = [0.0]

def _s2_get(url: str, session: requests.Session) -> dict:
    with _s2_lock:
        elapsed = time.time() - _s2_last[0]
        if elapsed < S2_DELAY:
            time.sleep(S2_DELAY - elapsed)
        _s2_last[0] = time.time()
    r = session.get(url, timeout=15)
    if r.status_code == 429:
        time.sleep(10)
        r = session.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_s2_by_doi(doi: str, session: requests.Session) -> str:
    try:
        data = _s2_get(f"{S2_BASE}/DOI:{doi}?fields={S2_FIELDS}", session)
        return clean_abstract(data.get("abstract") or "")
    except Exception:
        return ""


def fetch_s2_by_title(title: str, session: requests.Session) -> str:
    try:
        params = {"query": title, "fields": "abstract,title", "limit": 3}
        r = session.get(f"{S2_BASE}/search", params=params, timeout=15)
        if r.status_code != 200:
            return ""
        results = r.json().get("data", [])
        for hit in results:
            candidate = hit.get("title") or ""
            if fuzz.token_sort_ratio(title.lower(), candidate.lower()) >= FUZZY_MIN:
                return clean_abstract(hit.get("abstract") or "")
    except Exception:
        pass
    return ""


# ------------------------------------------------------------------ #
# Source 2: Wiley Online Library                                      #
# ------------------------------------------------------------------ #

def fetch_wiley(doi: str, session: requests.Session) -> str:
    url = f"{WILEY_BASE}/{doi}"
    for attempt in range(3):
        try:
            r = session.get(url, timeout=20, headers=HEADERS, allow_redirects=True)
            if r.status_code == 404:
                return ""
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # Try multiple selectors used by Wiley
            for sel in [
                "section.article-section__abstract",
                ".article-section__content",
                "[class*='abstract']",
                "#abstract",
            ]:
                el = soup.select_one(sel)
                if el:
                    text = el.get_text(separator=" ", strip=True)
                    text = clean_abstract(text)
                    if len(text) > 50:
                        return text
            return ""
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return ""


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main(limit: int | None, skip_wiley: bool) -> None:
    con = duckdb.connect(str(DB_PATH))
    df = con.execute(
        "SELECT doi, title, year, abstract FROM raw_jf "
        "WHERE abstract IS NULL OR abstract = ''"
    ).df()
    con.close()

    # Filter to genuine research papers
    df = df[~df["title"].apply(is_admin)].copy()
    print(f"Research papers missing abstracts: {len(df)}")

    if limit:
        df = df.head(limit)
        print(f"Test mode: processing {len(df)} papers")

    # Load cache (resumable)
    cache: dict[str, str] = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        already = sum(1 for v in cache.values() if v)
        print(f"Cache: {len(cache)} entries, {already} with abstracts")

    remaining = [row for _, row in df.iterrows() if row["doi"] not in cache]
    print(f"To fetch: {len(remaining)}")

    s2_session    = requests.Session()
    wiley_session = requests.Session()

    found_s2    = 0
    found_wiley = 0
    not_found   = 0

    for row in tqdm(remaining, desc="Fetching abstracts"):
        doi   = row["doi"]
        title = str(row["title"])

        # Try Semantic Scholar by DOI first, then title
        abstract = fetch_s2_by_doi(doi, s2_session)
        if not abstract:
            abstract = fetch_s2_by_title(title, s2_session)

        if abstract:
            found_s2 += 1
        elif not skip_wiley:
            abstract = fetch_wiley(doi, wiley_session)
            if abstract:
                found_wiley += 1
            else:
                not_found += 1
        else:
            not_found += 1

        cache[doi] = abstract

        # Save checkpoint every 50
        if len(cache) % 50 == 0:
            with open(CACHE_PATH, "w") as f:
                json.dump(cache, f)

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

    print(f"\nResults:")
    print(f"  Found via Semantic Scholar: {found_s2}")
    print(f"  Found via Wiley scraping:   {found_wiley}")
    print(f"  Not found:                  {not_found}")
    filled = sum(1 for v in cache.values() if v)
    print(f"  Total cache with abstract:  {filled}/{len(cache)}")

    # --- Update DuckDB and parquet ---
    updates = [(v, k) for k, v in cache.items() if v]
    if not updates:
        print("\nNo abstracts retrieved — nothing to update.")
        return

    print(f"\nUpdating raw_jf with {len(updates)} new abstracts...")
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE TEMP TABLE _jf_abs (doi VARCHAR, abstract VARCHAR)")
    con.executemany("INSERT INTO _jf_abs VALUES (?, ?)", updates)
    con.execute("""
        UPDATE raw_jf
        SET abstract = _jf_abs.abstract
        FROM _jf_abs
        WHERE raw_jf.doi = _jf_abs.doi
          AND (raw_jf.abstract IS NULL OR raw_jf.abstract = '')
          AND LENGTH(_jf_abs.abstract) > 0
    """)
    updated_n = con.execute(
        "SELECT COUNT(*) FROM raw_jf WHERE LENGTH(abstract) > 0 "
        "AND doi IN (SELECT doi FROM _jf_abs)"
    ).fetchone()[0]
    print(f"  Rows updated in raw_jf: {updated_n}")

    # Re-export parquet
    jf_df = con.execute("SELECT * FROM raw_jf").df()
    con.close()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    jf_df.to_parquet(OUT_PATH, index=False)
    print(f"  Exported {len(jf_df)} rows → {OUT_PATH}")

    # --- Check for new keyword hits ---
    print("\nChecking for new keyword hits among filled abstracts...")
    new_hits = []
    for doi, abstract in cache.items():
        if not abstract:
            continue
        row = df[df["doi"] == doi]
        if row.empty:
            continue
        title = str(row.iloc[0]["title"])
        combined = title + " " + abstract
        if keyword_hit(combined):
            new_hits.append({
                "doi": doi,
                "year": int(row.iloc[0]["year"]),
                "title": title,
                "abstract": abstract[:150],
            })

    if new_hits:
        print(f"\n*** {len(new_hits)} NEW KEYWORD HITS found! ***")
        print("These papers were missed in Pass 1 due to missing abstracts:")
        for h in new_hits:
            print(f"  {h['doi']} ({h['year']}): {h['title'][:70]}")
        print("\nNext steps: run 03_classify_inscope.py on these new candidates.")
    else:
        print("No new keyword hits — missing abstracts did not affect Pass 1 results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--skip-wiley", action="store_true")
    args = parser.parse_args()
    main(limit=args.limit, skip_wiley=args.skip_wiley)
