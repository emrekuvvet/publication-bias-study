"""
10e_retry_missing_abstracts.py
-------------------------------
Targeted abstract retrieval for AFA papers still missing abstracts in
afa_papers_enriched.parquet. Safe to run after ASSA scraping (10d) —
never overwrites rows that already have abstracts.

Sources (priority order):
  1. OpenAlex         — broad journal + preprint coverage
  2. CrossRef         — published papers
  3. Semantic Scholar — working papers / preprints

Timeout strategy: requests library with socket-level timeout.
  requests.get(url, timeout=N) enforces the timeout at the OS socket layer —
  it raises requests.exceptions.Timeout after N seconds regardless of what the
  library does internally. No threads or processes needed; no accumulation of
  zombie threads. httpx was abandoned because (a) its anyio internals are not
  fork-safe on macOS/Python 3.14 and (b) abandoned daemon threads accumulate
  and progressively slow the main loop.

Usage:
    python scripts/10e_retry_missing_abstracts.py              # all missing
    python scripts/10e_retry_missing_abstracts.py --year 2024  # single year
    python scripts/10e_retry_missing_abstracts.py --force      # ignore cache

Prerequisites:
    pip install requests pandas pyarrow rapidfuzz tqdm python-dotenv
"""

import argparse
import json
import os
import pathlib
import re
import time
import unicodedata

import pandas as pd
import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz
from tqdm import tqdm

ROOT       = pathlib.Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")

PARQUET    = ROOT / "data" / "raw" / "afa_papers_enriched.parquet"
CACHE_PATH = ROOT / "data" / "raw" / "afa_10e_cache.json"
LOG_PATH   = ROOT / "validation" / "abstract_10e_matches.csv"

OA_EMAIL     = "research@pubbiasstudy.org"
FUZZY_MIN    = 80
SS_DELAY = 1.2   # seconds between SS requests (free tier ~1 req/s)
TIMEOUT  = 8     # socket-level timeout passed to requests.get()

SS_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

UA = "PublicationBiasResearch/1.0 (mailto:research@pubbiasstudy.org)"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

STOP_WORDS = {
    "a","an","the","of","in","on","at","to","for","and","or","but",
    "with","by","from","that","this","is","are","was","were","be",
    "been","being","have","has","had","do","does","did","will","would",
    "can","could","should","may","might","shall","some","any","all",
}



def normalise(title: str) -> str:
    if not title:
        return ""
    s = unicodedata.normalize("NFD", str(title)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return " ".join(t for t in s.split() if t not in STOP_WORDS and len(t) > 1)


def title_match(norm_query: str, cand: str) -> int:
    return fuzz.token_sort_ratio(norm_query, normalise(cand))


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


def is_good(abstract: str) -> bool:
    return bool(abstract) and len(abstract.split()) >= 20


def openalex_search(title: str, year: int | None) -> str | None:
    q_norm = normalise(title)
    params: dict = {
        "search": title[:200],
        "per-page": 5,
        "select": "title,abstract_inverted_index",
        "mailto": OA_EMAIL,
    }
    if year:
        params["filter"] = f"publication_year:{year - 2}-{year + 4}"
    try:
        r = SESSION.get("https://api.openalex.org/works", params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
    except Exception:
        return None
    for item in results:
        cand = item.get("title") or ""
        if title_match(q_norm, cand) >= FUZZY_MIN:
            inv = item.get("abstract_inverted_index") or {}
            if inv:
                pos_word = [(p, w) for w, positions in inv.items() for p in positions]
                abstract = " ".join(w for _, w in sorted(pos_word))
                if is_good(abstract):
                    return abstract
    return None


def crossref_search(title: str, year: int | None) -> str | None:
    q_norm = normalise(title)
    params: dict = {
        "query.title": title[:200],
        "rows": 5,
        "select": "title,abstract",
        "mailto": OA_EMAIL,
    }
    if year:
        params["filter"] = f"from-pub-date:{year - 2},until-pub-date:{year + 4}"
    try:
        r = SESSION.get("https://api.crossref.org/works", params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        items = r.json().get("message", {}).get("items", [])
    except Exception:
        return None
    for item in items:
        titles = item.get("title") or []
        cand = titles[0] if titles else ""
        if title_match(q_norm, cand) >= FUZZY_MIN:
            raw = item.get("abstract") or ""
            if raw:
                abstract = strip_html(raw)
                if is_good(abstract):
                    return abstract
    return None


def ss_search(title: str, year: int | None) -> str | None:
    q_norm = normalise(title)
    params: dict = {
        "query": title[:200],
        "fields": "title,abstract,year",
        "limit": 5,
    }
    if year:
        params["year"] = f"{year - 2}-{year + 4}"
    headers: dict = {}
    if SS_API_KEY:
        headers["x-api-key"] = SS_API_KEY
    for attempt in range(2):
        try:
            r = SESSION.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params, headers=headers, timeout=TIMEOUT,
            )
            if r.status_code == 429:
                time.sleep(3)
                continue
            if r.status_code != 200:
                return None
            break
        except Exception:
            return None
    else:
        return None
    for paper in r.json().get("data", []):
        cand = paper.get("title") or ""
        if title_match(q_norm, cand) >= FUZZY_MIN:
            abstract = paper.get("abstract") or ""
            if is_good(abstract):
                return abstract
    return None


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year",  type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print(f"Loading {PARQUET.name} ...")
    df = pd.read_parquet(PARQUET)
    has_abs = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    print(f"  Starting coverage: {has_abs.sum():,} / {len(df):,} ({has_abs.mean()*100:.1f}%)")

    missing_mask = ~has_abs
    if args.year:
        missing_mask &= (df["year"] == args.year)
    missing = df[missing_mask].copy()
    print(f"  Papers to query: {len(missing):,}"
          + (f" (year={args.year})" if args.year else ""))

    cache: dict = {}
    if CACHE_PATH.exists() and not args.force:
        with open(CACHE_PATH) as f:
            cache = json.load(f)

    if SS_API_KEY:
        print("  Semantic Scholar API key loaded")

    hits: dict[str, int] = {"openalex": 0, "crossref": 0, "semantic_scholar": 0}
    log_rows = []

    for _, row in tqdm(missing.iterrows(), total=len(missing), desc="Fetching"):
        pid   = str(row["paper_id"])
        title = str(row.get("title", "") or "")
        year  = int(row["year"]) if pd.notna(row.get("year")) else None

        if pid in cache and not args.force:
            entry    = cache[pid]
            abstract = entry.get("abstract", "")
            source   = entry.get("source", "none")
        else:
            abstract, source = "", "none"

            if not abstract:
                abstract = openalex_search(title, year) or ""
                if abstract:
                    source = "openalex"

            if not abstract:
                abstract = crossref_search(title, year) or ""
                if abstract:
                    source = "crossref"

            if not abstract:
                abstract = ss_search(title, year) or ""
                if abstract:
                    source = "semantic_scholar"
                time.sleep(SS_DELAY)

            cache[pid] = {"abstract": abstract, "source": source}

        if abstract:
            df.loc[row.name, "abstract"]        = abstract
            df.loc[row.name, "abstract_source"] = source
            hits[source] = hits.get(source, 0) + 1
            log_rows.append({"paper_id": pid, "year": year,
                             "title": title, "source": source})

    # Save cache
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

    df.to_parquet(PARQUET, index=False)
    has_abs_new = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    gain = has_abs_new.sum() - has_abs.sum()

    print(f"\nResults:")
    for src, n in hits.items():
        if n:
            print(f"  {src}: {n}")
    print(f"  New abstracts added: {gain}")
    print(f"  Final coverage: {has_abs_new.sum():,} / {len(df):,} "
          f"({has_abs_new.mean()*100:.1f}%)")

    if log_rows:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(log_rows)
        if LOG_PATH.exists():
            new_df = pd.concat([pd.read_csv(LOG_PATH), new_df], ignore_index=True)
        new_df.to_csv(LOG_PATH, index=False)
        print(f"  Log: {LOG_PATH.name} ({len(log_rows)} new rows)")


if __name__ == "__main__":
    main()
