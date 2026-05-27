"""
10k_elsevier_abstracts.py
--------------------------
Fetches abstracts for remaining missing papers using the Elsevier API.

Pipeline for each missing paper:
  1. Search Scopus by title → get DOI
  2. Fetch abstract via api.elsevier.com/content/article/doi/{doi}?view=META_ABS
  3. Fuzzy-match title to confirm correct paper before saving

Requires ELSEVIER_API_KEY in .env (free Elsevier developer key works).

Usage:
    python scripts/10k_elsevier_abstracts.py
    python scripts/10k_elsevier_abstracts.py --force   # ignore cache
    python scripts/10k_elsevier_abstracts.py --limit N # process N papers
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
PARQUET    = ROOT / "data" / "raw" / "afa_papers_enriched.parquet"
CACHE_PATH = ROOT / "data" / "raw" / "afa_10k_cache.json"
LOG_PATH   = ROOT / "validation" / "abstract_10k_matches.csv"

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("ELSEVIER_API_KEY", "")

FUZZY_MIN = 78
TIMEOUT   = 15

SESSION = requests.Session()
SESSION.headers.update({
    "Accept": "application/json",
    "X-ELS-APIKey": API_KEY,
})

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


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


def is_good(abstract: str) -> bool:
    return bool(abstract) and len(abstract.split()) >= 20


def content_valid(title: str, abstract: str, min_overlap: int = 2) -> bool:
    title_words = {w for w in normalise(title).split() if len(w) >= 4}
    abs_text = normalise(abstract)
    return sum(1 for w in title_words if w in abs_text) >= min_overlap


def scopus_get_doi(title: str) -> tuple[str, str]:
    """Search Scopus by title → return (doi, found_title)."""
    try:
        r = SESSION.get("https://api.elsevier.com/content/search/scopus", params={
            "query": f'TITLE("{title}")',
            "field": "dc:title,prism:doi",
            "count": 5,
            "httpAccept": "application/json",
        }, timeout=TIMEOUT)
        if r.status_code != 200:
            return "", ""
        entries = r.json().get("search-results", {}).get("entry", [])
    except Exception:
        return "", ""

    q_norm = normalise(title)
    for e in entries:
        found_title = e.get("dc:title", "")
        doi = e.get("prism:doi", "")
        if not doi:
            continue
        score = fuzz.token_sort_ratio(q_norm, normalise(found_title))
        if score >= FUZZY_MIN:
            return doi, found_title
    return "", ""


def elsevier_abstract(doi: str) -> str:
    """Fetch abstract via Elsevier article/doi endpoint with META_ABS view."""
    try:
        r = SESSION.get(
            f"https://api.elsevier.com/content/article/doi/{doi}",
            params={"httpAccept": "application/json", "view": "META_ABS"},
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            return ""
        core = r.json().get("full-text-retrieval-response", {}).get("coredata", {})
        raw = core.get("dc:description", "")
        return strip_html(raw).strip() if raw else ""
    except Exception:
        return ""


def main():
    if not API_KEY:
        raise SystemExit("ELSEVIER_API_KEY not set in .env")

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore cache")
    ap.add_argument("--limit", type=int, default=0, help="max papers to process")
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    has_abs = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    noise = df["title"].str.startswith(("Session Chair", "Location:"))
    missing = df[~has_abs & ~noise].copy()

    if args.limit:
        missing = missing.head(args.limit)

    print(f"Papers missing abstracts : {len(missing)}")
    print(f"Starting coverage        : {has_abs.sum():,} / {len(df):,} ({has_abs.mean()*100:.1f}%)")

    cache: dict = {}
    if CACHE_PATH.exists() and not args.force:
        with open(CACHE_PATH) as f:
            cache = json.load(f)

    hits = 0
    log_rows = []

    for _, row in tqdm(missing.iterrows(), total=len(missing), desc="10k Elsevier"):
        pid   = str(row["paper_id"])
        title = str(row.get("title", "") or "")

        if pid in cache and not args.force:
            entry    = cache[pid]
            abstract = entry.get("abstract", "")
            source   = entry.get("source", "none")
        else:
            abstract, source = "", "none"

            doi, found_title = scopus_get_doi(title)
            time.sleep(0.3)

            if doi:
                abstract = elsevier_abstract(doi)
                time.sleep(0.3)
                if abstract and is_good(abstract) and content_valid(title, abstract):
                    source = "elsevier_10k"
                else:
                    abstract = ""

            cache[pid] = {"abstract": abstract, "source": source, "doi": doi}

            if len(cache) % 100 == 0:
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(CACHE_PATH, "w") as f:
                    json.dump(cache, f)

        if abstract:
            df.loc[row.name, "abstract"]        = abstract
            df.loc[row.name, "abstract_source"] = source
            hits += 1
            log_rows.append({
                "paper_id": pid, "year": row.get("year"),
                "title": title, "source": source,
            })

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

    df.to_parquet(PARQUET, index=False)
    has_new = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    gain = has_new.sum() - has_abs.sum()

    print(f"\nResults:")
    print(f"  New abstracts added : {gain}")
    print(f"  Final coverage      : {has_new.sum():,} / {len(df):,} ({has_new.mean()*100:.1f}%)")

    if log_rows:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(log_rows)
        if LOG_PATH.exists():
            new_df = pd.concat([pd.read_csv(LOG_PATH), new_df], ignore_index=True)
        new_df.to_csv(LOG_PATH, index=False)
        print(f"  Log: {LOG_PATH.name} ({len(log_rows)} new rows)")


if __name__ == "__main__":
    main()
