"""
10l_semantic_scholar_abstracts.py
-----------------------------------
Fetches abstracts for remaining missing papers using the Semantic Scholar API.

For each missing paper:
  1. Search S2 by title (conservative 2.5s delay to stay within free tier)
  2. Fuzzy-match returned title to confirm correct paper
  3. Content-validate abstract (≥2 title words must appear)
  4. Save to afa_papers_enriched.parquet

Usage:
    python scripts/10l_semantic_scholar_abstracts.py
    python scripts/10l_semantic_scholar_abstracts.py --force   # ignore cache
    python scripts/10l_semantic_scholar_abstracts.py --limit N # process N papers
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
CACHE_PATH = ROOT / "data" / "raw" / "afa_10l_cache.json"
LOG_PATH   = ROOT / "validation" / "abstract_10l_matches.csv"

load_dotenv(ROOT / ".env")
S2_API_KEY = os.getenv("S2_API_KEY", "")

FUZZY_MIN  = 78
TIMEOUT    = 15
DELAY      = 1.1   # 1 req/sec limit with small buffer

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "PublicationBiasResearch/1.0 (mailto:emrekuvvet@gmail.com)",
    "x-api-key": S2_API_KEY,
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


def is_good(abstract: str) -> bool:
    return bool(abstract) and len(abstract.split()) >= 20


def content_valid(title: str, abstract: str, min_overlap: int = 2) -> bool:
    title_words = {w for w in normalise(title).split() if len(w) >= 4}
    abs_text = normalise(abstract)
    return sum(1 for w in title_words if w in abs_text) >= min_overlap


def s2_search(title: str) -> tuple[str, str]:
    """Search S2 by title → return (abstract, matched_title). Returns ('','') on failure."""
    q_norm = normalise(title)
    try:
        r = SESSION.get("https://api.semanticscholar.org/graph/v1/paper/search", params={
            "query": title[:200],
            "fields": "title,abstract",
            "limit": 5,
        }, timeout=TIMEOUT)
    except Exception:
        return "", ""

    if r.status_code == 429:
        raise RuntimeError("429 rate limited")
    if r.status_code != 200:
        return "", ""

    for paper in r.json().get("data", []):
        cand_title = paper.get("title") or ""
        score = fuzz.token_sort_ratio(q_norm, normalise(cand_title))
        if score < FUZZY_MIN:
            continue
        abstract = paper.get("abstract") or ""
        if is_good(abstract) and content_valid(title, abstract):
            return abstract, cand_title
    return "", ""


def wait_for_s2(max_wait: int = 3600) -> bool:
    """Poll S2 until rate limit clears. Returns True when ready."""
    print("Waiting for Semantic Scholar rate limit to clear (polling every 60s) ...")
    waited = 0
    while waited < max_wait:
        time.sleep(60)
        waited += 60
        try:
            r = SESSION.get("https://api.semanticscholar.org/graph/v1/paper/search", params={
                "query": "test", "fields": "title", "limit": 1,
            }, timeout=TIMEOUT)
            if r.status_code == 200:
                print(f"  Rate limit cleared after {waited}s.")
                return True
            print(f"  {waited}s: still 429 ...")
        except Exception:
            print(f"  {waited}s: request failed, retrying ...")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore cache")
    ap.add_argument("--limit", type=int, default=0, help="max papers to process")
    ap.add_argument("--no-wait", action="store_true", help="fail immediately if rate limited instead of waiting")
    args = ap.parse_args()

    if not S2_API_KEY:
        print("WARNING: S2_API_KEY not set — running unauthenticated (strict rate limits)")
    else:
        print(f"Using S2 API key: {S2_API_KEY[:8]}...")

    # Check if S2 is accessible; wait if not
    try:
        r = SESSION.get("https://api.semanticscholar.org/graph/v1/paper/search", params={
            "query": "test", "fields": "title", "limit": 1,
        }, timeout=TIMEOUT)
        if r.status_code == 429:
            if args.no_wait:
                raise SystemExit("Semantic Scholar is rate limited. Run without --no-wait to wait automatically.")
            if not wait_for_s2():
                raise SystemExit("Timed out waiting for rate limit to clear.")
    except RuntimeError:
        pass

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

    for _, row in tqdm(missing.iterrows(), total=len(missing), desc="10l S2"):
        pid   = str(row["paper_id"])
        title = str(row.get("title", "") or "")

        if pid in cache and not args.force:
            entry    = cache[pid]
            abstract = entry.get("abstract", "")
            source   = entry.get("source", "none")
        else:
            abstract, source = "", "none"
            try:
                abstract, _ = s2_search(title)
                if abstract:
                    source = "semantic_scholar_10l"
            except RuntimeError:
                # Hit rate limit mid-run — save progress and wait
                print("\nRate limited mid-run — saving progress and waiting ...")
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(CACHE_PATH, "w") as f:
                    json.dump(cache, f)
                if not wait_for_s2():
                    raise SystemExit("Timed out waiting for rate limit.")
                # Retry this paper
                try:
                    abstract, _ = s2_search(title)
                    if abstract:
                        source = "semantic_scholar_10l"
                except RuntimeError:
                    abstract, source = "", "none"

            cache[pid] = {"abstract": abstract, "source": source}
            time.sleep(DELAY)

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
