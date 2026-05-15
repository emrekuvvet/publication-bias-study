"""
10b_enrich_afa_abstracts.py
---------------------------
Fetch abstracts for AFA conference papers from OpenAlex (primary) and
Semantic Scholar (fallback), so the AFA pipeline can apply the same
title+abstract keyword filter and LLM prompts as the NBER pipeline.

Input:  data/raw/afa_papers.parquet
Output: data/raw/afa_papers_enriched.parquet  (adds 'abstract', 'abstract_source')
Cache:  data/raw/afa_abstract_cache.json       (avoids re-fetching on re-runs)

Usage:
    python scripts/10b_enrich_afa_abstracts.py
    python scripts/10b_enrich_afa_abstracts.py --no-ss   # skip Semantic Scholar fallback
    python scripts/10b_enrich_afa_abstracts.py --no-cache  # ignore cache, re-fetch all

Prerequisites:
    pip install requests pandas pyarrow rapidfuzz tqdm
"""

import argparse
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
IN_PATH    = ROOT / "data" / "raw" / "afa_papers.parquet"
OUT_PATH   = ROOT / "data" / "raw" / "afa_papers_enriched.parquet"
CACHE_PATH = ROOT / "data" / "raw" / "afa_abstract_cache.json"

OA_EMAIL   = "research@pubbiasstudy.org"
FUZZY_MIN  = 80   # minimum title similarity to accept a match

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

STOP_WORDS = {
    "a","an","the","of","in","on","at","to","for","and","or","but",
    "with","by","from","that","this","is","are","was","were","be",
    "been","being","have","has","had","do","does","did","will","would",
    "can","could","should","may","might","shall","some","any","all",
}


def normalise(title: str) -> str:
    if not title:
        return ""
    nfd = unicodedata.normalize("NFD", str(title))
    s = nfd.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return " ".join(t for t in s.split() if t not in STOP_WORDS and len(t) > 1)


def decode_inverted_index(inv: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted index."""
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, pos_list in inv.items():
        for pos in pos_list:
            positions.append((pos, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def title_match(query_norm: str, candidate_title: str) -> float:
    return fuzz.token_sort_ratio(query_norm, normalise(candidate_title))


# ------------------------------------------------------------------ #
# OpenAlex lookup                                                      #
# ------------------------------------------------------------------ #

def openalex_search(title: str, year: int | None,
                    session: requests.Session) -> str | None:
    """
    Search OpenAlex for a paper by title. Returns decoded abstract or None.
    """
    q_norm = normalise(title)
    params = {
        "search": title[:200],
        "select": "title,abstract_inverted_index,publication_year",
        "per_page": 5,
        "mailto": OA_EMAIL,
    }
    if year:
        params["filter"] = f"publication_year:{year - 2}-{year + 4}"

    try:
        resp = session.get("https://api.openalex.org/works",
                           params=params, timeout=20)
        if resp.status_code != 200:
            return None
        works = resp.json().get("results", [])
    except Exception:
        return None

    for work in works:
        cand_title = work.get("title") or ""
        score = title_match(q_norm, cand_title)
        if score >= FUZZY_MIN:
            inv = work.get("abstract_inverted_index") or {}
            abstract = decode_inverted_index(inv)
            if abstract and len(abstract.split()) >= 20:
                return abstract
    return None


# ------------------------------------------------------------------ #
# Semantic Scholar fallback                                            #
# ------------------------------------------------------------------ #

def ss_search(title: str, year: int | None,
              session: requests.Session) -> str | None:
    """
    Search Semantic Scholar for a paper by title. Returns abstract or None.
    """
    q_norm = normalise(title)
    params: dict = {
        "query": title[:200],
        "fields": "title,abstract,year",
        "limit": 5,
    }
    if year:
        params["year"] = f"{year - 2}-{year + 4}"

    for attempt in range(3):
        try:
            resp = session.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params, timeout=20,
            )
            if resp.status_code == 429:
                time.sleep(60)
                continue
            if resp.status_code != 200:
                return None
            break
        except Exception:
            return None
    else:
        return None

    for paper in resp.json().get("data", []):
        cand_title = paper.get("title") or ""
        score = title_match(q_norm, cand_title)
        if score >= FUZZY_MIN:
            abstract = paper.get("abstract") or ""
            if abstract and len(abstract.split()) >= 20:
                return abstract
    return None


# ------------------------------------------------------------------ #
# Cache helpers                                                        #
# ------------------------------------------------------------------ #

def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            pass
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(no_ss: bool, no_cache: bool) -> None:
    df = pd.read_parquet(IN_PATH)
    print(f"Loaded {len(df):,} AFA papers")

    cache = {} if no_cache else load_cache()

    oa_session = requests.Session()
    oa_session.headers["User-Agent"] = (
        "PublicationBiasResearch/1.0 (mailto:research@pubbiasstudy.org)"
    )

    ss_session = requests.Session()
    ss_session.headers["User-Agent"] = "PublicationBiasResearch/1.0"

    abstracts: list[str]  = []
    sources:   list[str]  = []

    oa_hits = 0
    ss_hits = 0
    misses  = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching abstracts"):
        pid   = row["paper_id"]
        title = str(row.get("title", "") or "")
        year  = int(row["year"]) if pd.notna(row.get("year")) else None

        # Check cache first
        if pid in cache:
            entry = cache[pid]
            abstracts.append(entry.get("abstract", ""))
            sources.append(entry.get("source", "none"))
            continue

        abstract = None
        source   = "none"

        # --- OpenAlex ---
        abstract = openalex_search(title, year, oa_session)
        if abstract:
            source = "openalex"
            oa_hits += 1
        time.sleep(0.12)   # ~8 req/s within polite pool

        # --- Semantic Scholar fallback ---
        if not abstract and not no_ss:
            abstract = ss_search(title, year, ss_session)
            if abstract:
                source = "semantic_scholar"
                ss_hits += 1
            time.sleep(4.0)   # SS free tier ~15 req/min

        if not abstract:
            abstract = ""
            misses += 1

        cache[pid] = {"abstract": abstract, "source": source}
        abstracts.append(abstract)
        sources.append(source)

        # Save cache every 100 papers
        if len(abstracts) % 100 == 0:
            save_cache(cache)

    save_cache(cache)

    df["abstract"]        = abstracts
    df["abstract_source"] = sources

    has_abstract = (df["abstract"].str.len() > 0).sum()
    print(f"\nResults:")
    print(f"  OpenAlex hits:         {oa_hits:,}")
    print(f"  Semantic Scholar hits: {ss_hits:,}")
    print(f"  No abstract:           {misses:,}")
    print(f"  Total with abstract:   {has_abstract:,} / {len(df):,} "
          f"({has_abstract/len(df)*100:.1f}%)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich AFA papers with abstracts from OpenAlex and Semantic Scholar"
    )
    parser.add_argument("--no-ss", action="store_true",
                        help="Skip Semantic Scholar fallback (OpenAlex only)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore existing cache and re-fetch all abstracts")
    args = parser.parse_args()
    main(args.no_ss, args.no_cache)
