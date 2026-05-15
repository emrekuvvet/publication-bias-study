"""
10b_enrich_afa_abstracts.py
---------------------------
Fetch abstracts for AFA conference papers from four sources (in priority order):
  1. OpenAlex      — free, no key, good journal coverage
  2. CrossRef      — free, no key, good for published papers
  3. Semantic Scholar — free, covers preprints/working papers
  4. arXiv         — free, covers finance preprints

Input:  data/raw/afa_papers.parquet
Output: data/raw/afa_papers_enriched.parquet  (adds 'abstract', 'abstract_source')
Cache:  data/raw/afa_abstract_cache.json       (avoids re-fetching on re-runs)

Usage:
    python scripts/10b_enrich_afa_abstracts.py               # full run
    python scripts/10b_enrich_afa_abstracts.py --retry-misses # only re-query cache misses
    python scripts/10b_enrich_afa_abstracts.py --no-cache     # ignore cache, re-fetch all

Environment (optional):
    SEMANTIC_SCHOLAR_API_KEY  — free key from semanticscholar.org/product/api
                                 raises rate limit from 15→100 req/s

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
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz
from tqdm import tqdm

ROOT       = pathlib.Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")

IN_PATH    = ROOT / "data" / "raw" / "afa_papers.parquet"
OUT_PATH   = ROOT / "data" / "raw" / "afa_papers_enriched.parquet"
CACHE_PATH = ROOT / "data" / "raw" / "afa_abstract_cache.json"

OA_EMAIL  = "research@pubbiasstudy.org"
FUZZY_MIN = 80   # minimum title similarity to accept a match

SS_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

# ------------------------------------------------------------------ #
# Text helpers                                                         #
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


def title_match(query_norm: str, candidate_title: str) -> float:
    return fuzz.token_sort_ratio(query_norm, normalise(candidate_title))


def strip_html(text: str) -> str:
    """Remove XML/HTML tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


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


def is_good(abstract: str | None) -> bool:
    return bool(abstract) and len(abstract.split()) >= 20


# ------------------------------------------------------------------ #
# Source 1 — OpenAlex                                                  #
# ------------------------------------------------------------------ #

def openalex_search(title: str, year: int | None,
                    session: requests.Session) -> str | None:
    q_norm = normalise(title)
    params: dict = {
        "search": title[:200],
        "select": "title,abstract_inverted_index,publication_year",
        "per_page": 5,
        "mailto": OA_EMAIL,
    }
    if year:
        params["filter"] = f"publication_year:{year - 2}-{year + 4}"

    try:
        resp = session.get("https://api.openalex.org/works",
                           params=params, timeout=15)
        if resp.status_code != 200:
            return None
        works = resp.json().get("results", [])
    except Exception:
        return None

    for work in works:
        cand_title = work.get("title") or ""
        if title_match(q_norm, cand_title) >= FUZZY_MIN:
            abstract = decode_inverted_index(work.get("abstract_inverted_index") or {})
            if is_good(abstract):
                return abstract
    return None


# ------------------------------------------------------------------ #
# Source 2 — CrossRef                                                  #
# ------------------------------------------------------------------ #

def crossref_search(title: str, year: int | None,
                    session: requests.Session) -> str | None:
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
        resp = session.get("https://api.crossref.org/works",
                           params=params, timeout=12)
        if resp.status_code != 200:
            return None
        items = resp.json().get("message", {}).get("items", [])
    except Exception:
        return None

    for item in items:
        titles = item.get("title") or []
        cand_title = titles[0] if titles else ""
        if title_match(q_norm, cand_title) >= FUZZY_MIN:
            raw = item.get("abstract") or ""
            if raw:
                abstract = strip_html(raw)
                if is_good(abstract):
                    return abstract
    return None


# ------------------------------------------------------------------ #
# Source 3 — Semantic Scholar                                          #
# ------------------------------------------------------------------ #

def ss_search(title: str, year: int | None,
              session: requests.Session) -> str | None:
    q_norm = normalise(title)
    params: dict = {
        "query": title[:200],
        "fields": "title,abstract,year",
        "limit": 5,
    }
    if year:
        params["year"] = f"{year - 2}-{year + 4}"

    for attempt in range(2):
        try:
            resp = session.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
                timeout=8,          # tight timeout — don't let one paper stall the run
            )
            if resp.status_code == 429:
                time.sleep(10)      # short back-off (was 60s — caused multi-minute hangs)
                continue
            if resp.status_code != 200:
                return None
            break
        except requests.exceptions.Timeout:
            return None             # skip rather than hang
        except Exception:
            return None
    else:
        return None

    for paper in resp.json().get("data", []):
        cand_title = paper.get("title") or ""
        if title_match(q_norm, cand_title) >= FUZZY_MIN:
            abstract = paper.get("abstract") or ""
            if is_good(abstract):
                return abstract
    return None


# ------------------------------------------------------------------ #
# Source 4 — arXiv                                                     #
# ------------------------------------------------------------------ #

_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def arxiv_search(title: str, year: int | None,
                 session: requests.Session) -> str | None:
    q_norm = normalise(title)
    # Use title search — quote the first 10 words for precision
    short_title = " ".join(title.split()[:10])
    params = {
        "search_query": f'ti:"{short_title}"',
        "max_results": 3,
        "sortBy": "relevance",
    }

    try:
        resp = session.get("https://export.arxiv.org/api/query",
                           params=params, timeout=12)
        if resp.status_code != 200:
            return None
        root = ET.fromstring(resp.text)
    except Exception:
        return None

    for entry in root.findall("atom:entry", _ARXIV_NS):
        title_el = entry.find("atom:title", _ARXIV_NS)
        cand_title = (title_el.text or "").replace("\n", " ").strip()
        if title_match(q_norm, cand_title) >= FUZZY_MIN:
            summary_el = entry.find("atom:summary", _ARXIV_NS)
            if summary_el is not None:
                abstract = " ".join((summary_el.text or "").split())
                if is_good(abstract):
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

def main(retry_misses: bool, no_cache: bool,
         no_ss: bool = False, no_arxiv: bool = False) -> None:
    df = pd.read_parquet(IN_PATH)
    print(f"Loaded {len(df):,} AFA papers")

    cache = {} if no_cache else load_cache()

    if retry_misses:
        prior_misses = {pid for pid, v in cache.items() if not v.get("abstract")}
        print(f"--retry-misses: will re-query {len(prior_misses):,} cache misses "
              f"with CrossRef / Semantic Scholar / arXiv")
        # Evict miss entries so the main loop re-queries them
        for pid in prior_misses:
            del cache[pid]

    # Build sessions
    def make_session(extra_headers: dict | None = None) -> requests.Session:
        s = requests.Session()
        s.headers["User-Agent"] = (
            "PublicationBiasResearch/1.0 (mailto:research@pubbiasstudy.org)"
        )
        if extra_headers:
            s.headers.update(extra_headers)
        return s

    oa_session  = make_session()
    cr_session  = make_session()
    ss_headers  = {"x-api-key": SS_API_KEY} if SS_API_KEY else {}
    ss_session  = make_session(ss_headers)
    arx_session = make_session()

    if SS_API_KEY:
        print(f"  Semantic Scholar API key loaded — higher rate limits enabled")
    else:
        print(f"  No SEMANTIC_SCHOLAR_API_KEY found — using free tier (~1 req/s)")

    abstracts: list[str] = []
    sources:   list[str] = []
    hits: dict[str, int] = {"openalex": 0, "crossref": 0,
                             "semantic_scholar": 0, "arxiv": 0}
    misses = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching abstracts"):
        pid   = row["paper_id"]
        title = str(row.get("title", "") or "")
        year  = int(row["year"]) if pd.notna(row.get("year")) else None

        # Serve from cache if already resolved
        if pid in cache:
            entry = cache[pid]
            abstracts.append(entry.get("abstract", ""))
            sources.append(entry.get("source", "none"))
            continue

        abstract: str | None = None
        source = "none"

        # --- 1. OpenAlex ---
        abstract = openalex_search(title, year, oa_session)
        if abstract:
            source = "openalex"
            hits["openalex"] += 1
        time.sleep(0.12)

        # --- 2. CrossRef ---
        if not abstract:
            abstract = crossref_search(title, year, cr_session)
            if abstract:
                source = "crossref"
                hits["crossref"] += 1
            time.sleep(0.12)

        # --- 3. Semantic Scholar ---
        if not abstract and not no_ss:
            abstract = ss_search(title, year, ss_session)
            if abstract:
                source = "semantic_scholar"
                hits["semantic_scholar"] += 1
            # SS free tier: ~1 req/s; key tier: much faster
            time.sleep(1.1 if not SS_API_KEY else 0.05)

        # --- 4. arXiv ---
        if not abstract and not no_arxiv:
            abstract = arxiv_search(title, year, arx_session)
            if abstract:
                source = "arxiv"
                hits["arxiv"] += 1
            time.sleep(0.5)

        if not abstract:
            abstract = ""
            misses += 1

        cache[pid] = {"abstract": abstract, "source": source}
        abstracts.append(abstract)
        sources.append(source)

        if len(abstracts) % 100 == 0:
            save_cache(cache)
            total_found = sum(hits.values())
            processed = len(abstracts)
            tqdm.write(f"  [{processed}/{len(df)}] hits so far: "
                       f"OA={hits['openalex']} CR={hits['crossref']} "
                       f"SS={hits['semantic_scholar']} arXiv={hits['arxiv']} "
                       f"miss={misses} ({total_found/processed*100:.1f}% coverage)")

    save_cache(cache)

    df["abstract"]        = abstracts
    df["abstract_source"] = sources

    has_abstract = (df["abstract"].str.len() > 0).sum()
    total_hits   = sum(hits.values())
    print(f"\nResults:")
    print(f"  OpenAlex:         {hits['openalex']:,}")
    print(f"  CrossRef:         {hits['crossref']:,}")
    print(f"  Semantic Scholar: {hits['semantic_scholar']:,}")
    print(f"  arXiv:            {hits['arxiv']:,}")
    print(f"  No abstract:      {misses:,}")
    print(f"  Total with abstract: {has_abstract:,} / {len(df):,} "
          f"({has_abstract/len(df)*100:.1f}%)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich AFA papers with abstracts from OpenAlex, CrossRef, "
                    "Semantic Scholar, and arXiv"
    )
    parser.add_argument(
        "--retry-misses", action="store_true",
        help="Re-query only cache misses (source='none') with CrossRef/SS/arXiv. "
             "Skips papers already found by OpenAlex. Fast — ~10-30 min for 3,500 papers."
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore existing cache and re-fetch all abstracts from scratch."
    )
    parser.add_argument(
        "--no-ss", action="store_true",
        help="Skip Semantic Scholar (useful without an API key — free tier is slow)."
    )
    parser.add_argument(
        "--no-arxiv", action="store_true",
        help="Skip arXiv (low hit rate for finance papers)."
    )
    args = parser.parse_args()
    main(args.retry_misses, args.no_cache, args.no_ss, args.no_arxiv)
