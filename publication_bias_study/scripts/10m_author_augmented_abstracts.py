"""
10m_author_augmented_abstracts.py
-----------------------------------
Fetches abstracts for remaining missing papers using author-augmented search.

Searches CrossRef (primary) and OpenAlex (fallback) with BOTH the paper title
and the first author's last name in the query.  This recovers papers that
title-only scripts (10i-10l) miss because:
  - Short/ambiguous AFA titles like "Does it Work?" return wrong papers
    when searched by title alone; adding the author name pins the result.
  - The AFA title is often a short subtitle of the full published title;
    token_set_ratio handles this correctly (partial containment).

Per-paper pipeline:
  1. Parse first author's last name from the AFA authors field
  2. CrossRef: query.title + query.author → fuzzy title + author validation
  3. OpenAlex fallback: combined title+author free-text search → same checks
  4. Content-validate abstract (≥2 significant title words must appear)

Usage:
    python scripts/10m_author_augmented_abstracts.py
    python scripts/10m_author_augmented_abstracts.py --force
    python scripts/10m_author_augmented_abstracts.py --limit N
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
PARQUET    = ROOT / "data" / "raw" / "afa_papers_enriched.parquet"
CACHE_PATH = ROOT / "data" / "raw" / "afa_10m_cache.json"
LOG_PATH   = ROOT / "validation" / "abstract_10m_matches.csv"

FUZZY_MIN = 65   # lower threshold: author match compensates; token_set handles substrings
TIMEOUT   = 12
EMAIL     = "emrekuvvet@gmail.com"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": f"PublicationBiasResearch/1.0 (mailto:{EMAIL})"})

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


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


def is_good(abstract: str) -> bool:
    return bool(abstract) and len(abstract.split()) >= 20


def content_valid(title: str, abstract: str, min_overlap: int = 2) -> bool:
    title_words = {w for w in normalise(title).split() if len(w) >= 4}
    abs_text = normalise(abstract)
    return sum(1 for w in title_words if w in abs_text) >= min_overlap


def first_lastname(authors_str: str) -> str:
    """Extract the last name of the first author from the AFA authors string.

    Format: 'First [M.] Last, Affiliation [and First2 Last2, Affiliation2 ...]'
    """
    if not authors_str:
        return ""
    # Everything before the first comma is the first author's full name
    name_part = str(authors_str).split(",")[0].strip()
    words = name_part.split()
    return words[-1] if words else ""


def author_match(afa_authors: str, api_authors: list[str]) -> bool:
    """Return True if the first author's last name appears in any API author name."""
    lastname = first_lastname(afa_authors).lower()
    if not lastname:
        return True  # can't verify → don't reject
    for name in api_authors:
        if lastname in name.lower():
            return True
    return False


def crossref_search(title: str, lastname: str) -> tuple[str, str]:
    """CrossRef query.title + query.author → (abstract, matched_title)."""
    q_norm = normalise(title)
    try:
        r = SESSION.get("https://api.crossref.org/works", params={
            "query.title":  title[:200],
            "query.author": lastname,
            "rows": 5,
            "select": "title,abstract,author",
            "mailto": EMAIL,
        }, timeout=TIMEOUT)
        if r.status_code != 200:
            return "", ""
        items = r.json().get("message", {}).get("items", [])
    except Exception:
        return "", ""

    for item in items:
        cand_title = (item.get("title") or [""])[0]
        # Use token_set_ratio: handles AFA short title being a subset of full title
        score = fuzz.token_set_ratio(q_norm, normalise(cand_title))
        if score < FUZZY_MIN:
            continue
        api_authors = [
            f"{a.get('given','')} {a.get('family','')}".strip()
            for a in (item.get("author") or [])
        ]
        if not author_match(lastname, api_authors):
            continue
        raw = item.get("abstract") or ""
        if not raw:
            continue
        abstract = strip_html(raw)
        if is_good(abstract) and content_valid(title, abstract):
            return abstract, cand_title
    return "", ""


def openalex_search(title: str, lastname: str) -> tuple[str, str]:
    """OpenAlex free-text search combining title words + author last name."""
    q_norm = normalise(title)
    # Append lastname to the search query so OpenAlex ranks author-matching
    # results higher; we still validate author in post-processing
    combined_query = f"{title[:180]} {lastname}"
    try:
        r = SESSION.get("https://api.openalex.org/works", params={
            "search":   combined_query,
            "per-page": 10,
            "select":   "title,abstract_inverted_index,authorships",
            "mailto":   EMAIL,
        }, timeout=TIMEOUT)
        if r.status_code != 200:
            return "", ""
        results = r.json().get("results", [])
    except Exception:
        return "", ""

    for item in results:
        cand_title = item.get("title") or ""
        score = fuzz.token_set_ratio(q_norm, normalise(cand_title))
        if score < FUZZY_MIN:
            continue
        api_authors = [
            a.get("author", {}).get("display_name", "")
            for a in (item.get("authorships") or [])
        ]
        if not author_match(lastname, api_authors):
            continue
        inv = item.get("abstract_inverted_index") or {}
        if not inv:
            continue
        pos_word = [(p, w) for w, positions in inv.items() for p in positions]
        abstract = " ".join(w for _, w in sorted(pos_word))
        if is_good(abstract) and content_valid(title, abstract):
            return abstract, cand_title
    return "", ""


def main():
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

    log_rows = []

    for _, row in tqdm(missing.iterrows(), total=len(missing), desc="10m author+title"):
        pid      = str(row["paper_id"])
        title    = str(row.get("title", "") or "")
        authors  = str(row.get("authors", "") or "")
        lastname = first_lastname(authors)

        if pid in cache and not args.force:
            entry    = cache[pid]
            abstract = entry.get("abstract", "")
            source   = entry.get("source", "none")
        else:
            abstract, source = "", "none"

            # Primary: CrossRef author-augmented search
            if lastname:
                abstract, _ = crossref_search(title, lastname)
                if abstract:
                    source = "crossref_10m"

            # Fallback: OpenAlex combined search
            if not abstract and lastname:
                abstract, _ = openalex_search(title, lastname)
                if abstract:
                    source = "openalex_10m"

            cache[pid] = {"abstract": abstract, "source": source}
            time.sleep(0.15)

            if len(cache) % 100 == 0:
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(CACHE_PATH, "w") as f:
                    json.dump(cache, f)

        if abstract:
            df.loc[row.name, "abstract"]        = abstract
            df.loc[row.name, "abstract_source"] = source
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
