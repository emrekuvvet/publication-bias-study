"""
10g_lower_threshold_retry.py
-----------------------------
Retries abstract retrieval for in-scope AFA papers still missing abstracts
using a lower fuzzy-match threshold (65) with an additional content-keyword
validation gate to control false positives.

Motivation: The standard threshold (80) used in 10e misses papers where the
AFA program title differs from the published title. For in-scope papers—which
are critical for the analysis—a lower threshold with content validation is
preferable to leaving abstracts blank.

Content validation: at least 2 of the significant (>3 char, non-stopword)
title words must also appear in the abstract text.  This rejects clear
mismatches (e.g. physics papers matching finance titles) that score 65–74.

Sources: OpenAlex (wide year window), CrossRef (wide year window).

Usage:
    python scripts/10g_lower_threshold_retry.py          # in-scope missing only
    python scripts/10g_lower_threshold_retry.py --all    # all missing papers
    python scripts/10g_lower_threshold_retry.py --force  # ignore cache

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
PARQUET    = ROOT / "data" / "raw" / "afa_papers_enriched.parquet"
DIR_PARQ   = ROOT / "data" / "classified" / "afa_direction.parquet"
CACHE_PATH = ROOT / "data" / "raw" / "afa_10g_cache.json"
LOG_PATH   = ROOT / "validation" / "abstract_10g_matches.csv"

FUZZY_MIN  = 65
TIMEOUT    = 10
OA_EMAIL   = "research@pubbiasstudy.org"

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


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


def is_good(abstract: str) -> bool:
    return bool(abstract) and len(abstract.split()) >= 20


def content_valid(title: str, abstract: str, min_overlap: int = 2) -> bool:
    title_words = {w for w in normalise(title).split() if len(w) >= 4}
    abs_text = normalise(abstract)
    return sum(1 for w in title_words if w in abs_text) >= min_overlap


def openalex_search(title: str, year: int | None) -> str | None:
    q_norm = normalise(title)
    params: dict = {
        "search": title[:200],
        "per-page": 10,
        "select": "title,abstract_inverted_index",
        "mailto": OA_EMAIL,
    }
    if year:
        params["filter"] = f"publication_year:{year - 3}-{year + 5}"
    try:
        r = SESSION.get("https://api.openalex.org/works", params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
    except Exception:
        return None
    for item in results:
        cand = item.get("title") or ""
        score = fuzz.token_sort_ratio(q_norm, normalise(cand))
        if score < FUZZY_MIN:
            continue
        inv = item.get("abstract_inverted_index") or {}
        if not inv:
            continue
        pos_word = [(p, w) for w, positions in inv.items() for p in positions]
        abstract = " ".join(w for _, w in sorted(pos_word))
        if is_good(abstract) and content_valid(title, abstract):
            return abstract
    return None


def crossref_search(title: str, year: int | None) -> str | None:
    q_norm = normalise(title)
    params: dict = {
        "query.title": title[:200],
        "rows": 10,
        "select": "title,abstract",
        "mailto": OA_EMAIL,
    }
    if year:
        params["filter"] = f"from-pub-date:{year - 3},until-pub-date:{year + 5}"
    try:
        r = SESSION.get("https://api.crossref.org/works", params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        items = r.json().get("message", {}).get("items", [])
    except Exception:
        return None
    for item in items:
        cand = (item.get("title") or [""])[0]
        score = fuzz.token_sort_ratio(q_norm, normalise(cand))
        if score < FUZZY_MIN:
            continue
        raw = item.get("abstract") or ""
        if not raw:
            continue
        abstract = strip_html(raw)
        if is_good(abstract) and content_valid(title, abstract):
            return abstract
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all",   action="store_true", help="retry all missing, not just in-scope")
    ap.add_argument("--force", action="store_true", help="ignore cache")
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    has_abs = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    print(f"Starting coverage: {has_abs.sum():,} / {len(df):,} ({has_abs.mean()*100:.1f}%)")

    if args.all:
        missing = df[~has_abs].copy()
    else:
        # In-scope papers only (requires direction parquet from step 11)
        if not DIR_PARQ.exists():
            raise FileNotFoundError(
                f"{DIR_PARQ.name} not found — run 11_afa_analysis.py --step classify first"
            )
        dir_df = pd.read_parquet(DIR_PARQ)
        inscope_pids = set(dir_df["paper_id"])
        missing = df[~has_abs & df["paper_id"].isin(inscope_pids)].copy()
        print(f"  Restricting to {len(missing)} in-scope papers without abstracts")

    cache: dict = {}
    if CACHE_PATH.exists() and not args.force:
        with open(CACHE_PATH) as f:
            cache = json.load(f)

    hits = 0
    log_rows = []

    for _, row in tqdm(missing.iterrows(), total=len(missing), desc="10g retry"):
        pid   = str(row["paper_id"])
        title = str(row.get("title", "") or "")
        year  = int(row["year"]) if pd.notna(row.get("year")) else None

        if pid in cache and not args.force:
            entry    = cache[pid]
            abstract = entry.get("abstract", "")
            source   = entry.get("source", "none")
        else:
            abstract, source = "", "none"

            abstract = openalex_search(title, year) or ""
            if abstract:
                source = "openalex_10g"

            if not abstract:
                abstract = crossref_search(title, year) or ""
                if abstract:
                    source = "crossref_10g"

            cache[pid] = {"abstract": abstract, "source": source}
            time.sleep(0.1)

        if abstract:
            df.loc[row.name, "abstract"]        = abstract
            df.loc[row.name, "abstract_source"] = source
            hits += 1
            log_rows.append({"paper_id": pid, "year": year,
                             "title": title, "source": source})

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

    df.to_parquet(PARQUET, index=False)
    has_new = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    gain = has_new.sum() - has_abs.sum()

    print(f"\nResults:")
    print(f"  New abstracts added: {gain}")
    print(f"  Final coverage: {has_new.sum():,} / {len(df):,} ({has_new.mean()*100:.1f}%)")

    if log_rows:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(log_rows)
        if LOG_PATH.exists():
            new_df = pd.concat([pd.read_csv(LOG_PATH), new_df], ignore_index=True)
        new_df.to_csv(LOG_PATH, index=False)
        print(f"  Log: {LOG_PATH.name} ({len(log_rows)} new rows)")


if __name__ == "__main__":
    main()
