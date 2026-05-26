"""
10f_repec_abstracts.py
-----------------------
Retrieves abstracts for AFA papers still missing after 10e, using IDEAS/RePEC.

Strategy (two-step per paper):
  1. POST title (or first-author + keywords) to IDEAS CGI search
  2. Fetch the top result pages and extract <META NAME="citation_abstract">
  3. Accept if fuzzy title score >= FUZZY_MIN (70 — lower than other sources
     because paper titles evolve between AFA presentation and RepEc indexing)

RePEC is especially useful for the ~77% of missing papers that exist only as
SSRN working papers: SSRN blocks scrapers but IDEAS indexes many of the same
papers and is openly accessible.

Usage:
    python scripts/10f_repec_abstracts.py          # all still-missing
    python scripts/10f_repec_abstracts.py --force  # ignore cache

Prerequisites:
    pip install requests pandas pyarrow rapidfuzz tqdm python-dotenv
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
CACHE_PATH = ROOT / "data" / "raw" / "afa_10f_cache.json"
LOG_PATH   = ROOT / "validation" / "abstract_10f_matches.csv"

FUZZY_MIN     = 70   # lower than other sources — titles change from AFA to RePEC
SEARCH_DELAY  = 1.5  # seconds between search POSTs (avoid rate-limiting)
PAGE_DELAY    = 1.0  # seconds between individual paper-page fetches
TIMEOUT       = 12

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


def is_good(abstract: str) -> bool:
    return bool(abstract) and len(abstract.split()) >= 20


def make_query(title: str, authors: str) -> str:
    """Return up to 8 significant title words + first author's last name."""
    words = [w for w in normalise(title).split() if len(w) > 2][:8]
    if authors:
        last = str(authors).split(",")[0].split()
        if last:
            words.append(last[-1])
    return " ".join(words)


def repec_search(title: str, authors: str) -> str | None:
    """Search IDEAS/RePEC and return the first matching abstract, or None."""
    q_norm = normalise(title)

    for query in [title[:120], make_query(title, authors)]:
        try:
            r = SESSION.post(
                "https://ideas.repec.org/cgi-bin/htsearch2",
                data={"q": query},
                timeout=TIMEOUT,
            )
            time.sleep(SEARCH_DELAY)   # always delay after a search POST
            if r.status_code != 200:
                continue
        except Exception:
            time.sleep(SEARCH_DELAY)
            continue

        links = re.findall(
            r'href="(https://ideas\.repec\.org/[ap]/[^"]+)"', r.text
        )
        if not links:
            continue

        for link in links[:4]:
            time.sleep(PAGE_DELAY)
            try:
                rp = SESSION.get(link, timeout=TIMEOUT)
                if rp.status_code != 200:
                    continue
            except Exception:
                continue

            mt = re.search(
                r'META NAME="citation_title" content="([^"]+)"',
                rp.text, re.IGNORECASE
            )
            page_title = mt.group(1) if mt else ""
            score = fuzz.token_sort_ratio(q_norm, normalise(page_title))
            if score < FUZZY_MIN:
                continue

            ma = re.search(
                r'META NAME="citation_abstract" content="([^"]{60,})"',
                rp.text, re.IGNORECASE
            )
            if ma and is_good(ma.group(1)):
                return ma.group(1).strip()

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print(f"Loading {PARQUET.name} ...")
    df = pd.read_parquet(PARQUET)
    has_abs = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    print(f"  Starting coverage: {has_abs.sum():,} / {len(df):,} "
          f"({has_abs.mean()*100:.1f}%)")

    missing = df[~has_abs].copy()
    print(f"  Papers to query: {len(missing):,}")

    cache: dict = {}
    if CACHE_PATH.exists() and not args.force:
        with open(CACHE_PATH) as f:
            cache = json.load(f)

    hits = 0
    log_rows = []

    for _, row in tqdm(missing.iterrows(), total=len(missing), desc="RePEC"):
        pid     = str(row["paper_id"])
        title   = str(row.get("title", "") or "")
        authors = str(row.get("authors", "") or "")
        year    = int(row["year"]) if pd.notna(row.get("year")) else None

        if pid in cache and not args.force:
            abstract = cache[pid].get("abstract", "")
        else:
            abstract = repec_search(title, authors) or ""
            cache[pid] = {"abstract": abstract,
                          "source": "repec" if abstract else "none"}

        if abstract:
            df.loc[row.name, "abstract"]        = abstract
            df.loc[row.name, "abstract_source"] = "repec"
            hits += 1
            log_rows.append({"paper_id": pid, "year": year,
                             "title": title, "source": "repec"})

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

    df.to_parquet(PARQUET, index=False)
    has_new = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    gain = has_new.sum() - has_abs.sum()

    print(f"\nResults:")
    print(f"  New abstracts added: {gain}")
    print(f"  Final coverage: {has_new.sum():,} / {len(df):,} "
          f"({has_new.mean()*100:.1f}%)")

    if log_rows:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(log_rows)
        if LOG_PATH.exists():
            new_df = pd.concat([pd.read_csv(LOG_PATH), new_df], ignore_index=True)
        new_df.to_csv(LOG_PATH, index=False)
        print(f"  Log: {LOG_PATH.name} ({len(log_rows)} new rows)")


if __name__ == "__main__":
    main()
