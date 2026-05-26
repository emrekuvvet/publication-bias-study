"""
10h_aeaweb_abstracts.py
------------------------
Fetches abstracts for missing AFA 2024 papers by scraping the AEA 2024
conference program on aeaweb.org.

Steps:
  1. Scan session pages 1600–1950 to collect paper hash IDs
  2. Download each paper PDF from aeaweb.org/conference/2024/program/paper/HASH
  3. Extract title + abstract via pdfminer
  4. Fuzzy-match extracted titles to our missing AFA 2024 papers
  5. Save abstracts to afa_papers_enriched.parquet

Usage:
    python scripts/10h_aeaweb_abstracts.py
    python scripts/10h_aeaweb_abstracts.py --force   # ignore cache
"""

import argparse
import io
import json
import pathlib
import re
import time
import unicodedata

import pandas as pd
import requests
from pdfminer.high_level import extract_text as pdf_extract
from rapidfuzz import fuzz
from tqdm import tqdm

ROOT       = pathlib.Path(__file__).parent.parent.resolve()
PARQUET    = ROOT / "data" / "raw" / "afa_papers_enriched.parquet"
CACHE_PATH = ROOT / "data" / "raw" / "afa_10h_cache.json"
LOG_PATH   = ROOT / "validation" / "abstract_10h_matches.csv"

SESSION_START = 1600
SESSION_END   = 1950
FUZZY_MIN     = 80
TIMEOUT       = 20

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": UA})

STOP_WORDS = {
    "a","an","the","of","in","on","at","to","for","and","or","but","with",
    "by","from","that","this","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","can","could",
    "should","may","might","shall","some","any","all",
}


def normalise(title: str) -> str:
    if not title:
        return ""
    s = unicodedata.normalize("NFD", str(title)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return " ".join(t for t in s.split() if t not in STOP_WORDS and len(t) > 1)


def collect_paper_hashes() -> set:
    """Scan session pages and collect all paper hash IDs."""
    hashes = set()
    print(f"Scanning sessions {SESSION_START}–{SESSION_END} ...")
    for sid in tqdm(range(SESSION_START, SESSION_END + 1), desc="sessions"):
        try:
            r = HTTP.get(
                f"https://www.aeaweb.org/conference/2024/program/{sid}",
                timeout=10,
            )
            if r.status_code != 200 or len(r.text) < 3000:
                continue
            found = re.findall(
                r'/conference/2024/program/paper/([A-Za-z0-9]+)', r.text
            )
            hashes.update(found)
        except Exception:
            pass
        time.sleep(0.12)
    print(f"  Total unique paper hashes: {len(hashes)}")
    return hashes


def extract_title_abstract(pdf_bytes: bytes) -> tuple[str, str]:
    """Extract title (first non-empty line) and abstract from PDF bytes."""
    try:
        text = pdf_extract(io.BytesIO(pdf_bytes), maxpages=3)
    except Exception:
        return "", ""

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "", ""

    title = lines[0]

    # Find abstract: look for "Abstract" header then grab text
    full = " ".join(lines)
    m = re.search(
        r'[Aa]bstract[:\s]+(.{80,2000}?)(?:\n\n|\.\s{2,}[A-Z]|\Z)',
        full,
        re.DOTALL,
    )
    if m:
        abstract = re.sub(r'\s+', ' ', m.group(1)).strip()
    else:
        # Fall back: text after line containing "Abstract"
        abstract = ""
        capture = False
        chunks = []
        for line in lines:
            if re.match(r'^[Aa]bstract', line):
                rest = re.sub(r'^[Aa]bstract[:\s]*', '', line).strip()
                if rest:
                    chunks.append(rest)
                capture = True
                continue
            if capture:
                if len(" ".join(chunks)) > 1500:
                    break
                if re.match(r'^(Introduction|[1I][.\s])', line):
                    break
                chunks.append(line)
        abstract = " ".join(chunks).strip()

    return title, abstract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore cache")
    args = ap.parse_args()

    # Load missing 2024 papers
    df = pd.read_parquet(PARQUET)
    noise = df["title"].str.startswith(("Session Chair", "Location:"))
    has_abs = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    missing = df[~has_abs & (df["year"] == 2024) & ~noise].copy()
    print(f"Missing 2024 AFA papers: {len(missing)}")

    missing_norm = {
        row["paper_id"]: normalise(row["title"])
        for _, row in missing.iterrows()
    }

    # Load / init cache
    cache: dict = {}
    if CACHE_PATH.exists() and not args.force:
        with open(CACHE_PATH) as f:
            cache = json.load(f)

    # Step 1: collect paper hashes
    if "hashes" in cache and not args.force:
        hashes = set(cache["hashes"])
        print(f"Using cached {len(hashes)} paper hashes")
    else:
        hashes = collect_paper_hashes()
        cache["hashes"] = list(hashes)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)

    # Step 2 & 3: download PDFs, extract title+abstract
    pdf_cache: dict = cache.get("pdfs", {})
    new_pdfs = 0
    for h in tqdm(hashes, desc="fetching PDFs"):
        if h in pdf_cache and not args.force:
            continue
        try:
            r = HTTP.get(
                f"https://www.aeaweb.org/conference/2024/program/paper/{h}",
                timeout=TIMEOUT,
            )
            if r.status_code != 200 or "pdf" not in r.headers.get("Content-Type", ""):
                pdf_cache[h] = {"title": "", "abstract": ""}
                continue
            title, abstract = extract_title_abstract(r.content)
            pdf_cache[h] = {"title": title, "abstract": abstract}
            new_pdfs += 1
        except Exception:
            pdf_cache[h] = {"title": "", "abstract": ""}
        time.sleep(0.15)

    cache["pdfs"] = pdf_cache
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    print(f"  PDFs processed: {len(pdf_cache)} ({new_pdfs} new)")

    # Step 4: fuzzy match extracted titles to our missing papers
    hits = 0
    log_rows = []

    for h, entry in pdf_cache.items():
        ext_title = entry.get("title", "")
        abstract  = entry.get("abstract", "")
        if not ext_title or not abstract or len(abstract.split()) < 20:
            continue
        ext_norm = normalise(ext_title)
        if not ext_norm:
            continue

        best_pid, best_score = None, 0
        for pid, miss_norm in missing_norm.items():
            score = fuzz.token_sort_ratio(ext_norm, miss_norm)
            if score > best_score:
                best_score, best_pid = score, pid

        if best_score >= FUZZY_MIN and best_pid:
            idx = df[df["paper_id"] == best_pid].index
            if len(idx) and (
                pd.isna(df.loc[idx[0], "abstract"])
                or df.loc[idx[0], "abstract"].strip() == ""
            ):
                df.loc[idx[0], "abstract"]        = abstract
                df.loc[idx[0], "abstract_source"] = "aeaweb_10h"
                hits += 1
                orig_title = df.loc[idx[0], "title"]
                print(f"  MATCH ({best_score}): '{orig_title[:70]}' ← '{ext_title[:70]}'")
                log_rows.append({
                    "paper_id": best_pid, "hash": h,
                    "score": best_score, "aea_title": ext_title,
                    "afa_title": orig_title,
                })
                # Remove from missing pool to avoid double-assignment
                del missing_norm[best_pid]

    df.to_parquet(PARQUET, index=False)
    print(f"\nResults: {hits} abstracts added for 2024 AFA papers")

    if log_rows:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(log_rows)
        if LOG_PATH.exists():
            new_df = pd.concat([pd.read_csv(LOG_PATH), new_df], ignore_index=True)
        new_df.to_csv(LOG_PATH, index=False)
        print(f"Log: {LOG_PATH.name} ({len(log_rows)} new rows)")


if __name__ == "__main__":
    main()
