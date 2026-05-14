"""
01_collect_nber.py
------------------
Collect NBER working papers 2000–2024 via the public NBER search API.

Endpoint: https://www.nber.org/api/v1/search
          ?facet=contentType:working_paper
          &page=N&perPage=100&sortBy=public_date

No authentication required.
Total corpus: ~35,000+ working papers across all programs.

The program code column (programs) is populated from the NBER batch
incremental JSON files where available; otherwise left empty.  Program
filtering for analysis is performed downstream in script 03 (via keyword
filtering and LLM classification) rather than here.

Output: data/raw/nber_papers.parquet

Usage:
    python scripts/01_collect_nber.py
    python scripts/01_collect_nber.py --start 2000 --end 2024
    python scripts/01_collect_nber.py --test          # first 2 pages only
"""

import argparse
import json
import pathlib
import re
import time
from html.parser import HTMLParser

import pandas as pd
import requests
from tqdm import tqdm

ROOT     = pathlib.Path(__file__).parent.parent.resolve()
OUT_PATH = ROOT / "data" / "raw" / "nber_papers.parquet"

NBER_SEARCH = "https://www.nber.org/api/v1/search"
NBER_BATCH  = "https://api.nber.org/batch"

PAGE_SIZE   = 100
START_YEAR  = 2000
END_YEAR    = 2024
SLEEP_BETWEEN_PAGES = 1.0   # seconds — be polite

MONTH_MAP = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
}


# ------------------------------------------------------------------ #
# HTML stripping                                                       #
# ------------------------------------------------------------------ #

class _Stripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
    def handle_data(self, data: str):
        self._parts.append(data)
    def get_text(self) -> str:
        return " ".join(self._parts)

def strip_html(text: str) -> str:
    if not text:
        return ""
    s = _Stripper()
    s.feed(str(text))
    return " ".join(s.get_text().split())


# ------------------------------------------------------------------ #
# Date parsing                                                         #
# ------------------------------------------------------------------ #

def parse_display_date(date_str: str) -> tuple[int | None, int | None]:
    """Convert 'January 2012', 'May 2023', '2019' → (year, month)."""
    if not date_str:
        return None, None
    parts = date_str.strip().lower().split()
    year, month = None, None
    for p in parts:
        if p.isdigit() and len(p) == 4:
            year = int(p)
        elif p in MONTH_MAP:
            month = MONTH_MAP[p]
    return year, month


def wp_number_from_url(url: str) -> str:
    """Extract 'w12345' from '/papers/w12345'."""
    m = re.search(r'/papers/(w\d+)', url)
    return m.group(1) if m else url.split("/")[-1]


# ------------------------------------------------------------------ #
# Batch program enrichment                                             #
# ------------------------------------------------------------------ #

def load_batch_programs() -> dict[str, list[str]]:
    """
    Download all available NBER daily incremental batch files and return
    a dict {wp_number: [program_codes]}.  This covers recent papers only;
    historical papers will have an empty programs list.
    """
    programs_map: dict[str, list[str]] = {}

    # List batch directory
    try:
        resp = requests.get(f"{NBER_BATCH}/", timeout=15)
        resp.raise_for_status()
        batch_files = re.findall(r'working_paper_i\.[^"]+\.json', resp.text)
        batch_files = list(set(batch_files))   # deduplicate
    except Exception as e:
        print(f"  Warning: could not list batch directory: {e}")
        return programs_map

    print(f"  Downloading {len(batch_files)} batch file(s) for program codes …")
    for fname in batch_files:
        try:
            r = requests.get(f"{NBER_BATCH}/{fname}", timeout=30)
            r.raise_for_status()
            data = r.json().get("data", [])
            for rec in data:
                attrs = rec.get("attributes", rec)
                wp = attrs.get("paper", "")
                progs = attrs.get("programs_array", [])
                if wp and progs:
                    programs_map[wp] = progs
        except Exception as e:
            print(f"    Warning: {fname}: {e}")
        time.sleep(0.2)

    print(f"  Program codes loaded for {len(programs_map):,} papers (recent only)")
    return programs_map


# ------------------------------------------------------------------ #
# NBER search API pagination                                           #
# ------------------------------------------------------------------ #

def fetch_search_page(session: requests.Session, page: int) -> dict:
    params = {
        "q":       "",
        "page":    page,
        "perPage": PAGE_SIZE,
        "sortBy":  "public_date",
        "facet":   "contentType:working_paper",
    }
    resp = session.get(NBER_SEARCH, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_record(r: dict) -> dict | None:
    """Parse a single search result into a flat row."""
    url       = r.get("url", "")
    wp_number = wp_number_from_url(url)

    date_str  = r.get("displaydate", "")
    year, month = parse_display_date(date_str)

    if year is None:
        return None

    authors_raw = r.get("authors", [])
    if isinstance(authors_raw, list):
        authors = "; ".join(strip_html(a) for a in authors_raw)
    else:
        authors = strip_html(str(authors_raw))

    return {
        "wp_number": wp_number,
        "title":     r.get("title", "").strip(),
        "authors":   authors,
        "year":      year,
        "month":     month,
        "abstract":  strip_html(r.get("abstract", "")),
        "programs":  "",          # filled later from batch files
        "url":       f"https://www.nber.org{url}" if url.startswith("/") else url,
    }


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(start_year: int, end_year: int, test: bool) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "publication-bias-research/1.0"})

    # --- Step 1: get total count ---
    data = fetch_search_page(session, 1)
    total = data.get("totalResults", 0)
    print(f"NBER search API: {total:,} total working papers")
    if test:
        total = min(total, PAGE_SIZE * 2)
        print("  (test mode: limiting to 2 pages)")

    n_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    records: list[dict] = []

    # --- Step 2: paginate ---
    for page in tqdm(range(1, n_pages + 1), desc="Downloading NBER pages"):
        try:
            data = fetch_search_page(session, page)
        except Exception as e:
            print(f"  Error page {page}: {e}  — retrying once")
            time.sleep(5)
            try:
                data = fetch_search_page(session, page)
            except Exception as e2:
                print(f"  Failed page {page}: {e2}")
                continue

        for r in data.get("results", []):
            row = parse_record(r)
            if row:
                # In test mode collect everything; otherwise apply year filter
                if test or (start_year <= row["year"] <= end_year):
                    records.append(row)

        time.sleep(SLEEP_BETWEEN_PAGES)

    df = pd.DataFrame(records)
    if df.empty:
        print("No records collected — check API connectivity.")
        return

    df = df.drop_duplicates(subset="wp_number", keep="first")
    df = df.sort_values(["year", "wp_number"]).reset_index(drop=True)
    print(f"\nCollected {len(df):,} unique NBER working papers ({start_year}–{end_year})")
    print(df["year"].value_counts().sort_index().to_string())

    # Save immediately — before any enrichment that might fail
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"\nSaved → {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.1f} KB)")

    # --- Step 3: enrich with program codes from batch files ---
    programs_map = load_batch_programs()
    if programs_map:
        def safe_join(wp):
            progs = programs_map.get(wp, [])
            return ",".join(str(p) for p in progs if p is not None)
        df["programs"] = df["wp_number"].apply(safe_join)
        # Re-save with enriched programs
        df.to_parquet(OUT_PATH, index=False, compression="snappy")

    print(f"Columns: {list(df.columns)}")
    print(f"\nNote: 'programs' column populated for {(df['programs'] != '').sum():,} papers")
    print("  (recent papers only; program filtering done in script 03 via keyword/LLM)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect NBER working papers 2000–2024")
    parser.add_argument("--start", type=int, default=START_YEAR)
    parser.add_argument("--end",   type=int, default=END_YEAR)
    parser.add_argument("--test",  action="store_true",
                        help="Download 2 pages only (for testing)")
    args = parser.parse_args()
    main(args.start, args.end, args.test)
