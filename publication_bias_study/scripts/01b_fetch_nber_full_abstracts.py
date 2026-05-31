"""
01b_fetch_nber_full_abstracts.py
---------------------------------
Fetch full abstracts for all NBER working papers from nber.org.

The NBER search API (used in 01_collect_nber.py) truncates abstracts at
~300 characters (~45 words). This script fetches the full abstract for
every paper in raw_nber by scraping the individual paper page.

The full abstract lives in the CSS selector: .page-header__intro-inner

Progress is saved to data/raw/nber_full_abstracts.parquet so the script
is resumable if interrupted. After all papers are fetched, raw_nber in
publication_bias.duckdb is updated in-place.

Usage:
    python scripts/01b_fetch_nber_full_abstracts.py
    python scripts/01b_fetch_nber_full_abstracts.py --workers 12
    python scripts/01b_fetch_nber_full_abstracts.py --test 50   # first 50 only

Runtime: ~10–15 minutes with 8 workers for 30,000 papers.
"""

import argparse
import pathlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT        = pathlib.Path(__file__).parent.parent.resolve()
DB_PATH     = ROOT / "publication_bias.duckdb"
PROGRESS_PATH = ROOT / "data" / "raw" / "nber_full_abstracts.parquet"

NBER_BASE   = "https://www.nber.org/papers/{wp_number}"
HEADERS     = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
REQUEST_DELAY = 0.1   # seconds per worker between requests
WORKERS     = 8
SAVE_EVERY  = 500     # save progress checkpoint every N papers


def fetch_abstract(wp_number: str, session: requests.Session) -> str:
    url = NBER_BASE.format(wp_number=wp_number)
    for attempt in range(3):
        try:
            r = session.get(url, timeout=20, headers=HEADERS)
            if r.status_code == 404:
                return ""
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            el = soup.select_one(".page-header__intro-inner")
            if el:
                return el.get_text(separator=" ", strip=True)
            return ""
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return ""


def main(workers: int, test: int | None) -> None:
    # --- Load paper IDs from raw_nber ---
    con = duckdb.connect(str(DB_PATH))
    df = con.execute(
        "SELECT wp_number FROM raw_nber WHERE title IS NOT NULL AND title != ''"
    ).df()
    con.close()

    all_ids = df["wp_number"].tolist()
    if test:
        all_ids = all_ids[:test]
        print(f"Test mode: {len(all_ids)} papers")
    else:
        print(f"Total NBER papers: {len(all_ids):,}")

    # --- Resume from checkpoint ---
    done: dict[str, str] = {}
    if PROGRESS_PATH.exists():
        prev = pd.read_parquet(PROGRESS_PATH)
        done = dict(zip(prev["wp_number"], prev["full_abstract"]))
        print(f"Resuming: {len(done):,} already fetched, "
              f"{len(all_ids) - len(done):,} remaining")

    remaining = [p for p in all_ids if p not in done]
    if not remaining:
        print("All abstracts already fetched.")
    else:
        _tls = threading.local()

        def _get_session() -> requests.Session:
            if not hasattr(_tls, "sess"):
                _tls.sess = requests.Session()
            return _tls.sess

        def _fetch(wp_number: str) -> tuple[str, str]:
            time.sleep(REQUEST_DELAY)
            return wp_number, fetch_abstract(wp_number, _get_session())

        batch: list[tuple[str, str]] = []
        save_counter = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch, pid): pid for pid in remaining}
            with tqdm(total=len(remaining), desc="Fetching abstracts") as pbar:
                for future in as_completed(futures):
                    pid, abstract = future.result()
                    done[pid] = abstract
                    batch.append((pid, abstract))
                    save_counter += 1
                    pbar.update(1)

                    if save_counter % SAVE_EVERY == 0:
                        _save_progress(done, PROGRESS_PATH)

        _save_progress(done, PROGRESS_PATH)

    # --- Stats ---
    prog = pd.read_parquet(PROGRESS_PATH)
    filled  = (prog["full_abstract"].str.len() > 300).sum()
    empty   = (prog["full_abstract"].str.len() == 0).sum()
    print(f"\nFetch complete:")
    print(f"  Full abstracts (>300 chars): {filled:,}")
    print(f"  Empty/not found:             {empty:,}")
    print(f"  Truncated only (<=300):      {len(prog) - filled - empty:,}")

    # --- Update raw_nber in DuckDB ---
    print("\nUpdating raw_nber in DuckDB...")
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE TEMP TABLE _full_abstracts (wp_number VARCHAR, full_abstract VARCHAR)")
    con.executemany(
        "INSERT INTO _full_abstracts VALUES (?, ?)",
        prog[["wp_number", "full_abstract"]].values.tolist()
    )
    con.execute("""
        UPDATE raw_nber
        SET abstract = _full_abstracts.full_abstract
        FROM _full_abstracts
        WHERE raw_nber.wp_number = _full_abstracts.wp_number
          AND LENGTH(_full_abstracts.full_abstract) > LENGTH(raw_nber.abstract)
    """)
    updated = con.execute(
        "SELECT COUNT(*) FROM raw_nber WHERE LENGTH(abstract) > 300"
    ).fetchone()[0]
    con.close()
    print(f"  raw_nber rows with abstract >300 chars: {updated:,}")
    print(f"\nDone. Run 03c_refilter_nber_full_corpus.py next.")


def _save_progress(done: dict[str, str], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        list(done.items()), columns=["wp_number", "full_abstract"]
    ).to_parquet(path, index=False, compression="snappy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--test",    type=int, default=None,
                        help="Fetch only first N papers (for testing)")
    args = parser.parse_args()
    main(workers=args.workers, test=args.test)
