"""
02b_enrich_jfe_abstracts.py
----------------------------
Fetch missing JFE abstracts via the Elsevier ScienceDirect API
(free for research; requires API key from https://dev.elsevier.com).

The script:
  1. Loads all JFE papers missing abstracts from the DuckDB database.
  2. Fetches each abstract via the Elsevier Article Retrieval API.
  3. Rebuilds raw_jfe in the database with the updated abstracts.
  4. Re-exports data/raw/jfe_papers.parquet.

Setup:
    Add  ELSEVIER_API_KEY=<your_key>  to .env
    (Register free at https://dev.elsevier.com — API key issued instantly)

Usage:
    python scripts/02b_enrich_jfe_abstracts.py
    python scripts/02b_enrich_jfe_abstracts.py --limit 50   # test run
"""

import argparse
import os
import pathlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

ROOT    = pathlib.Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")

DB_PATH  = ROOT / "publication_bias.duckdb"
OUT_DIR  = ROOT / "data" / "raw"
OUT_PATH = OUT_DIR / "jfe_papers.parquet"

ELSEVIER_BASE = "https://api.elsevier.com/content/article/doi"
WORKERS = 5          # parallel threads — stay comfortably under 10 req/s limit


def clean_abstract(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)         # strip XML/HTML tags
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^Abstract\s*[:.]?\s*", "", text, flags=re.IGNORECASE)
    return text


def fetch_elsevier(session: requests.Session,
                   doi: str, api_key: str) -> str:
    """
    Fetch abstract via ScienceDirect Article Retrieval API.
    Docs: https://dev.elsevier.com/documentation/ArticleRetrievalAPI.wadl
    """
    url = f"{ELSEVIER_BASE}/{doi}"
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept":        "application/json",
    }
    params = {"view": "META_ABS"}
    try:
        resp = session.get(url, headers=headers, params=params, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            # Navigate the nested response structure
            msg  = data.get("full-text-retrieval-response", {})
            core = msg.get("coredata", {})
            abstract = core.get("dc:description", "") or ""
            return clean_abstract(abstract)
        elif resp.status_code == 401:
            raise RuntimeError("Invalid or expired API key. Check ELSEVIER_API_KEY in .env")
        elif resp.status_code == 403:
            # Institutional access required for full-text — but abstract is in META_ABS
            # Try the abstract endpoint instead
            return _fetch_abstract_endpoint(session, doi, api_key)
        elif resp.status_code == 429:
            print("  Rate limit hit — sleeping 10s")
            time.sleep(10)
        return ""
    except RuntimeError:
        raise
    except Exception as exc:
        return ""


def _fetch_abstract_endpoint(session: requests.Session,
                              doi: str, api_key: str) -> str:
    """Fallback: use the ABSTRACT endpoint (metadata only, no full text needed)."""
    url = f"https://api.elsevier.com/content/abstract/doi/{doi}"
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    try:
        resp = session.get(url, headers=headers,
                           params={"view": "META"}, timeout=20)
        if resp.status_code == 200:
            core = (resp.json()
                    .get("abstracts-retrieval-response", {})
                    .get("coredata", {}))
            abstract = core.get("dc:description", "") or ""
            return clean_abstract(abstract)
    except Exception:
        pass
    return ""


def main(limit: int | None) -> None:
    api_key = os.getenv("ELSEVIER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "\nELSEVIER_API_KEY not set.\n"
            "Register free at https://dev.elsevier.com then add:\n"
            "  ELSEVIER_API_KEY=<your-key>\n"
            "to the .env file in the project root."
        )

    con = duckdb.connect(str(DB_PATH))

    missing = con.execute(
        "SELECT doi, title, year FROM raw_jfe "
        "WHERE abstract IS NULL OR abstract = '' "
        "ORDER BY year DESC"      # newest first — better chance of success
    ).df()

    if limit:
        missing = missing.head(limit)

    total = len(missing)
    print(f"JFE papers missing abstracts: {total:,}")
    print(f"Using Elsevier API key: {api_key[:8]}...")

    results: dict[str, str] = {}
    errors = 0

    def fetch_one(row) -> tuple[str, str]:
        doi = str(row["doi"] or "").strip()
        if not doi:
            return doi, ""
        # Each thread gets its own session for connection-pool efficiency
        s = requests.Session()
        s.headers.update({"User-Agent": "publication-bias-research/1.0"})
        try:
            abstract = fetch_elsevier(s, doi, api_key)
            return doi, abstract
        except RuntimeError:
            raise
        except Exception:
            return doi, ""

    rows = [row for _, row in missing.iterrows()]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_one, row): row for row in rows}
        with tqdm(total=total, desc="Fetching") as pbar:
            for future in as_completed(futures):
                try:
                    doi, abstract = future.result()
                    if abstract:
                        results[doi] = abstract
                    else:
                        errors += 1
                except RuntimeError as exc:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise SystemExit(str(exc))
                except Exception:
                    errors += 1
                pbar.update(1)

    print(f"\nFetched {len(results):,} / {total:,} abstracts  ({errors:,} misses)")

    if not results:
        print("No abstracts retrieved — check API key and network.")
        con.close()
        return

    # Read full table, apply updates, rebuild
    df_full = con.execute("SELECT * FROM raw_jfe").df()
    before  = (df_full["abstract"].fillna("").str.len() > 0).sum()

    update_df = pd.DataFrame(list(results.items()),
                             columns=["doi", "new_abstract"])
    df_full = df_full.merge(update_df, on="doi", how="left")
    mask = (
        df_full["new_abstract"].notna() &
        (df_full["new_abstract"] != "") &
        (df_full["abstract"].isna() | (df_full["abstract"].fillna("") == ""))
    )
    df_full.loc[mask, "abstract"] = df_full.loc[mask, "new_abstract"]
    df_full = df_full.drop(columns=["new_abstract"])

    after = (df_full["abstract"].fillna("").str.len() > 0).sum()
    print(f"Abstract coverage: {before:,} → {after:,} / {len(df_full):,} "
          f"({after/len(df_full)*100:.1f}%)")

    # Rebuild — drop whatever object type raw_jfe currently is
    con.register("jfe_updated", df_full)
    for drop_sql in ("DROP VIEW IF EXISTS raw_jfe",
                     "DROP TABLE IF EXISTS raw_jfe"):
        try:
            con.execute(drop_sql)
        except Exception:
            pass
    con.execute("CREATE TABLE raw_jfe AS SELECT * FROM jfe_updated")
    print("DuckDB table rebuilt.")

    # Re-export parquet for downstream scripts
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_full.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"Saved → {OUT_PATH}")
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only N papers (for testing)")
    args = parser.parse_args()
    main(limit=args.limit)
