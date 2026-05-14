"""
02_collect_journals.py
----------------------
Collect paper metadata for JF, RFS, and JFE using the OpenAlex API
(free, no key required) and save one Parquet file per journal to
data/raw/.

OpenAlex: https://docs.openalex.org — 10 req/s polite rate, no auth.
Covers all three journals with full abstracts, authors, citations.

Fallback: if --csv is given, import a manual Web of Science CSV export
instead (see --help for format instructions).

Usage:
    python scripts/02_collect_journals.py              # all three journals
    python scripts/02_collect_journals.py --journal jf # one journal
    python scripts/02_collect_journals.py --csv data/manual/jf.csv --journal jf
    python scripts/02_collect_journals.py --test       # 50 papers per journal
"""

import argparse
import pathlib
import re
import time

import pandas as pd
import requests
from tqdm import tqdm

ROOT    = pathlib.Path(__file__).parent.parent.resolve()
OUT_DIR = ROOT / "data" / "raw"

OPENALEX_BASE = "https://api.openalex.org/works"
POLITE_MAILTO = "emrekuvvet@gmail.com"

JOURNAL_META = {
    "jf":  {"issn": "0022-1082", "name": "Journal of Finance"},
    "rfs": {"issn": "0893-9454", "name": "Review of Financial Studies"},
    "jfe": {"issn": "0304-405X", "name": "Journal of Financial Economics"},
}

START_YEAR = 2000
END_YEAR   = 2024
PER_PAGE   = 200    # OpenAlex max


# ------------------------------------------------------------------ #
# Abstract reconstruction from inverted index                         #
# ------------------------------------------------------------------ #

def reconstruct_abstract(inv_index: dict | None) -> str:
    """
    OpenAlex stores abstracts as {word: [position, ...]} dicts.
    Reconstruct into a readable string.
    """
    if not inv_index:
        return ""
    try:
        max_pos = max(pos for positions in inv_index.values() for pos in positions)
        words = [""] * (max_pos + 1)
        for word, positions in inv_index.items():
            for pos in positions:
                if 0 <= pos <= max_pos:
                    words[pos] = word
        return " ".join(w for w in words if w)
    except Exception:
        return ""


# ------------------------------------------------------------------ #
# Author / institution helpers                                         #
# ------------------------------------------------------------------ #

def parse_authors(authorships: list) -> tuple[str, str]:
    """Return (author_string, affiliation_string)."""
    names, affils = [], []
    for a in authorships:
        name = (a.get("author") or {}).get("display_name", "")
        if name:
            names.append(name)
        for inst in (a.get("institutions") or []):
            inst_name = inst.get("display_name", "")
            if inst_name and inst_name not in affils:
                affils.append(inst_name)
    return "; ".join(names), "; ".join(affils)


# ------------------------------------------------------------------ #
# OpenAlex cursor-based pagination                                     #
# ------------------------------------------------------------------ #

def fetch_page(session: requests.Session, issn: str,
               cursor: str) -> dict:
    params = {
        "filter":   f"primary_location.source.issn:{issn},"
                    f"publication_year:{START_YEAR}-{END_YEAR},"
                    "type:article",
        "per_page": PER_PAGE,
        "cursor":   cursor,
        "select":   ",".join([
            "id", "doi", "display_name", "publication_year",
            "publication_date", "biblio", "authorships",
            "abstract_inverted_index", "cited_by_count",
            "concepts", "primary_location",
        ]),
        "mailto": POLITE_MAILTO,
    }
    resp = session.get(OPENALEX_BASE, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_record(r: dict, journal_name: str) -> dict:
    doi = r.get("doi", "") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]

    authors, affils = parse_authors(r.get("authorships", []))
    abstract = reconstruct_abstract(r.get("abstract_inverted_index"))

    concepts = r.get("concepts") or []
    jel_proxy = "; ".join(
        c["display_name"] for c in concepts[:8] if c.get("display_name")
    )

    bib = r.get("biblio") or {}

    return {
        "doi":          doi,
        "title":        r.get("display_name", "").strip(),
        "authors":      authors,
        "affiliations": affils,
        "year":         r.get("publication_year"),
        "pub_date":     r.get("publication_date", ""),
        "volume":       bib.get("volume", ""),
        "issue":        bib.get("issue", ""),
        "first_page":   bib.get("first_page", ""),
        "last_page":    bib.get("last_page", ""),
        "abstract":     abstract,
        "jel_codes":    jel_proxy,
        "citations":    r.get("cited_by_count", 0),
        "journal":      journal_name,
        "openalex_id":  r.get("id", ""),
    }


def collect_journal(journal_key: str, test: bool) -> pd.DataFrame:
    meta    = JOURNAL_META[journal_key]
    issn    = meta["issn"]
    jname   = meta["name"]
    session = requests.Session()
    session.headers.update({"User-Agent": "publication-bias-research/1.0"})

    # Get total count first
    first = fetch_page(session, issn, "*")
    total = first["meta"]["count"]
    print(f"  {jname}: {total:,} papers in OpenAlex (2000–{END_YEAR})")

    if test:
        total = min(total, 50)

    n_pages = (total + PER_PAGE - 1) // PER_PAGE
    records = []
    cursor  = "*"

    with tqdm(total=total, desc=f"  {journal_key.upper()}", unit="papers") as pbar:
        for page_num in range(n_pages):
            if page_num == 0:
                data = first
            else:
                try:
                    data = fetch_page(session, issn, cursor)
                except Exception as e:
                    print(f"\n  Error page {page_num}: {e} — retrying")
                    time.sleep(5)
                    data = fetch_page(session, issn, cursor)

            for r in data["results"]:
                records.append(parse_record(r, jname))
                if test and len(records) >= total:
                    break

            cursor = (data["meta"] or {}).get("next_cursor", "")
            pbar.update(len(data["results"]))

            if not cursor or (test and len(records) >= total):
                break
            time.sleep(0.12)   # ~8 req/s — well within polite limit

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset="doi", keep="first")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["citations"] = pd.to_numeric(df["citations"], errors="coerce").astype("Int64")
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ #
# Manual WoS CSV fallback                                             #
# ------------------------------------------------------------------ #
# Web of Science export columns (used subset):
# TI=Title, AU=Authors, PY=Year, VL=Volume, IS=Issue,
# AB=Abstract, ID=Keywords Plus, TC=Times Cited, DI=DOI, SO=Source

WOS_MAP = {
    "TI": "title",  "AU": "authors", "PY": "year",
    "VL": "volume", "IS": "issue",   "AB": "abstract",
    "ID": "jel_codes", "TC": "citations", "DI": "doi", "SO": "journal",
}

def load_csv(csv_path: str, journal_key: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str,
                      low_memory=False, na_filter=False)
    present = {c: WOS_MAP[c] for c in WOS_MAP if c in raw.columns}
    df = raw[list(present.keys())].rename(columns=present)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "citations" in df.columns:
        df["citations"] = pd.to_numeric(df["citations"], errors="coerce").astype("Int64")
    if "journal" not in df.columns:
        df["journal"] = JOURNAL_META[journal_key]["name"]
    for col in ["doi","affiliations","pub_date","volume","issue",
                "first_page","last_page","jel_codes","openalex_id"]:
        if col not in df.columns:
            df[col] = ""
    df = df[df["year"].between(START_YEAR, END_YEAR)]
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(journals: list[str], csv_path: str | None, test: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for jkey in journals:
        print(f"\nProcessing {jkey.upper()} ({'test mode' if test else 'full'}) ...")
        out_path = OUT_DIR / f"{jkey}_papers.parquet"

        if csv_path:
            df = load_csv(csv_path, jkey)
            print(f"  Loaded {len(df):,} rows from CSV")
        else:
            df = collect_journal(jkey, test=test)

        df.to_parquet(out_path, index=False, compression="snappy")
        kb = out_path.stat().st_size / 1024
        print(f"  Saved → {out_path}  ({kb:.1f} KB,  {len(df):,} papers)")
        print(f"  Year range: {df['year'].min()} – {df['year'].max()}")
        print(f"  With abstract: {(df['abstract'].str.len() > 0).sum():,} / {len(df):,}")
        print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect JF/RFS/JFE paper metadata via OpenAlex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Manual CSV fallback (Web of Science export):
  1. Go to https://www.webofscience.com
  2. Search: SO="Journal of Finance" AND PY=2000-2024
  3. Export → All records → Full record → CSV
  4. python scripts/02_collect_journals.py --csv jf_wos.csv --journal jf
        """
    )
    parser.add_argument("--journal", choices=["jf","rfs","jfe"],
                        default=None,
                        help="Process one journal (default: all three)")
    parser.add_argument("--csv",  type=str, default=None,
                        help="Path to manual WoS CSV export")
    parser.add_argument("--test", action="store_true",
                        help="Collect only 50 papers per journal for testing")
    args = parser.parse_args()

    journals = [args.journal] if args.journal else ["jf", "rfs", "jfe"]
    main(journals, csv_path=args.csv, test=args.test)
