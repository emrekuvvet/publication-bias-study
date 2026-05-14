"""
10_collect_afa.py
-----------------
Scrape AFA annual meeting program pages (2006-2024) and save all
paper titles + authors to data/raw/afa_papers.parquet.

URL patterns:
  2006-2012,
  2014-2023: https://afajof.org/{YEAR}-preliminary-program/
  2013      : https://www.aeaweb.org/conference/2013/preliminary.php
              (AFA 2013 page on afajof.org returns 404; AEA hosts the full
               ASSA program with AFA sessions identifiable by sessionSource)
  2024+     : https://afajof.org/management/full-program{YEAR}.html

HTML structure (2006-2023, afajof.org):
  h3 = time block heading
  h5 = paper title
  p  = authors (first <p> after the h5, before "Discussant:")

HTML structure (2013, aeaweb.org):
  div.session with font.sessionSource="American Finance Association"
  div.paper > font.paperTitle = paper title
  div.author_prelim > font.name + font.affiliation = authors

HTML structure (2024+):
  h5 = session heading
  h6 = paper title
  p  = authors

Output: data/raw/afa_papers.parquet
Columns: paper_id, year, title, authors, session, source

Usage:
    python scripts/10_collect_afa.py
    python scripts/10_collect_afa.py --years 2018 2019 2020   # subset
    python scripts/10_collect_afa.py --dry-run                # print URLs only

Prerequisites:
    pip install requests beautifulsoup4 pandas pyarrow lxml
"""

import argparse
import hashlib
import pathlib
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT     = pathlib.Path(__file__).parent.parent.resolve()
OUT_PATH = ROOT / "data" / "raw" / "afa_papers.parquet"
CACHE_DIR = ROOT / "data" / "raw" / "afa_html_cache"

YEARS_OLD   = list(range(2006, 2024))   # h5 = title (2013 uses AEA URL)
YEARS_NEW   = [2024, 2025, 2026]        # h6 = title
ALL_YEARS   = sorted(set(YEARS_OLD) | {2024})  # full 2006-2024 coverage

RATE_LIMIT  = 2.0   # seconds between HTTP requests

SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PublicationBiasResearch/1.0; "
        "academic research; mailto:research@pubbiasstudy.org)"
    ),
    "Accept": "text/html,application/xhtml+xml",
}


# ------------------------------------------------------------------ #
# URL builder                                                          #
# ------------------------------------------------------------------ #

AEA_2013_URL = "https://www.aeaweb.org/conference/2013/preliminary.php"


def url_for_year(year: int) -> str:
    if year == 2013:
        return AEA_2013_URL
    if year <= 2023:
        return f"https://afajof.org/{year}-preliminary-program/"
    return f"https://afajof.org/management/full-program{year}.html"


# ------------------------------------------------------------------ #
# HTTP with cache                                                      #
# ------------------------------------------------------------------ #

def fetch_html(url: str, session: requests.Session,
               use_cache: bool = True) -> str | None:
    cache_key = hashlib.md5(url.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.html"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if use_cache and cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="replace")

    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            print(f"  404 → {url}")
            return None
        resp.raise_for_status()
        html = resp.text
        cache_file.write_text(html, encoding="utf-8")
        time.sleep(RATE_LIMIT)
        return html
    except Exception as exc:
        print(f"  Error fetching {url}: {exc}")
        return None


# ------------------------------------------------------------------ #
# Text helpers                                                         #
# ------------------------------------------------------------------ #

def clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def is_skip_line(text: str) -> bool:
    t = text.strip().lower()
    return (
        t.startswith("discussant")
        or t.startswith("session:")
        or t.startswith("session chair")
        or t.startswith("chair:")
        or t.startswith("president")
    )


_DASH = re.compile(r"\s*[–—]\s*")   # en-dash or em-dash
_INST_WORDS = re.compile(
    r"\b(university|school|college|institute|faculty|department|business|"
    r"insead|nber|stern|booth|sloan|fuqua|tuck|haas|gsb|hbs)\b",
    re.IGNORECASE,
)


def looks_like_institution(text: str) -> bool:
    return bool(_INST_WORDS.search(text))


def make_paper_id(year: int, title: str) -> str:
    slug = re.sub(r"[^a-z0-9]", "", title.lower())[:40]
    return f"afa_{year}_{slug}"


# ------------------------------------------------------------------ #
# Parsers                                                              #
# ------------------------------------------------------------------ #

def _append(papers: list, year: int, title: str, authors: str, session: str) -> None:
    title = clean(title)
    if title and len(title) > 8:
        papers.append({
            "paper_id": make_paper_id(year, title),
            "year": year,
            "title": title,
            "authors": clean(authors),
            "session": session,
            "source": "afa",
        })


def parse_dashed(html: str, year: int) -> list[dict]:
    """
    2006-2012 format: each paper is one <p> with
    "Author, Institution – Dept – Paper Title".
    Title is after the LAST en/em-dash.
    Session context from <h3> time headings.
    """
    soup = BeautifulSoup(html, "lxml")
    papers: list[dict] = []
    current_session = ""

    for tag in soup.find_all(["h3", "h4", "p"]):
        if tag.name in ("h3", "h4"):
            t = clean(tag.get_text())
            if t:
                current_session = t
            continue

        txt = clean(tag.get_text(separator=" "))
        if not txt or is_skip_line(txt):
            continue

        # Must contain a dash to have a parseable title
        if not _DASH.search(txt):
            continue

        # Title is after the last dash; authors are before it
        parts = _DASH.split(txt)
        title = parts[-1].strip()
        authors_raw = " – ".join(parts[:-1])

        # Sanity: title shouldn't look like an institution block
        if len(title.split()) < 2 or looks_like_institution(title):
            continue

        _append(papers, year, title, authors_raw, current_session)

    return papers


def parse_old_v2(html: str, year: int) -> list[dict]:
    """
    2014-2016 format: title is in a standalone <p>, authors in the next <p>.
    Session info embedded in a <p> starting with "Session:" or date patterns.
    Falls back to trying the dashed parser if no titles are found this way.
    """
    soup = BeautifulSoup(html, "lxml")
    papers: list[dict] = []
    current_session = ""

    content = (soup.find("article")
               or soup.find("div", class_="entry-content")
               or soup)
    ps = content.find_all("p")

    i = 0
    while i < len(ps):
        txt = clean(ps[i].get_text(separator=" "))
        i += 1

        if not txt or is_skip_line(txt):
            continue

        # Session header detection
        if re.match(r"(session\s*:|01/|0[1-9]/|\d{1,2}/\d{1,2}/20)", txt, re.IGNORECASE):
            current_session = txt
            continue

        # A potential title: doesn't look like an institution, >3 words
        word_count = len(txt.split())
        if word_count < 3:
            continue

        # If line contains a dash, try dashed-format extraction
        if _DASH.search(txt):
            parts = _DASH.split(txt)
            candidate_title = parts[-1].strip()
            if (len(candidate_title.split()) >= 2
                    and not looks_like_institution(candidate_title)):
                authors_raw = " – ".join(parts[:-1])
                _append(papers, year, candidate_title, authors_raw, current_session)
                continue

        # Heuristic: title p is followed by an author/institution p
        if i < len(ps):
            next_txt = clean(ps[i].get_text(separator=" "))
            if looks_like_institution(next_txt) and not is_skip_line(txt):
                # Current p is likely the title
                _append(papers, year, txt, next_txt, current_session)
                i += 1   # consume the authors p
                continue

    return papers


def parse_old(html: str, year: int) -> list[dict]:
    """
    2017-2023 format: <h5> = paper title, next <p> = authors.
    """
    soup = BeautifulSoup(html, "lxml")
    papers: list[dict] = []
    current_session = ""

    for tag in soup.find_all(["h3", "h4", "h5", "p", "hr"]):
        name = tag.name

        if name in ("h3", "h4"):
            text = clean(tag.get_text())
            if text:
                current_session = text
            continue

        if name == "h5":
            title = clean(tag.get_text())
            if not title or len(title) < 5:
                continue

            authors_text = ""
            for sib in tag.next_siblings:
                if not hasattr(sib, "name"):
                    continue
                if sib.name == "h5":
                    break
                if sib.name in ("h3", "h4", "hr"):
                    break
                if sib.name == "p":
                    txt = clean(sib.get_text())
                    if is_skip_line(txt):
                        break
                    if txt:
                        authors_text = txt
                        break

            _append(papers, year, title, authors_text, current_session)

    return papers


def parse_new(html: str, year: int) -> list[dict]:
    """
    Parse 2024+ AFA program pages.
    Session heading in h5; paper title in h6; authors in <p> immediately after.
    """
    soup = BeautifulSoup(html, "lxml")
    papers = []
    current_session = ""

    for tag in soup.find_all(["h5", "h6", "p"]):
        name = tag.name

        if name == "h5":
            text = clean(tag.get_text())
            if text:
                current_session = text
            continue

        if name == "h6":
            title = clean(tag.get_text())
            if not title or len(title) < 5:
                continue

            authors_text = ""
            for sib in tag.next_siblings:
                if not hasattr(sib, "name"):
                    continue
                if sib.name in ("h5", "h6"):
                    break
                if sib.name == "p":
                    txt = clean(sib.get_text())
                    if is_skip_line(txt):
                        break
                    if txt:
                        authors_text = txt
                        break

            _append(papers, year, title, authors_text, current_session)

    return papers


def parse_aea_2013(html: str) -> list[dict]:
    """
    Parse the AEA 2013 full ASSA program page and extract only AFA sessions.
    Structure: div.session with font.sessionSource = 'American Finance Association'
    Papers: div.paper > font.paperTitle; authors in div.author_prelim.
    """
    soup = BeautifulSoup(html, "lxml")
    papers: list[dict] = []

    for session in soup.find_all("div", class_="session"):
        src_tag = session.find("font", class_="sessionSource")
        if not src_tag or "American Finance Association" not in src_tag.get_text():
            continue

        title_tag = session.find("div", class_="sessionTitle")
        session_title = clean(title_tag.get_text(" ")) if title_tag else ""

        for paper_div in session.find_all("div", class_="paper"):
            pt = paper_div.find("font", class_="paperTitle")
            if not pt:
                continue
            title = clean(pt.get_text(" "))
            if not title or len(title) < 5:
                continue

            authors_list = []
            for ap in paper_div.find_all("div", class_="author_prelim"):
                name_tag = ap.find("font", class_="name")
                aff_tag  = ap.find("font", class_="affiliation")
                if name_tag:
                    name = name_tag.get_text(strip=True)
                    aff  = aff_tag.get_text(strip=True) if aff_tag else ""
                    authors_list.append(f"{name} {aff}".strip())
            authors = ", ".join(authors_list)

            _append(papers, 2013, title, authors, session_title)

    return papers


def parse_year(html: str, year: int) -> list[dict]:
    if year == 2013:
        return parse_aea_2013(html)
    if year <= 2012:
        records = parse_dashed(html, year)
        if not records:
            records = parse_old_v2(html, year)
        return records
    if year <= 2016:
        return parse_old_v2(html, year)
    if year <= 2023:
        return parse_old(html, year)
    return parse_new(html, year)


# ------------------------------------------------------------------ #
# Deduplication                                                        #
# ------------------------------------------------------------------ #

def deduplicate(records: list[dict]) -> list[dict]:
    """Drop exact-duplicate (year, normalised_title) pairs."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        key = f"{r['year']}|{re.sub(r'[^a-z0-9]', '', r['title'].lower())}"
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(years: list[int], dry_run: bool, no_cache: bool) -> None:
    session = requests.Session()
    session.headers.update(SESSION_HEADERS)

    all_records: list[dict] = []

    for year in tqdm(years, desc="Scraping AFA"):
        url = url_for_year(year)
        if dry_run:
            print(f"  {year}: {url}")
            continue

        html = fetch_html(url, session, use_cache=not no_cache)
        if html is None:
            print(f"  {year}: no data (skipped)")
            continue

        records = parse_year(html, year)
        print(f"  {year}: {len(records)} papers parsed from {url}")
        all_records.extend(records)

    if dry_run:
        return

    if not all_records:
        print("No records collected — check network access or URL patterns.")
        return

    all_records = deduplicate(all_records)
    df = pd.DataFrame(all_records)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False, compression="snappy")

    print(f"\nSaved {len(df):,} AFA papers → {OUT_PATH}")
    print(f"  Year range: {df['year'].min()} – {df['year'].max()}")
    print(f"  Papers with authors: {df['authors'].str.len().gt(0).sum():,}")
    year_counts = df.groupby("year").size()
    print("\nPapers per year:")
    print(year_counts.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape AFA annual meeting programs (2006-2024)"
    )
    parser.add_argument(
        "--years", nargs="+", type=int,
        default=ALL_YEARS,
        help="Years to scrape (default: 2006-2024)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print URLs only, do not fetch"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore cached HTML, re-fetch all pages"
    )
    args = parser.parse_args()
    main(args.years, args.dry_run, args.no_cache)
