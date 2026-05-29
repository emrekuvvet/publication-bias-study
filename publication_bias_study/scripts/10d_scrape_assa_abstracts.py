"""
10d_scrape_assa_abstracts.py
----------------------------
Scrape paper abstracts from the AEA/ASSA conference website for AFA sessions (2015-2024).

Three HTML formats across years:

  2015-2016  Old inline format: all sessions and abstracts embedded in the
             listing page HTML.  Structure: font.sessionSource / font.paperTitle /
             div#paper_NNNN_abstract (hidden).  Parsed directly without detail pages.

  2017-2020  New listing HTML, all sessions visible without AJAX.
             Structure: article.session-item on listing page; detail pages use
             article.paper > section.abstract.

  2021-2024  New listing HTML, paginated (50 per page).  The AFA-filtered session
             list is obtained via AJAX POST to /conf{year}/preliminary/search/index
             with search[assoc]=5.  Detail pages use the same structure as 2017-2020.

Usage:
    python scripts/10d_scrape_assa_abstracts.py              # all years 2015-2024
    python scripts/10d_scrape_assa_abstracts.py --year 2022  # single year
    python scripts/10d_scrape_assa_abstracts.py --force      # re-fetch even if cached

Prerequisites:
    pip install requests beautifulsoup4 rapidfuzz tqdm
"""

import argparse
import json
import pathlib
import re
import time
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from tqdm import tqdm

ROOT          = pathlib.Path(__file__).parent.parent.resolve()
OUT_PATH      = ROOT / "data" / "raw" / "afa_papers_enriched.parquet"
SESSION_CACHE = ROOT / "data" / "raw" / "assa_session_cache.json"
MATCH_LOG     = ROOT / "validation" / "assa_scrape_matches.csv"

FUZZY_MIN = 85   # minimum token_sort_ratio to accept a title match
DELAY     = 0.8  # seconds between HTTP requests

REQ_HEADERS = {"User-Agent": "Mozilla/5.0 (research bot; publication bias study)"}

# --------------------------------------------------------------------------- #
# Year → listing URL                                                           #
# --------------------------------------------------------------------------- #

def listing_url(year: int) -> str:
    if year <= 2022:
        return f"https://www.aeaweb.org/conference/{year}/preliminary"
    return f"https://www.aeaweb.org/conference/{year}/program"

# --------------------------------------------------------------------------- #
# Text normalisation                                                           #
# --------------------------------------------------------------------------- #

STOP = {"a","an","the","of","in","on","at","to","for","and","or","but","with",
        "by","from","that","this","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","can","could",
        "should","may","might","shall","some","any","all"}

def normalise(title: str) -> str:
    if not title:
        return ""
    s = unicodedata.normalize("NFD", str(title)).encode("ascii","ignore").decode()
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return " ".join(t for t in s.split() if t not in STOP and len(t) > 1)

# --------------------------------------------------------------------------- #
# Format A (2015-2016): parse listing page directly                           #
# --------------------------------------------------------------------------- #

def parse_old_listing_page(html: str, year: int) -> list[dict]:
    """
    Parse AFA papers + inline abstracts from the old conference listing page
    format used in 2015-2016.

    Structure:
      font.sessionSource  ← contains the association name
      div.sessionTitle    ← session title (sibling of font.sessionSource)
      div.paper           ← each paper
        font.paperTitle   ← paper title
        span.hyperlink    ← onclick="show_abstract(NNNN)"
      div#paper_NNNN_abstract  ← hidden abstract (already in HTML)
    """
    soup = BeautifulSoup(html, "html.parser")
    papers = []
    in_afa = False

    for el in soup.find_all(["font", "div"]):
        if el.name == "font" and "sessionSource" in el.get("class", []):
            in_afa = "American Finance Association" in el.get_text()
            continue

        if in_afa and el.name == "div" and "paper" in el.get("class", []):
            title_el = el.find("font", class_="paperTitle")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)

            paper_id = None
            toggle = el.find("span", id=re.compile(r"^toggle_\d+$"))
            if toggle:
                paper_id = toggle["id"].replace("toggle_", "")

            abstract = ""
            if paper_id:
                abs_div = soup.find("div", id=f"paper_{paper_id}_abstract")
                if abs_div:
                    abstract = abs_div.get_text(strip=True)

            papers.append({
                "title":       title,
                "abstract":    abstract,
                "session_url": listing_url(year),
            })

    return papers

# --------------------------------------------------------------------------- #
# Format B (2017-2020): extract session URLs from full listing HTML           #
# --------------------------------------------------------------------------- #

def fetch_afa_session_ids_html(html: str, year: int) -> list[str]:
    """
    Extract AFA session detail URLs from the listing page HTML.
    Used for 2017-2020 where all sessions are present without AJAX.
    """
    soup = BeautifulSoup(html, "html.parser")
    afa_urls = []
    seen: set[str] = set()
    for art in soup.find_all("article", class_="session-item"):
        if "American Finance Association" not in art.get_text():
            continue
        link = art.find("a", href=True)
        if not link:
            continue
        clean = re.sub(r"\?.*$", "", link["href"])
        if clean in seen:
            continue
        seen.add(clean)
        if clean.startswith("http"):
            afa_urls.append(clean)
        elif clean.startswith("/"):
            afa_urls.append("https://www.aeaweb.org" + clean)
        else:
            afa_urls.append(f"https://www.aeaweb.org/conference/{year}/{clean}")
    return afa_urls

# --------------------------------------------------------------------------- #
# Format C (2021+): get session URLs via AJAX                                 #
# --------------------------------------------------------------------------- #

def fetch_afa_session_ids_ajax(year: int, req_session: requests.Session) -> list[str]:
    """
    Call the AEA AJAX search endpoint with assoc=5 (AFA) and sessionType=session.
    The backend always lives at /conf{year}/preliminary/search/index.
    """
    base = listing_url(year)
    ajax_url = f"https://www.aeaweb.org/conf{year}/preliminary/search/index"

    r = req_session.get(base, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    meta = soup.find("meta", attrs={"name": "csrf-token"})
    if not meta:
        raise RuntimeError(f"No CSRF token found on {base}")
    csrf = meta["content"]

    ajax_headers = {
        "User-Agent":       REQ_HEADERS["User-Agent"],
        "X-Requested-With": "XMLHttpRequest",
        "X-Csrf-Token":     csrf,
        "Referer":          base,
    }

    afa_urls: list[str] = []
    seen: set[str] = set()

    for page in range(1, 20):
        data = {
            "_csrf":                        csrf,
            "search[assoc]":                "5",
            "search[sessionType][session]": "1",
            "search[page]":                 str(page),
        }
        r2 = req_session.post(ajax_url, data=data, headers=ajax_headers, timeout=20)
        r2.raise_for_status()
        html = r2.json().get("searchResultsHtml", "")
        articles = BeautifulSoup(html, "html.parser").find_all("article", class_="session-item")
        if not articles:
            break

        for art in articles:
            link = art.find("a", href=True)
            if not link:
                continue
            clean = re.sub(r"\?.*$", "", link["href"])
            if clean in seen:
                continue
            seen.add(clean)
            if clean.startswith("http"):
                afa_urls.append(clean)
            elif clean.startswith("/"):
                afa_urls.append("https://www.aeaweb.org" + clean)
            else:
                afa_urls.append(f"https://www.aeaweb.org/conference/{year}/{clean}")

        if len(articles) < 25:
            break

    return afa_urls

# --------------------------------------------------------------------------- #
# Parse one session detail page (2017+)                                       #
# --------------------------------------------------------------------------- #

def parse_session_page(html: str, session_url: str) -> list[dict]:
    """
    Parse papers from a session detail page (2017-2024 format).
    Returns list of {title, abstract, session_url}.
    """
    soup = BeautifulSoup(html, "html.parser")
    papers = []
    for art in soup.find_all("article", class_="paper"):
        title_el = art.find("h3", class_="paper-title") or art.find("h3")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)

        abstract = ""
        abs_sec = art.find("section", class_="abstract")
        if abs_sec:
            for h4 in abs_sec.find_all("h4"):
                h4.decompose()
            abstract = abs_sec.get_text(separator=" ", strip=True)

        papers.append({"title": title, "abstract": abstract, "session_url": session_url})
    return papers

# --------------------------------------------------------------------------- #
# Fuzzy match + update parquet                                                 #
# --------------------------------------------------------------------------- #

def update_enriched(year: int, scraped: list[dict], df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    missing_mask = (df["year"] == year) & (
        df["abstract"].isna() | (df["abstract"].str.strip() == "")
    )
    missing = df[missing_mask].copy()
    if missing.empty:
        return df, []

    missing_norm = {idx: normalise(row["title"]) for idx, row in missing.iterrows()}
    scraped_with_abs = [p for p in scraped if p["abstract"].strip()]

    log_rows = []
    for paper in scraped_with_abs:
        snorm = normalise(paper["title"])
        if not snorm:
            continue
        best_idx, best_score = None, 0
        for idx, mnorm in missing_norm.items():
            score = fuzz.token_sort_ratio(snorm, mnorm)
            if score > best_score:
                best_score, best_idx = score, idx

        if best_idx is not None and best_score >= FUZZY_MIN:
            df.at[best_idx, "abstract"]        = paper["abstract"]
            df.at[best_idx, "abstract_source"] = "assa_web"
            del missing_norm[best_idx]
            log_rows.append({
                "year":        year,
                "our_title":   df.at[best_idx, "title"],
                "assa_title":  paper["title"],
                "score":       best_score,
                "session_url": paper["session_url"],
            })

    return df, log_rows

# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year",  type=int, default=None,
                    help="Process only this year (default: 2015-2024)")
    ap.add_argument("--force", action="store_true",
                    help="Re-fetch session URLs even if already cached")
    args = ap.parse_args()

    years = [args.year] if args.year else list(range(2015, 2025))

    print(f"Loading {OUT_PATH.name} …")
    df = pd.read_parquet(OUT_PATH)
    has_abs = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    print(f"  Starting coverage: {has_abs.sum()} / {len(df)} ({has_abs.mean()*100:.1f}%)")

    session_cache: dict[str, list[str]] = {}
    if SESSION_CACHE.exists():
        with open(SESSION_CACHE) as f:
            session_cache = json.load(f)

    all_log: list[dict] = []

    for year in years:
        ykey = str(year)
        print(f"\n{'='*60}")
        print(f"Year {year}")

        missing_this_year = (
            (df["year"] == year) &
            (df["abstract"].isna() | (df["abstract"].str.strip() == ""))
        ).sum()
        print(f"  Papers missing abstract this year: {missing_this_year}")

        # ------------------------------------------------------------------ #
        # 2015-2016: old inline HTML format — parse listing page directly    #
        # ------------------------------------------------------------------ #
        if year <= 2016:
            print(f"  [Format A] Parsing inline listing page …")
            try:
                r = requests.get(listing_url(year), headers=REQ_HEADERS, timeout=30)
                r.raise_for_status()
                scraped = parse_old_listing_page(r.text, year)
                papers_with_abs = sum(1 for p in scraped if p["abstract"].strip())
                print(f"  Scraped {len(scraped)} AFA papers, {papers_with_abs} with abstracts")
                df, log_rows = update_enriched(year, scraped, df)
                all_log.extend(log_rows)
                print(f"  Matched and updated: {len(log_rows)} papers")
            except Exception as e:
                print(f"  ERROR for {year}: {e}")
            continue

        # ------------------------------------------------------------------ #
        # 2017-2020: all sessions in listing HTML                            #
        # 2021-2024: AJAX for session list                                   #
        # Both fetch detail pages using parse_session_page                   #
        # ------------------------------------------------------------------ #

        # Step 1: get AFA session URLs
        if args.force or ykey not in session_cache:
            try:
                if year <= 2020:
                    print(f"  [Format B] Extracting session URLs from listing HTML …")
                    r = requests.get(listing_url(year), headers=REQ_HEADERS, timeout=30)
                    r.raise_for_status()
                    urls = fetch_afa_session_ids_html(r.text, year)
                else:
                    print(f"  [Format C] Fetching session URLs via AJAX …")
                    req_sess = requests.Session()
                    req_sess.headers.update(REQ_HEADERS)
                    urls = fetch_afa_session_ids_ajax(year, req_sess)

                session_cache[ykey] = urls
                with open(SESSION_CACHE, "w") as f:
                    json.dump(session_cache, f, indent=2)
                print(f"  Found {len(urls)} AFA session URLs")
            except Exception as e:
                print(f"  ERROR fetching session URLs for {year}: {e}")
                if ykey not in session_cache:
                    print("  No cached URLs — skipping year")
                    continue
        else:
            print(f"  Using cached {len(session_cache[ykey])} session URLs")

        session_urls = session_cache[ykey]
        if not session_urls:
            print(f"  No AFA sessions found for {year}")
            continue

        # Step 2: fetch session detail pages
        print(f"  Fetching {len(session_urls)} session pages …")
        all_scraped: list[dict] = []
        for url in tqdm(session_urls, desc=f"  {year} sessions", leave=False):
            try:
                r = requests.get(url, headers=REQ_HEADERS, timeout=15)
                if r.status_code != 200:
                    print(f"  HTTP {r.status_code} for {url}")
                    continue
                papers = parse_session_page(r.text, url)
                all_scraped.extend(papers)
                time.sleep(DELAY)
            except Exception as e:
                print(f"  Request error {url}: {e}")
                continue

        papers_with_abs = sum(1 for p in all_scraped if p["abstract"].strip())
        print(f"  Scraped {len(all_scraped)} papers, {papers_with_abs} with abstracts")

        # Step 3: fuzzy match
        df, log_rows = update_enriched(year, all_scraped, df)
        all_log.extend(log_rows)
        print(f"  Matched and updated: {len(log_rows)} papers")

    # ---- Save ----
    df.to_parquet(OUT_PATH, index=False)
    has_abs_new = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    gain = has_abs_new.sum() - has_abs.sum()
    print(f"\n{'='*60}")
    print(f"Final coverage: {has_abs_new.sum()} / {len(df)} ({has_abs_new.mean()*100:.1f}%)")
    print(f"New abstracts added: {gain}")

    if all_log:
        MATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
        # Append to existing log if present
        if MATCH_LOG.exists():
            existing = pd.read_csv(MATCH_LOG)
            combined = pd.concat([existing, pd.DataFrame(all_log)], ignore_index=True)
            combined.to_csv(MATCH_LOG, index=False)
        else:
            pd.DataFrame(all_log).to_csv(MATCH_LOG, index=False)
        print(f"Match log saved to {MATCH_LOG.name} ({len(all_log)} new rows)")


if __name__ == "__main__":
    main()
