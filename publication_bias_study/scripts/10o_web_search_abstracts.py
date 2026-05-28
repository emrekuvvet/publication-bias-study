"""
10o_web_search_abstracts.py
----------------------------
Fetches abstracts for remaining missing papers.

Pipeline per paper:
  1. Extract author surnames from the `authors` column
  2. Search IDEAS/REPEC with "title + surnames" — no rate limiting, high recall
     a. Extract abstract from search result snippet (text after <hr/>)
     b. If no snippet, fetch the IDEAS paper page and extract abstract
  3. Fallback: Google Scholar plain requests (~3-5s delay to avoid CAPTCHA)
     a. Extract snippet from .gs_rs element
     b. If no snippet, fetch the first result URL

Usage:
    python scripts/10o_web_search_abstracts.py
    python scripts/10o_web_search_abstracts.py --force
    python scripts/10o_web_search_abstracts.py --limit N
"""

import argparse
import json
import pathlib
import random
import re
import time
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from tqdm import tqdm

ROOT       = pathlib.Path(__file__).parent.parent.resolve()
PARQUET    = ROOT / "data" / "raw" / "afa_papers_enriched.parquet"
CACHE_PATH = ROOT / "data" / "raw" / "afa_10o_cache.json"
LOG_PATH   = ROOT / "validation" / "abstract_10o_matches.csv"

FUZZY_MIN    = 68
MAX_RESULTS  = 4
TIMEOUT      = 15
IDEAS_DELAY  = (0.8, 1.5)    # IDEAS has no strict rate limit but be polite
SCHOLAR_DELAY = (3.0, 5.5)   # Scholar blocks at higher rates

SCHOLAR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

STOP_WORDS = {
    "a","an","the","of","in","on","at","to","for","and","or","but",
    "with","by","from","that","this","is","are","was","were","be",
    "been","being","have","has","had","do","does","did","will","would",
    "can","could","should","may","might","shall","some","any","all",
}

SKIP_DOMAINS = {
    "twitter.com","x.com","facebook.com","linkedin.com","youtube.com",
    "wikipedia.org","amazon.com","jstor.org",
    "springer.com","wiley.com","tandfonline.com",
}


def normalise(text: str) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFD", str(text)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return " ".join(t for t in s.split() if t not in STOP_WORDS and len(t) > 1)


def is_good(abstract: str) -> bool:
    return bool(abstract) and len(abstract.split()) >= 20


def content_valid(title: str, abstract: str, min_overlap: int = 2) -> bool:
    title_words = {w for w in normalise(title).split() if len(w) >= 4}
    abs_text = normalise(abstract)
    return sum(1 for w in title_words if w in abs_text) >= min_overlap


def should_skip(url: str) -> bool:
    return any(d in url for d in SKIP_DOMAINS)


def extract_surnames(authors_str: str) -> list[str]:
    """Extract up to 3 author surnames from 'First Last, Affil and First Last, Affil' strings."""
    if not authors_str:
        return []
    parts = re.split(r"\s+and\s+", str(authors_str))
    surnames = []
    for part in parts[:3]:
        name_token = part.split(",")[0].strip()
        words = name_token.split()
        if words:
            surnames.append(words[-1].rstrip("."))
    return surnames


# ── IDEAS/REPEC search ──────────────────────────────────────────────────────

IDEAS_SESSION = requests.Session()
IDEAS_SESSION.headers.update({"User-Agent": "Mozilla/5.0"})


def _ideas_abstract_from_li(li, title: str) -> str:
    """Extract abstract text from an IDEAS <li> result item (text after <hr/>)."""
    hr = li.find("hr")
    if not hr:
        return ""
    parts = []
    for sib in hr.next_siblings:
        if sib.name == "i":   # RepEC handle marker — stop here
            break
        if sib.name == "br":
            break
        if isinstance(sib, str):
            parts.append(sib)
        elif sib.name not in ("script", "span"):
            parts.append(sib.get_text(" "))
    txt = re.sub(r"\s+", " ", " ".join(parts)).strip()
    # Relax content_valid to 1 overlap: title fuzzy match already verified relevance
    if is_good(txt) and content_valid(title, txt, min_overlap=1):
        return txt
    return ""


def _ideas_abstract_from_page(url: str, title: str) -> str:
    """Fetch an IDEAS paper page and extract the abstract."""
    try:
        r = IDEAS_SESSION.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return ""
    except Exception:
        return ""
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(" ")
    idx = text.find("Abstract")
    if idx == -1:
        return ""
    chunk = text[idx + 8 : idx + 2500]
    for stop in ["Download", "JEL", "Keywords", "Suggested Citation",
                 "Related works", "References", "No abstract is available"]:
        si = chunk.find(stop)
        if si > 50:
            chunk = chunk[:si]
    txt = re.sub(r"\s+", " ", chunk).strip()
    if is_good(txt) and content_valid(title, txt):
        return txt
    return ""


def ideas_search(title: str, surnames: list[str]) -> tuple[str, str]:
    """Search IDEAS with title+surnames. Returns (abstract, source_label)."""
    query = f"{title} {' '.join(surnames)}" if surnames else title
    try:
        r = IDEAS_SESSION.post(
            "https://ideas.repec.org/cgi-bin/htsearch2",
            data={"q": query[:200], "method": "and", "fmt": "score"},
            timeout=TIMEOUT,
        )
    except Exception:
        return "", ""

    if r.status_code != 200:
        return "", ""

    soup = BeautifulSoup(r.text, "html.parser")
    ol = soup.find("ol", class_="list-group")
    if not ol:
        return "", ""

    q_norm = normalise(title)
    for li in ol.find_all("li")[:6]:
        link = li.find("a")
        if not link:
            continue
        # use get_text(" ") to preserve spaces between <b> tags
        cand_title = re.sub(r"\s+", " ", link.get_text(" ")).strip()
        score = fuzz.token_set_ratio(q_norm, normalise(cand_title))
        if score < FUZZY_MIN:
            continue

        # Try snippet in search result first
        abstract = _ideas_abstract_from_li(li, title)
        if abstract:
            return abstract, "ideas_snippet_10o"

        # Fallback: fetch the IDEAS paper page
        href = link.get("href", "")
        if href:
            abstract = _ideas_abstract_from_page(href, title)
            if abstract:
                return abstract, "ideas_page_10o"

    return "", ""


# ── Google Scholar ──────────────────────────────────────────────────────────

SCHOLAR_SESSION = requests.Session()
SCHOLAR_SESSION.headers.update(SCHOLAR_HEADERS)
_scholar_blocked = False   # flip to True on first CAPTCHA, skip rest of run


def scholar_search(title: str) -> tuple[str, list[str]]:
    """Search Google Scholar. Returns (snippet_or_empty, [result_urls])."""
    global _scholar_blocked
    if _scholar_blocked:
        return "", []
    try:
        r = SCHOLAR_SESSION.get(
            "https://scholar.google.com/scholar",
            params={"q": title[:120], "hl": "en"},
            timeout=20,
        )
    except Exception:
        return "", []

    if r.status_code != 200:
        return "", []

    if "captcha" in r.text.lower() or "/sorry/" in r.url:
        _scholar_blocked = True
        return "", []

    soup = BeautifulSoup(r.text, "html.parser")
    q_norm = normalise(title)
    snippet_abstract = ""
    urls = []

    result_blocks = soup.select(".gs_r.gs_or.gs_scl") or soup.select(".gs_r")
    for block in result_blocks[:6]:
        title_link = block.select_one(".gs_rt a")
        if title_link:
            href = title_link.get("href", "")
            cand_title = title_link.get_text(strip=True)
            if href.startswith("http") and not should_skip(href):
                if fuzz.token_set_ratio(q_norm, normalise(cand_title)) >= FUZZY_MIN:
                    urls.append(href)

        if not snippet_abstract:
            snip = block.select_one(".gs_rs")
            if snip:
                txt = re.sub(r"\s+", " ", snip.get_text(" ", strip=True)).strip()
                txt = re.sub(r"^[Aa]bstract[:\s]*", "", txt).strip()
                if is_good(txt) and content_valid(title, txt):
                    t_link = block.select_one(".gs_rt")
                    t_text = t_link.get_text(strip=True) if t_link else ""
                    if fuzz.token_set_ratio(q_norm, normalise(t_text)) >= FUZZY_MIN:
                        snippet_abstract = txt

    return snippet_abstract, urls


# ── General URL fetching ────────────────────────────────────────────────────

def extract_abstract_from_html(html: str, title: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for attr in [{"property": "og:description"}, {"name": "description"}]:
        tag = soup.find("meta", attr)
        if tag:
            txt = tag.get("content", "").strip()
            if is_good(txt) and content_valid(title, txt):
                return txt

    for sel in [
        ".abstract", ".abstract-text", "#abstract", "#abstractDiv",
        "[class*='abstract']", "[id*='abstract']",
        ".description", ".summary", ".paper-abstract", "dd",
    ]:
        try:
            el = soup.select_one(sel)
            if el:
                txt = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
                txt = re.sub(r"^[Aa]bstract[:\s]*", "", txt).strip()
                if is_good(txt) and content_valid(title, txt):
                    return txt
        except Exception:
            pass

    for heading in soup.find_all(["h2","h3","h4","h5","strong","b","dt"]):
        if re.match(r"^\s*abstract\s*$", heading.get_text(), re.I):
            sib = heading.find_next_sibling()
            if sib:
                txt = re.sub(r"\s+", " ", sib.get_text(" ", strip=True))
                if is_good(txt) and content_valid(title, txt):
                    return txt

    title_words = {w for w in normalise(title).split() if len(w) >= 4}
    best_p, best_overlap = "", 0
    for p in soup.find_all("p"):
        txt = re.sub(r"\s+", " ", p.get_text(" ", strip=True))
        if not is_good(txt):
            continue
        overlap = sum(1 for w in title_words if w in normalise(txt))
        if overlap >= 2 and overlap > best_overlap:
            best_overlap = overlap
            best_p = txt
    return best_p


def fetch_abstract(url: str, title: str) -> str:
    if should_skip(url):
        return ""
    try:
        r = requests.get(url, headers=FETCH_HEADERS, timeout=TIMEOUT,
                         allow_redirects=True)
        if r.status_code != 200:
            return ""
        ct = r.headers.get("content-type", "")
        if "html" not in ct and "text" not in ct:
            return ""
        return extract_abstract_from_html(r.text, title)
    except Exception:
        return ""


# ── Main processing loop ────────────────────────────────────────────────────

def run(missing: pd.DataFrame, cache: dict) -> tuple[dict, list]:
    log_rows = []

    for _, row in tqdm(missing.iterrows(), total=len(missing), desc="10o search"):
        pid     = str(row["paper_id"])
        title   = str(row.get("title", "") or "")
        authors = str(row.get("authors", "") or "")

        if pid in cache:
            entry    = cache[pid]
            abstract = entry.get("abstract", "")
            source   = entry.get("source", "none")
        else:
            abstract, source = "", "none"
            surnames = extract_surnames(authors)

            # ── 1. IDEAS primary ──
            abstract, source = ideas_search(title, surnames)
            time.sleep(random.uniform(*IDEAS_DELAY))

            # ── 2. Google Scholar fallback ──
            if not abstract:
                snippet, urls = scholar_search(title)
                if not _scholar_blocked:
                    time.sleep(random.uniform(*SCHOLAR_DELAY))
                if snippet:
                    abstract = snippet
                    source   = "scholar_snippet_10o"
                else:
                    for url in urls[:MAX_RESULTS]:
                        abstract = fetch_abstract(url, title)
                        if abstract:
                            source = "web_10o"
                            break
                        time.sleep(0.5)

            cache[pid] = {"abstract": abstract, "source": source}

            if len(cache) % 50 == 0:
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(CACHE_PATH, "w") as f:
                    json.dump(cache, f)

        if abstract:
            log_rows.append({
                "paper_id": pid,
                "year":     row.get("year"),
                "title":    title,
                "source":   source,
            })

    return cache, log_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore cache")
    ap.add_argument("--limit", type=int, default=0, help="max papers to process")
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    has_abs = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    noise = df["title"].str.contains(
        r'(?i)^(?:Session Chair|Location:|AFA Lecture)'
        r'|\d{4}.*\bthrough\b'
        r'|^(?:Finance|Economics|Accounting|Real Estate)\s+(?:Group|Unit|Section)\s*$'
        r'|^Finance and Economics|^Economics,\s*Finance',
        regex=True, na=False)
    missing = df[~has_abs & ~noise].copy()

    if args.limit:
        missing = missing.head(args.limit)

    print(f"Papers missing abstracts : {len(missing)}")
    print(f"Starting coverage        : {has_abs[~noise].sum():,} / {(~noise).sum():,} "
          f"({has_abs[~noise].mean()*100:.1f}%)")

    cache: dict = {}
    if CACHE_PATH.exists() and not args.force:
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    if not args.force:
        missing = missing[~missing["paper_id"].astype(str).isin(cache)]
        print(f"After cache filter       : {len(missing)} to process")

    cache, log_rows = run(missing, cache)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

    for pid, entry in cache.items():
        abstract = entry.get("abstract", "")
        source   = entry.get("source", "none")
        if not abstract:
            continue
        mask = df["paper_id"].astype(str) == pid
        if mask.any() and (df.loc[mask, "abstract"].isna().all() or
                           (df.loc[mask, "abstract"].str.strip() == "").all()):
            df.loc[mask, "abstract"]        = abstract
            df.loc[mask, "abstract_source"] = source

    df.to_parquet(PARQUET, index=False)
    has_new = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    gain = has_new[~noise].sum() - has_abs[~noise].sum()

    print(f"\nResults:")
    print(f"  New abstracts added : {gain}")
    print(f"  Final coverage      : {has_new[~noise].sum():,} / {(~noise).sum():,} "
          f"({has_new[~noise].mean()*100:.1f}%)")

    if log_rows:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(log_rows)
        if LOG_PATH.exists():
            new_df = pd.concat([pd.read_csv(LOG_PATH), new_df], ignore_index=True)
        new_df.to_csv(LOG_PATH, index=False)
        print(f"  Log: {LOG_PATH.name} ({len(log_rows)} new rows)")


if __name__ == "__main__":
    main()
