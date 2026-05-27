"""
10j_researchgate_abstracts.py
------------------------------
Fetches abstracts for remaining missing papers by searching ResearchGate.

Steps:
  1. For each missing paper title, query ResearchGate search API
  2. Find best matching publication URL from search results
  3. Fetch publication page and extract abstract from HTML
  4. Fuzzy-match extracted title to confirm correct paper
  5. Save abstracts to afa_papers_enriched.parquet

ResearchGate returns abstracts in HTML without login when using
browser-like headers (abstract is in meta og:description or page body).

Usage:
    python scripts/10j_researchgate_abstracts.py
    python scripts/10j_researchgate_abstracts.py --force   # ignore cache
    python scripts/10j_researchgate_abstracts.py --limit N # process N papers
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

ROOT       = pathlib.Path(__file__).parent.parent.resolve()
PARQUET    = ROOT / "data" / "raw" / "afa_papers_enriched.parquet"
CACHE_PATH = ROOT / "data" / "raw" / "afa_10j_cache.json"
LOG_PATH   = ROOT / "validation" / "abstract_10j_matches.csv"

FUZZY_MIN  = 75
TIMEOUT    = 15

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

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


def content_valid(title: str, abstract: str, min_overlap: int = 2) -> bool:
    title_words = {w for w in normalise(title).split() if len(w) >= 4}
    abs_text = normalise(abstract)
    return sum(1 for w in title_words if w in abs_text) >= min_overlap


def extract_abstract_from_html(html: str, title: str) -> tuple[str, str]:
    """Extract abstract and page title from ResearchGate HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # og:description contains "Request PDF | Title | ABSTRACT... | Find, read..."
    # Strip the boilerplate and extract just the abstract portion.
    for attr in [{"property": "og:description"}, {"name": "description"}]:
        tag = soup.find("meta", attr)
        if not tag:
            continue
        raw = tag.get("content", "").strip()
        # Strip RG framing: "Request PDF | Title | ABSTRACT | Find, read..."
        m = re.match(
            r'^(?:Request(?:\s+full-text)?\s+PDF|Article)\s*\|\s*.+?\|\s*(.+?)\s*\|\s*Find,\s*read',
            raw, re.DOTALL,
        )
        candidate = re.sub(r"\.\.\.$", "", m.group(1)).strip() if m else raw
        if is_good(candidate) and content_valid(title, candidate):
            page_title = ""
            og_title = soup.find("meta", property="og:title")
            if og_title:
                page_title = og_title.get("content", "").strip()
            return candidate, page_title

    # Try known abstract div classes
    for cls in [
        "research-detail-header-section__abstract",
        "publication-abstract",
        "abstractSection",
        "abstract",
    ]:
        div = soup.find(attrs={"class": re.compile(cls, re.I)})
        if div:
            text = div.get_text(" ", strip=True)
            text = re.sub(r"^[Aa]bstract[:\s]*", "", text).strip()
            if is_good(text) and content_valid(title, text):
                return text, ""

    # Fallback: JSON abstract in page source
    m = re.search(r'"abstract"\s*:\s*"([^"]{100,2000})"', html)
    if m:
        candidate = m.group(1).replace("\\n", " ").replace('\\"', '"').strip()
        if is_good(candidate) and content_valid(title, candidate):
            return candidate, ""

    return "", ""


def rg_search(title: str) -> list[dict]:
    """Use DuckDuckGo HTML search to find ResearchGate publication URLs."""
    q = f'site:researchgate.net/publication "{title[:120]}"'
    try:
        r = SESSION.get(
            "https://html.duckduckgo.com/html/",
            params={"q": q},
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            return []
        html = r.text
    except Exception:
        return []

    found = re.findall(r'researchgate\.net/publication/(\d+[^\s"&<]*)', html)
    seen = set()
    results = []
    for slug in found:
        pub_id = re.match(r"(\d+)", slug)
        if pub_id and pub_id.group(1) not in seen:
            seen.add(pub_id.group(1))
            clean = slug.split("?")[0].rstrip(".,)")
            results.append({
                "url": f"https://www.researchgate.net/publication/{clean}",
                "pub_id": pub_id.group(1),
            })
        if len(results) >= 3:
            break
    return results


def fetch_publication(url: str, title: str) -> tuple[str, str]:
    """Fetch a ResearchGate publication page and extract abstract + page title."""
    try:
        r = SESSION.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return "", ""
        return extract_abstract_from_html(r.text, title)
    except Exception:
        return "", ""


def wait_for_rg(max_wait: int = 86400) -> bool:
    """Poll ResearchGate every 5 minutes until Cloudflare block clears."""
    print("Waiting for ResearchGate to become accessible (polling every 5 min) ...")
    waited = 0
    while waited < max_wait:
        time.sleep(300)
        waited += 300
        try:
            r = SESSION.get(
                "https://www.researchgate.net/publication/256012960",
                timeout=TIMEOUT,
            )
            if r.status_code == 200:
                print(f"  ResearchGate accessible after {waited//60} min.")
                return True
            print(f"  {waited//60} min elapsed: still {r.status_code} ...")
        except Exception:
            print(f"  {waited//60} min elapsed: request failed, retrying ...")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore cache")
    ap.add_argument("--limit", type=int, default=0, help="max papers to process")
    ap.add_argument("--no-wait", action="store_true", help="fail immediately if blocked")
    args = ap.parse_args()

    # Check accessibility; wait if Cloudflare-blocked
    try:
        r = SESSION.get("https://www.researchgate.net/publication/256012960", timeout=TIMEOUT)
        if r.status_code != 200:
            if args.no_wait:
                raise SystemExit(f"ResearchGate blocked ({r.status_code}). Run without --no-wait to wait.")
            if not wait_for_rg():
                raise SystemExit("Timed out waiting for ResearchGate to become accessible.")
    except SystemExit:
        raise
    except Exception:
        pass

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

    print(f"Papers missing abstracts: {len(missing)}")
    print(f"Starting coverage: {has_abs.sum():,} / {len(df):,} ({has_abs.mean()*100:.1f}%)")

    cache: dict = {}
    if CACHE_PATH.exists() and not args.force:
        with open(CACHE_PATH) as f:
            cache = json.load(f)

    hits = 0
    log_rows = []

    for _, row in tqdm(missing.iterrows(), total=len(missing), desc="10j ResearchGate"):
        pid   = str(row["paper_id"])
        title = str(row.get("title", "") or "")

        if pid in cache and not args.force:
            entry    = cache[pid]
            abstract = entry.get("abstract", "")
            source   = entry.get("source", "none")
        else:
            abstract, source = "", "none"

            candidates = rg_search(title)
            time.sleep(2.0)

            q_norm = normalise(title)
            for cand in candidates:
                page_abstract, page_title = fetch_publication(cand["url"], title)
                time.sleep(1.5)

                if not page_abstract:
                    continue

                # If we got a page title, fuzzy-match it
                if page_title:
                    score = fuzz.token_sort_ratio(q_norm, normalise(page_title))
                    if score < FUZZY_MIN:
                        continue

                abstract = page_abstract
                source = "researchgate_10j"
                break

            cache[pid] = {"abstract": abstract, "source": source}

            # Conservative delay with jitter to avoid Cloudflare re-triggering
            time.sleep(3.0 + (hash(pid) % 20) / 10.0)

            # Save cache periodically
            if len(cache) % 50 == 0:
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(CACHE_PATH, "w") as f:
                    json.dump(cache, f)

        if abstract:
            df.loc[row.name, "abstract"]        = abstract
            df.loc[row.name, "abstract_source"] = source
            hits += 1
            log_rows.append({
                "paper_id": pid, "year": row.get("year"),
                "title": title, "source": source,
            })

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

    df.to_parquet(PARQUET, index=False)
    has_new = df["abstract"].notna() & (df["abstract"].str.strip() != "")
    gain = has_new.sum() - has_abs.sum()

    print(f"\nResults:")
    print(f"  New abstracts added : {gain}")
    print(f"  Final coverage      : {has_new.sum():,} / {len(df):,} ({has_new.mean()*100:.1f}%)")

    if log_rows:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(log_rows)
        if LOG_PATH.exists():
            new_df = pd.concat([pd.read_csv(LOG_PATH), new_df], ignore_index=True)
        new_df.to_csv(LOG_PATH, index=False)
        print(f"  Log: {LOG_PATH.name} ({len(log_rows)} new rows)")


if __name__ == "__main__":
    main()
