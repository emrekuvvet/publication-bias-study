"""
05_link_nber_to_published.py
----------------------------
Multi-strategy matching: published journal papers → NBER working papers.

Strategy 1 (local fuzzy)    : RapidFuzz token_sort_ratio against ALL 30,219
                               raw NBER papers (previously only 113 in-scope).
Strategy 2 (OpenAlex API)   : Use each paper's openalex_id to fetch OA
                               locations; extract nber.org/papers/wXXXX URLs.
Strategy 3 (Semantic Scholar): Title+year search; check externalIds for NBER
                               DOIs (10.3386/wXXXX).
Strategy 4 (NBER search API): Keyword+author search on nber.org REST endpoint.

Results merged in priority order: fuzzy > OpenAlex > SS > NBER API.
API results are cached in validation/api_match_cache.json so re-runs skip
already-queried papers.

Output:
    data/final/analysis_dataset.parquet
    validation/match_review_queue.parquet
    validation/api_match_log.csv          (one row per successful API hit)

Usage:
    python scripts/05_link_nber_to_published.py
    python scripts/05_link_nber_to_published.py --skip-api
    python scripts/05_link_nber_to_published.py --threshold 80
"""

import argparse
import json
import pathlib
import re
import time
import unicodedata
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz, process
from tqdm import tqdm

ROOT     = pathlib.Path(__file__).parent.parent.resolve()
OUT_PATH = ROOT / "data" / "final" / "analysis_dataset.parquet"
CACHE_PATH = ROOT / "validation" / "api_match_cache.json"
LOG_PATH   = ROOT / "validation" / "api_match_log.csv"

AUTO_MATCH_THRESHOLD = 90.0
REVIEW_THRESHOLD     = 75.0

# OpenAlex polite pool email (speeds up requests)
OA_EMAIL = "research@pubbiasstudy.org"

# ------------------------------------------------------------------ #
# Text helpers                                                         #
# ------------------------------------------------------------------ #

STOP_WORDS = {
    "a","an","the","of","in","on","at","to","for","and","or","but",
    "with","by","from","that","this","is","are","was","were","be",
    "been","being","have","has","had","do","does","did","will","would",
    "can","could","should","may","might","shall","some","any","all",
    "its","it","we","our","their","there","when","where","which","who",
}


def normalise_title(title: str) -> str:
    if not title:
        return ""
    nfd = unicodedata.normalize("NFD", str(title))
    ascii_str = nfd.encode("ascii", "ignore").decode("ascii")
    clean = re.sub(r"[^a-z0-9\s]", " ", ascii_str.lower())
    tokens = [t for t in clean.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def extract_surnames(authors_str: str) -> set:
    surnames = set()
    for name in str(authors_str or "").split(";"):
        name = name.strip()
        if not name:
            continue
        if "," in name:
            last = name.split(",")[0].strip().lower()
        else:
            parts = name.split()
            last = parts[-1].lower() if parts else ""
        last = re.sub(r"[^a-z]", "", last)
        if last:
            surnames.add(last)
    return surnames


# ------------------------------------------------------------------ #
# NBER URL / DOI parsing                                               #
# ------------------------------------------------------------------ #

_NBER_URL_RE = re.compile(
    r"(?:nber\.org/papers?/|10\.3386/)([hw]\d+)", re.IGNORECASE
)


def extract_wp_from_text(text: str) -> str | None:
    """Return NBER wp_number like 'w12345' from any URL/DOI string."""
    if not text:
        return None
    m = _NBER_URL_RE.search(text)
    if m:
        return m.group(1).lower()
    return None


def wp_to_int(wp: str) -> int | None:
    """'w12345' → 12345"""
    try:
        return int(re.sub(r"[^0-9]", "", wp))
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------ #
# Strategy 1 — Local fuzzy matching against ALL raw NBER papers        #
# ------------------------------------------------------------------ #

def build_nber_index(raw_nber: pd.DataFrame) -> tuple[dict, dict]:
    """
    Returns:
      nber_index       : {wp_number: normalised_title}
      nber_authors_map : {wp_number: authors_string}
    """
    nber_index, nber_authors_map = {}, {}
    for _, row in raw_nber.iterrows():
        wp = str(row["wp_number"])
        title = row.get("title", "") or ""
        authors = row.get("authors", "") or ""
        if title:
            nber_index[wp] = normalise_title(title)
        nber_authors_map[wp] = authors
    return nber_index, nber_authors_map


def fuzzy_match_one(
    journal_title_norm: str,
    journal_authors: str,
    nber_index: dict,
    nber_authors_map: dict,
    review_threshold: float = REVIEW_THRESHOLD,
    limit: int = 20,
) -> tuple[str | None, float, str]:
    """
    Returns (wp_number, score, quality) where quality ∈ {auto, review, none}.
    Fixes the RapidFuzz dict-mode unpacking: process.extract against a dict
    returns (value, score, key) — we need the key (wp_number), not the value.
    """
    if not journal_title_norm or not nber_index:
        return None, 0.0, "none"

    candidates = process.extract(
        journal_title_norm,
        nber_index,
        scorer=fuzz.token_sort_ratio,
        limit=limit,
    )
    # RapidFuzz dict mode: each item is (matched_value, score, key)
    j_surnames = extract_surnames(journal_authors)
    best_wp, best_score = None, 0.0

    for _matched_val, score, wp_num in candidates:
        if score < review_threshold:
            break
        n_surnames = extract_surnames(nber_authors_map.get(str(wp_num), ""))
        if j_surnames and n_surnames and not (j_surnames & n_surnames):
            continue
        if score > best_score:
            best_score = score
            best_wp = str(wp_num)

    if best_wp is None:
        return None, 0.0, "none"
    quality = "auto" if best_score >= AUTO_MATCH_THRESHOLD else "review"
    return best_wp, best_score, quality


# ------------------------------------------------------------------ #
# Strategy 2 — OpenAlex API                                            #
# ------------------------------------------------------------------ #

def _oa_extract_nber(data: dict) -> str | None:
    """Check OpenAlex work record for any nber.org location URL."""
    # best_oa_location
    boa = data.get("best_oa_location") or {}
    for field in ("url", "pdf_url", "landing_page_url"):
        wp = extract_wp_from_text(boa.get(field, ""))
        if wp:
            return wp
    # all locations
    for loc in data.get("locations", []):
        for field in ("url", "pdf_url", "landing_page_url"):
            wp = extract_wp_from_text(loc.get(field, ""))
            if wp:
                return wp
    # open_access.oa_url
    oa = data.get("open_access") or {}
    wp = extract_wp_from_text(oa.get("oa_url", ""))
    if wp:
        return wp
    # primary_location
    pl = data.get("primary_location") or {}
    wp = extract_wp_from_text(pl.get("pdf_url", "") or pl.get("landing_page_url", ""))
    if wp:
        return wp
    return None


def openalex_lookup_batch(
    openalex_ids: list[str], session: requests.Session
) -> dict[str, str]:
    """
    Batch-query OpenAlex for up to 25 works at a time.
    Returns {openalex_id: wp_number} for papers with NBER locations found.
    """
    results = {}
    BATCH = 25
    for i in range(0, len(openalex_ids), BATCH):
        batch = openalex_ids[i : i + BATCH]
        ids_filter = "|".join(oa.replace("https://openalex.org/", "") for oa in batch)
        try:
            resp = session.get(
                "https://api.openalex.org/works",
                params={
                    "filter": f"openalex_id:{ids_filter}",
                    "select": "id,best_oa_location,locations,open_access,primary_location",
                    "per_page": BATCH,
                    "mailto": OA_EMAIL,
                },
                timeout=20,
            )
            if resp.status_code != 200:
                continue
            for work in resp.json().get("results", []):
                wp = _oa_extract_nber(work)
                if wp:
                    oa_id = work.get("id", "")
                    results[oa_id] = wp
        except Exception as exc:
            print(f"  OpenAlex batch error (offset {i}): {exc}")
        time.sleep(0.12)   # ≈8 req/s — within polite pool limits
    return results


# ------------------------------------------------------------------ #
# Strategy 3 — Semantic Scholar API                                    #
# ------------------------------------------------------------------ #

def ss_lookup_one(
    title: str, year: int, authors: str, session: requests.Session
) -> str | None:
    """
    Search SS for papers matching title+year.  If any result has a
    10.3386/wXXXX DOI and shares ≥1 author surname → return wp_number.
    """
    try:
        year_i = int(year) if year else None
        params: dict = {
            "query": title[:200],
            "fields": "externalIds,year,authors,title",
            "limit": 8,
        }
        if year_i:
            params["year"] = f"{year_i - 1}-{year_i + 1}"

        for _attempt in range(3):
            resp = session.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
                timeout=20,
            )
            if resp.status_code == 429:
                time.sleep(8)   # back off on rate limit, then retry
                continue
            break
        else:
            return None
        if resp.status_code != 200:
            return None

        j_title_norm = normalise_title(title)
        j_surnames = extract_surnames(authors)

        for paper in resp.json().get("data", []):
            ext = paper.get("externalIds") or {}
            doi = ext.get("DOI", "")
            wp = extract_wp_from_text(doi)
            if not wp:
                continue
            # Title similarity guard
            score = fuzz.token_sort_ratio(
                j_title_norm, normalise_title(paper.get("title", ""))
            )
            if score < 70:
                continue
            # Author guard
            ss_names = "; ".join(
                a.get("name", "") for a in (paper.get("authors") or [])
            )
            ss_surnames = extract_surnames(ss_names)
            if j_surnames and ss_surnames and not (j_surnames & ss_surnames):
                continue
            return wp
    except Exception:
        pass
    return None


# ------------------------------------------------------------------ #
# Strategy 4 — NBER search REST endpoint                               #
# ------------------------------------------------------------------ #

def nber_search_one(
    title: str, year: int, authors: str, session: requests.Session
) -> str | None:
    """
    Query the NBER papers search API with title keywords + year range.
    Returns wp_number if a high-similarity match is found.
    """
    try:
        title_words = normalise_title(title).split()
        # Use 3 most distinctive (longest) words as query
        query_words = sorted(set(title_words), key=len, reverse=True)[:4]
        query = " ".join(query_words)
        if not query:
            return None

        year_i = int(year) if year else None
        params: dict = {"q": query, "perPage": 15}

        resp = session.get(
            "https://www.nber.org/api/v1/dataset_metadata/research/working_papers",
            params=params,
            timeout=15,
        )
        if resp.status_code != 200:
            return None

        j_title_norm = normalise_title(title)
        j_surnames = extract_surnames(authors)

        for item in resp.json().get("results", [])[:15]:
            item_title = item.get("title", "")
            score = fuzz.token_sort_ratio(normalise_title(item_title), j_title_norm)
            if score < 80:
                continue

            # Year guard (±3 years)
            item_year = item.get("year") or item.get("publishedYear")
            if year_i and item_year:
                try:
                    if abs(int(item_year) - year_i) > 3:
                        continue
                except (ValueError, TypeError):
                    pass

            # Author guard
            raw_authors = item.get("authors") or []
            if isinstance(raw_authors, list):
                item_author_str = "; ".join(
                    a.get("name", "") if isinstance(a, dict) else str(a)
                    for a in raw_authors
                )
            else:
                item_author_str = str(raw_authors)

            item_surnames = extract_surnames(item_author_str)
            if j_surnames and item_surnames and not (j_surnames & item_surnames):
                continue

            # Extract WP number
            wp_num = item.get("workingPaperNumber") or item.get("wp_number")
            if wp_num:
                wp = f"w{int(wp_num)}"
                return wp
            url = item.get("url", "")
            wp = extract_wp_from_text(url)
            if wp:
                return wp

    except Exception:
        pass
    return None


# ------------------------------------------------------------------ #
# Cache helpers                                                        #
# ------------------------------------------------------------------ #

def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            pass
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(review_threshold: float, skip_api: bool) -> None:
    global REVIEW_THRESHOLD
    REVIEW_THRESHOLD = review_threshold

    # ---- Load datasets -----------------------------------------------
    inscope = pd.read_parquet(ROOT / "data" / "classified" / "inscope_papers.parquet")
    inscope = inscope[inscope["in_scope"]].copy()

    direction_path = ROOT / "data" / "classified" / "coded_direction.parquet"
    if direction_path.exists():
        direction = pd.read_parquet(direction_path)
        merged = inscope.merge(
            direction[["paper_id", "direction", "confidence", "rationale"]],
            on="paper_id", how="left"
        )
    else:
        print("  Note: coded_direction.parquet missing — direction columns will be NULL")
        merged = inscope.copy()
        merged["direction"] = pd.NA
        merged["confidence"] = pd.NA
        merged["rationale"] = pd.NA

    # Restrict both samples to 2000+ (study design: post-deregulation era)
    nber_inscope_df = merged[
        (merged["source"] == "nber") &
        (pd.to_numeric(merged["year"], errors="coerce") >= 2000)
    ].copy()
    journal_df = merged[
        merged["source"].isin(["jf", "rfs", "jfe"]) &
        (pd.to_numeric(merged["year"], errors="coerce") >= 2000)
    ].copy()
    print(f"NBER in-scope papers:    {len(nber_inscope_df):,}  (year ≥ 2000)")
    print(f"Journal in-scope papers: {len(journal_df):,}  (year ≥ 2000)")

    # ---- Enrich with raw authors + openalex_id -----------------------
    raw_nber = pd.read_parquet(
        ROOT / "data" / "raw" / "nber_papers.parquet",
        columns=["wp_number", "authors", "title", "year"],
    )

    journal_meta = []
    for src in ["jf", "rfs", "jfe"]:
        rpath = ROOT / "data" / "raw" / f"{src}_papers.parquet"
        if rpath.exists():
            rj = pd.read_parquet(
                rpath, columns=["doi", "authors", "openalex_id"]
            ).rename(columns={"doi": "paper_id"})
            journal_meta.append(rj)
    if journal_meta:
        jmeta = pd.concat(journal_meta, ignore_index=True).drop_duplicates("paper_id")
        journal_df = journal_df.merge(
            jmeta[["paper_id", "authors", "openalex_id"]],
            on="paper_id", how="left"
        )
    if "authors" not in journal_df.columns:
        journal_df["authors"] = ""
    if "openalex_id" not in journal_df.columns:
        journal_df["openalex_id"] = pd.NA

    nber_inscope_df = nber_inscope_df.merge(
        raw_nber[["wp_number", "authors"]].rename(columns={"wp_number": "paper_id"}),
        on="paper_id", how="left"
    )

    # ---- Strategy 1: fuzzy match against ALL 30k NBER papers ---------
    print("\n[Strategy 1] Fuzzy title matching against ALL raw NBER papers …")
    nber_index, nber_authors_map = build_nber_index(raw_nber)
    print(f"  NBER index size: {len(nber_index):,}")

    fuzzy_results: dict[str, tuple] = {}  # paper_id → (wp, score, quality)
    for _, jrow in tqdm(journal_df.iterrows(), total=len(journal_df)):
        pid = jrow["paper_id"]
        j_norm = normalise_title(jrow.get("title", ""))
        j_auth = jrow.get("authors", "") or ""
        wp, score, quality = fuzzy_match_one(
            j_norm, j_auth, nber_index, nber_authors_map, review_threshold
        )
        fuzzy_results[pid] = (wp, score, quality)

    auto_n   = sum(1 for _, s, q in fuzzy_results.values() if q == "auto")
    review_n = sum(1 for _, s, q in fuzzy_results.values() if q == "review")
    print(f"  Auto  (≥{AUTO_MATCH_THRESHOLD}): {auto_n}  |  "
          f"Review ({review_threshold}–{AUTO_MATCH_THRESHOLD-1}): {review_n}  |  "
          f"None: {len(fuzzy_results)-auto_n-review_n}")

    if skip_api:
        print("  --skip-api: skipping API strategies")
        final_matches = {
            pid: (wp, score, quality, "fuzzy")
            for pid, (wp, score, quality) in fuzzy_results.items()
            if quality != "none"
        }
    else:
        # ---- Strategy 2: OpenAlex ------------------------------------
        print("\n[Strategy 2] OpenAlex API (OA location lookup) …")
        unmatched_pids = [
            pid for pid, (_, _, q) in fuzzy_results.items() if q == "none"
        ]
        oa_id_to_pid = {}
        for _, jrow in journal_df.iterrows():
            oa = jrow.get("openalex_id", "")
            if oa and jrow["paper_id"] in unmatched_pids:
                oa_id_to_pid[str(oa)] = jrow["paper_id"]

        cache = load_cache()
        oa_wp_map: dict[str, str] = {}  # paper_id → wp_number

        if oa_id_to_pid:
            session = requests.Session()
            session.headers["User-Agent"] = (
                "PublicationBiasStudy/1.0 (mailto:research@pubbiasstudy.org)"
            )
            oa_ids_to_query = [
                oa for oa in oa_id_to_pid
                if oa not in cache.get("openalex", {})
            ]
            print(f"  Querying {len(oa_ids_to_query)} papers "
                  f"({len(oa_id_to_pid)-len(oa_ids_to_query)} cached) …")

            if oa_ids_to_query:
                found = openalex_lookup_batch(oa_ids_to_query, session)
                if "openalex" not in cache:
                    cache["openalex"] = {}
                for oa_id in oa_ids_to_query:
                    cache["openalex"][oa_id] = found.get(oa_id, "")
                save_cache(cache)

            for oa_id, pid in oa_id_to_pid.items():
                wp = cache["openalex"].get(oa_id, "")
                if wp:
                    oa_wp_map[pid] = wp

        oa_new = sum(
            1 for pid, wp in oa_wp_map.items()
            if fuzzy_results.get(pid, (None, 0, "none"))[2] == "none"
        )
        print(f"  OpenAlex new matches: {oa_new}")

        # ---- Strategy 3: Semantic Scholar ----------------------------
        print("\n[Strategy 3] Semantic Scholar title search …")
        still_unmatched = [
            pid for pid in unmatched_pids if pid not in oa_wp_map
        ]
        ss_wp_map: dict[str, str] = {}

        if "ss" not in cache:
            cache["ss"] = {}

        pids_to_query_ss = [p for p in still_unmatched if p not in cache["ss"]]
        print(f"  Querying {len(pids_to_query_ss)} papers "
              f"({len(still_unmatched)-len(pids_to_query_ss)} cached) …")

        j_lookup = journal_df.set_index("paper_id")
        session_ss = requests.Session()
        session_ss.headers["User-Agent"] = "PublicationBiasResearch/1.0"

        for pid in tqdm(pids_to_query_ss):
            row = j_lookup.loc[pid] if pid in j_lookup.index else None
            if row is None:
                cache["ss"][pid] = ""
                continue
            wp = ss_lookup_one(
                row.get("title", ""),
                row.get("year"),
                row.get("authors", "") or "",
                session_ss,
            )
            cache["ss"][pid] = wp or ""
            time.sleep(4.0)   # SS free tier: ~15 req/min without key

        save_cache(cache)

        for pid in still_unmatched:
            wp = cache["ss"].get(pid, "")
            if wp:
                ss_wp_map[pid] = wp

        ss_new = len(ss_wp_map)
        print(f"  Semantic Scholar new matches: {ss_new}")

        # ---- Strategy 4: NBER search API -----------------------------
        print("\n[Strategy 4] NBER search API …")
        still_unmatched2 = [
            pid for pid in still_unmatched if pid not in ss_wp_map
        ]
        nber_api_wp_map: dict[str, str] = {}

        if "nber_api" not in cache:
            cache["nber_api"] = {}

        pids_to_query_nb = [p for p in still_unmatched2 if p not in cache["nber_api"]]
        print(f"  Querying {len(pids_to_query_nb)} papers "
              f"({len(still_unmatched2)-len(pids_to_query_nb)} cached) …")

        session_nb = requests.Session()
        session_nb.headers["User-Agent"] = "PublicationBiasResearch/1.0"

        for pid in tqdm(pids_to_query_nb):
            row = j_lookup.loc[pid] if pid in j_lookup.index else None
            if row is None:
                cache["nber_api"][pid] = ""
                continue
            wp = nber_search_one(
                row.get("title", ""),
                row.get("year"),
                row.get("authors", "") or "",
                session_nb,
            )
            cache["nber_api"][pid] = wp or ""
            time.sleep(0.5)

        save_cache(cache)

        for pid in still_unmatched2:
            wp = cache["nber_api"].get(pid, "")
            if wp:
                nber_api_wp_map[pid] = wp

        nber_api_new = len(nber_api_wp_map)
        print(f"  NBER API new matches: {nber_api_new}")

        # ---- Merge all strategies (priority: fuzzy > OA > SS > NBER) -
        final_matches: dict[str, tuple] = {}
        for pid, (wp, score, quality) in fuzzy_results.items():
            if quality != "none":
                final_matches[pid] = (wp, score, quality, "fuzzy")
        for pid, wp in oa_wp_map.items():
            if pid not in final_matches:
                final_matches[pid] = (wp, 0.0, "api", "openalex")
        for pid, wp in ss_wp_map.items():
            if pid not in final_matches:
                final_matches[pid] = (wp, 0.0, "api", "semantic_scholar")
        for pid, wp in nber_api_wp_map.items():
            if pid not in final_matches:
                final_matches[pid] = (wp, 0.0, "api", "nber_api")

    # ---- Build match_records dataframe --------------------------------
    match_records = []
    for _, jrow in journal_df.iterrows():
        pid = jrow["paper_id"]
        if pid in final_matches:
            wp, score, quality, source = final_matches[pid]
        else:
            wp, score, quality, source = None, 0.0, "none", "none"
        match_records.append({
            "journal_paper_id": pid,
            "nber_wp_number":   wp,
            "match_score":      score,
            "match_quality":    quality,
            "match_source":     source,
        })

    matches = pd.DataFrame(match_records)
    total_matched = (matches["match_quality"] != "none").sum()
    auto_n   = (matches["match_quality"] == "auto").sum()
    review_n = (matches["match_quality"] == "review").sum()
    api_n    = (matches["match_quality"] == "api").sum()
    none_n   = (matches["match_quality"] == "none").sum()

    print(f"\n{'='*55}")
    print(f"Final match summary ({len(matches)} journal papers):")
    print(f"  Auto fuzzy   (≥{AUTO_MATCH_THRESHOLD}): {auto_n:3d}  ({auto_n/len(matches)*100:.1f}%)")
    print(f"  Review fuzzy ({review_threshold}–{AUTO_MATCH_THRESHOLD-1}): {review_n:3d}  ({review_n/len(matches)*100:.1f}%)")
    print(f"  API match              : {api_n:3d}  ({api_n/len(matches)*100:.1f}%)")
    print(f"  No match               : {none_n:3d}  ({none_n/len(matches)*100:.1f}%)")
    print(f"  Total matched          : {total_matched:3d}  ({total_matched/len(matches)*100:.1f}%)")
    if not skip_api:
        src_counts = matches[matches["match_quality"]!="none"]["match_source"].value_counts()
        print(f"  By source: {src_counts.to_dict()}")
    print(f"{'='*55}")

    # ---- Save API match log ------------------------------------------
    if not skip_api:
        api_log = matches[matches["match_source"] != "none"].copy()
        # Attach titles for review
        title_map = journal_df.set_index("paper_id")["title"].to_dict()
        api_log["journal_title"] = api_log["journal_paper_id"].map(title_map)
        api_log = api_log.merge(
            raw_nber[["wp_number","title","authors","year"]].rename(
                columns={"wp_number":"nber_wp_number","title":"nber_title",
                         "authors":"nber_authors","year":"nber_year"}
            ),
            on="nber_wp_number", how="left"
        )
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        api_log.to_csv(LOG_PATH, index=False)
        print(f"Match log saved → {LOG_PATH}")

    # ---- Build final analysis dataset --------------------------------
    j = journal_df.copy()
    j["published_top3"] = True
    j["journal"] = j["source"].str.upper()
    j = j.merge(
        matches.rename(columns={"journal_paper_id": "paper_id"}),
        on="paper_id", how="left"
    )

    n = nber_inscope_df.copy()
    n["published_top3"] = False
    n["journal"] = pd.NA
    n["nber_wp_number"] = n["paper_id"]
    n["match_score"]    = pd.NA
    n["match_quality"]  = "n/a"
    n["match_source"]   = "n/a"

    # Mark in-scope NBER papers that were matched to journal papers
    matched_wp_ids = set(
        matches.loc[matches["match_quality"].isin(["auto", "review", "api"]),
                    "nber_wp_number"].dropna()
    )
    n.loc[n["paper_id"].isin(matched_wp_ids), "published_top3"] = True

    all_df = pd.concat([j, n], ignore_index=True)
    all_df["positive"] = all_df["direction"] == 1
    all_df["negative"] = all_df["direction"] == -1
    all_df["abstract_len"] = all_df["abstract"].str.split().str.len()

    # Compute n_authors from the authors field (count ";" delimiters + 1)
    if "authors" in all_df.columns:
        all_df["n_authors"] = all_df["authors"].apply(
            lambda x: len(str(x).split(";")) if pd.notna(x) and str(x).strip() else pd.NA
        )
    for col in ["id_strategy", "top_institution", "n_authors",
                "nber_citations", "jel_codes", "editor"]:
        if col not in all_df.columns:
            all_df[col] = pd.NA

    out_cols = [
        "paper_id", "source", "published_top3", "journal", "year",
        "title", "abstract",
        "direction", "positive", "negative",
        "nber_wp_number", "match_score", "match_quality", "match_source",
        "id_strategy", "top_institution", "n_authors",
        "nber_citations", "jel_codes", "editor", "abstract_len",
    ]
    out_df = all_df[[c for c in out_cols if c in all_df.columns]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"\nSaved → {OUT_PATH}  ({len(out_df):,} rows)")
    print(f"  Published (journal): {out_df[out_df['source'].isin(['jf','rfs','jfe'])]['published_top3'].sum()}")
    print(f"  NBER papers (now marked published): {out_df[out_df['source']=='nber']['published_top3'].sum()}")
    print(f"  NBER papers (unpublished control):  {(out_df['source']=='nber').sum() - out_df[out_df['source']=='nber']['published_top3'].sum()}")

    # ---- Save review queue -------------------------------------------
    review_q = matches[matches["match_quality"] == "review"]
    rq_path = ROOT / "validation" / "match_review_queue.parquet"
    review_q.to_parquet(rq_path, index=False)
    print(f"Review queue → {rq_path}  ({len(review_q):,} pairs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=REVIEW_THRESHOLD,
                        help=f"Lower fuzzy-match threshold (default {REVIEW_THRESHOLD})")
    parser.add_argument("--skip-api", action="store_true",
                        help="Run only local fuzzy matching (no API calls)")
    args = parser.parse_args()
    main(args.threshold, args.skip_api)
