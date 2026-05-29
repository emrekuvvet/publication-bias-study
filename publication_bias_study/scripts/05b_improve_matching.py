"""
05b_improve_matching.py
-----------------------
Improves NBER → JF/RFS/JFE matching using five additional strategies that
go beyond the fuzzy-title matching in script 05.  Run *after* script 05.

WHY: Title changes between working-paper and publication versions cause fuzzy
title matching to miss many true pairs.  All 296 papers have abstracts and
journal papers have DOIs, enabling more robust strategies.

New strategies (applied to the 157 currently unmatched journal papers):
  S5  CrossRef NBER-DOI  → journal DOI   query 10.3386/wN for 'is-preprint-of'
  S6  CrossRef journal-DOI → NBER        check journal record's 'has-preprint'
  S7  Unpaywall journal-DOI              free OA versions often list NBER URLs
  S8  OpenAlex NBER-DOI  → all versions  OA tracks multi-version works
  S9  Abstract TF-IDF cosine similarity  robust to title changes; author gate

The script is additive: it only promotes papers from 'none' to 'api' quality.
Existing auto/review matches are never overwritten.

Output:
    data/final/analysis_dataset.parquet   (updated match columns + published_top3)
    validation/improved_match_log.csv     (one row per new match found)

Usage:
    python scripts/05b_improve_matching.py
    python scripts/05b_improve_matching.py --skip-api     # S9 TF-IDF only
    python scripts/05b_improve_matching.py --skip-tfidf   # API strategies only
    python scripts/05b_improve_matching.py --min-sim 0.60 # stricter TF-IDF
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
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

ROOT      = pathlib.Path(__file__).parent.parent.resolve()
DS_PATH   = ROOT / "data" / "final" / "analysis_dataset.parquet"
LOG_PATH  = ROOT / "validation" / "improved_match_log.csv"
CACHE_PATH = ROOT / "validation" / "improve_match_cache.json"

UA      = "PublicationBiasResearch/1.0 (mailto:research@pubbiasstudy.org)"
OA_EMAIL = "research@pubbiasstudy.org"

_NBER_URL_RE = re.compile(r"(?:nber\.org/papers?/|10\.3386/)([hw]\d+)", re.IGNORECASE)

STOP_WORDS = {
    "a","an","the","of","in","on","at","to","for","and","or","but",
    "with","by","from","that","this","is","are","was","were","be",
    "been","being","have","has","had","do","does","did","will","would",
    "can","could","should","may","might","shall","some","any","all",
    "its","it","we","our","their","there","when","where","which","who",
}


# ------------------------------------------------------------------ #
# Text helpers (same as script 05)                                    #
# ------------------------------------------------------------------ #

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


def extract_wp(text: str) -> str | None:
    if not text:
        return None
    m = _NBER_URL_RE.search(str(text))
    return m.group(1).lower() if m else None


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


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
# S5 — CrossRef: NBER DOI → published journal DOI                     #
# ------------------------------------------------------------------ #

def s5_crossref_nber_doi(
    wp_number: str, journal_doi_set: set, session: requests.Session, cache: dict
) -> str | None:
    """
    Query CrossRef for the NBER working paper DOI (10.3386/wN) and
    return the published journal DOI if CrossRef knows the 'is-preprint-of' relation.
    """
    key = f"s5_{wp_number}"
    if key in cache:
        return cache[key] or None

    doi = f"10.3386/{wp_number}"
    result = None
    try:
        resp = session.get(
            f"https://api.crossref.org/works/{doi}",
            timeout=15,
        )
        if resp.status_code == 200:
            msg = resp.json().get("message", {})
            relation = msg.get("relation", {})
            for rel_type in ("is-preprint-of", "has-expression-in", "is-version-of"):
                for item in relation.get(rel_type, []):
                    if item.get("id-type") == "doi":
                        candidate = item["id"].lower().strip()
                        if candidate in journal_doi_set:
                            result = candidate
                            break
                if result:
                    break
    except Exception:
        pass

    cache[key] = result or ""
    return result


# ------------------------------------------------------------------ #
# S6 — CrossRef: journal DOI → NBER preprint relation / references    #
# ------------------------------------------------------------------ #

def s6_crossref_journal_doi(
    journal_doi: str, session: requests.Session, cache: dict
) -> str | None:
    """
    Query CrossRef for a journal paper's DOI and check the 'has-preprint' relation
    or in-text DOI references for an NBER working paper link (10.3386/wN).
    """
    key = f"s6_{journal_doi}"
    if key in cache:
        return cache[key] or None

    result = None
    try:
        resp = session.get(
            f"https://api.crossref.org/works/{journal_doi}",
            timeout=15,
        )
        if resp.status_code == 200:
            msg = resp.json().get("message", {})
            # Check relation fields
            relation = msg.get("relation", {})
            for rel_type in ("has-preprint", "is-expression-of", "is-based-on"):
                for item in relation.get(rel_type, []):
                    if item.get("id-type") == "doi":
                        wp = extract_wp(item["id"])
                        if wp:
                            result = wp
                            break
                if result:
                    break
            # Check first 30 references for NBER DOIs
            if not result:
                for ref in (msg.get("reference") or [])[:30]:
                    doi_val = ref.get("DOI", "")
                    if doi_val and "10.3386" in doi_val:
                        wp = extract_wp(doi_val)
                        if wp:
                            result = wp
                            break
    except Exception:
        pass

    cache[key] = result or ""
    return result


# ------------------------------------------------------------------ #
# S7 — Unpaywall: journal DOI → NBER / institutional URL              #
# ------------------------------------------------------------------ #

def s7_unpaywall(
    journal_doi: str, session: requests.Session, cache: dict
) -> str | None:
    """
    Query Unpaywall for a journal paper's DOI.  Unpaywall lists all free
    versions including NBER pages, SSRN, and university repositories.
    Returns an NBER wp_number if found in any OA location URL.
    """
    key = f"s7_{journal_doi}"
    if key in cache:
        return cache[key] or None

    result = None
    try:
        resp = session.get(
            f"https://api.unpaywall.org/v2/{journal_doi}",
            params={"email": OA_EMAIL},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            for loc in data.get("oa_locations", []):
                for field in ("url", "url_for_pdf", "url_for_landing_page"):
                    url_val = loc.get(field, "")
                    wp = extract_wp(url_val)
                    if wp:
                        result = wp
                        break
                if result:
                    break
            # Also check best_oa_location
            if not result:
                boa = data.get("best_oa_location") or {}
                for field in ("url", "url_for_pdf", "url_for_landing_page"):
                    wp = extract_wp(boa.get(field, ""))
                    if wp:
                        result = wp
                        break
    except Exception:
        pass

    cache[key] = result or ""
    return result


# ------------------------------------------------------------------ #
# S8 — OpenAlex: NBER DOI → all versions / locations                  #
# ------------------------------------------------------------------ #

def s8_openalex_nber_doi(
    wp_number: str, journal_doi_set: set, session: requests.Session, cache: dict
) -> str | None:
    """
    Query OpenAlex for an NBER working paper's DOI.  OpenAlex tracks multi-version
    works and may list the published journal version in its locations array.
    """
    key = f"s8_{wp_number}"
    if key in cache:
        return cache[key] or None

    result = None
    try:
        resp = session.get(
            f"https://api.openalex.org/works/https://doi.org/10.3386/{wp_number}",
            params={"mailto": OA_EMAIL},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Check all locations for a non-NBER DOI in our journal set
            for loc in data.get("locations", []):
                for field in ("doi",):
                    doi_val = (loc.get(field) or "").replace("https://doi.org/", "").lower().strip()
                    if doi_val and "nber.org" not in doi_val and doi_val in journal_doi_set:
                        result = doi_val
                        break
                if not result:
                    # Try extracting DOI from URL
                    for field in ("url", "landing_page_url"):
                        url_val = loc.get(field, "") or ""
                        doi_match = re.search(r"10\.\d{4,}/[^\s\"'<>]+", url_val)
                        if doi_match:
                            candidate = doi_match.group().rstrip(".,)")
                            if candidate in journal_doi_set:
                                result = candidate
                                break
                if result:
                    break
            # Also check top-level DOI
            if not result:
                top_doi = (data.get("doi") or "").replace("https://doi.org/", "").lower().strip()
                if top_doi and "nber.org" not in top_doi and top_doi in journal_doi_set:
                    result = top_doi
    except Exception:
        pass

    cache[key] = result or ""
    return result


# ------------------------------------------------------------------ #
# S9 — Abstract TF-IDF cosine similarity                              #
# ------------------------------------------------------------------ #

def s9_abstract_tfidf(
    unmatched_journal: pd.DataFrame,
    nber_raw: pd.DataFrame,
    min_sim: float = 0.55,
    min_title_sim: int = 50,
) -> dict[str, tuple[str, float, int]]:  # paper_id → (wp_number, cosine_sim, title_sim)
    """
    TF-IDF cosine similarity between unmatched journal paper abstracts and all
    30k NBER paper abstracts.  Guards: ≥1 common author surname AND title
    similarity ≥ min_title_sim (allows for major title changes between WP/pub).
    """
    j_valid = unmatched_journal.dropna(subset=["abstract"]).copy()
    n_valid = nber_raw[nber_raw["abstract"].notna() & (nber_raw["abstract"].str.len() > 100)].copy()

    if j_valid.empty or n_valid.empty:
        return {}

    print(f"  TF-IDF: {len(j_valid)} journal × {len(n_valid):,} NBER abstracts …")

    all_texts = list(j_valid["abstract"]) + list(n_valid["abstract"])
    vec = TfidfVectorizer(
        max_features=25000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    tfidf_matrix = vec.fit_transform(all_texts)
    j_tfidf = tfidf_matrix[: len(j_valid)]
    n_tfidf = tfidf_matrix[len(j_valid) :]

    j_ids    = list(j_valid["paper_id"])
    n_wps    = list(n_valid["wp_number"])
    j_auth   = j_valid.set_index("paper_id")["authors"].to_dict()
    n_auth   = n_valid.set_index("wp_number")["authors"].to_dict()
    j_titles = j_valid.set_index("paper_id")["title"].to_dict()
    n_titles = n_valid.set_index("wp_number")["title"].to_dict()

    matches: dict[str, tuple[str, float, int]] = {}
    BATCH = 30
    for i in range(0, len(j_ids), BATCH):
        batch_vecs = j_tfidf[i : i + BATCH]
        sims = cosine_similarity(batch_vecs, n_tfidf)
        for bi, j_pid in enumerate(j_ids[i : i + BATCH]):
            row_sims = sims[bi]
            top_k = np.argsort(row_sims)[::-1][:50]
            j_surn = extract_surnames(str(j_auth.get(j_pid, "") or ""))

            for idx in top_k:
                if row_sims[idx] < min_sim:
                    break
                n_wp = n_wps[idx]
                n_surn = extract_surnames(str(n_auth.get(n_wp, "") or ""))

                # Mandatory author overlap gate
                if j_surn and n_surn and not (j_surn & n_surn):
                    continue

                # Title similarity guard (relaxed: 50+ allows substantial title change)
                t_score = fuzz.token_sort_ratio(
                    normalise_title(j_titles.get(j_pid, "")),
                    normalise_title(n_titles.get(n_wp, "")),
                )
                if t_score < min_title_sim:
                    continue

                matches[j_pid] = (n_wp, float(row_sims[idx]), t_score)
                break

    return matches


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(skip_api: bool, skip_tfidf: bool, min_sim: float) -> None:
    if not DS_PATH.exists():
        raise FileNotFoundError(f"Missing {DS_PATH}. Run script 05 first.")

    df = pd.read_parquet(DS_PATH)
    print(f"Loaded analysis_dataset: {len(df)} rows")

    # --- Identify targets --------------------------------------------
    journal_mask = df["source"].isin(["jf", "rfs", "jfe"])
    unmatched_mask = df["match_quality"] == "none"
    unmatched_journal = df[journal_mask & unmatched_mask].copy()
    nber_inscope = df[df["source"] == "nber"].copy()
    print(f"Unmatched journal papers: {len(unmatched_journal)}")
    print(f"In-scope NBER papers:     {len(nber_inscope)}")

    # Build raw data lookups
    nber_raw = pd.read_parquet(
        ROOT / "data" / "raw" / "nber_papers.parquet",
        columns=["wp_number", "title", "authors", "abstract", "year"],
    )
    # Build journal DOI set (normalized)
    journal_raw = []
    for src in ["jf", "rfs", "jfe"]:
        p = ROOT / "data" / "raw" / f"{src}_papers.parquet"
        if p.exists():
            j = pd.read_parquet(p, columns=["doi"])
            journal_raw.append(j)
    journal_doi_set = set()
    if journal_raw:
        all_j = pd.concat(journal_raw, ignore_index=True)
        journal_doi_set = set(all_j["doi"].dropna().str.lower().str.strip())
    print(f"Journal DOI set size: {len(journal_doi_set)}")

    # Build wp → paper_id lookup for quick annotation
    nber_wp_set = set(nber_inscope["paper_id"].str.lower())

    # Author lookup maps (for validation of API matches)
    nber_author_map = nber_raw.set_index("wp_number")["authors"].to_dict()
    journal_author_raw: dict[str, str] = {}
    for src in ["jf", "rfs", "jfe"]:
        p = ROOT / "data" / "raw" / f"{src}_papers.parquet"
        if p.exists():
            jm = pd.read_parquet(p, columns=["doi", "authors"])
            for _, r in jm.iterrows():
                journal_author_raw[r["doi"]] = str(r["authors"] or "")

    def api_match_valid(journal_pid: str, wp: str) -> bool:
        """Return True if at least 1 author surname overlaps between journal paper and NBER WP."""
        j_auth = journal_author_raw.get(journal_pid, "") or ""
        n_auth = nber_author_map.get(wp, "") or ""
        j_surn = extract_surnames(j_auth)
        n_surn = extract_surnames(n_auth)
        if not j_surn or not n_surn:
            return True  # cannot validate — assume OK
        return bool(j_surn & n_surn)

    # new_matches: paper_id (journal) → {wp, strategy, confidence}
    new_matches: dict[str, dict] = {}
    cache = load_cache()

    session = make_session()

    if not skip_api:
        # === S5: CrossRef NBER DOI → journal DOI (NBER direction) ===
        print(f"\n[S5] CrossRef NBER-DOI → journal DOI …")
        s5_hits = 0
        for _, nrow in tqdm(nber_inscope.iterrows(), total=len(nber_inscope)):
            wp = nrow["paper_id"]
            journal_doi = s5_crossref_nber_doi(wp, journal_doi_set, session, cache)
            time.sleep(0.12)
            if journal_doi:
                # Find which journal paper this DOI corresponds to
                j_match = unmatched_journal[
                    unmatched_journal["paper_id"].str.lower() == journal_doi
                ]
                if not j_match.empty:
                    pid = j_match.iloc[0]["paper_id"]
                    if pid not in new_matches and api_match_valid(pid, wp):
                        new_matches[pid] = {"wp": wp, "strategy": "s5_crossref_nber",
                                            "confidence": "high"}
                        s5_hits += 1
        save_cache(cache)
        print(f"  S5 new matches: {s5_hits}")

        # === S6: CrossRef journal DOI → NBER preprint ===
        print(f"\n[S6] CrossRef journal-DOI → NBER preprint …")
        s6_hits = 0
        remaining = [
            row for _, row in unmatched_journal.iterrows()
            if row["paper_id"] not in new_matches
        ]
        for row in tqdm(remaining):
            j_doi = row["paper_id"]
            wp = s6_crossref_journal_doi(j_doi, session, cache)
            time.sleep(0.12)
            if wp and wp in nber_wp_set and api_match_valid(j_doi, wp):
                new_matches[j_doi] = {"wp": wp, "strategy": "s6_crossref_journal",
                                      "confidence": "high"}
                s6_hits += 1
        save_cache(cache)
        print(f"  S6 new matches: {s6_hits}")

        # === S7: Unpaywall journal DOI → NBER/institutional URL ===
        print(f"\n[S7] Unpaywall journal-DOI → NBER URL …")
        s7_hits = 0
        remaining = [
            row for _, row in unmatched_journal.iterrows()
            if row["paper_id"] not in new_matches
        ]
        for row in tqdm(remaining):
            j_doi = row["paper_id"]
            wp = s7_unpaywall(j_doi, session, cache)
            time.sleep(0.12)
            if wp and wp in nber_wp_set and api_match_valid(j_doi, wp):
                new_matches[j_doi] = {"wp": wp, "strategy": "s7_unpaywall",
                                      "confidence": "high"}
                s7_hits += 1
        save_cache(cache)
        print(f"  S7 new matches: {s7_hits}")

        # === S8: OpenAlex NBER-DOI → all locations ===
        print(f"\n[S8] OpenAlex NBER-DOI → published location …")
        s8_hits = 0
        for _, nrow in tqdm(nber_inscope.iterrows(), total=len(nber_inscope)):
            wp = nrow["paper_id"]
            journal_doi = s8_openalex_nber_doi(wp, journal_doi_set, session, cache)
            time.sleep(0.12)
            if journal_doi:
                j_match = unmatched_journal[
                    unmatched_journal["paper_id"].str.lower() == journal_doi
                ]
                if not j_match.empty:
                    pid = j_match.iloc[0]["paper_id"]
                    if pid not in new_matches and api_match_valid(pid, wp):
                        new_matches[pid] = {"wp": wp, "strategy": "s8_openalex_nber",
                                            "confidence": "high"}
                        s8_hits += 1
        save_cache(cache)
        print(f"  S8 new matches: {s8_hits}")

    # === S9: Abstract TF-IDF cosine similarity ===
    if not skip_tfidf:
        print(f"\n[S9] Abstract TF-IDF cosine similarity (min_sim={min_sim}) …")
        still_unmatched = unmatched_journal[
            ~unmatched_journal["paper_id"].isin(new_matches)
        ]

        # Attach authors from raw journal data for unmatched papers
        journal_meta = []
        for src in ["jf", "rfs", "jfe"]:
            p = ROOT / "data" / "raw" / f"{src}_papers.parquet"
            if p.exists():
                jm = pd.read_parquet(p, columns=["doi", "authors"]).rename(
                    columns={"doi": "paper_id"}
                )
                journal_meta.append(jm)
        if journal_meta:
            jmeta = pd.concat(journal_meta, ignore_index=True).drop_duplicates("paper_id")
            still_unmatched = still_unmatched.merge(
                jmeta, on="paper_id", how="left", suffixes=("", "_raw")
            )
            if "authors_raw" in still_unmatched.columns:
                still_unmatched["authors"] = still_unmatched["authors"].fillna(
                    still_unmatched["authors_raw"]
                )
                still_unmatched = still_unmatched.drop(columns=["authors_raw"])

        tfidf_matches = s9_abstract_tfidf(still_unmatched, nber_raw, min_sim=min_sim)

        s9_hits = 0
        for j_pid, (n_wp, cos_sim, t_sim) in tfidf_matches.items():
            # Only accept if NBER paper is in our in-scope set
            if n_wp in nber_wp_set and j_pid not in new_matches:
                new_matches[j_pid] = {
                    "wp": n_wp, "strategy": "s9_tfidf",
                    "confidence": "medium",
                    "cosine_sim": round(cos_sim, 4),
                    "title_sim": t_sim,
                }
                s9_hits += 1
        print(f"  S9 new matches (in-scope NBER): {s9_hits}")

        # Also report TF-IDF matches to non-inscope NBER papers (informational)
        s9_ext = sum(
            1 for j_pid, (n_wp, _, _) in tfidf_matches.items()
            if n_wp not in nber_wp_set and j_pid not in new_matches
        )
        print(f"  S9 matches to non-inscope NBER (informational only): {s9_ext}")

    # --- Summary of new matches --------------------------------------
    print(f"\n{'='*55}")
    print(f"New matches found: {len(new_matches)}")
    by_strategy: dict[str, int] = {}
    for v in new_matches.values():
        s = v["strategy"]
        by_strategy[s] = by_strategy.get(s, 0) + 1
    for strat, cnt in sorted(by_strategy.items()):
        print(f"  {strat}: {cnt}")
    print(f"{'='*55}")

    if not new_matches:
        print("No new matches — analysis_dataset unchanged.")
        return

    # --- Update analysis_dataset ------------------------------------
    df = df.copy()

    # Index for fast lookup
    df_idx = df.set_index("paper_id")
    updated_rows = []

    for j_pid, info in new_matches.items():
        wp = info["wp"]
        strategy = info["strategy"]

        # Update journal paper row
        if j_pid in df_idx.index:
            df.loc[df["paper_id"] == j_pid, "nber_wp_number"] = wp
            df.loc[df["paper_id"] == j_pid, "match_quality"]  = "api"
            df.loc[df["paper_id"] == j_pid, "match_source"]   = strategy
            df.loc[df["paper_id"] == j_pid, "match_score"]    = info.get("cosine_sim", 0.0)

        # If the matched NBER paper is in our in-scope set, mark it as published
        if wp in nber_wp_set:
            df.loc[df["paper_id"] == wp, "published_top3"] = True

        # Record for log
        j_row = df_idx.loc[j_pid] if j_pid in df_idx.index else {}
        updated_rows.append({
            "journal_doi":      j_pid,
            "journal_title":    j_row.get("title", ""),
            "journal_source":   j_row.get("source", ""),
            "nber_wp_number":   wp,
            "strategy":         strategy,
            "confidence":       info.get("confidence", ""),
            "cosine_sim":       info.get("cosine_sim", ""),
            "title_sim":        info.get("title_sim", ""),
        })

    # Enrich log with NBER titles
    nber_title_map = nber_raw.set_index("wp_number")["title"].to_dict()
    for row in updated_rows:
        row["nber_title"] = nber_title_map.get(row["nber_wp_number"], "")

    # Save
    df.to_parquet(DS_PATH, index=False, compression="snappy")
    print(f"\nSaved updated analysis_dataset → {DS_PATH}")

    log_df = pd.DataFrame(updated_rows)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(LOG_PATH, index=False)
    print(f"Match log saved → {LOG_PATH}")

    # Print final stats
    df_updated = pd.read_parquet(DS_PATH)
    journal_updated = df_updated[df_updated["source"].isin(["jf","rfs","jfe"])]
    nber_updated = df_updated[df_updated["source"] == "nber"]
    print(f"\nUpdated match summary ({len(journal_updated)} journal papers):")
    mq = journal_updated["match_quality"].value_counts()
    print(f"  auto    : {mq.get('auto',0)}")
    print(f"  review  : {mq.get('review',0)}")
    print(f"  api     : {mq.get('api',0)}")
    print(f"  none    : {mq.get('none',0)}")
    total_matched = mq.get("auto",0) + mq.get("review",0) + mq.get("api",0)
    print(f"  TOTAL matched: {total_matched} / {len(journal_updated)} "
          f"({total_matched/len(journal_updated)*100:.1f}%)")
    print(f"\nNBER papers now marked published: "
          f"{nber_updated['published_top3'].sum()} / {len(nber_updated)}")
    print(f"NBER unpublished control:         "
          f"{(~nber_updated['published_top3']).sum()} / {len(nber_updated)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Improve NBER→journal matching with abstract similarity and DOI APIs"
    )
    parser.add_argument(
        "--skip-api", action="store_true",
        help="Skip CrossRef/Unpaywall/OpenAlex (run TF-IDF only)"
    )
    parser.add_argument(
        "--skip-tfidf", action="store_true",
        help="Skip abstract TF-IDF (run API strategies only)"
    )
    parser.add_argument(
        "--min-sim", type=float, default=0.55,
        help="Minimum TF-IDF cosine similarity to accept (default 0.55)"
    )
    args = parser.parse_args()
    main(args.skip_api, args.skip_tfidf, args.min_sim)
