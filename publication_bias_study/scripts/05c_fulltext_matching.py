#!/usr/bin/env python3
"""
05c_fulltext_matching.py — Full-text NBER→journal matching

For the 155 currently-unmatched published papers, attempts to find their NBER
working paper counterpart by reading both papers in full.

Pipeline per unmatched published paper:
  1. TF-IDF pre-filter:  top-40 NBER candidates from all 30K abstracts
  2. Keyword expansion:  add any candidates matching fingerprint terms
  3. NBER PDF download:  first 4 pages (intro + data section), cached to disk
  4. LLM comparison:     Claude reads published abstract vs NBER full intro
  5. Author gate:        ≥1 common surname required before accepting match

Handles substantial revisions, new co-authors, and completely changed titles.

Usage:
    python scripts/05c_fulltext_matching.py [--max-cands N] [--skip-pdf] [--dry-run]
    python scripts/05c_fulltext_matching.py --resume   # skip already-processed pairs
"""

import argparse
import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import anthropic
import pandas as pd
import pdfplumber
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
RAW  = DATA / "raw"
FINAL = DATA / "final"
VAL  = ROOT / "validation"
VAL.mkdir(exist_ok=True)

PDF_CACHE_DIR = VAL / "nber_pdf_text"
PDF_CACHE_DIR.mkdir(exist_ok=True)

MATCH_CACHE_PATH = VAL / "fulltext_match_cache.json"
LOG_PATH         = VAL / "fulltext_match_log.csv"

ANALYSIS_PATH = FINAL / "analysis_dataset.parquet"

# ── tuning ──────────────────────────────────────────────────────────────────
MAX_CANDS    = 40      # NBER candidates per published paper (TF-IDF top-N)
MAX_PDF_PAGES = 4      # pages to extract from NBER PDF
DOWNLOAD_DELAY = 0.6   # seconds between NBER PDF requests
HTTP_TIMEOUT   = 30    # seconds

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (research bot; academic use; "
        "contact: emrekuvvet@gmail.com)"
    )
}

# ── logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def normalise(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation."""
    text = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def extract_surnames(author_str: str) -> set[str]:
    """Extract lowercase surnames from 'First Last; First Last' strings."""
    surnames: set[str] = set()
    if not author_str:
        return surnames
    for name in re.split(r"[;,]", author_str):
        parts = name.strip().split()
        if parts:
            surnames.add(parts[-1].lower().rstrip("."))
    return surnames


def nber_pdf_url(wp: str) -> str:
    return f"https://www.nber.org/system/files/working_papers/{wp}/{wp}.pdf"


def download_nber_text(wp: str, skip_pdf: bool = False) -> str:
    """
    Return text from the first MAX_PDF_PAGES of the NBER working paper PDF.
    Results are cached as plain text files under PDF_CACHE_DIR.
    """
    cache_file = PDF_CACHE_DIR / f"{wp}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    if skip_pdf:
        return ""

    url = nber_pdf_url(wp)
    try:
        time.sleep(DOWNLOAD_DELAY)
        r = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            log.debug(f"  PDF {wp}: HTTP {r.status_code}")
            cache_file.write_text("", encoding="utf-8")
            return ""
        pages_text: list[str] = []
        # Suppress pdfplumber MediaBox warnings
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pdfplumber.open(io.BytesIO(r.content)) as pdf:
                for page in pdf.pages[:MAX_PDF_PAGES]:
                    try:
                        t = page.extract_text() or ""
                    except Exception:
                        t = ""
                    pages_text.append(t)
        text = "\n\n".join(pages_text)
        cache_file.write_text(text, encoding="utf-8")
        return text
    except Exception as exc:
        log.debug(f"  PDF {wp}: {exc}")
        cache_file.write_text("", encoding="utf-8")
        return ""


def load_match_cache() -> dict:
    if MATCH_CACHE_PATH.exists():
        return json.loads(MATCH_CACHE_PATH.read_text())
    return {}


def save_match_cache(cache: dict) -> None:
    MATCH_CACHE_PATH.write_text(json.dumps(cache, indent=2))


# ═══════════════════════════════════════════════════════════════════════════
# LLM comparison
# ═══════════════════════════════════════════════════════════════════════════

COMPARE_PROMPT = """\
You are a research assistant helping identify whether a published finance journal \
article is a revised version of an NBER working paper.

Published papers often:
- Have a substantially different title than the working-paper version
- Add or remove co-authors during peer review
- Rewrite the abstract almost completely
- Change methodology, scope, or framing

But they almost always share:
- The same core research question
- The same primary dataset or data source
- The same main identification strategy (instrument, cutoff, event, etc.)
- The same institutional/regulatory focus

---
PUBLISHED PAPER
Title: {pub_title}
Journal: {pub_journal}, Year: {pub_year}
Abstract:
{pub_abstract}

---
NBER WORKING PAPER
Title: {nber_title}
Authors: {nber_authors}
Abstract:
{nber_abstract}

Full text excerpt (first pages):
{nber_fulltext}

---
TASK
Determine whether these two documents are the same research project \
(possibly revised between working-paper and publication stages).

Look for matching signals:
1. Same dataset or primary data source (e.g., CRSP, Compustat, specific regulatory database)
2. Same identification strategy or natural experiment
3. Same institutional/country/firm scope
4. Same dependent variable or main outcome
5. Any cross-references ("previously circulated as", "based on NBER WP …")

Return ONLY valid JSON (no markdown, no prose before or after):
{{"match": true_or_false, "confidence": "high"/"medium"/"low", "reason": "one concise sentence"}}
"""


def llm_compare(
    client: anthropic.Anthropic,
    pub_title: str,
    pub_journal: str,
    pub_year: int,
    pub_abstract: str,
    nber_title: str,
    nber_authors: str,
    nber_abstract: str,
    nber_fulltext: str,
) -> dict:
    """Ask Claude to compare a published paper vs an NBER working paper."""
    # Truncate full text to avoid token overflow (~2000 chars ≈ 400 tokens)
    ft_excerpt = nber_fulltext[:2500].strip()

    prompt = COMPARE_PROMPT.format(
        pub_title=pub_title,
        pub_journal=pub_journal,
        pub_year=pub_year,
        pub_abstract=(pub_abstract or "")[:600],
        nber_title=nber_title,
        nber_authors=nber_authors,
        nber_abstract=(nber_abstract or "")[:400],
        nber_fulltext=ft_excerpt,
    )

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # Extract JSON robustly
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as exc:
        log.debug(f"  LLM compare failed: {exc}")
    return {"match": False, "confidence": "low", "reason": "error"}


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main(max_cands: int, skip_pdf: bool, dry_run: bool, resume: bool, review_only: bool) -> None:
    client = anthropic.Anthropic()

    # ── load data ──────────────────────────────────────────────────────────
    df = pd.read_parquet(ANALYSIS_PATH)
    nber_raw = pd.read_parquet(RAW / "nber_papers.parquet")

    # Load author info from raw journal files (for validation gate)
    journal_dfs = []
    for jname in ("jf", "rfs", "jfe"):
        p = RAW / f"{jname}_papers.parquet"
        if p.exists():
            j = pd.read_parquet(p)[["doi", "authors"]]
            j["doi"] = j["doi"].str.strip().str.lower()
            journal_dfs.append(j)
    journal_authors = (
        pd.concat(journal_dfs).drop_duplicates("doi").set_index("doi")["authors"]
        if journal_dfs else pd.Series(dtype=str)
    )

    # Unmatched published papers
    unmatched = df[(df["published_top3"] == True) & (df["nber_wp_number"].isna())].copy()
    log.info(f"Unmatched published papers: {len(unmatched)}")

    # NBER working papers already matched to a published paper — exclude from candidates
    already_matched_wps = set(df["nber_wp_number"].dropna().astype(str))
    log.info(f"Already-matched NBER WPs (excluded from candidates): {len(already_matched_wps)}")

    # NBER pool (papers with 'w' prefix = standard working papers)
    nber_pool = nber_raw[
        nber_raw["wp_number"].str.startswith("w")
        & ~nber_raw["wp_number"].isin(already_matched_wps)
    ].copy()
    nber_pool["text"] = (
        nber_pool["title"].fillna("") + " " + nber_pool["abstract"].fillna("")
    )
    log.info(f"NBER candidate pool: {len(nber_pool)}")

    # Build TF-IDF index
    log.info("Building TF-IDF index over NBER pool …")
    vec = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words="english",
    )
    nber_matrix = vec.fit_transform(nber_pool["text"].tolist())
    nber_wps    = nber_pool["wp_number"].tolist()
    nber_lookup = nber_pool.set_index("wp_number")

    # ── load caches ────────────────────────────────────────────────────────
    llm_cache = load_match_cache()

    # Results accumulator
    log_rows: list[dict] = []
    new_matches: list[dict] = []

    # ── process each unmatched published paper ─────────────────────────────
    for idx, row in enumerate(unmatched.itertuples(), 1):
        pid   = str(row.paper_id)
        title = str(row.title)
        abstr = str(row.abstract or "")
        journ = str(row.journal)
        year  = int(row.year)

        log.info(f"[{idx}/{len(unmatched)}] {journ} {year} — {title[:70]}")

        query_text = title + " " + abstr

        # ── TF-IDF top candidates ──────────────────────────────────────
        q_vec = vec.transform([query_text])
        sims  = cosine_similarity(q_vec, nber_matrix).flatten()
        top_idx = sims.argsort()[::-1][:max_cands]
        candidates = [(nber_wps[i], float(sims[i])) for i in top_idx if sims[i] > 0.05]

        if not candidates:
            log.info("  No TF-IDF candidates found — skipping")
            continue

        # Published paper author surnames (for validation)
        doi_key = pid.strip().lower()
        pub_authors_str = journal_authors.get(doi_key, "") or ""
        pub_surnames = extract_surnames(pub_authors_str)

        # Year filter: NBER WP should predate or match publication year
        candidates = [
            (wp, sc) for wp, sc in candidates
            if wp in nber_lookup.index
            and int(nber_lookup.at[wp, "year"] or 9999) <= year + 1
        ]

        if dry_run:
            log.info(f"  [DRY-RUN] Top candidate: {candidates[0] if candidates else 'none'}")
            continue

        # ── per-candidate LLM comparison ───────────────────────────────
        matched_wp   = None
        match_reason = ""
        match_conf   = ""

        for wp, tfidf_score in candidates:
            nber_row = nber_lookup.loc[wp]
            nber_title   = str(nber_row["title"] or "")
            nber_abstract = str(nber_row["abstract"] or "")
            nber_authors_str = str(nber_row["authors"] or "")

            # Check LLM cache
            cache_key = f"{pid}||{wp}"
            if resume and cache_key in llm_cache:
                result = llm_cache[cache_key]
            else:
                # Download NBER full text
                nber_fulltext = download_nber_text(wp, skip_pdf=skip_pdf)

                result = llm_compare(
                    client=client,
                    pub_title=title,
                    pub_journal=journ,
                    pub_year=year,
                    pub_abstract=abstr,
                    nber_title=nber_title,
                    nber_authors=nber_authors_str,
                    nber_abstract=nber_abstract,
                    nber_fulltext=nber_fulltext,
                )
                llm_cache[cache_key] = result
                save_match_cache(llm_cache)

            is_match  = bool(result.get("match", False))
            confidence = str(result.get("confidence", "low"))
            reason     = str(result.get("reason", ""))

            log_rows.append({
                "pub_pid":      pid,
                "pub_title":    title[:80],
                "wp_number":    wp,
                "nber_title":   nber_title[:80],
                "tfidf_score":  round(tfidf_score, 4),
                "llm_match":    is_match,
                "confidence":   confidence,
                "reason":       reason,
            })

            if not is_match:
                continue
            if confidence == "low":
                log.debug(f"  {wp}: LLM match=True but low confidence — skip")
                continue

            # Author overlap gate — require ≥50% of the smaller author set
            nber_surnames = extract_surnames(nber_authors_str)
            if pub_surnames and nber_surnames:
                overlap = pub_surnames & nber_surnames
                min_size = min(len(pub_surnames), len(nber_surnames))
                overlap_ratio = len(overlap) / min_size
                if overlap_ratio < 0.5:
                    log.info(
                        f"  {wp}: LLM match={confidence} but insufficient author overlap "
                        f"{len(overlap)}/{min_size}={overlap_ratio:.0%} "
                        f"(pub={pub_surnames}, nber={nber_surnames}) — rejected"
                    )
                    continue
            elif pub_surnames:
                # NBER authors unknown — skip to avoid false positives
                log.debug(f"  {wp}: no NBER author info — rejected")
                continue
            else:
                overlap = nber_surnames

            # ── CONFIRMED MATCH ────────────────────────────────────────
            log.info(
                f"  ✓ MATCH {wp} [{confidence}] — {reason[:80]}"
                f" | overlap={overlap}"
            )
            matched_wp   = wp
            match_reason = reason
            match_conf   = confidence
            break  # take the first (highest TF-IDF) confirmed match

        if matched_wp:
            new_matches.append({
                "pid":         pid,
                "wp_number":   matched_wp,
                "confidence":  match_conf,
                "reason":      match_reason,
                "method":      "s10_fulltext",
            })

    # ── write log ──────────────────────────────────────────────────────────
    if log_rows:
        pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)
        log.info(f"Log written → {LOG_PATH}  ({len(log_rows)} comparisons)")

    # ── write proposed matches to review file ──────────────────────────────
    if not new_matches:
        log.info("No new matches found.")
        return

    review_path = VAL / "fulltext_proposed_matches.csv"
    pd.DataFrame(new_matches).to_csv(review_path, index=False)
    log.info(f"\n{'='*50}")
    log.info(f"New matches found: {len(new_matches)} — written to {review_path}")
    for m in new_matches:
        log.info(f"  {m['pid']}  ←→  {m['wp_number']}  [{m['confidence']}]")
        log.info(f"     {m['reason']}")

    if review_only:
        log.info("--review-only: dataset NOT updated. Inspect the review file first.")
        return

    # Update analysis_dataset
    for m in new_matches:
        pid = m["pid"]
        wp  = m["wp_number"]
        mask = df["paper_id"] == pid
        if not mask.any():
            log.warning(f"  PID {pid} not found in analysis_dataset — skipping update")
            continue

        nber_mask = df["paper_id"] == wp
        if nber_mask.any():
            # Mark the NBER paper as published (if it's in our in-scope sample)
            df.loc[nber_mask, "published_top3"] = True
            df.loc[nber_mask, "nber_wp_number"] = wp

        df.loc[mask, "nber_wp_number"] = wp
        df.loc[mask, "match_quality"]  = "fulltext"
        df.loc[mask, "match_source"]   = m["method"]
        df.loc[mask, "match_score"]    = 1.0

    df.to_parquet(ANALYSIS_PATH, index=False)
    log.info(f"Updated {ANALYSIS_PATH} with {len(new_matches)} new matches.")

    # Summary
    matched_now = df[(df["published_top3"] == True) & (df["nber_wp_number"].notna())]
    total_pub   = df[df["published_top3"] == True]
    log.info(f"\nFinal match rate: {len(matched_now)}/{len(total_pub)} "
             f"= {100*len(matched_now)/len(total_pub):.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-cands", type=int, default=MAX_CANDS,
        help=f"NBER candidates per published paper via TF-IDF (default {MAX_CANDS})"
    )
    parser.add_argument(
        "--skip-pdf", action="store_true",
        help="Skip NBER PDF download; use abstract only for comparison"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show candidate list without calling LLM or updating data"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Use cached LLM results; skip already-compared pairs"
    )
    parser.add_argument(
        "--review-only", action="store_true",
        help="Write proposed matches to review file without updating analysis_dataset"
    )
    args = parser.parse_args()

    main(
        max_cands=args.max_cands,
        skip_pdf=args.skip_pdf,
        dry_run=args.dry_run,
        resume=args.resume,
        review_only=args.review_only,
    )
