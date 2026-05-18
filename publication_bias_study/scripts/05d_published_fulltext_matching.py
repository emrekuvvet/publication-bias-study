#!/usr/bin/env python3
"""
05d_published_fulltext_matching.py — Match unmatched published papers to NBER WPs
using full text extracted from user-supplied published PDFs.

Unlike 05c (which only had published abstracts), here we have the full published
paper text. This gives the LLM much richer signal to identify NBER counterparts
even when title and abstract were completely rewritten during peer review.

Pipeline per unmatched published paper:
  1. PDF mapping:    match each PDF in missing_paper/ to a published paper by title
  2. PDF extraction: extract first MAX_PUB_PAGES from the published PDF
  3. TF-IDF filter:  top MAX_CANDS NBER candidates from all abstracts
  4. NBER PDF:       download NBER full text (first 4 pages), cached to disk
  5. LLM comparison: Claude reads both full texts to identify same research project
  6. Author gate:    ≥50% of smaller author set must overlap

Usage:
    python scripts/05d_published_fulltext_matching.py
    python scripts/05d_published_fulltext_matching.py --resume       # skip cached pairs
    python scripts/05d_published_fulltext_matching.py --review-only  # don't update dataset
    python scripts/05d_published_fulltext_matching.py --dry-run      # TF-IDF only, no LLM
    python scripts/05d_published_fulltext_matching.py --max-cands 60
"""

import argparse
import io
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path

import anthropic
import pandas as pd
import pdfplumber
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
DATA  = ROOT / "data"
RAW   = DATA / "raw"
FINAL = DATA / "final"
VAL   = ROOT / "validation"
VAL.mkdir(exist_ok=True)

PUB_PDF_DIR   = ROOT / "missing_paper"          # user-supplied published PDFs
NBER_CACHE    = VAL  / "nber_pdf_text"          # shared with 05c
NBER_CACHE.mkdir(exist_ok=True)

MATCH_CACHE_PATH = VAL / "fulltext_05d_cache.json"
LOG_PATH         = VAL / "fulltext_05d_log.csv"
REVIEW_PATH      = VAL / "fulltext_05d_proposed.csv"

ANALYSIS_PATH = FINAL / "analysis_dataset.parquet"

# ── tuning ───────────────────────────────────────────────────────────────────
MAX_CANDS     = 50     # NBER candidates per paper (slightly more than 05c)
MAX_PUB_PAGES = 4      # pages to extract from published PDF
MAX_NBER_PAGES = 4     # pages to extract from NBER PDF
DOWNLOAD_DELAY = 0.6
HTTP_TIMEOUT   = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (research bot; academic use; "
        "contact: emrekuvvet@gmail.com)"
    )
}

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
    text = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def extract_surnames(author_str: str) -> set[str]:
    surnames: set[str] = set()
    if not author_str:
        return surnames
    for name in re.split(r"[;,]", author_str):
        parts = name.strip().split()
        if parts:
            surnames.add(parts[-1].lower().rstrip("."))
    return surnames


def pdf_text(path: Path, max_pages: int) -> str:
    """Extract text from first max_pages of a local PDF."""
    try:
        pages: list[str] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages[:max_pages]:
                    try:
                        t = page.extract_text() or ""
                    except Exception:
                        t = ""
                    pages.append(t)
        return "\n\n".join(pages)
    except Exception as exc:
        log.debug(f"  PDF extract failed {path.name}: {exc}")
        return ""


def nber_pdf_url(wp: str) -> str:
    return f"https://www.nber.org/system/files/working_papers/{wp}/{wp}.pdf"


def download_nber_text(wp: str) -> str:
    """Download and cache first MAX_NBER_PAGES of an NBER working paper PDF."""
    cache_file = NBER_CACHE / f"{wp}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    url = nber_pdf_url(wp)
    try:
        time.sleep(DOWNLOAD_DELAY)
        r = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            cache_file.write_text("", encoding="utf-8")
            return ""
        pages: list[str] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pdfplumber.open(io.BytesIO(r.content)) as pdf:
                for page in pdf.pages[:MAX_NBER_PAGES]:
                    try:
                        t = page.extract_text() or ""
                    except Exception:
                        t = ""
                    pages.append(t)
        text = "\n\n".join(pages)
        cache_file.write_text(text, encoding="utf-8")
        return text
    except Exception as exc:
        log.debug(f"  NBER PDF {wp}: {exc}")
        cache_file.write_text("", encoding="utf-8")
        return ""


def load_cache() -> dict:
    if MATCH_CACHE_PATH.exists():
        return json.loads(MATCH_CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict) -> None:
    MATCH_CACHE_PATH.write_text(json.dumps(cache, indent=2))


# ═══════════════════════════════════════════════════════════════════════════
# PDF ↔ paper mapping
# ═══════════════════════════════════════════════════════════════════════════

def map_pdfs_to_papers(
    pdf_dir: Path,
    papers: pd.DataFrame,
) -> dict[str, Path]:
    """
    Return {paper_id: pdf_path} by matching PDF filenames / extracted titles
    to rows in `papers` (which has paper_id, title, year, journal).

    Strategy (in priority order):
      A. Oxford RFS suffix: 'hh*.pdf' → match against RFS DOI suffix
      B. Elsevier PII:      '1-s2.0-S0304405X...' → match against JFE DOI
      C. JF verbatim name:  'The Journal of Finance - YEAR - AUTHOR - Title.pdf'
      D. Named PDF:         'Surname-TitleWords-Year.pdf' → fuzzy title match
      E. Fallback:          extract first-page text, match title by overlap
    """
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    log.info(f"PDFs in {pdf_dir.name}: {len(pdfs)}")

    # Build lookup tables
    # Oxford RFS DOIs end in e.g. /hhy052 → last path component
    rfs = papers[papers["journal"] == "RFS"].copy()
    rfs["doi_suffix"] = rfs["paper_id"].str.split("/").str[-1].str.lower()
    rfs_by_suffix = rfs.set_index("doi_suffix")["paper_id"].to_dict()

    # JFE DOIs contain the Elsevier PII like S0304405X99000616
    jfe = papers[papers["journal"] == "JFE"].copy()

    # Title → paper_id for fuzzy fallback
    papers_by_title = {normalise(r.title): r.paper_id for r in papers.itertuples()}

    mapping: dict[str, Path] = {}
    unmapped_pdfs: list[Path] = []

    for pdf in pdfs:
        name = pdf.stem.lower()
        matched_pid = None

        # ── Strategy A: Oxford RFS suffix ──────────────────────────────
        # filenames like hhaa007.pdf, hhy052.pdf, hhad086.pdf
        if re.match(r"^hh[a-z0-9]+$", name):
            pid = rfs_by_suffix.get(name)
            if pid:
                matched_pid = pid

        # ── Strategy B: Elsevier PII ────────────────────────────────────
        # filenames like 1-s2.0-S0304405X01000836-main
        if matched_pid is None:
            m = re.search(r"s0304405x([0-9a-z]+)", name)
            if m:
                pii_frag = m.group(1)
                # DOIs look like 10.1016/s0304-405x(01)00083-6
                # PII fragment is the numeric part; do substring match
                for _, row in jfe.iterrows():
                    doi_norm = row["paper_id"].replace("-", "").replace("(", "").replace(")", "").lower()
                    if pii_frag in doi_norm:
                        matched_pid = row["paper_id"]
                        break

        # ── Strategy C: JF verbose filename ────────────────────────────
        # "The Journal of Finance - 2008 - JOHN - Corporate Governance..."
        if matched_pid is None and "journal of finance" in name:
            # Extract year and first keyword from filename
            year_m = re.search(r"(\d{4})", name)
            yr = int(year_m.group(1)) if year_m else None
            # Remove prefix up to year+author, get title words
            title_part = re.sub(r".*?(\d{4})[^-]*-\s*", "", pdf.stem, count=1).lower()
            title_norm = normalise(title_part)
            # Match against JF papers by year + title overlap
            jf_papers = papers[(papers["journal"] == "JF") & (papers["year"] == yr)] if yr else papers[papers["journal"] == "JF"]
            best_score = 0.0
            best_pid   = None
            for _, row in jf_papers.iterrows():
                row_norm = normalise(row["title"])
                row_words = set(row_norm.split())
                query_words = set(title_norm.split())
                if not query_words:
                    continue
                overlap = len(row_words & query_words) / len(query_words)
                if overlap > best_score:
                    best_score = overlap
                    best_pid   = row["paper_id"]
            if best_pid and best_score >= 0.4:
                matched_pid = best_pid

        # ── Strategy D: named PDF (Surname-TitleWords-Year) ─────────────
        if matched_pid is None:
            # e.g. Choi-ConvertibleBondArbitrageurs-2010
            year_m = re.search(r"(\d{4})", name)
            yr = int(year_m.group(1)) if year_m else None
            name_words = set(re.sub(r"\d{4}", "", name).replace("-", " ").split())
            subset = papers[papers["year"] == yr] if yr else papers
            best_score = 0.0
            best_pid   = None
            for _, row in subset.iterrows():
                row_words = set(normalise(row["title"]).split())
                if not name_words:
                    continue
                overlap = len(row_words & name_words) / len(name_words)
                if overlap > best_score:
                    best_score = overlap
                    best_pid   = row["paper_id"]
            if best_pid and best_score >= 0.35:
                matched_pid = best_pid

        if matched_pid:
            if matched_pid in mapping:
                log.warning(
                    f"  Duplicate PDF mapping: {matched_pid} already mapped to "
                    f"{mapping[matched_pid].name}, now also {pdf.name} — keeping first"
                )
            else:
                mapping[matched_pid] = pdf
        else:
            unmapped_pdfs.append(pdf)

    # ── Strategy E: full-text title extraction fallback ──────────────────
    if unmapped_pdfs:
        log.info(f"Strategy E fallback for {len(unmapped_pdfs)} unmapped PDFs …")
        for pdf in unmapped_pdfs:
            text = pdf_text(pdf, max_pages=1)
            if not text:
                log.warning(f"  Could not extract text from {pdf.name}")
                continue
            # First non-empty line is usually the title
            first_lines = [ln.strip() for ln in text.split("\n") if ln.strip()][:8]
            best_score = 0.0
            best_pid   = None
            for line in first_lines:
                line_norm = normalise(line)
                line_words = set(line_norm.split())
                for t_norm, pid in papers_by_title.items():
                    if pid in mapping:
                        continue
                    t_words = set(t_norm.split())
                    if not t_words:
                        continue
                    overlap = len(line_words & t_words) / len(t_words)
                    if overlap > best_score:
                        best_score = overlap
                        best_pid   = pid
            if best_pid and best_score >= 0.5:
                mapping[best_pid] = pdf
                log.info(f"  E: {pdf.name} → {best_pid} (score={best_score:.2f})")
            else:
                log.warning(f"  E: failed to map {pdf.name} (best={best_score:.2f})")

    already_mapped = set(mapping.keys())
    still_missing  = [r.paper_id for r in papers.itertuples() if r.paper_id not in already_mapped]
    log.info(
        f"PDF mapping: {len(mapping)}/{len(papers)} papers mapped"
        + (f" | {len(still_missing)} unmapped" if still_missing else "")
    )
    if still_missing:
        log.info(f"  Unmapped paper_ids: {still_missing[:10]}")
    return mapping


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
Abstract: {pub_abstract}

Full text excerpt (first pages of published paper):
{pub_fulltext}

---
NBER WORKING PAPER
Title: {nber_title}
Authors: {nber_authors}
Abstract: {nber_abstract}

Full text excerpt (first pages of NBER working paper):
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
    pub_fulltext: str,
    nber_title: str,
    nber_authors: str,
    nber_abstract: str,
    nber_fulltext: str,
) -> dict:
    prompt = COMPARE_PROMPT.format(
        pub_title=pub_title,
        pub_journal=pub_journal,
        pub_year=pub_year,
        pub_abstract=(pub_abstract or "")[:500],
        pub_fulltext=pub_fulltext[:2000].strip(),
        nber_title=nber_title,
        nber_authors=nber_authors,
        nber_abstract=(nber_abstract or "")[:400],
        nber_fulltext=nber_fulltext[:2000].strip(),
    )
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as exc:
        log.debug(f"  LLM compare failed: {exc}")
    return {"match": False, "confidence": "low", "reason": "error"}


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main(
    max_cands: int,
    dry_run: bool,
    resume: bool,
    review_only: bool,
    skip_pdf: bool,
) -> None:
    client = anthropic.Anthropic()

    # ── load data ──────────────────────────────────────────────────────────
    df       = pd.read_parquet(ANALYSIS_PATH)
    nber_raw = pd.read_parquet(RAW / "nber_papers.parquet")

    # Published author info (from raw journal parquets)
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
    unmatched = df[
        df["source"].isin(["jf", "rfs", "jfe"]) & df["nber_wp_number"].isna()
    ].copy()
    unmatched["journal"] = unmatched["source"].str.upper()
    log.info(f"Unmatched published papers: {len(unmatched)}")

    # ── map PDFs to papers ─────────────────────────────────────────────────
    pdf_map = map_pdfs_to_papers(PUB_PDF_DIR, unmatched)

    # ── NBER candidate pool ────────────────────────────────────────────────
    already_matched_wps = set(df["nber_wp_number"].dropna().astype(str))
    nber_pool = nber_raw[
        nber_raw["wp_number"].str.startswith("w")
        & ~nber_raw["wp_number"].isin(already_matched_wps)
    ].copy()
    nber_pool["text"] = (
        nber_pool["title"].fillna("") + " " + nber_pool["abstract"].fillna("")
    )
    log.info(f"NBER candidate pool: {len(nber_pool)} | already matched excluded: {len(already_matched_wps)}")

    # Build TF-IDF index
    log.info("Building TF-IDF index …")
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

    # ── caches ─────────────────────────────────────────────────────────────
    llm_cache = load_cache()

    log_rows:    list[dict] = []
    new_matches: list[dict] = []

    # ── process each unmatched paper ───────────────────────────────────────
    for idx, row in enumerate(unmatched.itertuples(), 1):
        pid   = str(row.paper_id)
        title = str(row.title)
        abstr = str(row.abstract or "")
        journ = str(row.journal)
        year  = int(row.year)

        log.info(f"[{idx}/{len(unmatched)}] {journ} {year} — {title[:70]}")

        # Extract published PDF text
        pdf_path = pdf_map.get(pid)
        if pdf_path is None:
            log.warning(f"  No PDF found for {pid} — using abstract only")
            pub_fulltext = ""
        else:
            pub_fulltext = pdf_text(pdf_path, max_pages=MAX_PUB_PAGES)
            log.info(f"  PDF: {pdf_path.name} ({len(pub_fulltext)} chars)")

        # TF-IDF query: title + abstract + published full text (extra signal)
        query_text = title + " " + abstr + " " + pub_fulltext[:3000]
        q_vec = vec.transform([query_text])
        sims  = cosine_similarity(q_vec, nber_matrix).flatten()
        top_idx = sims.argsort()[::-1][:max_cands]
        candidates = [(nber_wps[i], float(sims[i])) for i in top_idx if sims[i] > 0.05]

        # Year filter: NBER WP should predate or match publication year
        candidates = [
            (wp, sc) for wp, sc in candidates
            if wp in nber_lookup.index
            and int(nber_lookup.at[wp, "year"] or 9999) <= year + 1
        ]

        log.info(f"  TF-IDF candidates: {len(candidates)}")
        if not candidates:
            log.info("  No candidates — skipping")
            continue

        if dry_run:
            top5 = candidates[:5]
            for wp, sc in top5:
                log.info(f"  [{sc:.3f}] {wp}: {nber_lookup.at[wp, 'title'][:60]}")
            continue

        # Published author surnames
        doi_key = pid.strip().lower()
        pub_authors_str = journal_authors.get(doi_key, "") or ""
        pub_surnames = extract_surnames(pub_authors_str)

        # ── per-candidate LLM comparison ───────────────────────────────
        matched_wp   = None
        match_reason = ""
        match_conf   = ""

        for wp, tfidf_score in candidates:
            nber_row         = nber_lookup.loc[wp]
            nber_title       = str(nber_row["title"] or "")
            nber_abstract    = str(nber_row["abstract"] or "")
            nber_authors_str = str(nber_row["authors"] or "")

            cache_key = f"{pid}||{wp}"
            if resume and cache_key in llm_cache:
                result = llm_cache[cache_key]
            else:
                nber_fulltext = "" if skip_pdf else download_nber_text(wp)
                result = llm_compare(
                    client=client,
                    pub_title=title,
                    pub_journal=journ,
                    pub_year=year,
                    pub_abstract=abstr,
                    pub_fulltext=pub_fulltext,
                    nber_title=nber_title,
                    nber_authors=nber_authors_str,
                    nber_abstract=nber_abstract,
                    nber_fulltext=nber_fulltext,
                )
                llm_cache[cache_key] = result
                save_cache(llm_cache)

            is_match   = bool(result.get("match", False))
            confidence = str(result.get("confidence", "low"))
            reason     = str(result.get("reason", ""))

            log_rows.append({
                "pub_pid":     pid,
                "pub_title":   title[:80],
                "wp_number":   wp,
                "nber_title":  nber_title[:80],
                "tfidf_score": round(tfidf_score, 4),
                "llm_match":   is_match,
                "confidence":  confidence,
                "reason":      reason,
            })

            if not is_match:
                continue
            if confidence == "low":
                log.debug(f"  {wp}: match=True but low confidence — skip")
                continue

            # Author overlap gate — ≥50% of smaller set
            nber_surnames = extract_surnames(nber_authors_str)
            if pub_surnames and nber_surnames:
                overlap    = pub_surnames & nber_surnames
                min_size   = min(len(pub_surnames), len(nber_surnames))
                overlap_ratio = len(overlap) / min_size
                if overlap_ratio < 0.5:
                    log.info(
                        f"  {wp}: match={confidence} but author overlap "
                        f"{len(overlap)}/{min_size}={overlap_ratio:.0%} "
                        f"(pub={pub_surnames}, nber={nber_surnames}) — rejected"
                    )
                    continue
            elif pub_surnames:
                log.debug(f"  {wp}: no NBER author info — rejected")
                continue
            else:
                overlap = nber_surnames

            log.info(f"  ✓ MATCH {wp} [{confidence}] — {reason[:80]} | overlap={overlap}")
            matched_wp   = wp
            match_reason = reason
            match_conf   = confidence
            break

        if matched_wp:
            new_matches.append({
                "pid":        pid,
                "wp_number":  matched_wp,
                "confidence": match_conf,
                "reason":     match_reason,
                "method":     "s11_pub_fulltext",
            })

    # ── write log ──────────────────────────────────────────────────────────
    if log_rows:
        pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)
        log.info(f"Log → {LOG_PATH} ({len(log_rows)} comparisons)")

    if not new_matches:
        log.info("No new matches found.")
        return

    # ── write review file ──────────────────────────────────────────────────
    pd.DataFrame(new_matches).to_csv(REVIEW_PATH, index=False)
    log.info(f"\n{'='*50}")
    log.info(f"Proposed matches: {len(new_matches)} — see {REVIEW_PATH}")
    for m in new_matches:
        log.info(f"  {m['pid']}  ←→  {m['wp_number']}  [{m['confidence']}]")
        log.info(f"     {m['reason']}")

    if review_only:
        log.info("--review-only: dataset NOT updated.")
        return

    # ── update analysis_dataset ────────────────────────────────────────────
    for m in new_matches:
        pid = m["pid"]
        wp  = m["wp_number"]
        mask = df["paper_id"] == pid
        if not mask.any():
            log.warning(f"  {pid} not found in dataset — skipping")
            continue
        nber_mask = df["paper_id"] == wp
        if nber_mask.any():
            df.loc[nber_mask, "published_top3"] = True
            df.loc[nber_mask, "nber_wp_number"] = wp
        df.loc[mask, "nber_wp_number"] = wp
        df.loc[mask, "match_quality"]  = "fulltext"
        df.loc[mask, "match_source"]   = m["method"]
        df.loc[mask, "match_score"]    = 1.0

    df.to_parquet(ANALYSIS_PATH, index=False)
    log.info(f"Updated {ANALYSIS_PATH} with {len(new_matches)} new matches.")

    matched_now = df[df["source"].isin(["jf","rfs","jfe"]) & df["nber_wp_number"].notna()]
    total_pub   = df[df["source"].isin(["jf","rfs","jfe"])]
    log.info(
        f"Final match rate: {len(matched_now)}/{len(total_pub)} "
        f"= {100*len(matched_now)/len(total_pub):.1f}%"
    )


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-cands", type=int, default=MAX_CANDS,
        help=f"NBER candidates per paper (default {MAX_CANDS})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run TF-IDF only; print top candidates without calling LLM"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Use cached LLM results; skip already-compared pairs"
    )
    parser.add_argument(
        "--review-only", action="store_true",
        help="Write proposed matches to review file without updating dataset"
    )
    parser.add_argument(
        "--skip-pdf", action="store_true",
        help="Skip NBER PDF downloads; use published full text + NBER abstract only"
    )
    args = parser.parse_args()
    main(
        max_cands=args.max_cands,
        dry_run=args.dry_run,
        resume=args.resume,
        review_only=args.review_only,
        skip_pdf=args.skip_pdf,
    )
