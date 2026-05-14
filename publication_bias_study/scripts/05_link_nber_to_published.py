"""
05_link_nber_to_published.py
----------------------------
Fuzzy-match NBER working papers to published journal papers using
title similarity (RapidFuzz) + author overlap cross-check.

Thresholds (from implementation plan §6c):
  ≥ 90  → automatic match
  75–89 → flagged for human review
  < 75  → no match

Output: data/final/analysis_dataset.parquet

Matching logic:
1. Candidate retrieval: for each journal paper, find NBER candidates
   where at least one token in the title overlaps (fast pre-filter).
2. Scoring: RapidFuzz token_sort_ratio on normalised titles.
3. Author cross-check: at least one surname in common.
4. Disambiguation: if multiple NBER candidates score ≥ 75, keep the
   highest-scoring one.

Usage:
    python scripts/05_link_nber_to_published.py
    python scripts/05_link_nber_to_published.py --threshold 80
"""

import argparse
import pathlib
import re
import unicodedata

import duckdb
import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm

ROOT     = pathlib.Path(__file__).parent.parent.resolve()
DB_PATH  = ROOT / "publication_bias.duckdb"
OUT_PATH = ROOT / "data" / "final" / "analysis_dataset.parquet"

AUTO_MATCH_THRESHOLD   = 90.0
REVIEW_THRESHOLD       = 75.0

# ------------------------------------------------------------------ #
# Text normalisation                                                   #
# ------------------------------------------------------------------ #

STOP_WORDS = {
    "a","an","the","of","in","on","at","to","for","and","or","but",
    "with","by","from","that","this","is","are","was","were","be",
    "been","being","have","has","had","do","does","did","will","would",
    "can","could","should","may","might","shall","some","any","all",
    "its","it","we","our","their","there","when","where","which","who",
}


def normalise_title(title: str) -> str:
    """Lower-case, strip accents, remove punctuation, drop stop-words."""
    if not title:
        return ""
    # NFD → ASCII approximation
    nfd = unicodedata.normalize("NFD", title)
    ascii_str = nfd.encode("ascii", "ignore").decode("ascii")
    # lower, remove non-alphanumeric
    clean = re.sub(r"[^a-z0-9\s]", " ", ascii_str.lower())
    tokens = [t for t in clean.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def extract_surnames(authors_str: str) -> set[str]:
    """Return a set of lower-case last names from a semicolon-delimited string."""
    surnames = set()
    for name in authors_str.split(";"):
        name = name.strip()
        if not name:
            continue
        # Handle "Last, First" and "First Last" formats
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
# Matching                                                             #
# ------------------------------------------------------------------ #

def build_nber_index(nber: pd.DataFrame) -> dict[str, str]:
    """Return {paper_id: normalised_title} for fast candidate retrieval."""
    return {
        row["paper_id"]: normalise_title(row["title"])
        for _, row in nber.iterrows()
        if row.get("title", "")
    }


def find_best_match(journal_title_norm: str,
                    journal_authors: str,
                    nber_index: dict[str, str],
                    nber_authors_map: dict[str, str],
                    choices_limit: int = 10) -> tuple[str | None, float, str]:
    """
    Returns (nber_paper_id, score, match_quality) where match_quality is
    'auto', 'review', or 'none'.
    """
    if not journal_title_norm or not nber_index:
        return None, 0.0, "none"

    candidates = process.extract(
        journal_title_norm,
        nber_index,
        scorer=fuzz.token_sort_ratio,
        limit=choices_limit,
    )
    # candidates: [(key, score, _), ...]

    best_id, best_score = None, 0.0
    j_surnames = extract_surnames(journal_authors)

    for nber_id, score, _ in candidates:
        if score < REVIEW_THRESHOLD:
            break
        # Author cross-check
        n_surnames = extract_surnames(nber_authors_map.get(nber_id, ""))
        overlap = j_surnames & n_surnames
        if not overlap and j_surnames and n_surnames:
            # No author overlap at all → reject this candidate
            continue
        if score > best_score:
            best_score = score
            best_id = nber_id

    if best_id is None:
        return None, 0.0, "none"
    if best_score >= AUTO_MATCH_THRESHOLD:
        return best_id, best_score, "auto"
    return best_id, best_score, "review"


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(threshold: float) -> None:
    global REVIEW_THRESHOLD
    REVIEW_THRESHOLD = threshold

    con = duckdb.connect(str(DB_PATH), read_only=True)

    # Load inscope papers by source (sample covers 2000–2024)
    inscope = pd.read_parquet(ROOT / "data" / "classified" / "inscope_papers.parquet")
    inscope = inscope[inscope["in_scope"] == True]

    direction_path = ROOT / "data" / "classified" / "coded_direction.parquet"
    if direction_path.exists():
        direction = pd.read_parquet(direction_path)
        merged = inscope.merge(direction[["paper_id","direction","confidence","rationale"]],
                               on="paper_id", how="left")
    else:
        print("  Note: coded_direction.parquet not yet available — direction columns will be NULL")
        print("  Run 04_code_direction.py after adding ANTHROPIC_API_KEY to .env")
        merged = inscope.copy()
        merged["direction"]  = pd.NA
        merged["confidence"] = pd.NA
        merged["rationale"]  = pd.NA

    # Split into NBER and journal subsets
    nber_df = merged[merged["source"] == "nber"].copy()
    journal_df = merged[merged["source"].isin(["jf","rfs","jfe"])].copy()

    print(f"NBER in-scope papers:    {len(nber_df):,}")
    print(f"Journal in-scope papers: {len(journal_df):,}")

    # Enrich NBER subset with authors from the raw table (inscope lacks authors)
    raw_nber = pd.read_parquet(ROOT / "data" / "raw" / "nber_papers.parquet",
                               columns=["wp_number", "authors", "title"])
    raw_nber = raw_nber.rename(columns={"wp_number": "paper_id"})
    nber_df  = nber_df.merge(raw_nber[["paper_id","authors"]],
                             on="paper_id", how="left")

    # Enrich journal subsets with authors from raw journal tables
    journal_authors = []
    for src, col_id in [("jf","doi"),("rfs","doi"),("jfe","doi")]:
        rpath = ROOT / "data" / "raw" / f"{src}_papers.parquet"
        if rpath.exists():
            rj = pd.read_parquet(rpath, columns=[col_id, "authors", "title"])
            rj = rj.rename(columns={col_id: "paper_id"})
            journal_authors.append(rj)
    if journal_authors:
        ja_df = pd.concat(journal_authors, ignore_index=True)
        journal_df = journal_df.merge(ja_df[["paper_id","authors"]],
                                      on="paper_id", how="left")

    # Build matching index
    nber_index       = build_nber_index(nber_df)
    nber_authors_map = dict(zip(nber_df["paper_id"],
                                nber_df["authors"].fillna("")))

    # Match journal papers → NBER
    match_records = []
    print("Matching journal papers to NBER ...")
    for _, jrow in tqdm(journal_df.iterrows(), total=len(journal_df)):
        j_title_norm = normalise_title(jrow.get("title", ""))
        j_authors    = jrow.get("authors", "")
        nber_id, score, quality = find_best_match(
            j_title_norm, j_authors, nber_index, nber_authors_map
        )
        match_records.append({
            "journal_paper_id": jrow["paper_id"],
            "nber_wp_number":   nber_id,
            "match_score":      score,
            "match_quality":    quality,
        })

    matches = pd.DataFrame(match_records)
    auto_n   = (matches["match_quality"] == "auto").sum()
    review_n = (matches["match_quality"] == "review").sum()
    none_n   = (matches["match_quality"] == "none").sum()
    print(f"\nMatch summary:")
    print(f"  Auto  (≥{AUTO_MATCH_THRESHOLD}):   {auto_n:,}  ({auto_n/len(matches)*100:.1f}%)")
    print(f"  Review ({REVIEW_THRESHOLD}–{AUTO_MATCH_THRESHOLD-1}): {review_n:,}  ({review_n/len(matches)*100:.1f}%)")
    print(f"  None  (<{REVIEW_THRESHOLD}):    {none_n:,}  ({none_n/len(matches)*100:.1f}%)")

    # ---------------------------------------------------------------- #
    # Build final analysis dataset                                      #
    # ---------------------------------------------------------------- #
    # Row = one unique paper (NBER or journal-only)
    # Probit outcome: published_top3 = True for journal papers

    # Journal-paper rows
    j = journal_df.copy()
    j["published_top3"] = True
    j["journal"] = j["source"].str.upper()
    j = j.merge(matches.rename(columns={"journal_paper_id":"paper_id"}),
                on="paper_id", how="left")

    # NBER-paper rows (published_top3 depends on whether matched)
    n = nber_df.copy()
    n["published_top3"] = False
    n["journal"] = pd.NA
    n["nber_wp_number"] = n["paper_id"]
    n["match_score"] = pd.NA
    n["match_quality"] = "n/a"

    # Mark NBER papers that have a matched journal counterpart as published
    matched_nber_ids = set(matches.loc[
        matches["match_quality"].isin(["auto","review"]),
        "nber_wp_number"
    ].dropna())
    n.loc[n["paper_id"].isin(matched_nber_ids), "published_top3"] = True

    # Union
    all_df = pd.concat([j, n], ignore_index=True)
    all_df["positive"] = all_df["direction"] == 1
    all_df["negative"] = all_df["direction"] == -1

    # Placeholder columns to be filled later (editor data, id_strategy etc.)
    for col in ["id_strategy","top_institution","n_authors",
                "nber_citations","jel_codes","editor"]:
        if col not in all_df.columns:
            all_df[col] = pd.NA

    out_cols = [
        "paper_id","source","published_top3","journal","year",
        "title","abstract",
        "direction","positive","negative",
        "nber_wp_number","match_score","match_quality",
        "id_strategy","top_institution","n_authors",
        "nber_citations","jel_codes","editor",
    ]
    out_df = all_df[[c for c in out_cols if c in all_df.columns]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"\nSaved → {OUT_PATH}  ({len(out_df):,} rows)")

    # Save review queue for human inspection
    review_queue = matches[matches["match_quality"] == "review"]
    review_path = ROOT / "validation" / "match_review_queue.parquet"
    review_queue.to_parquet(review_path, index=False)
    print(f"Review queue → {review_path}  ({len(review_queue):,} pairs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=REVIEW_THRESHOLD,
                        help="Lower bound for 'review' match quality (default 75)")
    args = parser.parse_args()
    main(args.threshold)
