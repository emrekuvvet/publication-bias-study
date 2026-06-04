"""
Script 30: Collect top-institution flags via OpenAlex API.

For each paper in the analysis dataset, queries OpenAlex for the first
author's institutional affiliation and checks whether it appears in a list
of top-50 finance/economics departments.

Outputs:
    data/final/analysis_dataset.parquet  (top_institution column updated)
    output/tables/top_institution_coverage.csv
"""

import time, re
import pandas as pd
import numpy as np
import requests
from pathlib import Path

DATA  = Path("data/final/analysis_dataset.parquet")
OUT   = Path("output/tables/top_institution_coverage.csv")
NBER_BASE = "https://api.semanticscholar.org/graph/v1/paper/arXiv:"

# Top-50 finance/economics departments (any name variant that matches)
TOP_INSTITUTIONS = {
    "mit", "massachusetts institute of technology",
    "harvard", "harvard university",
    "chicago", "university of chicago",
    "stanford", "stanford university",
    "columbia", "columbia university",
    "nyu", "new york university",
    "wharton", "university of pennsylvania",
    "yale", "yale university",
    "princeton", "princeton university",
    "kellogg", "northwestern university",
    "berkeley", "uc berkeley", "university of california, berkeley",
    "ucla", "university of california, los angeles",
    "ross", "university of michigan",
    "fuqua", "duke university",
    "darden", "university of virginia",
    "london school of economics", "lse",
    "oxford", "university of oxford",
    "cambridge", "university of cambridge",
    "london business school", "lbs",
    "insead",
    "booth", # Chicago Booth already covered by 'chicago'
    "stern", # NYU Stern already covered
    "sloan", # MIT Sloan already covered
    "cornell", "cornell university",
    "carnegie mellon", "cmu",
    "dartmouth", "tuck",
    "unc", "university of north carolina",
    "emory",
    "washington university", "olin",
    "texas", "university of texas",
    "ohio state", "fisher",
    "bocconi",
    "ecb", "european central bank",
    "federal reserve", "fed ",
    "imf", "international monetary fund",
    "world bank",
    "bank for international settlements", "bis",
}

def is_top_institution(affil_str: str) -> bool:
    if not affil_str:
        return False
    s = affil_str.lower()
    return any(t in s for t in TOP_INSTITUTIONS)

def openalex_lookup(doi: str = None, title: str = None) -> dict:
    """Query OpenAlex for a paper; return first-author affiliation string."""
    headers = {"User-Agent": "mailto:research@example.com"}
    if doi:
        url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    else:
        safe = requests.utils.quote(title[:100]) if title else ""
        url  = f"https://api.openalex.org/works?filter=title.search:{safe}&per-page=1"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return {}
        data = r.json()
        # unwrap search result
        if "results" in data:
            if not data["results"]:
                return {}
            data = data["results"][0]
        return data
    except Exception:
        return {}

def get_first_author_affil(work: dict) -> str:
    authors = work.get("authorships", [])
    if not authors:
        return ""
    first = authors[0]
    instits = first.get("institutions", [])
    if instits:
        return instits[0].get("display_name", "")
    # fallback: raw affil string
    return first.get("raw_affiliation_string", "")

def nber_lookup(wp_number: str) -> str:
    """Query NBER API for a working paper, extract first author affiliation."""
    if not wp_number or pd.isna(wp_number):
        return ""
    num = re.sub(r"[^\d]", "", str(wp_number))
    url = f"https://api.nber.org/papers/{num}.json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return ""
        data = r.json()
        authors = data.get("authors", [])
        if authors:
            return authors[0].get("affil", "")
        return ""
    except Exception:
        return ""

# ── Main ─────────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA)
print(f"Loaded {len(df)} papers. Collecting institution data...")

affils = []
for i, row in df.iterrows():
    pid    = str(row.get("paper_id", ""))
    title  = str(row.get("title", ""))
    source = str(row.get("source", ""))
    wp_num = row.get("nber_wp_number", "")

    affil = ""

    # For published papers: try DOI lookup on OpenAlex
    if source in ("jf", "rfs", "jfe") and pid.startswith("10."):
        work  = openalex_lookup(doi=pid)
        affil = get_first_author_affil(work)
        time.sleep(0.15)

    # For NBER papers: try NBER API first, then OpenAlex title search
    if not affil and source == "nber" and wp_num and not pd.isna(wp_num):
        affil = nber_lookup(wp_num)
        time.sleep(0.15)

    # Fallback: OpenAlex title search
    if not affil and title:
        work  = openalex_lookup(title=title)
        affil = get_first_author_affil(work)
        time.sleep(0.2)

    affils.append(affil)
    if (i + 1) % 30 == 0:
        covered = sum(1 for a in affils if a)
        print(f"  {i+1}/{len(df)} done | affil found: {covered}")

df["affil_raw"]      = affils
df["top_institution"] = [1 if is_top_institution(a) else 0 for a in affils]

# Save
df.to_parquet(DATA, index=False)

# Coverage report
coverage = pd.DataFrame({
    "source": df["source"],
    "affil_found":    df["affil_raw"].apply(lambda x: 1 if x else 0),
    "top_institution": df["top_institution"],
})
cov_summary = coverage.groupby("source").agg(
    n=("affil_found","count"),
    affil_found=("affil_found","sum"),
    top_institution=("top_institution","sum"),
).reset_index()
cov_summary["affil_pct"]  = (100 * cov_summary["affil_found"] / cov_summary["n"]).round(1)
cov_summary["top_inst_pct"] = (100 * cov_summary["top_institution"] / cov_summary["n"]).round(1)
cov_summary.to_csv(OUT, index=False)

print("\n=== Institution Coverage ===")
print(cov_summary.to_string(index=False))
print(f"\ntop_institution rate: {df['top_institution'].mean():.1%}")
print(f"affil found rate:     {(df['affil_raw'] != '').mean():.1%}")
print(f"\nSaved → {DATA} and {OUT}")
