"""
Script 30: Collect top-institution flags via OpenAlex API.

For each paper in the analysis dataset, queries OpenAlex for the first
author's institutional affiliation and checks whether it appears in the
UTD Top 100 Business School Research Rankings (2021-2025 period).

Source: UTD Top 100 Business School Research Rankings™, Naveen Jindal
School of Management, University of Texas at Dallas.
https://jsom.utdallas.edu/the-utd-top-100-business-school-research-rankings/
Accessed: June 2026.

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

# UTD Top 100 Business School Research Rankings (2021-2025 worldwide ranking).
# Each tuple is (rank, canonical_name, match_keywords).
# Matching is substring-based on the lowercased OpenAlex affiliation string.
UTD_TOP100 = [
    (1,   "University of Pennsylvania",              ["university of pennsylvania", "wharton"]),
    (2,   "University of Texas at Dallas",           ["university of texas at dallas", "utdallas"]),
    (3,   "Columbia University",                     ["columbia university"]),
    (4,   "Harvard University",                      ["harvard university"]),
    (5,   "University of Chicago",                   ["university of chicago", "booth school"]),
    (6,   "New York University",                     ["new york university", "nyu stern"]),
    (7,   "University of Southern California",       ["university of southern california", "marshall school"]),
    (8,   "Massachusetts Institute of Technology",   ["massachusetts institute of technology", "mit sloan"]),
    (9,   "INSEAD",                                  ["insead"]),
    (10,  "Indiana University",                      ["indiana university", "kelley school"]),
    (11,  "Stanford University",                     ["stanford university"]),
    (12,  "University of Texas at Austin",           ["university of texas at austin", "mccombs school"]),
    (13,  "Cornell University",                      ["cornell university"]),
    (14,  "University of Toronto",                   ["university of toronto", "rotman school"]),
    (15,  "University of Hong Kong",                 ["university of hong kong"]),
    (16,  "Washington University in St. Louis",      ["washington university in st. louis", "olin school of business"]),
    (17,  "University of Washington",                ["university of washington", "foster school"]),
    (18,  "London Business School",                  ["london business school"]),
    (19,  "University of Michigan",                  ["university of michigan", "ross school"]),
    (20,  "Arizona State University",                ["arizona state university", "w.p. carey"]),
    (21,  "Duke University",                         ["duke university", "fuqua school"]),
    (22,  "University of North Carolina",            ["university of north carolina", "kenan-flagler"]),
    (23,  "University of Minnesota",                 ["university of minnesota", "carlson school"]),
    (24,  "Ohio State University",                   ["ohio state university", "fisher college of business"]),
    (25,  "Pennsylvania State University",           ["pennsylvania state university", "penn state", "smeal college"]),
    (26,  "Northwestern University",                 ["northwestern university", "kellogg school"]),
    (27,  "Hong Kong University of Science and Technology", ["hong kong university of science and technology", "hkust"]),
    (28,  "National University of Singapore",        ["national university of singapore", "nus business"]),
    (29,  "University of Maryland",                  ["university of maryland", "smith school of business"]),
    (30,  "University of Illinois at Urbana-Champaign", ["university of illinois at urbana-champaign", "gies college"]),
    (31,  "Chinese University of Hong Kong",         ["chinese university of hong kong"]),
    (32,  "Purdue University",                       ["purdue university", "daniels school", "krannert"]),
    (33,  "Texas A&M University",                    ["texas a&m university", "mays business"]),
    (34,  "Yale University",                         ["yale university"]),
    (35,  "University of California, Los Angeles",   ["university of california, los angeles", "ucla", "anderson school"]),
    (36,  "City University of Hong Kong",            ["city university of hong kong"]),
    (37,  "Erasmus University Rotterdam",            ["erasmus university", "rotterdam school of management"]),
    (38,  "Hong Kong Polytechnic University",        ["hong kong polytechnic university"]),
    (39,  "Boston University",                       ["boston university", "questrom school"]),
    (40,  "University of Notre Dame",                ["university of notre dame", "mendoza college"]),
    (41,  "University of Colorado",                  ["university of colorado", "leeds school of business"]),
    (42,  "Tilburg University",                      ["tilburg university"]),
    (43,  "University of Florida",                   ["university of florida", "warrington college"]),
    (44,  "University of California, Berkeley",      ["university of california, berkeley", "uc berkeley", "haas school"]),
    (45,  "Temple University",                       ["temple university", "fox school of business"]),
    (46,  "Imperial College London",                 ["imperial college london"]),
    (47,  "Emory University",                        ["emory university", "goizueta"]),
    (48,  "McGill University",                       ["mcgill university", "desautels"]),
    (49,  "Boston College",                          ["boston college", "carroll school of management"]),
    (50,  "University of Wisconsin",                 ["university of wisconsin", "wisconsin school of business"]),
    (51,  "University of British Columbia",          ["university of british columbia", "sauder school"]),
    (52,  "Carnegie Mellon University",              ["carnegie mellon university", "tepper school"]),
    (53,  "University of Georgia",                   ["university of georgia", "terry college"]),
    (54,  "University of Utah",                      ["university of utah", "eccles school"]),
    (55,  "HEC Paris",                               ["hec paris"]),
    (56,  "Georgia Institute of Technology",         ["georgia institute of technology", "georgia tech", "scheller college"]),
    (57,  "Johns Hopkins University",                ["johns hopkins university", "carey business"]),
    (58,  "Tsinghua University",                     ["tsinghua university"]),
    (59,  "University of Connecticut",               ["university of connecticut"]),
    (60,  "University of Miami",                     ["university of miami", "herbert business"]),
    (61,  "University of Houston",                   ["university of houston", "bauer college"]),
    (62,  "Singapore Management University",         ["singapore management university"]),
    (63,  "University of South Carolina",            ["university of south carolina", "moore school"]),
    (64,  "Copenhagen Business School",              ["copenhagen business school"]),
    (65,  "Fudan University",                        ["fudan university"]),
    (66,  "Georgia State University",                ["georgia state university", "robinson college"]),
    (67,  "Georgetown University",                   ["georgetown university", "mcdonough school"]),
    (68,  "Northeastern University",                 ["northeastern university"]),
    (69,  "University of Science and Technology of China", ["university of science and technology of china", "ustc"]),
    (70,  "Rice University",                         ["rice university", "jones graduate school"]),
    (71,  "City University of New York, Baruch College", ["baruch college", "zicklin school"]),
    (72,  "City University of London",               ["bayes business school", "city, university of london"]),
    (73,  "Chinese University of Hong Kong, Shenzhen", ["cuhk shenzhen", "chinese university of hong kong, shenzhen"]),
    (74,  "Michigan State University",               ["michigan state university", "eli broad college"]),
    (75,  "Dartmouth College",                       ["dartmouth college", "tuck school"]),
    (76,  "Nanyang Technological University",        ["nanyang technological university", "nanyang business school"]),
    (77,  "University of Arizona",                   ["university of arizona", "eller college"]),
    (78,  "University of Warwick",                   ["university of warwick", "warwick business school"]),
    (79,  "University of Tennessee",                 ["university of tennessee", "haslam college"]),
    (80,  "University of Melbourne",                 ["university of melbourne", "melbourne business school"]),
    (81,  "Shanghai Jiao Tong University",           ["shanghai jiao tong university"]),
    (82,  "University of California, Irvine",        ["university of california, irvine", "merage school"]),
    (83,  "University College London",               ["university college london", "ucl school of management"]),
    (84,  "University of Navarra",                   ["university of navarra", "iese business school"]),
    (85,  "University of Oxford",                    ["university of oxford", "said business school"]),
    (86,  "London School of Economics",              ["london school of economics", "lse"]),
    (87,  "Bocconi University",                      ["bocconi university", "universita bocconi"]),
    (88,  "Frankfurt School of Finance and Management", ["frankfurt school of finance"]),
    (89,  "Virginia Tech",                           ["virginia tech", "virginia polytechnic", "pamplin college"]),
    (90,  "University of Illinois at Chicago",       ["university of illinois at chicago"]),
    (91,  "Peking University",                       ["peking university", "guanghua school"]),
    (92,  "University of Cambridge",                 ["university of cambridge", "judge business school"]),
    (93,  "University of California, San Diego",     ["university of california, san diego", "rady school"]),
    (94,  "Western University Canada",               ["western university", "ivey business school"]),
    (95,  "University of Rochester",                 ["university of rochester", "simon business school"]),
    (96,  "Indian School of Business",               ["indian school of business"]),
    (97,  "Technical University of Munich",          ["technical university of munich", "tum school of management"]),
    (98,  "University of Amsterdam",                 ["university of amsterdam", "amsterdam business school"]),
    (99,  "University of Pittsburgh",                ["university of pittsburgh", "katz graduate school"]),
    (100, "Rutgers University",                      ["rutgers university"]),
]

# Flat set of all match keywords (lowercase) for fast lookup
_TOP100_KEYWORDS = {kw for _, _, kws in UTD_TOP100 for kw in kws}

def is_top_institution(affil_str: str) -> bool:
    if not affil_str:
        return False
    s = affil_str.lower()
    return any(kw in s for kw in _TOP100_KEYWORDS)

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
