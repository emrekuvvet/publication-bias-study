"""
07b_extract_tstats.py
---------------------
Extract primary t-statistics from published paper abstracts using a
two-pass approach:

Pass 1 — Regex: catch explicit t/z/F-stat patterns in the abstract text.
Pass 2 — Claude Haiku: for remaining papers, ask the model to extract or
          infer the primary test statistic from the abstract.

Output: adds/updates 'primary_tstat' column in
        data/final/analysis_dataset.parquet

Usage:
    python scripts/07b_extract_tstats.py
    python scripts/07b_extract_tstats.py --skip-llm   # regex only
"""

import argparse
import json
import math
import os
import pathlib
import re
import time

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

ROOT    = pathlib.Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")

IN_PATH = ROOT / "data" / "final" / "analysis_dataset.parquet"

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

# ------------------------------------------------------------------ #
# Pass 1 — Regex extraction                                            #
# ------------------------------------------------------------------ #

# Patterns that appear in finance paper abstracts
_PATTERNS = [
    # t = 3.12 / t-stat = 3.12 / t-statistic of 3.12
    r't[\s\-]?(?:stat(?:istic)?)?[\s=of]{1,5}([\d]+\.[\d]+)',
    # z = 2.45 / z-score = 2.45
    r'z[\s\-]?(?:score|stat)?[\s=of]{1,5}([\d]+\.[\d]+)',
    # F = 9.1  → convert to t = sqrt(F)
    r'F[\s\-]?(?:stat(?:istic)?)?[\s=of]{1,5}([\d]+\.[\d]+)',
    # p < 0.01 / p = 0.003 → infer |t| from p-value
    r'p[\s\-]?(?:value)?[\s=<>]{1,4}(0\.0[\d]+)',
    # significant at the 1% / 5% level → lower-bound t
    r'significant at the (1|5|10)[\s%]',
    # χ²(1) = 12.3 → convert chi2(1) to t = sqrt(chi2)
    r'chi[\s\-]?2?\s*\(\s*1\s*\)[\s=of]{1,5}([\d]+\.[\d]+)',
]

_F_PAT  = re.compile(r'F[\s\-]?(?:stat(?:istic)?)?[\s=of]{1,5}([\d]+\.[\d]+)', re.I)
_P_PAT  = re.compile(r'p[\s\-]?(?:value)?[\s=<>]{1,4}(0\.0[\d]+)', re.I)
_SIG_PAT= re.compile(r'significant at the (1|5|10)[\s%]', re.I)
_CHI_PAT= re.compile(r'chi[\s\-]?2?\s*\(\s*1\s*\)[\s=of]{1,5}([\d]+\.[\d]+)', re.I)
_TZ_PAT = re.compile(r'(?<![a-z])(?:t|z)[\s\-]?(?:stat(?:istic)?|score)?[\s=of]{1,5}([\d]+\.[\d]+)', re.I)


def regex_extract(abstract: str) -> float | None:
    if not abstract or not abstract.strip():
        return None
    txt = abstract[:2000]

    # Direct t or z statistic
    m = _TZ_PAT.search(txt)
    if m:
        val = float(m.group(1))
        if 0.5 <= val <= 20:
            return val

    # F-stat(1 dof) → t = sqrt(F)
    m = _F_PAT.search(txt)
    if m:
        f = float(m.group(1))
        if 0 < f <= 400:
            return math.sqrt(f)

    # chi2(1) → t = sqrt(chi2)
    m = _CHI_PAT.search(txt)
    if m:
        c = float(m.group(1))
        if 0 < c <= 400:
            return math.sqrt(c)

    # p-value → infer |t| from normal quantile
    m = _P_PAT.search(txt)
    if m:
        p = float(m.group(1))
        if 0 < p < 0.5:
            from scipy.stats import norm
            return abs(norm.ppf(p / 2))

    # Significance level → conservative lower bound
    m = _SIG_PAT.search(txt)
    if m:
        level = int(m.group(1))
        return {1: 2.58, 5: 1.96, 10: 1.645}.get(level)

    return None


# ------------------------------------------------------------------ #
# Pass 2 — LLM extraction                                             #
# ------------------------------------------------------------------ #

TSTAT_SYSTEM = (
    "You are a research assistant extracting statistical results from "
    "finance paper abstracts. Return valid JSON only — no other text."
)

TSTAT_PROMPT = """\
Extract the PRIMARY test statistic from the abstract below.
The primary statistic is the main empirical result reported (for the \
headline finding about the government intervention studied).

Return the ABSOLUTE value of the t-statistic (or z-statistic).
If the abstract reports an F-statistic with 1 numerator df, return sqrt(F).
If only a p-value is given, return the implied |z| = |Φ⁻¹(p/2)|.
If no test statistic can be inferred, return null.

Abstract:
\"\"\"
{abstract}
\"\"\"

Return JSON with exactly:
{{
  "t_stat": <number or null>,
  "stat_type": "t" | "z" | "F" | "p-implied" | "none",
  "notes": "one short phrase"
}}"""


@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=2, max=20))
def llm_extract(client: Anthropic, abstract: str) -> float | None:
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=150,
        system=TSTAT_SYSTEM,
        messages=[{
            "role": "user",
            "content": TSTAT_PROMPT.format(abstract=abstract[:2500])
        }],
    )
    raw = msg.content[0].text.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(m.group()) if m else {}

    val = result.get("t_stat")
    if val is None:
        return None
    try:
        val = float(val)
        return val if 0.5 <= val <= 20 else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(skip_llm: bool) -> None:
    df = pd.read_parquet(IN_PATH)

    # Only process published papers
    pub = df[df["published_top3"] == True].copy()
    print(f"Published papers to process: {len(pub):,}")

    # Initialise column if missing
    if "primary_tstat" not in df.columns:
        df["primary_tstat"] = float("nan")

    # Pass 1: regex
    regex_hits = 0
    for idx, row in pub.iterrows():
        abstract = str(row.get("abstract", "") or "")
        val = regex_extract(abstract)
        if val is not None:
            df.at[idx, "primary_tstat"] = val
            regex_hits += 1

    print(f"Pass 1 (regex): {regex_hits:,} / {len(pub):,} extracted")

    if not skip_llm:
        # Pass 2: LLM for remaining papers
        still_missing = df[(df["published_top3"] == True) &
                           df["primary_tstat"].isna()]
        print(f"Pass 2 (LLM): processing {len(still_missing):,} remaining papers …")

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        llm_hits = 0

        for idx, row in tqdm(still_missing.iterrows(), total=len(still_missing)):
            abstract = str(row.get("abstract", "") or "")
            if not abstract.strip():
                continue
            try:
                val = llm_extract(client, abstract)
                if val is not None:
                    df.at[idx, "primary_tstat"] = val
                    llm_hits += 1
            except Exception as exc:
                pass
            time.sleep(0.05)

        print(f"Pass 2 (LLM): {llm_hits:,} additional extracted")

    # Summary
    covered = df["primary_tstat"].notna() & (df["published_top3"] == True)
    print(f"\nTotal coverage: {covered.sum():,} / {len(pub):,} published papers "
          f"({covered.sum()/len(pub)*100:.1f}%)")
    print(f"  Mean |t|: {df.loc[covered, 'primary_tstat'].mean():.2f}")
    print(f"  Median |t|: {df.loc[covered, 'primary_tstat'].median():.2f}")

    df.to_parquet(IN_PATH, index=False, compression="snappy")
    print(f"\nUpdated → {IN_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    main(skip_llm=args.skip_llm)
