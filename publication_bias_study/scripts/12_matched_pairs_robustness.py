"""
12_matched_pairs_robustness.py
-------------------------------
Matched-pairs robustness check (Referee Point 1c).

For each of the 25 published papers that have a confirmed NBER WP match,
retrieve the NBER draft abstract from nber_raw, LLM-code its direction,
then compare draft vs. published direction codes.

This tests whether direction codes are stable across the draft→publication
transition for the same paper, and whether the main probit result holds
within the matched-pairs subsample (N=50: 25 NBER drafts + 25 published
versions).

Output
------
- data/classified/matched_pairs_direction.parquet  — per-pair comparison
- output/tables/robustness_matched_pairs.csv       — probit results

Usage
------
    python scripts/12_matched_pairs_robustness.py           # full API run
    python scripts/12_matched_pairs_robustness.py --no-api  # dry run (direction=0)
"""

import argparse
import json
import os
import pathlib
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from anthropic import Anthropic
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT       = pathlib.Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")

ANA_PATH   = ROOT / "data" / "final"   / "analysis_dataset.parquet"
NBER_PATH  = ROOT / "data" / "raw"     / "nber_papers.parquet"
OUT_PATH   = ROOT / "data" / "classified" / "matched_pairs_direction.parquet"
TABLE_PATH = ROOT / "output" / "tables"   / "robustness_matched_pairs.csv"

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
WORKERS         = 4

DIRECTION_SYSTEM = (
    "You are a research assistant classifying finance paper findings. "
    "Return valid JSON only — no other text."
)

DIRECTION_PROMPT = """\
The following is the abstract of a finance paper that studies the effect of a \
government law, regulation, or public policy.

Classify the MAIN empirical finding:
  +1  The government intervention produced a POSITIVE / BENEFICIAL outcome \
(e.g. improved market quality, reduced systemic risk, better investor protection, \
increased firm value, lower cost of capital)
  -1  The government intervention produced a NEGATIVE / HARMFUL outcome \
(e.g. reduced liquidity, higher costs, unintended consequences, welfare losses)
   0  NULL, MIXED, or INCONCLUSIVE finding (no statistically significant effect, \
evidence on both sides, or the paper explicitly refuses to take a directional stance)

Coding rules:
- Code the PRIMARY intervention and PRIMARY outcome (named in title or first sentence)
- If paper is purely theoretical with no empirical test → set direction=0, confidence=low
- Do NOT infer direction from normative framing; code the sign of the estimated coefficient

Abstract:
\"\"\"
{abstract}
\"\"\"

Return JSON with exactly:
{{
  "direction": 1 | -1 | 0,
  "confidence": "high" | "medium" | "low",
  "rationale": "one sentence"
}}"""


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
def code_one(client: Anthropic, abstract: str) -> dict:
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=200,
        temperature=0,
        system=DIRECTION_SYSTEM,
        messages=[{"role": "user", "content": DIRECTION_PROMPT.format(abstract=abstract[:3000])}],
    )
    raw = msg.content[0].text.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group()) if match else {
            "direction": 0, "confidence": "low", "rationale": "parse error"
        }
    direction = result.get("direction", 0)
    if isinstance(direction, str):
        direction = int(direction)
    result["direction"] = int(direction)
    return result


def run_probit(formula: str, df: pd.DataFrame, label: str) -> list[dict]:
    try:
        model = smf.probit(formula, data=df).fit(disp=False, method="bfgs", maxiter=500)
    except Exception as exc:
        print(f"  Probit failed [{label}]: {exc}")
        return []
    rows = []
    for name in model.params.index:
        coef = model.params[name]
        se   = model.bse[name]
        z    = model.tvalues[name]
        p    = model.pvalues[name]
        ci_lo, ci_hi = model.conf_int().loc[name]
        stars = ("***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "")
        rows.append({
            "Specification": label,
            "Variable":      name,
            "Coefficient":   round(coef, 4),
            "Std Error":     round(se,   4),
            "z-stat":        round(z,    3),
            "p-value":       round(p,    4),
            "CI 2.5%":       round(ci_lo, 4),
            "CI 97.5%":      round(ci_hi, 4),
            "Significance":  stars,
            "N":             int(model.nobs),
            "Pseudo R2":     round(model.prsquared, 4),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-api", action="store_true")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────
    df       = pd.read_parquet(ANA_PATH)
    nber_raw = pd.read_parquet(NBER_PATH)

    # Normalise WP numbers in nber_raw
    nber_raw["wp_clean"] = nber_raw["wp_number"].str.extract(r"(w\d+)", expand=False)

    # Matched published papers
    matched_pub = df[
        (df["published_top3"] == True) &
        df["match_quality"].isin(["auto", "fulltext", "manual", "api"])
    ].copy()
    print(f"Matched published papers: {len(matched_pub)}")

    # Find their NBER drafts
    drafts = nber_raw[nber_raw["wp_clean"].isin(matched_pub["nber_wp_number"].tolist())].copy()
    drafts = drafts.rename(columns={"wp_clean": "nber_wp_number"})
    print(f"NBER drafts found in nber_raw: {len(drafts)}")

    # ── LLM-code NBER draft abstracts ─────────────────────────────
    if args.no_api:
        drafts["direction_draft"] = 0
        drafts["confidence_draft"] = "low"
    else:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        client = Anthropic(api_key=api_key)

        results = {}
        futures = {}
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            for _, row in drafts.iterrows():
                abstract = str(row.get("abstract", "") or "")
                if len(abstract.split()) < 10:
                    results[row["nber_wp_number"]] = {"direction": 0, "confidence": "low"}
                    continue
                fut = executor.submit(code_one, client, abstract)
                futures[fut] = row["nber_wp_number"]

            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="LLM coding NBER drafts"):
                wp = futures[fut]
                try:
                    results[wp] = fut.result()
                except Exception as exc:
                    print(f"  Error {wp}: {exc}")
                    results[wp] = {"direction": 0, "confidence": "low"}

        drafts["direction_draft"]  = drafts["nber_wp_number"].map(
            lambda x: results.get(x, {}).get("direction", 0))
        drafts["confidence_draft"] = drafts["nber_wp_number"].map(
            lambda x: results.get(x, {}).get("confidence", "low"))

    # ── Build per-pair comparison table ───────────────────────────
    pairs = matched_pub[["paper_id", "title", "nber_wp_number", "direction", "year"]].merge(
        drafts[["nber_wp_number", "direction_draft", "confidence_draft"]],
        on="nber_wp_number", how="inner"
    ).rename(columns={"direction": "direction_pub"})

    pairs["stable"]  = (pairs["direction_pub"] == pairs["direction_draft"]).astype(int)
    pairs["shifted"] = (pairs["direction_pub"] != pairs["direction_draft"]).astype(int)
    pairs["draft_directional"] = (pairs["direction_draft"].abs() == 1).astype(int)
    pairs["pub_directional"]   = (pairs["direction_pub"].abs() == 1).astype(int)

    print(f"\nMatched pairs: {len(pairs)}")
    print(f"  Draft directional rate:     {pairs['draft_directional'].mean()*100:.1f}%")
    print(f"  Published directional rate: {pairs['pub_directional'].mean()*100:.1f}%")
    print(f"  Direction-stable pairs:     {pairs['stable'].sum()} / {len(pairs)} "
          f"({pairs['stable'].mean()*100:.1f}%)")
    print("\nDirection transition matrix (rows=draft, cols=published):")
    print(pd.crosstab(pairs["direction_draft"], pairs["direction_pub"],
                      rownames=["Draft"], colnames=["Published"]))

    pairs.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")

    # ── Matched-pairs probit: N=50 (25 NBER drafts + 25 published) ─
    # Build a 50-row dataset: each paper appears twice (draft=0, published=1)
    draft_rows = pairs[["nber_wp_number", "year", "direction_draft"]].copy()
    draft_rows["published_top3"] = False
    draft_rows["direction"] = draft_rows["direction_draft"]

    pub_rows = pairs[["nber_wp_number", "year", "direction_pub"]].copy()
    pub_rows["published_top3"] = True
    pub_rows["direction"] = pub_rows["direction_pub"]

    paired_df = pd.concat([draft_rows, pub_rows], ignore_index=True)
    paired_df["year_c"]      = paired_df["year"] - paired_df["year"].mean()
    paired_df["directional"] = (paired_df["direction"].abs() == 1).astype(int)
    paired_df["pos"]         = (paired_df["direction"] == 1).astype(int)
    paired_df["neg"]         = (paired_df["direction"] == -1).astype(int)
    paired_df["pub"]         = paired_df["published_top3"].astype(int)

    print(f"\n=== Matched-pairs probit (N={len(paired_df)}) ===")
    print(paired_df.groupby("pub")["directional"].value_counts().to_string())

    probit_rows = []

    # Bivariate: pub ~ directional
    probit_rows += run_probit("pub ~ directional", paired_df, "Matched-pairs (1): Binary")

    # Three-way: pub ~ pos + neg
    probit_rows += run_probit("pub ~ pos + neg", paired_df, "Matched-pairs (2): Three-way")

    # + year trend
    probit_rows += run_probit("pub ~ directional + year_c", paired_df, "Matched-pairs (3): + Year")

    if probit_rows:
        result_df = pd.DataFrame(probit_rows)
        print("\nRegression table (matched-pairs robustness):")
        print(result_df.to_string(index=False))
        result_df.to_csv(TABLE_PATH, index=False)
        print(f"\nSaved → {TABLE_PATH}")
    else:
        print("No probit results (likely perfect separation at N=50).")
        # Save direction stability summary as fallback
        summary = pd.DataFrame([{
            "Specification": "Matched-pairs direction stability",
            "Variable": "% stable",
            "Coefficient": round(pairs["stable"].mean(), 4),
            "N": len(pairs),
        }])
        summary.to_csv(TABLE_PATH, index=False)
        print(f"Saved stability summary → {TABLE_PATH}")


if __name__ == "__main__":
    main()
