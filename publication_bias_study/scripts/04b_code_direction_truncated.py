"""
04b_code_direction_truncated.py
--------------------------------
Robustness check: re-code direction for all 200 in-scope papers using only
the FIRST 100 WORDS of each abstract, then re-run the primary probit.

Motivation (referee concern)
------------------------------
Published papers in the main sample average ~104 words per abstract while
NBER working-paper abstracts average ~45 words. A concern is that the longer
published abstracts give the LLM more text to detect directional language,
potentially inflating the directional-coding rate for published papers relative
to NBER papers and biasing the probit estimates upward.

Symmetric 100-word truncation equalises the information available to the LLM
across both groups. If the main result is robust to this restriction, the
abstract-length asymmetry cannot explain the finding.

Logic
------
1. Load data/final/analysis_dataset.parquet (200 in-scope papers).
2. For each paper, truncate the abstract to the first 100 words.
3. Re-run the same LLM direction-coding prompt as script 04 (claude-haiku-4-5
   with temperature=0).
4. If the truncated abstract has fewer than 20 words, skip the LLM and assign
   direction=0 (null) as a conservative default.
5. Save to data/classified/direction_truncated100.parquet.
6. Run the primary probit: Published ~ Positive_trunc + Negative_trunc + year_c
7. Save coefficients to output/tables/robustness_truncated100.csv and print.

Usage
------
    python scripts/04b_code_direction_truncated.py           # full run with LLM
    python scripts/04b_code_direction_truncated.py --no-api  # dry run, direction=0 for all
    python scripts/04b_code_direction_truncated.py --sample 20  # dev test

Prerequisites
--------------
    ANTHROPIC_API_KEY in .env
    data/final/analysis_dataset.parquet (scripts 01-05)
"""

import argparse
import json
import os
import pathlib
import re
import threading
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

ROOT = pathlib.Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")

IN_PATH  = ROOT / "data" / "final" / "analysis_dataset.parquet"
OUT_PATH = ROOT / "data" / "classified" / "direction_truncated100.parquet"
TABLE_PATH = ROOT / "output" / "tables" / "robustness_truncated100.csv"

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
TRUNCATE_WORDS  = 100   # symmetric window size
MIN_WORDS       = 20    # skip LLM if fewer words after truncation
WORKERS         = 8


# ------------------------------------------------------------------ #
# Prompt — identical to script 04                                     #
# ------------------------------------------------------------------ #

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


def truncate_abstract(text: str, n_words: int = TRUNCATE_WORDS) -> str:
    """Return the first *n_words* whitespace-delimited tokens of *text*."""
    tokens = str(text).split()
    return " ".join(tokens[:n_words])


@retry(stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=1, min=2, max=30))
def code_one(client: Anthropic, abstract_truncated: str) -> dict:
    """Call the LLM on the truncated abstract. Returns a dict with direction/confidence/rationale."""
    prompt = DIRECTION_PROMPT.format(abstract=abstract_truncated[:3000])
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=200,
        temperature=0,
        system=DIRECTION_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
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


# ------------------------------------------------------------------ #
# Probit helpers                                                       #
# ------------------------------------------------------------------ #

def run_probit(formula: str, df: pd.DataFrame, label: str) -> list[dict]:
    """Fit a probit and return a list of coefficient-row dicts."""
    try:
        model = smf.probit(formula, data=df).fit(
            disp=False, method="bfgs", maxiter=500
        )
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
        stars = ("***" if p < 0.01 else "**" if p < 0.05 else
                 "*"   if p < 0.10 else "")
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


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(sample: int | None, no_api: bool) -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"Missing {IN_PATH}. Run scripts 01-05 first."
        )

    df = pd.read_parquet(IN_PATH)
    print(f"Loaded analysis_dataset: {len(df):,} papers")
    print(f"  Published (top-3 journal): {df['published_top3'].sum():,}")
    print(f"  NBER / pre-pub:            {(~df['published_top3']).sum():,}")

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
        print(f"  Dev sample: {len(df):,} papers")

    # Build truncated abstracts and flag short ones
    df["abstract_raw"]      = df["abstract"].fillna("")
    df["abstract_trunc"]    = df["abstract_raw"].apply(truncate_abstract)
    df["trunc_word_count"]  = df["abstract_trunc"].apply(lambda x: len(x.split()))

    n_short = (df["trunc_word_count"] < MIN_WORDS).sum()
    n_long  = (df["trunc_word_count"] >= MIN_WORDS).sum()
    print(f"\nTruncated-abstract stats (first {TRUNCATE_WORDS} words):")
    print(f"  Papers with >= {MIN_WORDS} words (will use LLM): {n_long:,}")
    print(f"  Papers with <  {MIN_WORDS} words (direction=0):  {n_short:,}")
    print(f"  Mean word count after truncation (published):  "
          f"{df.loc[df['published_top3'], 'trunc_word_count'].mean():.1f}")
    print(f"  Mean word count after truncation (NBER/pre-pub): "
          f"{df.loc[~df['published_top3'], 'trunc_word_count'].mean():.1f}")

    # ---- Direction coding ----
    rows_to_code = df[df["trunc_word_count"] >= MIN_WORDS].copy()
    rows_short   = df[df["trunc_word_count"] <  MIN_WORDS].copy()

    results: dict[str, dict] = {}

    if no_api:
        print("\n--no-api: assigning direction=0 for all papers (dry run)")
        for pid in df["paper_id"]:
            results[pid] = {"direction": 0, "confidence": "low",
                            "rationale": "dry-run (--no-api)"}
    else:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to .env or set env variable."
            )

        # Papers too short → conservative null without API call
        for _, row in rows_short.iterrows():
            results[row["paper_id"]] = {
                "direction":  0,
                "confidence": "low",
                "rationale":  f"truncated abstract < {MIN_WORDS} words — assigned null conservatively",
            }

        print(f"\nCoding direction on truncated abstracts ({TRUNCATE_WORDS}-word window) "
              f"with {WORKERS} parallel workers ...")

        _tls = threading.local()

        def _get_client() -> Anthropic:
            if not hasattr(_tls, "client"):
                _tls.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            return _tls.client

        def _code_row(row) -> tuple[str, dict]:
            try:
                res = code_one(_get_client(), row["abstract_trunc"])
                return row["paper_id"], res
            except Exception as exc:
                print(f"\n  Error on {row['paper_id']}: {exc}")
                return row["paper_id"], {
                    "direction": 0, "confidence": "low",
                    "rationale": f"error: {exc}",
                }

        code_rows = [r for _, r in rows_to_code.iterrows()]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(_code_row, row): row["paper_id"]
                       for row in code_rows}
            with tqdm(total=len(code_rows), desc="Coding direction (truncated)") as pbar:
                for future in as_completed(futures):
                    pid, res = future.result()
                    results[pid] = res
                    pbar.update(1)

    # ---- Assemble output parquet ----
    out_records = []
    for _, row in df.iterrows():
        res = results.get(row["paper_id"], {"direction": 0, "confidence": "low",
                                            "rationale": "missing"})
        out_records.append({
            "paper_id":           row["paper_id"],
            "direction_truncated": res["direction"],
            "confidence_truncated": res.get("confidence", "low"),
            "rationale_truncated": res.get("rationale", ""),
            "trunc_word_count":    row["trunc_word_count"],
        })

    out_df = pd.DataFrame(out_records)
    out_df["direction_truncated"] = out_df["direction_truncated"].astype("Int8")

    print(f"\nTruncated-abstract direction distribution:")
    print(out_df["direction_truncated"].value_counts().sort_index().to_string())
    print(f"Confidence distribution:")
    print(out_df["confidence_truncated"].value_counts().to_string())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"\nSaved → {OUT_PATH}  ({len(out_df):,} records)")

    # ---- Primary probit ----
    print("\n=== Primary probit: Published ~ Positive_trunc + Negative_trunc + year_c ===")

    # Merge truncated codes back onto analysis_dataset
    reg_df = df.merge(
        out_df[["paper_id", "direction_truncated"]],
        on="paper_id", how="left"
    )
    reg_df["pub"]       = reg_df["published_top3"].astype(int)
    reg_df["pos_trunc"] = (reg_df["direction_truncated"] == 1).astype(int)
    reg_df["neg_trunc"] = (reg_df["direction_truncated"] == -1).astype(int)
    reg_df["directional_trunc"] = (reg_df["direction_truncated"] != 0).astype(int)
    reg_df["year_c"]    = reg_df["year"] - reg_df["year"].median()

    # Drop rows with missing outcome or regressors
    reg_df = reg_df.dropna(subset=["pub", "year_c"])

    # Original direction codes for comparison
    reg_df["pos_orig"] = reg_df["positive"].fillna(False).astype(int)
    reg_df["neg_orig"] = reg_df["negative"].fillna(False).astype(int)

    # Raw publication rates by truncated direction
    print("\nPublication rates by truncated direction:")
    for dir_val, label in [(1, "Positive"), (-1, "Negative"), (0, "Null")]:
        sub = reg_df[reg_df["direction_truncated"] == dir_val]
        if len(sub) == 0:
            continue
        rate = sub["pub"].mean()
        print(f"  {label:10s}: {len(sub):3d} papers, pub rate = {rate:.3f}")

    all_rows: list[dict] = []

    specs = [
        ("Trunc100 (1): Bivariate",       "pub ~ pos_trunc + neg_trunc"),
        ("Trunc100 (2): + Year",           "pub ~ pos_trunc + neg_trunc + year_c"),
        ("Trunc100 (3): Binary directional", "pub ~ directional_trunc + year_c"),
    ]

    for label, formula in specs:
        rows = run_probit(formula, reg_df, label)
        all_rows.extend(rows)

    if not all_rows:
        print("  No probit results to save.")
        return

    result_df = pd.DataFrame(all_rows)

    # Print regression table to stdout
    print("\nRegression table (truncated-abstract robustness):")
    print("-" * 100)
    print(result_df.to_string(index=False))
    print("-" * 100)

    # Compare directional beta with original
    orig_rows = run_probit(
        "pub ~ pos_orig + neg_orig + year_c", reg_df,
        "Original coding (full abstract)"
    )
    if orig_rows:
        orig_df = pd.DataFrame(orig_rows)
        result_df = pd.concat([result_df, orig_df], ignore_index=True)
        print("\nComparison: original (full-abstract) vs. truncated-100-word coding:")
        print("-" * 80)
        cmp = result_df[result_df["Variable"].isin(["pos_trunc", "neg_trunc",
                                                     "pos_orig", "neg_orig"])].copy()
        print(cmp[["Specification", "Variable", "Coefficient",
                    "Std Error", "p-value", "Significance"]].to_string(index=False))

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(TABLE_PATH, index=False)
    print(f"\nSaved → {TABLE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robustness: re-code direction on 100-word truncated abstracts."
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Dev test: code only a random subsample of N papers")
    parser.add_argument("--no-api", action="store_true",
                        help="Dry run: skip LLM, assign direction=0 for all papers")
    args = parser.parse_args()
    main(sample=args.sample, no_api=args.no_api)
