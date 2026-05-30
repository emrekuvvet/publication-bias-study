"""
11c_recode_afa_direction_llm.py
--------------------------------
Robustness check: re-run LLM direction coding uniformly for ALL AFA in-scope
papers that have abstracts, replacing keyword-rule codes for unmatched papers.

Motivation (referee concern)
------------------------------
In the main AFA analysis (script 11), the --no-api path uses LLM direction codes
for papers that matched to published journals (copied from the analysis_dataset)
but falls back to keyword-rule coding for unmatched (pre-publication) AFA papers.
This creates a potential coding asymmetry: matched papers are coded by the LLM,
while unmatched papers are coded by simpler rules that may have different precision
and recall properties.

This script eliminates that asymmetry by running the *same* LLM direction-coding
prompt uniformly across all 108 AFA in-scope papers — regardless of match status.
Papers without a usable abstract (< 20 words) receive direction=0 (null)
conservatively.

If the AFA baseline result is robust to uniform LLM coding, the asymmetry cannot
explain the finding. We also report the change in the key coefficient β_Directional
relative to the published value of 0.077.

Logic
------
1. Load data/classified/afa_inscope.parquet (108 AFA in-scope papers).
2. Load data/classified/afa_matched.parquet (match status + existing direction codes).
3. For matched AFA papers: use the existing LLM direction codes (they already come
   from the main pipeline, which used the same prompt as script 04).
4. For unmatched AFA papers WITH abstracts (>= 20 words): run LLM direction coding.
5. For unmatched AFA papers WITHOUT abstracts (< 20 words): direction=0 (null).
6. Save to data/classified/afa_direction_llm.parquet.
7. Re-run the AFA baseline probit: Published_top3 ~ Directional + year_c.
8. Save to output/tables/robustness_afa_uniform_llm.csv.
9. Print comparison: old AFA beta (0.077) vs. new beta with uniform LLM coding.

Usage
------
    python scripts/11c_recode_afa_direction_llm.py           # full run
    python scripts/11c_recode_afa_direction_llm.py --no-api  # dry run (direction=0)
    python scripts/11c_recode_afa_direction_llm.py --sample 10  # dev test

Prerequisites
--------------
    ANTHROPIC_API_KEY in .env
    data/classified/afa_inscope.parquet  (from 11_afa_analysis.py --step classify)
    data/classified/afa_matched.parquet  (from 11_afa_analysis.py --step match)
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

AFA_SCOPE_PATH   = ROOT / "data" / "classified" / "afa_inscope.parquet"
AFA_MATCHED_PATH = ROOT / "data" / "classified" / "afa_matched.parquet"
OUT_PATH         = ROOT / "data" / "classified" / "afa_direction_llm.parquet"
TABLE_PATH       = ROOT / "output" / "tables" / "robustness_afa_uniform_llm.csv"

ANTHROPIC_MODEL  = "claude-haiku-4-5-20251001"
MIN_WORDS        = 20   # minimum abstract length to attempt LLM coding
WORKERS          = 1    # AFA pipeline convention (rate-limit safety)

# Published value of β_Directional from the existing AFA baseline result
AFA_BETA_PUBLISHED = 0.077


# ------------------------------------------------------------------ #
# Direction coding prompt — identical to scripts 04 and 11            #
# ------------------------------------------------------------------ #

DIRECTION_SYSTEM = (
    "You are a research assistant reading finance research papers. "
    "Return valid JSON only — no other text."
)

DIRECTION_PROMPT_ABSTRACT = """\
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


@retry(stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=1, min=2, max=30))
def code_direction_llm(client: Anthropic, abstract: str) -> dict:
    """Call the LLM to code direction from an abstract. Returns direction/confidence/rationale."""
    prompt = DIRECTION_PROMPT_ABSTRACT.format(abstract=abstract[:1500])
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
# Probit helper                                                        #
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
    # ---- Load inputs ----
    if not AFA_SCOPE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {AFA_SCOPE_PATH}. "
            "Run: python scripts/11_afa_analysis.py --step classify"
        )
    if not AFA_MATCHED_PATH.exists():
        raise FileNotFoundError(
            f"Missing {AFA_MATCHED_PATH}. "
            "Run: python scripts/11_afa_analysis.py --step match"
        )

    scope_df   = pd.read_parquet(AFA_SCOPE_PATH)
    matched_df = pd.read_parquet(AFA_MATCHED_PATH)

    inscope = scope_df[scope_df["in_scope"]].copy().reset_index(drop=True)
    inscope["abstract"] = inscope.get("abstract", pd.Series("", index=inscope.index)).fillna("")

    print(f"AFA in-scope papers:    {len(inscope):,}")
    print(f"afa_matched records:    {len(matched_df):,}")

    # ---- Separate matched vs. unmatched ----
    matched_published = matched_df[matched_df["published_top3"]].copy()
    matched_unpub     = matched_df[~matched_df["published_top3"]].copy()

    matched_ids   = set(matched_df["paper_id"])
    unmatched_ids = set(inscope["paper_id"]) - matched_ids

    # All 108 in-scope papers are already in afa_matched (matched_df covers all in-scope);
    # "unmatched" in the referee's sense means published_top3=False.
    # We re-code direction only for those (unmatched / unpublished) papers.
    print(f"\nStatus of AFA in-scope papers:")
    print(f"  Matched to top-3 journal (published_top3=True):  {len(matched_published):,}")
    print(f"  Unmatched (pre-publication / unpublished):        {len(matched_unpub):,}")

    # Attach abstract to matched_df rows
    abs_map = inscope.set_index("paper_id")["abstract"].to_dict()
    matched_unpub = matched_unpub.copy()
    matched_unpub["abstract"] = matched_unpub["paper_id"].map(abs_map).fillna("")
    matched_unpub["abstract_word_count"] = (
        matched_unpub["abstract"].str.split().str.len().fillna(0).astype(int)
    )

    with_abs  = (matched_unpub["abstract_word_count"] >= MIN_WORDS).sum()
    no_abs    = (matched_unpub["abstract_word_count"] <  MIN_WORDS).sum()
    print(f"\n  Unmatched papers with abstract >= {MIN_WORDS} words: {with_abs:,}")
    print(f"  Unmatched papers without usable abstract:          {no_abs:,}")

    if sample:
        matched_unpub = matched_unpub.sample(
            n=min(sample, len(matched_unpub)), random_state=42
        ).reset_index(drop=True)
        print(f"  Dev sample: coding {len(matched_unpub):,} unmatched papers")

    # ---- Direction coding for unmatched papers ----
    new_codes: dict[str, dict] = {}

    if no_api:
        print("\n--no-api: assigning direction=0 for all unmatched papers (dry run)")
        for pid in matched_unpub["paper_id"]:
            new_codes[pid] = {"direction": 0, "confidence": "low",
                              "rationale": "dry-run (--no-api)"}
    else:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to .env or set env variable."
            )

        # Papers without abstracts → conservative null
        no_abs_rows = matched_unpub[
            matched_unpub["abstract_word_count"] < MIN_WORDS
        ]
        for _, row in no_abs_rows.iterrows():
            new_codes[row["paper_id"]] = {
                "direction":  0,
                "confidence": "low",
                "rationale":  f"abstract < {MIN_WORDS} words — assigned null conservatively",
            }

        # Papers with abstracts → LLM
        to_code = matched_unpub[
            matched_unpub["abstract_word_count"] >= MIN_WORDS
        ].copy()

        print(f"\nRunning LLM direction coding for {len(to_code):,} unmatched "
              f"papers with abstracts ...")

        _tls = threading.local()

        def _get_client() -> Anthropic:
            if not hasattr(_tls, "client"):
                _tls.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            return _tls.client

        def _code_row(row) -> tuple[str, dict]:
            try:
                res = code_direction_llm(_get_client(), row["abstract"])
                return row["paper_id"], res
            except Exception as exc:
                print(f"\n  Error on {row['paper_id']}: {exc}")
                return row["paper_id"], {
                    "direction": 0, "confidence": "low",
                    "rationale": f"error: {exc}",
                }

        code_rows = [r for _, r in to_code.iterrows()]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(_code_row, row): row["paper_id"]
                       for row in code_rows}
            with tqdm(total=len(code_rows),
                      desc="LLM direction (unmatched AFA)") as pbar:
                for future in as_completed(futures):
                    pid, res = future.result()
                    new_codes[pid] = res
                    pbar.update(1)

    # ---- Assemble full dataset with uniform LLM codes ----
    # For matched (published) papers: keep existing LLM direction codes from pipeline
    # For unmatched papers: use new_codes (LLM or null)
    all_records = []

    for _, row in matched_published.iterrows():
        # Already coded by LLM in main pipeline
        all_records.append({
            "paper_id":        row["paper_id"],
            "published_top3":  True,
            "year":            row["year"],
            "title":           row.get("title", ""),
            "authors":         row.get("authors", ""),
            "direction_llm":   int(row["direction"]) if pd.notna(row["direction"]) else 0,
            "confidence_llm":  row.get("dir_conf", "low"),
            "rationale_llm":   row.get("dir_rationale", ""),
            "coding_source":   "pipeline_llm",
        })

    for _, row in matched_unpub.iterrows():
        pid = row["paper_id"]
        res = new_codes.get(pid, {"direction": 0, "confidence": "low",
                                  "rationale": "missing"})
        coding_src = "new_llm" if (
            not no_api and row["abstract_word_count"] >= MIN_WORDS
        ) else "no_abstract_null"
        all_records.append({
            "paper_id":        pid,
            "published_top3":  False,
            "year":            row["year"],
            "title":           row.get("title", ""),
            "authors":         row.get("authors", ""),
            "direction_llm":   int(res["direction"]),
            "confidence_llm":  res.get("confidence", "low"),
            "rationale_llm":   res.get("rationale", ""),
            "coding_source":   coding_src,
        })

    out_df = pd.DataFrame(all_records)
    out_df["direction_llm"] = out_df["direction_llm"].astype("Int8")

    print(f"\nUniform-LLM direction distribution (all {len(out_df):,} AFA in-scope):")
    print(out_df["direction_llm"].value_counts().sort_index().to_string())
    print(f"\nCoding source breakdown:")
    print(out_df["coding_source"].value_counts().to_string())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"\nSaved → {OUT_PATH}  ({len(out_df):,} records)")

    # ---- AFA baseline probit with uniform LLM codes ----
    print("\n=== AFA baseline probit: Published_top3 ~ Directional + year_c ===")

    reg_df = out_df.copy()
    reg_df["pub"]         = reg_df["published_top3"].astype(int)
    reg_df["directional"] = (reg_df["direction_llm"] != 0).astype(int)
    reg_df["pos"]         = (reg_df["direction_llm"] == 1).astype(int)
    reg_df["neg"]         = (reg_df["direction_llm"] == -1).astype(int)
    reg_df["year"]        = pd.to_numeric(reg_df["year"], errors="coerce")
    reg_df["year_c"]      = reg_df["year"] - reg_df["year"].median()
    reg_df["log_nauth"]   = np.log1p(
        reg_df["authors"].str.split(",").str.len().clip(lower=1)
    )
    reg_df = reg_df.dropna(subset=["pub", "year_c"])

    # Publication rates
    print("\nPublication rates by uniform LLM direction:")
    for dir_val, label in [(1, "Positive"), (-1, "Negative"), (0, "Null")]:
        sub = reg_df[reg_df["direction_llm"] == dir_val]
        if len(sub) == 0:
            continue
        rate = sub["pub"].mean()
        print(f"  {label:10s}: {len(sub):3d} papers, pub rate = {rate:.3f}")

    all_rows: list[dict] = []

    specs = [
        ("AFA Uniform-LLM (1): Bivariate",  "pub ~ directional"),
        ("AFA Uniform-LLM (2): + Year",      "pub ~ directional + year_c"),
        ("AFA Uniform-LLM (3): + Authors",   "pub ~ directional + year_c + log_nauth"),
        ("AFA Uniform-LLM (4): 3-way",       "pub ~ pos + neg + year_c"),
    ]

    for label, formula in specs:
        rows = run_probit(formula, reg_df, label)
        all_rows.extend(rows)

    if not all_rows:
        print("  No probit results to save.")
        return

    result_df = pd.DataFrame(all_rows)

    # Print regression table
    print("\nRegression table (uniform LLM coding robustness):")
    print("-" * 110)
    print(result_df.to_string(index=False))
    print("-" * 110)

    # Key comparison: old beta vs. new beta
    spec2_rows = [r for r in all_rows
                  if r["Specification"] == "AFA Uniform-LLM (2): + Year"
                  and r["Variable"] == "directional"]
    if spec2_rows:
        new_beta = spec2_rows[0]["Coefficient"]
        new_se   = spec2_rows[0]["Std Error"]
        new_p    = spec2_rows[0]["p-value"]
        new_sig  = spec2_rows[0]["Significance"]
        print(f"\nComparison of β_Directional (AFA + year_c probit):")
        print(f"  Old (existing AFA baseline):   β = {AFA_BETA_PUBLISHED:.3f}")
        print(f"  New (uniform LLM coding):      β = {new_beta:.3f}  "
              f"(SE = {new_se:.4f}, p = {new_p:.4f}{new_sig})")
        delta = new_beta - AFA_BETA_PUBLISHED
        print(f"  Change:                        Δβ = {delta:+.3f}")

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(TABLE_PATH, index=False)
    print(f"\nSaved → {TABLE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Robustness: re-code direction uniformly via LLM for all AFA in-scope "
            "papers, eliminating the keyword-rule asymmetry for unmatched papers."
        )
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Dev test: code only N unmatched papers")
    parser.add_argument("--no-api", action="store_true",
                        help="Dry run: skip LLM, assign direction=0 for all unmatched papers")
    args = parser.parse_args()
    main(sample=args.sample, no_api=args.no_api)
