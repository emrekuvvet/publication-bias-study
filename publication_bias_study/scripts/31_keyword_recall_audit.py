"""
Estimate the keyword-filter recall by sampling keyword-negative papers and
running the same in-scope LLM prompt that gates the keyword-positive pool.

Output:
  output/tables/keyword_recall_audit.csv  -- per-paper LLM verdict on the sample
  output/tables/keyword_recall_audit_summary.csv  -- aggregate counts and the
    implied false-negative rate and 95% CI

Run:
  python3 scripts/31_keyword_recall_audit.py --n 200 --seed 42
"""

import argparse
import json
import os
import pathlib
import re

import numpy as np
import pandas as pd
import scipy.stats as stats
from anthropic import Anthropic
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

ROOT = pathlib.Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")

MODEL = "claude-haiku-4-5-20251001"

SCOPE_SYSTEM = (
    "You are a research assistant classifying finance research papers. "
    "Return valid JSON only — no other text."
)

SCOPE_PROMPT_TEMPLATE = """\
A finance research paper is 'in scope' for a study of publication bias in \
government intervention research if and only if its PRIMARY research question \
is the CAUSAL EFFECT of a specific government law, regulation, or public \
policy on financial markets, firms, or investors.

Exclusion criteria (mark in_scope=false):
- Purely theoretical papers (no empirical test of a policy)
- Papers that only DESCRIBE a regulation without estimating its effect
- Papers whose main topic is private firm/market behaviour, with policy as incidental context
- Papers studying central bank communication WITHOUT a specific policy instrument
- Papers whose primary policy variable is a rule imposed by a stock exchange (NYSE, NASDAQ, etc.) without a specific government or regulatory mandate

Title: \"\"\"{title}\"\"\"
Abstract:
\"\"\"
{abstract}
\"\"\"

Return JSON with exactly these keys:
{{
  "in_scope": true or false,
  "confidence": "high" | "medium" | "low",
  "reason": "one sentence"
}}"""


@retry(stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=1, min=2, max=30))
def classify_one(client: Anthropic, title: str, abstract: str) -> dict:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=SCOPE_SYSTEM,
        messages=[{
            "role": "user",
            "content": SCOPE_PROMPT_TEMPLATE.format(
                title=(title or "")[:300],
                abstract=(abstract or "")[:6000],
            ),
        }],
    )
    raw = msg.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"in_scope": False, "confidence": "low", "reason": "parse error"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ins = pd.read_parquet(ROOT / "data" / "classified" / "inscope_papers.parquet")
    neg = ins[(ins["keyword_hit"] == False)].copy()
    # Restrict to papers with usable abstracts (>= 20 words) for a fair test
    neg["words"] = neg["abstract"].fillna("").str.split().str.len()
    neg = neg[neg["words"] >= 20].copy()
    print(f"Keyword-negative pool with usable abstract: {len(neg):,}")

    sample = neg.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    print(f"Sampling {len(sample)} papers (seed={args.seed})")

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    verdicts = []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="LLM audit"):
        try:
            res = classify_one(client, str(row["title"]), str(row["abstract"]))
        except Exception as exc:
            res = {"in_scope": None, "confidence": "low", "reason": f"error: {exc}"}
        verdicts.append({
            "paper_id":   row["paper_id"],
            "source":     row["source"],
            "title":      row["title"],
            "in_scope":   res.get("in_scope"),
            "confidence": res.get("confidence"),
            "reason":     res.get("reason"),
        })

    out = pd.DataFrame(verdicts)
    out.to_csv(ROOT / "output" / "tables" / "keyword_recall_audit.csv", index=False)
    print(f"\nSaved per-paper verdicts to output/tables/keyword_recall_audit.csv")

    # Headline summary: false-negative rate (LLM says in-scope despite keyword miss)
    in_scope_true = out["in_scope"].fillna(False).eq(True).sum()
    total         = out["in_scope"].notna().sum()
    p             = in_scope_true / total if total else float("nan")
    lo, hi        = stats.binom.interval(0.95, total, p) if total else (np.nan, np.nan)
    ci_lo, ci_hi  = lo / total, hi / total

    # Implied count of false negatives in the full 38,111-paper keyword-negative pool
    full_pool = len(neg)
    implied_count        = full_pool * p
    implied_count_lo     = full_pool * ci_lo
    implied_count_hi     = full_pool * ci_hi

    summary = pd.DataFrame([{
        "n_sampled":               total,
        "n_llm_in_scope":          in_scope_true,
        "false_negative_rate":     round(p, 4),
        "ci95_lo":                 round(ci_lo, 4),
        "ci95_hi":                 round(ci_hi, 4),
        "keyword_negative_pool":   full_pool,
        "implied_false_negatives": round(implied_count, 1),
        "implied_fn_ci_lo":        round(implied_count_lo, 1),
        "implied_fn_ci_hi":        round(implied_count_hi, 1),
    }])
    summary.to_csv(
        ROOT / "output" / "tables" / "keyword_recall_audit_summary.csv",
        index=False,
    )
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
