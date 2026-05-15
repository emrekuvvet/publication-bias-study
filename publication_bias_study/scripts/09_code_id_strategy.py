"""
09_code_id_strategy.py
----------------------
Classifies identification strategy for each in-scope paper from its abstract.

This addresses the quality-control concern: null-finding papers might rely on
weaker empirical designs that are less likely to detect effects regardless of the
true effect size. The quasi_exp dummy allows us to include an identification-
quality control in the probit specifications.

NOTE: This coding is itself imperfect — fewer than half of abstracts explicitly
name their identification strategy, so the LLM infers it from other signals.
The quasi_exp dummy should be read as a noisy quality control, not a definitive
classification. Results are robust to including vs. excluding this control
(see Table III Column 6 vs. Columns 1-5 in the paper).

Classification scheme:
    quasi_exp   : RDD, IV, DiD, Event Study, Natural Experiment
    ols_reduced : OLS, panel regression, reduced-form without instrument
    structural  : Structural/calibration model
    unclear     : Cannot determine from abstract

Outputs:
    Updates data/final/analysis_dataset.parquet with id_strategy and quasi_exp
    columns populated.

Usage:
    python scripts/09_code_id_strategy.py
"""

import json
import os
import pathlib
import time
import dotenv

import anthropic
import pandas as pd

dotenv.load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

ROOT    = pathlib.Path(__file__).parent.parent.resolve()
IN_PATH = ROOT / "data" / "final" / "analysis_dataset.parquet"

SYSTEM_PROMPT = """You are a research assistant classifying empirical identification
strategies in finance and economics papers. You will be given a paper title and abstract.

Classify the PRIMARY identification strategy into exactly one of these four categories:

- quasi_exp: The paper uses a quasi-experimental method that addresses endogeneity:
  Regression Discontinuity (RDD), Instrumental Variables (IV),
  Difference-in-Differences (DiD), Event Study with causal interpretation,
  or Natural Experiment with exogenous variation.

- ols_reduced: The paper relies primarily on OLS, panel fixed effects without
  an instrument, reduced-form correlation, or descriptive regression without
  a credible source of exogenous variation.

- structural: The paper uses a structural model, calibration, or theoretical
  analysis as the primary empirical method.

- unclear: The abstract does not contain enough information to determine
  the identification strategy, or the paper is primarily theoretical.

Return ONLY a JSON object with exactly two keys:
{
  "id_strategy": "<one of: quasi_exp, ols_reduced, structural, unclear>",
  "rationale": "<one sentence explaining your classification>"
}"""


def classify_paper(client: anthropic.Anthropic, title: str, abstract: str) -> dict:
    prompt = f"Title: {title}\n\nAbstract: {abstract}"
    for attempt in range(4):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            # Extract JSON robustly
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError(f"No JSON in response: {text[:100]}")
            result = json.loads(text[start:end])
            # Accept event_study as an alias for quasi_exp
            if result.get("id_strategy") == "event_study":
                result["id_strategy"] = "quasi_exp"
            if result.get("id_strategy") not in ("quasi_exp", "ols_reduced", "structural", "unclear"):
                raise ValueError(f"Invalid id_strategy value: {result.get('id_strategy')}")
            return result
        except Exception as exc:
            wait = 2 ** attempt
            print(f"    Attempt {attempt+1} failed: {exc} — retrying in {wait}s")
            time.sleep(wait)
    return {"id_strategy": "unclear", "rationale": "classification failed after retries"}


def main():
    df = pd.read_parquet(IN_PATH)
    print(f"Loaded {len(df)} papers. Classifying identification strategies...")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    strategies = []
    rationales = []

    for i, row in df.iterrows():
        title    = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        if not abstract or abstract == "nan":
            strategies.append("unclear")
            rationales.append("no abstract available")
            continue

        result = classify_paper(client, title, abstract)
        strategies.append(result["id_strategy"])
        rationales.append(result.get("rationale", ""))

        if (len(strategies)) % 50 == 0:
            print(f"  Classified {len(strategies)}/{len(df)} papers...")
        time.sleep(0.05)  # gentle rate limiting

    df["id_strategy"]    = strategies
    df["id_rationale"]   = rationales
    df["quasi_exp"]      = (df["id_strategy"] == "quasi_exp").astype(int)

    df.to_parquet(IN_PATH, index=False)
    print(f"\nDone. Results saved to {IN_PATH}")
    print("\nIdentification strategy distribution:")
    print(df["id_strategy"].value_counts())
    print(f"\nQuasi-experimental by published status:")
    print(df.groupby("published_top3")["quasi_exp"].mean().round(3))


if __name__ == "__main__":
    main()
