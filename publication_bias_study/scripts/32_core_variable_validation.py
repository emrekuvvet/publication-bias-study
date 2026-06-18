"""
Two targeted validations of the core LLM-coded variables that drive the
main estimates:

(A) Null-coded paper stability.  Samples null-coded papers and re-codes
    them with a STRICTER prompt that requires any directional signal in
    the abstract to flip the verdict.  Catches the failure mode where
    the original LLM was swayed by hedging language into coding null
    despite a directional finding.

(B) QE-coded paper validation.  Samples papers coded as
    quasi-experimental and re-checks each abstract against an explicit
    rubric: must NAME an identification strategy (RDD, IV, DiD,
    natural experiment, event study with causal claim) AND be the
    paper's primary method.  Catches the failure mode where the
    18-20 pp QE premium reflects abstract-style differences in whether
    published papers explicitly NAME their identification strategy
    rather than actual differences in research-design quality.

Each audit samples a balanced number of NBER and published papers so
corpus-side asymmetries can be detected.

Outputs:
  output/tables/null_stability_audit.csv      -- per-paper verdicts
  output/tables/qe_validation_audit.csv       -- per-paper verdicts
  output/tables/core_variable_validation_summary.csv -- summary table

Run:
  python3 scripts/32_core_variable_validation.py
"""

import json
import os
import pathlib
import re

import numpy as np
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

ROOT = pathlib.Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")

MODEL = "claude-haiku-4-5-20251001"

# ── (A) NULL STABILITY PROMPT ──────────────────────────────────────────────
NULL_SYSTEM = (
    "You are a research assistant re-checking whether a finance paper has "
    "any directional finding. Return JSON only."
)

NULL_PROMPT = """\
The following abstract was previously coded as NULL / MIXED / INCONCLUSIVE \
(direction = 0). Re-examine the abstract under a STRICTER rule: a paper \
should remain coded 0 only if it explicitly states no significant effect, \
mixed evidence, or refuses to take a directional stance. Hedging language \
('we find some evidence', 'suggests', 'may') with a substantive \
directional claim should NOT be coded 0.

Return one of:
  +1  Abstract contains a positive/beneficial finding (intervention worked)
  -1  Abstract contains a negative/harmful finding (intervention failed or backfired)
   0  Abstract genuinely reports null, mixed, or inconclusive results

Title: \"\"\"{title}\"\"\"
Abstract:
\"\"\"
{abstract}
\"\"\"

Return JSON: {{"direction": 1|-1|0, "rationale": "one sentence"}}"""


# ── (B) QE VALIDATION PROMPT ────────────────────────────────────────────────
QE_SYSTEM = (
    "You are a research assistant validating whether a finance paper "
    "actually uses a quasi-experimental identification strategy. "
    "Return JSON only."
)

QE_PROMPT = """\
The following abstract was previously coded as QUASI-EXPERIMENTAL. Validate \
this coding under a STRICT rule. A paper qualifies as QE only if its \
abstract:
  (i) NAMES an identification strategy: regression discontinuity (RDD), \
      instrumental variables (IV), difference-in-differences (DiD), \
      natural experiment with an exogenous shock, or event study with an \
      explicit causal interpretation; AND
  (ii) Uses that strategy as the PRIMARY empirical method (not as a \
       robustness check or auxiliary test).

Descriptive event-window analyses without a causal claim, OLS with fixed \
effects, simple before/after comparisons, or generic 'we estimate the \
effect of X on Y' language without a named identification strategy do \
NOT qualify as QE.

Return:
  qe = true   if the abstract clearly meets BOTH (i) and (ii)
  qe = false  otherwise (including ambiguous cases)
  named_method  -- the specific identification strategy NAMED in the \
                   abstract, or 'none' if no strategy is explicitly named

Title: \"\"\"{title}\"\"\"
Abstract:
\"\"\"
{abstract}
\"\"\"

Return JSON: {{"qe": true|false, "named_method": "...", "rationale": "one sentence"}}"""


@retry(stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=1, min=2, max=30))
def call_llm(client: Anthropic, system: str, user: str) -> dict:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = msg.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"error": "parse error", "raw": raw[:300]}


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "final" / "analysis_dataset.parquet")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # ───────── (A) Null-coded paper stability ─────────
    nulls = df[df["direction"] == 0].copy()
    nbr   = nulls[nulls["published_top3"] == False].sample(15, random_state=42)
    pub   = nulls[nulls["published_top3"] == True].sample(15, random_state=42)
    null_sample = pd.concat([nbr, pub]).reset_index(drop=True)
    print(f"(A) Null-coded audit: sampling {len(null_sample)} papers")

    null_verdicts = []
    for _, row in tqdm(null_sample.iterrows(), total=len(null_sample),
                       desc="Null-stability audit"):
        prompt = NULL_PROMPT.format(
            title=str(row["title"])[:300],
            abstract=str(row["abstract"])[:6000],
        )
        res = call_llm(client, NULL_SYSTEM, prompt)
        null_verdicts.append({
            "paper_id": row["paper_id"],
            "published_top3": bool(row["published_top3"]),
            "original_direction": int(row["direction"]),
            "strict_direction":   res.get("direction"),
            "rationale": res.get("rationale", res.get("raw", "")),
        })

    null_df = pd.DataFrame(null_verdicts)
    null_df.to_csv(ROOT / "output" / "tables" / "null_stability_audit.csv",
                   index=False)

    # ───────── (B) QE-coded validation ─────────
    qes = df[df["quasi_exp"] == 1].copy()
    nbr_qe = qes[qes["published_top3"] == False].sample(15, random_state=42)
    pub_qe = qes[qes["published_top3"] == True].sample(15, random_state=42)
    qe_sample = pd.concat([nbr_qe, pub_qe]).reset_index(drop=True)
    print(f"(B) QE-coded audit: sampling {len(qe_sample)} papers")

    qe_verdicts = []
    for _, row in tqdm(qe_sample.iterrows(), total=len(qe_sample),
                       desc="QE-validation audit"):
        prompt = QE_PROMPT.format(
            title=str(row["title"])[:300],
            abstract=str(row["abstract"])[:6000],
        )
        res = call_llm(client, QE_SYSTEM, prompt)
        qe_verdicts.append({
            "paper_id":         row["paper_id"],
            "published_top3":   bool(row["published_top3"]),
            "original_qe":      1,
            "strict_qe":        res.get("qe"),
            "named_method":     res.get("named_method", "none"),
            "rationale":        res.get("rationale", res.get("raw", "")),
        })

    qe_df = pd.DataFrame(qe_verdicts)
    qe_df.to_csv(ROOT / "output" / "tables" / "qe_validation_audit.csv",
                 index=False)

    # ───────── Summary ─────────
    null_df["agreed"] = null_df["strict_direction"].fillna(99) == 0
    null_pub  = null_df[null_df["published_top3"]]
    null_nber = null_df[~null_df["published_top3"]]
    null_pub_flip  = (~null_pub["agreed"]).sum()
    null_nber_flip = (~null_nber["agreed"]).sum()

    qe_df["agreed"] = qe_df["strict_qe"] == True
    qe_pub  = qe_df[qe_df["published_top3"]]
    qe_nber = qe_df[~qe_df["published_top3"]]
    qe_pub_flip  = (~qe_pub["agreed"]).sum()
    qe_nber_flip = (~qe_nber["agreed"]).sum()

    summary = pd.DataFrame([
        {"audit": "Null-stability (Published)", "n": len(null_pub),  "agreed": int(null_pub["agreed"].sum()),  "flipped": int(null_pub_flip),  "agree_rate": round(null_pub["agreed"].mean(),  3)},
        {"audit": "Null-stability (NBER)",      "n": len(null_nber), "agreed": int(null_nber["agreed"].sum()), "flipped": int(null_nber_flip), "agree_rate": round(null_nber["agreed"].mean(), 3)},
        {"audit": "QE-validation (Published)",  "n": len(qe_pub),    "agreed": int(qe_pub["agreed"].sum()),    "flipped": int(qe_pub_flip),    "agree_rate": round(qe_pub["agreed"].mean(),    3)},
        {"audit": "QE-validation (NBER)",       "n": len(qe_nber),   "agreed": int(qe_nber["agreed"].sum()),   "flipped": int(qe_nber_flip),   "agree_rate": round(qe_nber["agreed"].mean(),   3)},
    ])
    summary.to_csv(
        ROOT / "output" / "tables" / "core_variable_validation_summary.csv",
        index=False,
    )
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
