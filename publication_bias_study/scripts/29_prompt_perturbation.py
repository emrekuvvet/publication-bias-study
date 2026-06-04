"""
Script 29: Prompt-perturbation robustness check for direction coding.

Uses the same 60-paper stratified sample as the model-choice replication
(Script 13 / llm_replication_kappa.csv) and re-runs direction coding under
three alternative prompt wordings.  κ across prompt variants measures
prompt-sensitivity rather than model-sensitivity.

Outputs:
    output/tables/prompt_perturbation_kappa.csv
"""

import json, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from anthropic import Anthropic

client = Anthropic()
MODEL  = "claude-haiku-4-5-20251001"
SYSTEM = "You are a research assistant classifying finance paper findings. Return valid JSON only — no other text."
OUT    = Path("output/tables/prompt_perturbation_kappa.csv")

# ── Load dataset and reconstruct the same stratified 60-paper sample ─────────
df = pd.read_parquet("data/final/analysis_dataset.parquet")
np.random.seed(42)
sample = (df.groupby(["source","direction"], group_keys=False)
            .apply(lambda g: g.sample(min(5, len(g))))
            .reset_index(drop=True))
print(f"Sample: {len(sample)} papers")
true_labels = sample["direction"].tolist()

# ── Three prompt variants ──────────────────────────────────────────────────────
PROMPTS = {
    "baseline": """\
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
\"\"\"{abstract}\"\"\"

Return JSON with exactly:
{{"direction": 1 | -1 | 0, "confidence": "high" | "medium" | "low", "rationale": "one sentence"}}""",

    "variant_a": """\
Read this abstract from an empirical finance paper about a government policy or regulation.

What is the paper's primary empirical conclusion about the policy's effect?
  +1  The policy had a POSITIVE or BENEFICIAL effect on the outcome studied
  -1  The policy had a NEGATIVE or HARMFUL effect on the outcome studied
   0  The paper finds NO SIGNIFICANT EFFECT, MIXED results, or is inconclusive

Instructions:
- Focus on the main stated result, not secondary findings
- If no clear directional causal estimate is reported, use 0
- Code based on what the paper actually estimates, not whether the policy is desirable

Abstract:
\"\"\"{abstract}\"\"\"

Respond with JSON only:
{{"direction": 1 | -1 | 0, "confidence": "high" | "medium" | "low", "rationale": "one sentence"}}""",

    "variant_b": """\
You will classify the direction of the main finding in a finance research abstract.

The paper examines a specific government intervention, law, or regulation.

Classification:
  POSITIVE (+1): The main estimated effect of the intervention is beneficial,
    improving outcomes such as market efficiency, investor welfare, firm value,
    financial stability, or credit access.
  NEGATIVE (-1): The main estimated effect is harmful, such as reducing liquidity,
    increasing costs, creating distortions, or producing unintended negative
    consequences.
  NULL/MIXED (0): No statistically significant main effect, mixed evidence, or
    the paper is primarily descriptive/theoretical without a clear directional claim.

Abstract to classify:
\"\"\"{abstract}\"\"\"

Return only valid JSON:
{{"direction": 1 | -1 | 0, "confidence": "high" | "medium" | "low", "rationale": "one sentence"}}""",
}

def code_one(abstract, prompt_template, retries=3):
    prompt = prompt_template.format(abstract=abstract[:3000])
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=200, system=SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            result = json.loads(raw)
            return int(result.get("direction", 0))
        except Exception as e:
            if attempt == retries - 1:
                print(f"  Error: {e}")
                return 0
            time.sleep(2)

def weighted_kappa(y_true, y_pred):
    labels = [-1, 0, 1]
    y_true = [labels.index(y) for y in y_true]
    y_pred = [labels.index(y) for y in y_pred]
    try:
        return cohen_kappa_score(y_true, y_pred, weights="linear", labels=[0,1,2])
    except Exception:
        return np.nan

# ── Run all three prompt variants ─────────────────────────────────────────────
results = []
for variant_name, prompt_template in PROMPTS.items():
    print(f"\nRunning variant: {variant_name}")
    preds = []
    for i, (_, row) in enumerate(sample.iterrows()):
        abstract = row.get("abstract", "")
        if not abstract or pd.isna(abstract):
            abstract = row.get("title", "")
        pred = code_one(abstract, prompt_template)
        preds.append(pred)
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(sample)} done")
        time.sleep(0.3)

    agreed = sum(p == t for p, t in zip(preds, true_labels))
    pct    = 100 * agreed / len(true_labels)
    kw     = weighted_kappa(true_labels, preds)

    # Per-category agreement
    for cat, label in [(-1,"Negative"), (0,"Null/mixed"), (1,"Positive")]:
        mask = [t == cat for t in true_labels]
        cat_preds  = [p for p,m in zip(preds, mask) if m]
        cat_trues  = [t for t,m in zip(true_labels, mask) if m]
        cat_agreed = sum(p == t for p, t in zip(cat_preds, cat_trues))
        print(f"  {label}: {cat_agreed}/{len(cat_trues)} ({100*cat_agreed/max(1,len(cat_trues)):.0f}%)")

    print(f"  Overall: {agreed}/{len(true_labels)} ({pct:.1f}%) | κ_w = {kw:.3f}")
    results.append({
        "Prompt variant": variant_name,
        "N":              len(true_labels),
        "N agreed":       agreed,
        "Agreement %":    round(pct, 1),
        "kappa_weighted": round(kw, 3),
    })
    sample[f"pred_{variant_name}"] = preds

out_df = pd.DataFrame(results)
out_df.to_csv(OUT, index=False)
print(f"\nSaved → {OUT}")
print(out_df.to_string(index=False))
