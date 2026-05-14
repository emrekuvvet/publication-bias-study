"""
00_validate_direction_coding.py
--------------------------------
Validate LLM direction codes against a second independent LLM pass.

Protocol:
  1. Randomly sample 60 papers from analysis_dataset.parquet
     (stratified by source × direction, seed=42).
  2. Strip the orignal codes and re-run direction coding with the
     same prompt but fresh context (temperature=0).
  3. Compute percent agreement and linear-weighted Cohen's kappa
     between original and replication codes.
  4. Save:
       output/tables/table_validation_irf.csv   — summary stats
       output/tables/table_validation_cm.csv    — confusion matrix
       validation/validation_sample.parquet     — full sample with both codes

Usage:
    python3 validation/00_validate_direction_coding.py
"""

import os, json, pathlib, time, random
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Try anthropic SDK
try:
    import anthropic
except ImportError:
    raise SystemExit("pip install anthropic")

# Try to load .env
try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).parent.parent / ".env")
except ImportError:
    pass

ROOT     = pathlib.Path(__file__).parent.parent.resolve()
DATA     = ROOT / "data" / "final" / "analysis_dataset.parquet"
OUT_DIR  = ROOT / "output" / "tables"
VAL_DIR  = ROOT / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIRECTION_PROMPT = """\
The following is the abstract of a finance paper that studies the effect \
of a government law, regulation, or public policy. Classify the MAIN \
empirical finding as:
  +1  if the government intervention produced a POSITIVE or BENEFICIAL \
outcome (e.g., improved market quality, reduced systemic risk, better \
investor protection);
  -1  if it produced a NEGATIVE or HARMFUL outcome (e.g., reduced \
liquidity, higher costs, unintended consequences, welfare losses);
   0  if the finding is NULL, MIXED, or INCONCLUSIVE.

Coding rule: code the sign of the estimated coefficient on the primary \
intervention variable, NOT the authors' normative framing.

Return JSON with exactly three keys:
  direction  : integer (1, -1, or 0)
  confidence : string ("high", "medium", or "low")
  rationale  : one sentence

Abstract:
{abstract}"""


def code_one(client, abstract: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": DIRECTION_PROMPT.format(abstract=abstract[:2000])
                }]
            )
            text = msg.content[0].text.strip()
            # Strip markdown code fences
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Failed after {max_retries} attempts: {exc}")
                return {"direction": 0, "confidence": "low", "rationale": "API error"}


def weighted_kappa(y1, y2):
    return cohen_kappa_score(y1, y2, labels=[-1, 0, 1], weights="linear")


def main():
    print("Loading analysis dataset …")
    df = pd.read_parquet(DATA)
    df = df[df["abstract"].notna() & (df["abstract"].str.len() > 20)].copy()
    df["direction"] = pd.to_numeric(df["direction"], errors="coerce")
    df = df[df["direction"].notna()].copy()

    # Stratified sample: 10 per (source × direction) cell, capped at available
    random.seed(42)
    np.random.seed(42)
    sample_ids = []
    for src in ["jf", "rfs", "jfe", "nber"]:
        for d in [-1, 0, 1]:
            sub = df[(df["source"] == src) & (df["direction"] == d)]
            n = min(5, len(sub))
            if n > 0:
                sample_ids.extend(sub.sample(n, random_state=42)["paper_id"].tolist())

    sample = df[df["paper_id"].isin(sample_ids)].copy().reset_index(drop=True)
    print(f"Validation sample: {len(sample)} papers")
    print(f"  Source distribution: {sample['source'].value_counts().to_dict()}")
    print(f"  Original direction distribution: {sample['direction'].value_counts().to_dict()}")

    # Run second-pass coding
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rep_directions = []
    rep_confidence = []

    for i, row in sample.iterrows():
        result = code_one(client, str(row["abstract"]))
        rep_directions.append(result.get("direction", 0))
        rep_confidence.append(result.get("confidence", "low"))
        if (i + 1) % 10 == 0:
            print(f"  Coded {i+1}/{len(sample)} …")
        time.sleep(0.3)

    sample["direction_rep"] = rep_directions
    sample["confidence_rep"] = rep_confidence
    sample["agreed"] = (sample["direction"] == sample["direction_rep"]).astype(int)

    # Agreement statistics
    orig = sample["direction"].astype(int).values
    repl = np.array(rep_directions, dtype=int)

    pct  = float((orig == repl).mean())
    kap  = weighted_kappa(orig, repl)
    cm   = confusion_matrix(orig, repl, labels=[-1, 0, 1])

    print(f"\nReplication reliability ({len(sample)} papers):")
    print(f"  Percent agreement : {pct*100:.1f}%")
    print(f"  Linear-weighted κ : {kap:.3f}")
    print("\nConfusion matrix (rows = original, cols = replication):")
    cm_df = pd.DataFrame(cm,
                         index=["Orig −1", "Orig 0", "Orig +1"],
                         columns=["Rep −1", "Rep 0", "Rep +1"])
    print(cm_df.to_string())

    # Save summary
    summary = pd.DataFrame([{
        "N": len(sample),
        "Percent agreement (%)": round(pct * 100, 1),
        "Linear-weighted kappa": round(kap, 3),
        "Meets threshold (κ≥0.70)": "YES" if kap >= 0.70 else "NO",
    }])
    summary.to_csv(OUT_DIR / "table_validation_irf.csv", index=False)
    cm_df.to_csv(OUT_DIR / "table_validation_cm.csv")
    sample.to_parquet(VAL_DIR / "validation_sample.parquet", index=False)

    print(f"\nSaved summary → {OUT_DIR / 'table_validation_irf.csv'}")
    print(f"Saved confusion matrix → {OUT_DIR / 'table_validation_cm.csv'}")
    print(f"Saved full sample → {VAL_DIR / 'validation_sample.parquet'}")


if __name__ == "__main__":
    main()
