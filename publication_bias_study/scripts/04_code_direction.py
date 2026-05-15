"""
04_code_direction.py
--------------------
For every in-scope paper, call the Claude API to code the DIRECTION
of the main finding:
    +1  — positive / beneficial effect of government intervention
    -1  — negative / harmful effect
     0  — null, mixed, or inconclusive

Output: data/classified/coded_direction.parquet

Usage:
    python scripts/04_code_direction.py
    python scripts/04_code_direction.py --sample 100   # dev test
    python scripts/04_code_direction.py --resume       # skip already-coded

Prerequisites:
    ANTHROPIC_API_KEY in .env
    data/classified/inscope_papers.parquet must exist (script 03)
"""

import argparse
import json
import os
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

ROOT = pathlib.Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")

IN_PATH  = ROOT / "data" / "classified" / "inscope_papers.parquet"
OUT_PATH = ROOT / "data" / "classified" / "coded_direction.parquet"

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
WORKERS = 8

# ------------------------------------------------------------------ #
# Prompt (from implementation plan §3a)                               #
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

DIRECTION_PROMPT_TITLE_ONLY = """\
The following is the title of a finance paper that studies the effect of a \
government law, regulation, or public policy. No abstract is available.

Based on the title alone, infer the likely direction of the main empirical finding:
  +1  The intervention likely produced a POSITIVE / BENEFICIAL outcome
  -1  The intervention likely produced a NEGATIVE / HARMFUL outcome
   0  NULL, MIXED, or cannot be determined from the title alone

Coding rules:
- Default to 0 (null) if the title is neutral or provides no directional signal
- Only assign +1 or -1 if the title clearly implies a directional finding
- Set confidence to "low" unless the directional signal is unambiguous

Title:
\"\"\"
{title}
\"\"\"

Return JSON with exactly:
{{
  "direction": 1 | -1 | 0,
  "confidence": "high" | "medium" | "low",
  "rationale": "one sentence"
}}"""


@retry(stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=1, min=2, max=30))
def code_one(client: Anthropic, text: str, title_only: bool = False) -> dict:
    prompt = (DIRECTION_PROMPT_TITLE_ONLY.format(title=text[:1200])
              if title_only
              else DIRECTION_PROMPT.format(abstract=text[:3000]))
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=200,
        system=DIRECTION_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(match.group()) if match else {
            "direction": 0, "confidence": "low",
            "rationale": "parse error"
        }
    direction = result.get("direction", 0)
    if isinstance(direction, str):
        direction = int(direction)
    result["direction"] = int(direction)
    return result


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(sample: int | None, resume: bool) -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"Missing {IN_PATH}. Run 03_classify_inscope.py first."
        )

    inscope = pd.read_parquet(IN_PATH)
    inscope = inscope[inscope["in_scope"] == True].copy()
    print(f"In-scope papers to code: {len(inscope):,}")

    already_coded: set = set()
    if resume and OUT_PATH.exists():
        prev = pd.read_parquet(OUT_PATH)
        already_coded = set(prev["paper_id"].tolist())
        print(f"Resuming: {len(already_coded):,} already coded, "
              f"{len(inscope) - len(already_coded):,} remaining")
        inscope = inscope[~inscope["paper_id"].isin(already_coded)]

    if sample:
        inscope = inscope.sample(n=min(sample, len(inscope)),
                                 random_state=42).reset_index(drop=True)
        print(f"Subsample: {len(inscope):,}")

    print(f"Coding direction with {WORKERS} parallel workers ...")

    _tls = threading.local()

    def _get_client() -> Anthropic:
        if not hasattr(_tls, "client"):
            _tls.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return _tls.client

    def _code_row(row) -> dict:
        abstract = str(row.get("abstract", ""))
        has_abstract = bool(abstract.strip()) and len(abstract.split()) >= 20
        title_only = not has_abstract
        text = abstract if has_abstract else str(row.get("title", ""))
        try:
            res = code_one(_get_client(), text, title_only=title_only)
            return {
                "paper_id":   row["paper_id"],
                "direction":  res["direction"],
                "confidence": res.get("confidence", "low"),
                "rationale":  res.get("rationale", ""),
                "human_code": None,
                "agreed":     None,
            }
        except Exception as exc:
            print(f"\n  Error on {row['paper_id']}: {exc}")
            return {
                "paper_id":   row["paper_id"],
                "direction":  0,
                "confidence": "low",
                "rationale":  f"error: {exc}",
                "human_code": None,
                "agreed":     None,
            }

    results = []
    rows = [row for _, row in inscope.iterrows()]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_code_row, row): row["paper_id"] for row in rows}
        with tqdm(total=len(rows), desc="Coding direction") as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    new_df = pd.DataFrame(results)

    # Merge with previous results if resuming
    if resume and OUT_PATH.exists():
        prev = pd.read_parquet(OUT_PATH)
        combined = pd.concat([prev, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="paper_id", keep="last")
    else:
        combined = new_df

    # Cast direction to Int8
    combined["direction"] = combined["direction"].astype("Int8")

    print(f"\nDirection distribution:")
    print(combined["direction"].value_counts().sort_index().to_string())
    print(f"Confidence distribution:")
    print(combined["confidence"].value_counts().to_string())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"\nSaved → {OUT_PATH}  ({len(combined):,} records)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-coded papers")
    args = parser.parse_args()
    main(sample=args.sample, resume=args.resume)
