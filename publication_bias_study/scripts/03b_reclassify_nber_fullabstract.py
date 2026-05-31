"""
03b_reclassify_nber_fullabstract.py
------------------------------------
Re-classify NBER in-scope papers using FULL abstracts (not the truncated
~300-char API abstracts that script 03 used).

Full abstracts for the 68 in-analysis NBER papers come from analysis_dataset.parquet.
The 3 pre-2000 papers (w5256, w6451, w7403) are excluded from the analysis anyway
and are not re-classified here.

Output:
  - Prints change summary
  - Updates data/classified/inscope_papers.parquet in-place if any flips occur
  - Writes data/classified/inscope_nber_recheck.csv for audit trail
"""

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

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
WORKERS = 6

SCOPE_SYSTEM = (
    "You are a research assistant helping to classify finance research papers. "
    "You must return valid JSON only — no other text."
)

SCOPE_PROMPT_TEMPLATE = """\
A finance research paper is 'in scope' for a study of publication bias in government \
intervention research if and only if its PRIMARY research question is the CAUSAL EFFECT \
of a specific government law, regulation, or public policy on financial markets, firms, \
or investors.

Exclusion criteria (mark in_scope=false):
- Purely theoretical papers (no empirical test of a policy)
- Papers that only DESCRIBE a regulation without estimating its effect
- Papers whose main topic is private firm/market behaviour, with policy as incidental context
- Papers studying central bank communication WITHOUT a specific policy instrument

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
        model=ANTHROPIC_MODEL,
        max_tokens=200,
        system=SCOPE_SYSTEM,
        messages=[{
            "role": "user",
            "content": SCOPE_PROMPT_TEMPLATE.format(
                title=title[:300] if title else "",
                abstract=abstract
            )
        }],
    )
    raw = msg.content[0].text.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(match.group()) if match else {
            "in_scope": False, "confidence": "low", "reason": "parse error"
        }
    return result


def main():
    inscope = pd.read_parquet(ROOT / "data" / "classified" / "inscope_papers.parquet")
    ad      = pd.read_parquet(ROOT / "data" / "final" / "analysis_dataset.parquet")

    # In-analysis NBER papers with full abstracts
    nber_full = ad[ad["source"] == "nber"][["paper_id", "title", "abstract"]].copy()
    nber_full = nber_full[nber_full["abstract"].notna() & (nber_full["abstract"] != "")]
    print(f"NBER papers to re-classify: {len(nber_full)}")
    print(f"Full abstract word count: mean={nber_full['abstract'].str.split().str.len().mean():.1f}")

    # Compare with truncated versions in inscope_papers
    orig = inscope[inscope["paper_id"].isin(nber_full["paper_id"])][
        ["paper_id", "abstract", "in_scope", "scope_conf", "scope_reason"]
    ].copy()
    orig = orig.rename(columns={
        "abstract": "trunc_abstract",
        "in_scope": "orig_in_scope",
        "scope_conf": "orig_conf",
        "scope_reason": "orig_reason"
    })

    _tls = threading.local()

    def _get_client():
        if not hasattr(_tls, "client"):
            _tls.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return _tls.client

    def _classify_row(row) -> tuple:
        try:
            res = classify_one(_get_client(), str(row["title"]), str(row["abstract"]))
            return (row["paper_id"],
                    bool(res.get("in_scope", False)),
                    str(res.get("confidence", "low")),
                    str(res.get("reason", "")))
        except Exception as exc:
            print(f"\n  Error on {row['paper_id']}: {exc}")
            return row["paper_id"], True, "low", f"error: {exc}"  # keep as in-scope on error

    results_map: dict[str, tuple] = {}
    rows = [row for _, row in nber_full.iterrows()]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_classify_row, row): row["paper_id"] for row in rows}
        with tqdm(total=len(rows), desc="Re-classifying with full abstracts") as pbar:
            for future in as_completed(futures):
                pid, in_sc, conf, reason = future.result()
                results_map[pid] = (in_sc, conf, reason)
                pbar.update(1)

    nber_full["new_in_scope"] = nber_full["paper_id"].map(lambda p: results_map[p][0])
    nber_full["new_conf"]     = nber_full["paper_id"].map(lambda p: results_map[p][1])
    nber_full["new_reason"]   = nber_full["paper_id"].map(lambda p: results_map[p][2])

    # Merge with original to find flips
    merged = nber_full.merge(orig, on="paper_id", how="left")
    flipped = merged[merged["new_in_scope"] != merged["orig_in_scope"]]

    print(f"\n{'='*60}")
    print(f"Re-classification results:")
    print(f"  Still in-scope:    {(merged['new_in_scope']==True).sum()}")
    print(f"  Flipped to OUT:    {((merged['orig_in_scope']==True) & (merged['new_in_scope']==False)).sum()}")
    print(f"  Unchanged:         {(merged['new_in_scope']==merged['orig_in_scope']).sum()}")
    print(f"{'='*60}")

    if len(flipped) > 0:
        print("\nFLIPPED PAPERS:")
        for _, row in flipped.iterrows():
            print(f"\n  {row['paper_id']}: {row['title'][:60]}")
            print(f"    Old: {row['orig_in_scope']} ({row['orig_conf']}) — {row['orig_reason'][:80]}")
            print(f"    New: {row['new_in_scope']} ({row['new_conf']}) — {row['new_reason'][:80]}")
    else:
        print("\nNo classification changes — all 68 papers confirmed in-scope with full abstracts.")

    # Save audit trail
    audit_path = ROOT / "data" / "classified" / "inscope_nber_recheck.csv"
    audit = merged[["paper_id", "orig_in_scope", "orig_conf", "orig_reason",
                    "new_in_scope", "new_conf", "new_reason"]].copy()
    audit.to_csv(audit_path, index=False)
    print(f"\nAudit trail saved → {audit_path}")

    # Update inscope_papers.parquet if any flips
    if len(flipped) > 0:
        print("\nUpdating inscope_papers.parquet...")
        update_map = {row["paper_id"]: (row["new_in_scope"], row["new_conf"], row["new_reason"])
                      for _, row in merged.iterrows()}
        for pid, (new_sc, new_conf, new_reason) in update_map.items():
            mask = inscope["paper_id"] == pid
            inscope.loc[mask, "in_scope"]     = new_sc
            inscope.loc[mask, "scope_conf"]   = new_conf
            inscope.loc[mask, "scope_reason"] = new_reason
        out_path = ROOT / "data" / "classified" / "inscope_papers.parquet"
        inscope.to_parquet(out_path, index=False, compression="snappy")
        print(f"Updated → {out_path}")
    else:
        print("No updates needed to inscope_papers.parquet.")

    return flipped


if __name__ == "__main__":
    main()
