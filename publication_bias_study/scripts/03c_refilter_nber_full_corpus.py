"""
03c_refilter_nber_full_corpus.py
----------------------------------
Re-run Pass 1 (keyword filter) and Pass 2 (LLM classification) on NBER
papers using their newly fetched full abstracts.

The original Pass 1 ran on ~300-char (~45-word) truncated abstracts from
the NBER search API. After 01b_fetch_nber_full_abstracts.py updates raw_nber
with full abstracts, this script identifies NBER papers that now pass the
keyword filter but did NOT pass on their truncated abstracts, and sends
those new candidates through the LLM scope classifier.

Outputs:
  - Prints summary of new candidates and new in-scope papers
  - Updates data/classified/inscope_papers.parquet with new in-scope papers
  - Writes data/classified/nber_refilter_newcandidates.csv for audit

Usage:
    python scripts/03c_refilter_nber_full_corpus.py
    python scripts/03c_refilter_nber_full_corpus.py --skip-llm   # keyword pass only
"""

import argparse
import json
import os
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

ROOT = pathlib.Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")

DB_PATH        = ROOT / "publication_bias.duckdb"
INSCOPE_PATH   = ROOT / "data" / "classified" / "inscope_papers.parquet"
AUDIT_PATH     = ROOT / "data" / "classified" / "nber_refilter_newcandidates.csv"

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
WORKERS = 8

# Import keyword filter from script 03
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module
_s03 = import_module("03_classify_inscope".replace("-", "_"))
keyword_hit = _s03.keyword_hit

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
- Papers whose primary policy variable is a rule imposed by a stock exchange (NYSE, NASDAQ, \
etc.) without a specific government or regulatory mandate (e.g., exchange-designed trading \
halts, voluntary tick size conventions, exchange fee structures)

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
        messages=[{"role": "user", "content": SCOPE_PROMPT_TEMPLATE.format(
            title=title[:300],
            abstract=abstract
        )}],
    )
    raw = msg.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(match.group()) if match else {
            "in_scope": False, "confidence": "low", "reason": "parse error"
        }


def main(skip_llm: bool) -> None:
    # --- Load current state ---
    con = duckdb.connect(str(DB_PATH))
    nber_df = con.execute(
        "SELECT wp_number AS paper_id, title, abstract, year FROM raw_nber "
        "WHERE title IS NOT NULL AND title != ''"
    ).df()
    con.close()

    inscope = pd.read_parquet(INSCOPE_PATH)
    already_hit = set(inscope[(inscope["source"] == "nber") & (inscope["keyword_hit"] == True)]["paper_id"])

    print(f"Total NBER papers in raw_nber:     {len(nber_df):,}")
    print(f"Already processed by keyword hit:  {len(already_hit):,}")

    # --- Re-run keyword filter on full abstracts ---
    nber_df["combined_text"] = (
        nber_df["title"].fillna("") + " " + nber_df["abstract"].fillna("")
    )
    nber_df["kw_hit_full"] = nber_df["combined_text"].apply(keyword_hit)

    new_candidates = nber_df[
        nber_df["kw_hit_full"] & ~nber_df["paper_id"].isin(already_hit)
    ].copy()

    print(f"\nPass 1 (full abstract) total hits: {nber_df['kw_hit_full'].sum():,}")
    print(f"  Already in system:               {nber_df['kw_hit_full'].sum() - len(new_candidates):,}")
    print(f"  NEW candidates (not seen before):{len(new_candidates):,}")

    if len(new_candidates) == 0:
        print("\nNo new candidates — full abstracts didn't add any new keyword hits.")
        return

    if skip_llm:
        print("\n--skip-llm: not running LLM classification.")
        new_candidates["in_scope"] = True
        new_candidates["scope_conf"] = "n/a"
        new_candidates["scope_reason"] = "keyword pass only"
    else:
        # --- Pass 2: LLM classification ---
        print(f"\nPass 2 (LLM): classifying {len(new_candidates):,} new candidates...")
        _tls = threading.local()

        def _get_client() -> Anthropic:
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
                return row["paper_id"], False, "low", f"error: {exc}"

        results_map: dict[str, tuple] = {}
        rows = [row for _, row in new_candidates.iterrows()]

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(_classify_row, row): row["paper_id"] for row in rows}
            with tqdm(total=len(rows), desc="Classifying new NBER candidates") as pbar:
                for future in as_completed(futures):
                    pid, in_sc, conf, reason = future.result()
                    results_map[pid] = (in_sc, conf, reason)
                    pbar.update(1)

        new_candidates["in_scope"]     = new_candidates["paper_id"].map(lambda p: results_map[p][0])
        new_candidates["scope_conf"]   = new_candidates["paper_id"].map(lambda p: results_map[p][1])
        new_candidates["scope_reason"] = new_candidates["paper_id"].map(lambda p: results_map[p][2])

    n_new_inscope = new_candidates["in_scope"].sum()
    print(f"\nNew candidates: {len(new_candidates)}")
    print(f"New in-scope:   {n_new_inscope}")

    if n_new_inscope > 0:
        print("\nNEW IN-SCOPE PAPERS:")
        for _, row in new_candidates[new_candidates["in_scope"]].iterrows():
            print(f"  {row['paper_id']} ({row.get('year','?')}): {str(row['title'])[:70]}")
            print(f"    Conf: {row['scope_conf']} — {row['scope_reason'][:80]}")

    # --- Save audit trail ---
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_candidates[["paper_id","title","year","in_scope","scope_conf","scope_reason"]].to_csv(
        AUDIT_PATH, index=False
    )
    print(f"\nAudit trail → {AUDIT_PATH}")

    # --- Update inscope_papers.parquet ---
    if n_new_inscope > 0:
        new_rows = new_candidates[new_candidates["in_scope"]].copy()
        new_rows["source"] = "nber"
        new_rows["keyword_hit"] = True
        new_rows = new_rows.rename(columns={"scope_conf": "scope_conf", "scope_reason": "scope_reason"})

        cols = ["paper_id","source","title","abstract","year",
                "in_scope","scope_conf","scope_reason","keyword_hit"]
        new_rows = new_rows[[c for c in cols if c in new_rows.columns]]
        for c in cols:
            if c not in new_rows.columns:
                new_rows[c] = None

        updated = pd.concat([inscope, new_rows[cols]], ignore_index=True)
        updated.to_parquet(INSCOPE_PATH, index=False, compression="snappy")
        print(f"Updated inscope_papers.parquet: {len(inscope)} → {len(updated)} rows")
        print("\nNEXT STEPS:")
        print("  1. Manually review new in-scope papers (see audit CSV above)")
        print("  2. Run 04_code_direction.py for new papers only")
        print("  3. Run 05_link_nber_to_published.py to check publication status")
        print("  4. Re-run 06_analysis_main.py and 08_robustness.py")
    else:
        print("\nNo new in-scope papers found — analysis dataset unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    main(skip_llm=args.skip_llm)
