"""
03_classify_inscope.py
----------------------
Two-pass pipeline that identifies "government intervention" papers.

Pass 1 — keyword filter on title + abstract (fast, no API cost).
Pass 2 — Claude API classifies each keyword-flagged candidate.

Output: data/classified/inscope_papers.parquet

Usage:
    python scripts/03_classify_inscope.py
    python scripts/03_classify_inscope.py --skip-llm   # keyword pass only
    python scripts/03_classify_inscope.py --sample 200  # quick test on 200 papers

Prerequisites:
    ANTHROPIC_API_KEY in .env
"""

import argparse
import json
import pathlib
import os
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

DB_PATH  = ROOT / "publication_bias.duckdb"
OUT_PATH = ROOT / "data" / "classified" / "inscope_papers.parquet"

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
WORKERS = 8

# ------------------------------------------------------------------ #
# Keyword taxonomy (from implementation plan §2a)                     #
# ------------------------------------------------------------------ #

INCLUSION_KEYWORDS = [
    # Financial regulation
    "regulation", "regulatory", "dodd-frank", "basel", "glass-steagall",
    "capital requirements", "leverage ratio", "bank capital",
    # Securities law
    "sec ", "securities and exchange", "disclosure requirement",
    "reg fd", "sarbanes-oxley", "sox", "insider trading law",
    "short-selling ban", "short sale ban", "short-sale restriction",
    # Banking policy
    "fdic", "deposit insurance", "lender of last resort", "bailout",
    "tarp", "stress test", "bank resolution", "government guarantee",
    # Monetary / fiscal policy
    "quantitative easing", "qe", "fed intervention", "government subsidy",
    "public guarantee", "state guarantee", "fiscal stimulus",
    # Market microstructure rules
    "circuit breaker", "tick size", "trading halt", "uptick rule",
    "limit-down", "limit-up", "trading curb",
    # Corporate governance law
    "board independence requirement", "say-on-pay", "proxy access",
    "mandatory audit", "governance mandate", "clawback provision",
    "mandatory disclosure",
    # General government intervention signals
    "government intervention", "government policy", "public policy",
    "law and finance", "legal protection", "shareholder rights",
    "creditor rights", "investor protection", "financial regulation reform",
]

# These are too broad without additional context — exclude lone matches
AMBIGUOUS_ALONE = {"regulation", "regulatory", "sec "}


def keyword_hit(text: str) -> bool:
    t = text.lower()
    for kw in INCLUSION_KEYWORDS:
        if kw in t:
            if kw in AMBIGUOUS_ALONE:
                # require at least one other keyword too
                others = [k for k in INCLUSION_KEYWORDS
                          if k != kw and k in t]
                if others:
                    return True
            else:
                return True
    return False


# ------------------------------------------------------------------ #
# LLM classification                                                   #
# ------------------------------------------------------------------ #

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
- Papers whose primary policy variable is a rule imposed by a stock exchange (NYSE, NASDAQ, etc.) without a specific government or regulatory mandate (e.g., exchange-designed trading halts, voluntary tick size conventions, exchange fee structures)

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
def classify_one(client: Anthropic, paper_id: str,
                 combined_text: str, title: str = "") -> dict:
    abstract = combined_text[len(title):].strip() if title and combined_text.startswith(title) else combined_text
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
        # Attempt to extract JSON from messy output
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(match.group()) if match else {
            "in_scope": False, "confidence": "low",
            "reason": "parse error"
        }
    return result


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def load_all_papers(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Union all source tables into one working frame."""
    # Each source has a different id column — handle separately
    queries = {
        "nber": "SELECT wp_number AS paper_id, 'nber' AS source, title, abstract, year FROM raw_nber WHERE title IS NOT NULL AND title != ''",
        "jf":   "SELECT doi       AS paper_id, 'jf'   AS source, title, abstract, year FROM raw_jf   WHERE title IS NOT NULL AND title != ''",
        "rfs":  "SELECT doi       AS paper_id, 'rfs'  AS source, title, abstract, year FROM raw_rfs  WHERE title IS NOT NULL AND title != ''",
        "jfe":  "SELECT doi       AS paper_id, 'jfe'  AS source, title, abstract, year FROM raw_jfe  WHERE title IS NOT NULL AND title != ''",
    }
    sources = []
    for src, sql in queries.items():
        try:
            df = con.execute(sql).df()
            sources.append(df)
            print(f"  Loaded {len(df):,} papers from {src}")
        except Exception as e:
            print(f"  Warning: could not load raw_{src}: {e}")
    if not sources:
        raise RuntimeError("No source tables found. Run scripts 01 and 02 first.")
    return pd.concat(sources, ignore_index=True)


def main(skip_llm: bool, sample: int | None) -> None:
    con = duckdb.connect(str(DB_PATH))
    df  = load_all_papers(con)
    con.close()

    print(f"Loaded {len(df):,} total papers")

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
        print(f"Subsample: {len(df):,} papers")

    # --- Pass 1: keyword filter ---
    df["combined_text"] = (df["title"].fillna("") + " " +
                           df["abstract"].fillna(""))
    df["keyword_hit"] = df["combined_text"].apply(keyword_hit)
    print(f"\nPass 1 (keyword): {df['keyword_hit'].sum():,} candidates "
          f"({df['keyword_hit'].mean()*100:.1f}%)")

    if skip_llm:
        df["in_scope"]   = df["keyword_hit"]
        df["scope_conf"] = "n/a"
        df["scope_reason"] = "keyword pass only"
    else:
        # --- Pass 2: LLM classification of keyword-flagged papers ---
        candidates = df[df["keyword_hit"]].copy()
        print(f"Pass 2 (LLM): classifying {len(candidates):,} candidates "
              f"with {WORKERS} parallel workers ...")

        _tls = threading.local()

        def _get_client() -> Anthropic:
            if not hasattr(_tls, "client"):
                _tls.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            return _tls.client

        def _classify_row(row) -> tuple:
            try:
                res = classify_one(_get_client(), row["paper_id"],
                                   row["combined_text"], row.get("title", ""))
                return (row["paper_id"],
                        bool(res.get("in_scope", False)),
                        str(res.get("confidence", "low")),
                        str(res.get("reason", "")))
            except Exception as exc:
                print(f"\n  Error on {row['paper_id']}: {exc}")
                return row["paper_id"], False, "low", f"error: {exc}"

        results_map: dict[str, tuple] = {}
        rows = [row for _, row in candidates.iterrows()]

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(_classify_row, row): row["paper_id"] for row in rows}
            with tqdm(total=len(rows), desc="Classifying") as pbar:
                for future in as_completed(futures):
                    pid, in_sc, conf, reason = future.result()
                    results_map[pid] = (in_sc, conf, reason)
                    pbar.update(1)

        in_scope_list = [results_map[pid][0] for pid in candidates["paper_id"]]
        conf_list     = [results_map[pid][1] for pid in candidates["paper_id"]]
        reason_list   = [results_map[pid][2] for pid in candidates["paper_id"]]

        candidates["in_scope"]    = in_scope_list
        candidates["scope_conf"]  = conf_list
        candidates["scope_reason"]= reason_list

        # Merge back; non-keyword-hit papers are out-of-scope
        df = df.merge(
            candidates[["paper_id","in_scope","scope_conf","scope_reason"]],
            on="paper_id", how="left"
        )
        df["in_scope"]    = df["in_scope"].fillna(False)
        df["scope_conf"]  = df["scope_conf"].fillna("n/a")
        df["scope_reason"]= df["scope_reason"].fillna("failed keyword filter")

    print(f"\nIn-scope papers: {df['in_scope'].sum():,}")
    print(df[df["in_scope"]]["source"].value_counts().to_string())

    out_df = df[[
        "paper_id","source","title","abstract","year",
        "in_scope","scope_conf","scope_reason","keyword_hit"
    ]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true",
                        help="Run keyword pass only (no LLM API calls)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Limit to N papers for testing")
    args = parser.parse_args()
    main(skip_llm=args.skip_llm, sample=args.sample)
