"""
11_afa_analysis.py
------------------
Full AFA-baseline analysis pipeline.  Replaces NBER as the pre-publication
reference population with AFA annual meeting papers (2006-2024).

Steps
-----
  A  classify  -- keyword filter + LLM in-scope classification
                  (title+abstract when abstract available; title-only otherwise)
  B  direction -- LLM direction coding (abstract-based when available; title otherwise)
  C  match     -- fuzzy-match AFA papers to JF/RFS/JFE publications
  D  analyze   -- binary directional probit (same spec as script 06)

Usage:
    python scripts/11_afa_analysis.py --step all           # LLM (~$15)
    python scripts/11_afa_analysis.py --step all --no-api  # FREE: keyword rules only
    python scripts/11_afa_analysis.py --step classify
    python scripts/11_afa_analysis.py --step direction
    python scripts/11_afa_analysis.py --step match
    python scripts/11_afa_analysis.py --step analyze

Abstract enrichment:
    Run scripts/10b_enrich_afa_abstracts.py first (produces afa_papers_enriched.parquet).
    When that file exists, this script automatically loads it and applies the same
    title+abstract keyword filter as the NBER pipeline (script 03).  When it does
    not exist, the script falls back to afa_papers.parquet and title-only filtering.

--no-api mode (free):
    Step A uses keyword filter only (no LLM scope check).
    Step B uses analysis_dataset direction codes for matched papers; rule-based otherwise.
    Steps C and D are unchanged (local computation only).

Prerequisites:
    data/raw/afa_papers.parquet (from 10_collect_afa.py)
    data/raw/afa_papers_enriched.parquet (from 10b_enrich_afa_abstracts.py) [optional]
    data/raw/{jf,rfs,jfe}_papers.parquet
    ANTHROPIC_API_KEY in .env (only needed without --no-api)
"""

import argparse
import json
import pathlib
import os
import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from anthropic import Anthropic
from dotenv import load_dotenv
from rapidfuzz import fuzz, process
from scipy import stats
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

ROOT = pathlib.Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")

# Input paths — prefer enriched parquet (has abstracts) when available
AFA_ENRICHED = ROOT / "data" / "raw" / "afa_papers_enriched.parquet"
AFA_RAW      = ROOT / "data" / "raw" / "afa_papers.parquet"

# Intermediate
AFA_SCOPE     = ROOT / "data" / "classified" / "afa_inscope.parquet"
AFA_DIRECTION = ROOT / "data" / "classified" / "afa_direction.parquet"
AFA_MATCHED   = ROOT / "data" / "classified" / "afa_matched.parquet"

# Output tables
OUT_DIR = ROOT / "output" / "tables"

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
WORKERS = 8

FUZZY_THRESHOLD = 85.0   # slightly tighter than NBER (85 vs 80) for AFA

# ------------------------------------------------------------------ #
# Shared text helpers                                                  #
# ------------------------------------------------------------------ #

STOP_WORDS = {
    "a","an","the","of","in","on","at","to","for","and","or","but",
    "with","by","from","that","this","is","are","was","were","be",
    "been","being","have","has","had","do","does","did","will","would",
    "can","could","should","may","might","shall","some","any","all",
    "its","it","we","our","their","there","when","where","which","who",
}


def normalise_title(title: str) -> str:
    if not title:
        return ""
    nfd = unicodedata.normalize("NFD", str(title))
    ascii_str = nfd.encode("ascii", "ignore").decode("ascii")
    clean = re.sub(r"[^a-z0-9\s]", " ", ascii_str.lower())
    tokens = [t for t in clean.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


# ------------------------------------------------------------------ #
# Step A — Keyword filter + LLM in-scope classification               #
# ------------------------------------------------------------------ #

# Reuse keyword taxonomy from script 03
INCLUSION_KEYWORDS = [
    "regulation", "regulatory", "dodd-frank", "basel", "glass-steagall",
    "capital requirements", "leverage ratio", "bank capital",
    "sec ", "securities and exchange", "disclosure requirement",
    "reg fd", "sarbanes-oxley", "sox", "insider trading law",
    "short-selling ban", "short sale ban", "short-sale restriction",
    "fdic", "deposit insurance", "lender of last resort", "bailout",
    "tarp", "stress test", "bank resolution", "government guarantee",
    "quantitative easing", "qe", "fed intervention", "government subsidy",
    "public guarantee", "state guarantee", "fiscal stimulus",
    "circuit breaker", "tick size", "trading halt", "uptick rule",
    "limit-down", "limit-up", "trading curb",
    "board independence requirement", "say-on-pay", "proxy access",
    "mandatory audit", "governance mandate", "clawback provision",
    "mandatory disclosure",
    "government intervention", "government policy", "public policy",
    "law and finance", "legal protection", "shareholder rights",
    "creditor rights", "investor protection", "financial regulation reform",
]

AMBIGUOUS_ALONE = {"regulation", "regulatory", "sec "}

# Broader keyword set for title-only mode (no abstract available)
# Includes single-word policy signals that are too vague in abstracts but fine in titles
TITLE_ONLY_EXTRA = [
    "deregulation", "reform", "enforcement", "legislation", "mandate",
    "compliance", "oversight", "supervision", "anti-takeover",
    "federal reserve", "central bank", "cfpb", "finra",
    "tax ", " tax", "subsidy", "subsidies", "tariff",
    "sanctions", "antitrust", "merger law", "securities law",
    "banking law", "short sale", "proxy rule", "clawback",
    "fiduciary", "volcker", "mifid", "gdpr",
]


def keyword_hit_title_only(title: str) -> bool:
    """
    Looser keyword filter for AFA title-only mode.
    Accepts ambiguous-alone keywords and adds extra policy signal words.
    """
    t = title.lower()
    # Accept all base keywords (including ambiguous ones — titles are shorter so
    # a lone "regulation" in a title is a stronger signal than in an abstract)
    for kw in INCLUSION_KEYWORDS:
        if kw in t:
            return True
    for kw in TITLE_ONLY_EXTRA:
        if kw in t:
            return True
    return False


# ---- Title-only keyword sets (short, unambiguous signals) ----
_POSITIVE_TITLE = [
    "improves", "improvement", "beneficial", "benefit", "benefits",
    "enhances", "enhancement", "strengthens", "boosts", "stabilizes",
    "reduces risk", "lowers cost", "reduces cost", "promotes",
    "protects investors", "better governance", "positive effect",
    "increases efficiency", "improves market", "improves liquidity",
    "reduces systemic", "reduces fraud",
]
_NEGATIVE_TITLE = [
    "reduces liquidity", "reduces market quality", "harms", "harmful",
    "distortion", "distorts", "crowds out", "crowding out",
    "unintended consequences", "unintended effects",
    "increases costs", "costly regulation", "costly compliance",
    "decreases", "impairs", "burdens", "worsens",
    "negative effect", "negative consequences",
    "deadweight", "inefficient", "regulatory burden",
    "adverse effect", "adverse effects",
]

# ---- Abstract keyword sets (richer; abstracts state findings explicitly) ----
# Null signals — check first; override directional signals when present
_ABS_NULL = [
    "no significant effect", "no significant impact", "no significant association",
    "not statistically significant", "statistically insignificant",
    "no evidence of", "find no evidence", "we find no",
    "cannot reject the null", "fail to reject",
    "mixed evidence", "mixed results", "inconclusive",
    "no systematic effect", "no robust evidence",
    "insignificant effect", "not significant",
]
_ABS_POSITIVE = [
    "positive effect", "positive impact", "positive association",
    "significantly positive", "positive and significant",
    "beneficial effect", "beneficial impact",
    "improves liquidity", "improves market quality", "improves efficiency",
    "increases firm value", "increases market quality", "increases liquidity",
    "reduces systemic risk", "lowers cost of capital", "lower cost of capital",
    "better investor protection", "improved investor protection",
    "enhances market", "strengthens market", "significant increase in",
    "significant positive", "we find that.*increas",
]
_ABS_NEGATIVE = [
    "negative effect", "negative impact", "negative association",
    "significantly negative", "negative and significant",
    "reduces liquidity", "decreased liquidity", "lower liquidity",
    "unintended consequences", "unintended consequence",
    "welfare loss", "deadweight loss",
    "increases costs", "higher compliance cost", "costly regulation",
    "distorts", "crowding out", "crowds out",
    "harmful effect", "harmful impact",
    "adverse effect", "adverse impact",
    "significant decrease in", "significant reduction in",
    "regulatory burden", "negative and significant",
]


def rule_based_direction(title: str) -> int:
    """
    Code direction ∈ {-1, 0, 1} from title keywords alone.
    Conservative: returns 0 unless a keyword is unambiguous.
    Most finance titles ("The Effect of X on Y") correctly return 0.
    """
    t = title.lower()
    pos = any(kw in t for kw in _POSITIVE_TITLE)
    neg = any(kw in t for kw in _NEGATIVE_TITLE)
    if pos and not neg:
        return 1
    if neg and not pos:
        return -1
    return 0


def rule_based_direction_abstract(abstract: str) -> int:
    """
    Code direction ∈ {-1, 0, 1} from abstract text.
    More accurate than title-only because abstracts state findings explicitly.
    Null signals take precedence over directional signals (conservative on ambiguity).
    """
    t = abstract.lower()
    if any(kw in t for kw in _ABS_NULL):
        return 0
    pos = any(kw in t for kw in _ABS_POSITIVE)
    neg = any(kw in t for kw in _ABS_NEGATIVE)
    if pos and not neg:
        return 1
    if neg and not pos:
        return -1
    return 0


def rule_based_direction_best(title: str, abstract: str = "") -> int:
    """
    Use abstract-based rules when abstract is available (≥20 words),
    fall back to title-only rules otherwise. Mirrors the LLM prompt hierarchy.
    """
    if abstract and len(abstract.split()) >= 20:
        return rule_based_direction_abstract(abstract)
    return rule_based_direction(title)


def keyword_hit(title: str) -> bool:
    t = title.lower()
    for kw in INCLUSION_KEYWORDS:
        if kw in t:
            if kw in AMBIGUOUS_ALONE:
                others = [k for k in INCLUSION_KEYWORDS if k != kw and k in t]
                if others:
                    return True
            else:
                return True
    return False


SCOPE_SYSTEM = (
    "You are a research assistant helping to classify finance research papers. "
    "You must return valid JSON only — no other text."
)

AFA_SCOPE_PROMPT_ABSTRACT = """\
The following is the title and abstract of a finance paper presented at the AFA
annual meeting. Classify whether this paper's primary research question is the
CAUSAL EFFECT of a government law, regulation, or public policy on financial
markets, firms, or investors.

Return in_scope=true if the paper clearly studies a government-policy causal-effect.
If uncertain, return in_scope=false (conservative default).

Title: \"\"\"{title}\"\"\"
Abstract: \"\"\"{abstract}\"\"\"

Return JSON with exactly these keys:
{{
  "in_scope": true or false,
  "confidence": "high" | "medium" | "low",
  "reason": "one sentence"
}}"""

AFA_SCOPE_PROMPT_TITLE_ONLY = """\
The following is the TITLE of a finance paper presented at the AFA annual meeting.
No abstract is available. Based on the title alone, classify whether this paper's
primary research question is likely the CAUSAL EFFECT of a government law,
regulation, or public policy on financial markets, firms, or investors.

Return in_scope=true ONLY if the title CLEARLY INDICATES a government-policy
causal-effect study. If uncertain, return in_scope=false (conservative default).

Title:
\"\"\"{title}\"\"\"

Return JSON with exactly these keys:
{{
  "in_scope": true or false,
  "confidence": "high" | "medium" | "low",
  "reason": "one sentence"
}}"""


@retry(stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=1, min=2, max=30))
def classify_one_afa(client: Anthropic, title: str, abstract: str = "") -> dict:
    if abstract and len(abstract.split()) >= 20:
        prompt = AFA_SCOPE_PROMPT_ABSTRACT.format(
            title=title[:400], abstract=abstract[:1200]
        )
    else:
        prompt = AFA_SCOPE_PROMPT_TITLE_ONLY.format(title=title[:500])

    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=200,
        system=SCOPE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group()) if m else {
            "in_scope": False, "confidence": "low", "reason": "parse error"
        }


def _load_afa_papers() -> pd.DataFrame:
    """Load AFA papers, preferring enriched parquet (with abstracts) when available."""
    if AFA_ENRICHED.exists():
        df = pd.read_parquet(AFA_ENRICHED)
        n_abstract = (df.get("abstract", pd.Series(dtype=str)).str.len() > 0).sum()
        print(f"  Loaded {len(df):,} AFA papers from enriched parquet "
              f"({n_abstract:,} with abstracts)")
    else:
        df = pd.read_parquet(AFA_RAW)
        df["abstract"]        = ""
        df["abstract_source"] = "none"
        print(f"  Loaded {len(df):,} AFA papers (no abstracts — run 10b_enrich_afa_abstracts.py)")
    return df


def step_classify(no_api: bool = False) -> None:
    print("=== Step A: Keyword filter + in-scope classification ===")
    df = _load_afa_papers()

    # Build combined text (title + abstract) — same approach as script 03
    df["abstract"] = df.get("abstract", pd.Series("", index=df.index)).fillna("")
    df["combined_text"] = df["title"].fillna("") + " " + df["abstract"]
    df["has_abstract"]  = df["abstract"].str.split().str.len() >= 20

    # Standard keyword filter on combined text (identical to script 03)
    df["keyword_hit"] = df["combined_text"].apply(keyword_hit)
    # Broader title-only filter for papers without abstracts (catches policy signals
    # that are unambiguous in short titles but too broad in full text)
    df["keyword_hit_title"] = df.apply(
        lambda r: keyword_hit_title_only(r["title"]) if not r["has_abstract"] else False,
        axis=1,
    )
    df["any_keyword_hit"] = df["keyword_hit"] | df["keyword_hit_title"]

    kw_n = df["keyword_hit"].sum()
    kw_title_n = df["keyword_hit_title"].sum()
    total_hits = df["any_keyword_hit"].sum()
    print(f"  Keyword hits (title+abstract): {kw_n:,} ({kw_n/len(df)*100:.1f}%)")
    print(f"  Broader title-only hits (no abstract): {kw_title_n:,}")
    print(f"  Total keyword candidates: {total_hits:,} ({total_hits/len(df)*100:.1f}%)")

    candidates = df[df["any_keyword_hit"]].copy().reset_index(drop=True)

    if no_api:
        candidates["in_scope"]   = True
        candidates["confidence"] = "keyword_only"
        candidates["reason"]     = "keyword filter (title+abstract where available) — no LLM (--no-api)"
        print(f"  --no-api: accepting all {len(candidates):,} keyword hits as in-scope")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
        client = Anthropic(api_key=api_key)
        lock   = threading.Lock()
        results: dict[str, dict] = {}

        def classify_row(row):
            res = classify_one_afa(client, row["title"],
                                   row.get("abstract", "") or "")
            with lock:
                results[row["paper_id"]] = res

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(classify_row, r): r["paper_id"]
                    for _, r in candidates.iterrows()}
            for f in tqdm(as_completed(futs), total=len(futs),
                          desc="LLM classify"):
                f.result()

        candidates["in_scope"]   = candidates["paper_id"].map(
            lambda pid: results.get(pid, {}).get("in_scope", False))
        candidates["confidence"] = candidates["paper_id"].map(
            lambda pid: results.get(pid, {}).get("confidence", "low"))
        candidates["reason"]     = candidates["paper_id"].map(
            lambda pid: results.get(pid, {}).get("reason", ""))

    # title_only = True when no usable abstract was available at classification time
    candidates["title_only"] = ~candidates["has_abstract"]

    inscope = candidates[candidates["in_scope"]].copy()
    if not no_api:
        with_abs = inscope["has_abstract"].sum()
        print(f"  In-scope after LLM: {len(inscope):,} / {len(candidates):,}")
        print(f"  In-scope with abstract: {with_abs:,} / {len(inscope):,}")

    AFA_SCOPE.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(AFA_SCOPE, index=False, compression="snappy")
    print(f"  Saved → {AFA_SCOPE}")


# ------------------------------------------------------------------ #
# Step B — Direction coding (abstract-based when available; title     #
#           fallback when not)                                        #
# ------------------------------------------------------------------ #

DIRECTION_SYSTEM = (
    "You are a research assistant reading finance research papers. "
    "Return valid JSON only — no other text."
)

# Abstract-based prompt — identical to script 04 DIRECTION_PROMPT
AFA_DIRECTION_PROMPT_ABSTRACT = """\
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

# Title-only fallback — used when no abstract is available
AFA_DIRECTION_PROMPT_TITLE = """\
The following is the TITLE of a finance paper that studies the effect of a
government law, regulation, or public policy. No abstract is available.
Based on the title alone, classify the LIKELY direction of the main finding.

  +1  Title strongly implies a POSITIVE / BENEFICIAL effect of the intervention
  -1  Title strongly implies a NEGATIVE / HARMFUL effect
   0  Title is neutral, ambiguous, or does not reveal direction

IMPORTANT: Use direction=0 (null/unclear) whenever the title does not clearly
signal a direction. Neutral titles like "The Effect of X on Y" should be coded 0.
Only assign +1 or -1 when the direction is unmistakable from the title itself.

Title:
\"\"\"{title}\"\"\"

Return JSON with exactly these keys:
{{
  "direction": 1 | -1 | 0,
  "confidence": "high" | "medium" | "low",
  "rationale": "one sentence"
}}"""


@retry(stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=1, min=2, max=30))
def code_direction_afa(client: Anthropic, title: str, abstract: str = "") -> dict:
    if abstract and len(abstract.split()) >= 20:
        prompt = AFA_DIRECTION_PROMPT_ABSTRACT.format(abstract=abstract[:1500])
    else:
        prompt = AFA_DIRECTION_PROMPT_TITLE.format(title=title[:500])

    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=200,
        system=DIRECTION_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group()) if m else {
            "direction": 0, "confidence": "low", "rationale": "parse error"
        }


def step_direction(no_api: bool = False) -> None:
    print("=== Step B: Direction coding ===")
    if not AFA_SCOPE.exists():
        raise FileNotFoundError(f"{AFA_SCOPE} not found — run --step classify first")

    scope_df = pd.read_parquet(AFA_SCOPE)
    inscope  = scope_df[scope_df["in_scope"]].copy().reset_index(drop=True)
    inscope["abstract"] = inscope.get("abstract", pd.Series("", index=inscope.index)).fillna("")
    with_abs = (inscope["abstract"].str.split().str.len() >= 20).sum()
    print(f"  In-scope papers to code: {len(inscope):,} "
          f"({with_abs:,} with abstract, {len(inscope)-with_abs:,} title-only)")

    if no_api:
        print("  --no-api: direction from analysis_dataset for matched papers; "
              "abstract-based rules for papers with abstracts; title rules otherwise")

        analysis_path = ROOT / "data" / "final" / "analysis_dataset.parquet"
        dir_map: dict[str, int] = {}

        if analysis_path.exists():
            adf = pd.read_parquet(analysis_path)
            coded = adf[
                adf["source"].isin(["jf", "rfs", "jfe"])
                & adf["direction"].notna()
            ].copy()
            j_index: dict[str, str] = {
                row["paper_id"]: normalise_title(row.get("title", ""))
                for _, row in coded.iterrows()
                if normalise_title(row.get("title", ""))
            }
            dir_lookup = coded.set_index("paper_id")["direction"]

            n_dataset = 0
            n_abstract = 0
            n_title    = 0

            for _, afa_row in inscope.iterrows():
                pid      = afa_row["paper_id"]
                title    = afa_row["title"]
                abstract = afa_row.get("abstract", "") or ""
                afa_norm = normalise_title(title)

                # Priority 1: direction from analysis_dataset fuzzy match
                matched_dir = None
                if afa_norm and j_index:
                    candidates = process.extract(
                        afa_norm, j_index,
                        scorer=fuzz.token_sort_ratio,
                        limit=3,
                    )
                    for _val, score, paper_id in candidates:
                        if score >= FUZZY_THRESHOLD and paper_id in dir_lookup.index:
                            matched_dir = int(dir_lookup[paper_id])
                            break

                if matched_dir is not None:
                    dir_map[pid] = matched_dir
                    n_dataset += 1
                elif abstract and len(abstract.split()) >= 20:
                    # Priority 2: abstract-based keyword rules (same logic as LLM prompt)
                    dir_map[pid] = rule_based_direction_abstract(abstract)
                    n_abstract += 1
                else:
                    # Priority 3: title-only keyword rules (last resort)
                    dir_map[pid] = rule_based_direction(title)
                    n_title += 1

            print(f"  Direction source: {n_dataset} from dataset match, "
                  f"{n_abstract} from abstract rules, {n_title} from title rules")
        else:
            print("  Warning: analysis_dataset.parquet not found — using abstract/title rules only")
            for _, afa_row in inscope.iterrows():
                abstract = afa_row.get("abstract", "") or ""
                dir_map[afa_row["paper_id"]] = rule_based_direction_best(
                    afa_row["title"], abstract
                )

        inscope["direction"]     = inscope["paper_id"].map(dir_map).fillna(0).astype(int)
        inscope["dir_conf"]      = "dataset_match_or_abstract_rules_or_title_rules"
        inscope["dir_rationale"] = ("analysis_dataset abstract coding if matched; "
                                    "abstract keyword rules if abstract available; "
                                    "title keyword rules otherwise")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
        client = Anthropic(api_key=api_key)
        lock   = threading.Lock()
        results: dict[str, dict] = {}

        def code_row(row):
            res = code_direction_afa(client, row["title"],
                                     row.get("abstract", "") or "")
            with lock:
                results[row["paper_id"]] = res

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(code_row, r): r["paper_id"]
                    for _, r in inscope.iterrows()}
            for f in tqdm(as_completed(futs), total=len(futs),
                          desc="LLM direction"):
                f.result()

        inscope["direction"]   = inscope["paper_id"].map(
            lambda pid: results.get(pid, {}).get("direction", 0))
        inscope["dir_conf"]    = inscope["paper_id"].map(
            lambda pid: results.get(pid, {}).get("confidence", "low"))
        inscope["dir_rationale"] = inscope["paper_id"].map(
            lambda pid: results.get(pid, {}).get("rationale", ""))

        n_abs_coded = (
            inscope["abstract"].str.split().str.len() >= 20
        ).sum()
        n_title_coded = len(inscope) - n_abs_coded
        print(f"  Coded from abstract: {n_abs_coded:,} | from title only: {n_title_coded:,}")

    counts = inscope["direction"].value_counts()
    print(f"  Direction distribution: {counts.to_dict()}")
    print(f"  Directional (±1): {(inscope['direction'] != 0).sum():,}")

    AFA_DIRECTION.parent.mkdir(parents=True, exist_ok=True)
    inscope.to_parquet(AFA_DIRECTION, index=False, compression="snappy")
    print(f"  Saved → {AFA_DIRECTION}")


# ------------------------------------------------------------------ #
# Step C — Fuzzy-match AFA papers to JF/RFS/JFE publications          #
# ------------------------------------------------------------------ #

def step_match() -> None:
    print("=== Step C: Fuzzy-match AFA → JF/RFS/JFE ===")
    if not AFA_DIRECTION.exists():
        raise FileNotFoundError(f"{AFA_DIRECTION} not found — run --step direction first")

    afa_df = pd.read_parquet(AFA_DIRECTION)
    print(f"  AFA in-scope papers: {len(afa_df):,}")

    # Load all journal paper titles
    journal_records = []
    for src in ["jf", "rfs", "jfe"]:
        rpath = ROOT / "data" / "raw" / f"{src}_papers.parquet"
        if not rpath.exists():
            print(f"  Warning: {rpath} not found — skipping {src}")
            continue
        jdf = pd.read_parquet(rpath, columns=["doi", "title", "year"])
        jdf["journal"] = src
        journal_records.append(jdf)

    if not journal_records:
        raise FileNotFoundError("No journal parquet files found in data/raw/")

    journals = pd.concat(journal_records, ignore_index=True)
    journals = journals.dropna(subset=["title"])
    print(f"  Journal papers (JF+RFS+JFE): {len(journals):,}")

    # Build normalised title index for journals
    j_norm_map: dict[str, str] = {}   # doi → normalised_title
    for _, row in journals.iterrows():
        j_norm_map[row["doi"]] = normalise_title(row["title"])

    # Reverse index: normalised_title → doi (for process.extract)
    j_index = {doi: norm for doi, norm in j_norm_map.items() if norm}

    match_records = []
    for _, afa_row in tqdm(afa_df.iterrows(), total=len(afa_df),
                            desc="Fuzzy matching"):
        afa_norm = normalise_title(afa_row["title"])
        afa_year = int(afa_row["year"]) if pd.notna(afa_row["year"]) else None

        if not afa_norm:
            match_records.append({
                "afa_paper_id": afa_row["paper_id"],
                "journal_doi": None,
                "match_score": 0.0,
                "published_top3": False,
            })
            continue

        candidates = process.extract(
            afa_norm, j_index,
            scorer=fuzz.token_sort_ratio,
            limit=10,
        )

        best_doi, best_score = None, 0.0
        for _val, score, doi in candidates:
            if score < FUZZY_THRESHOLD:
                break
            # Year guard: AFA presentation ≤ publication year + 2
            j_year = journals.loc[journals["doi"] == doi, "year"]
            if len(j_year) > 0 and afa_year:
                try:
                    jy = int(j_year.iloc[0])
                    if not (-1 <= jy - afa_year <= 4):   # AFA then published within 4 years
                        continue
                except (ValueError, TypeError):
                    pass
            if score > best_score:
                best_score = score
                best_doi = doi

        match_records.append({
            "afa_paper_id": afa_row["paper_id"],
            "journal_doi": best_doi,
            "match_score": best_score,
            "published_top3": best_doi is not None,
        })

    match_df = pd.DataFrame(match_records)
    out_df = afa_df.merge(match_df, left_on="paper_id", right_on="afa_paper_id",
                          how="left")
    out_df["published_top3"] = out_df["published_top3"].fillna(False)

    matched_n = out_df["published_top3"].sum()
    total_n   = len(out_df)
    print(f"\n  Match rate: {matched_n}/{total_n} = {matched_n/total_n*100:.1f}%")
    if matched_n > 0:
        print(f"  Median match score (matched only): "
              f"{out_df.loc[out_df['published_top3'], 'match_score'].median():.1f}")

    AFA_MATCHED.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(AFA_MATCHED, index=False, compression="snappy")
    print(f"  Saved → {AFA_MATCHED}")


# ------------------------------------------------------------------ #
# Step D — Probit analysis                                             #
# ------------------------------------------------------------------ #

def run_probit(formula: str, df: pd.DataFrame, label: str) -> dict | None:
    """Fit a probit and return a result dict. Returns None if estimation fails."""
    try:
        model  = smf.probit(formula, data=df).fit(disp=False, maxiter=200)
        params = model.params
        bse    = model.bse
        pvals  = model.pvalues
        ci     = model.conf_int()
        n      = int(model.nobs)
        pr2    = model.prsquared

        rows = []
        for var in params.index:
            coef = params[var]
            se   = bse[var]
            z    = coef / se if se > 0 else np.nan
            p    = pvals[var]
            lo   = ci.loc[var, 0]
            hi   = ci.loc[var, 1]
            sig  = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
            rows.append({
                "Specification": label,
                "Variable": var,
                "Coefficient": round(coef, 4),
                "Std Error": round(se, 4),
                "z-stat": round(z, 4),
                "p-value": round(p, 4),
                "CI 2.5%": round(lo, 4),
                "CI 97.5%": round(hi, 4),
                "Significance": sig,
                "N": n,
                "Pseudo R2": round(pr2, 4),
            })
        return rows
    except Exception as exc:
        print(f"  Probit failed for '{label}': {exc}")
        return None


def save_table(rows: list[dict], stem: str) -> None:
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{stem}.csv"
    tex_path = OUT_DIR / f"{stem}.tex"
    df.to_csv(csv_path, index=False)

    # Build LaTeX tabular
    lines = [
        r"\begin{tabular}{llrrrrrrlrr}",
        r"\toprule",
        r"Specification & Variable & Coefficient & Std Error & z-stat & "
        r"p-value & CI 2.5\% & CI 97.5\% & Significance & N & Pseudo R2 \\",
        r"\midrule",
        r"\multicolumn{11}{l}{\textit{AFA conference baseline robustness. "
        r"Binary directional probit; pre-publication = AFA presentation.}} \\[2pt]",
    ]
    for _, row in df.iterrows():
        line = (
            f"{row['Specification']} & {row['Variable']} & "
            f"{row['Coefficient']} & {row['Std Error']} & "
            f"{row['z-stat']} & {row['p-value']} & "
            f"{row['CI 2.5%']} & {row['CI 97.5%']} & "
            f"{row['Significance']} & {row['N']} & {row['Pseudo R2']} \\\\"
        )
        lines.append(line)
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved {csv_path.name} and {tex_path.name}")


def step_analyze() -> None:
    print("=== Step D: Probit analysis (AFA baseline) ===")
    if not AFA_MATCHED.exists():
        raise FileNotFoundError(f"{AFA_MATCHED} not found — run --step match first")

    df = pd.read_parquet(AFA_MATCHED)
    print(f"  AFA matched dataset: {len(df):,} papers")
    print(f"  Published (matched to journal): {df['published_top3'].sum():,}")

    # Prepare analysis variables
    df["pub"]        = df["published_top3"].astype(int)
    df["directional"] = (df["direction"] != 0).astype(int)
    df["pos"]        = (df["direction"] == 1).astype(int)
    df["neg"]        = (df["direction"] == -1).astype(int)
    df["year_c"]     = df["year"] - df["year"].median()
    df["log_nauth"]  = np.log1p(
        df["authors"].str.split(",").str.len().clip(lower=1)
    )

    # Publication rates
    rates = df.groupby("directional")["pub"].agg(["mean", "count"]).reset_index()
    rates.columns = ["directional", "pub_rate", "n"]
    rates["group"] = rates["directional"].map({1: "Directional", 0: "Null"})
    print("\n  AFA Publication rates:")
    print(rates[["group", "pub_rate", "n"]].to_string(index=False))

    # --- Binary directional probit (PRIMARY for AFA) ---
    all_rows: list[dict] = []

    r1 = run_probit("pub ~ directional", df, "AFA (1): Bivariate")
    r2 = run_probit("pub ~ directional + year_c", df, "AFA (2): + Year")
    r3 = run_probit("pub ~ directional + year_c + log_nauth", df,
                    "AFA (3): + Authors")

    for r in [r1, r2, r3]:
        if r:
            all_rows.extend(r)

    if all_rows:
        save_table(all_rows, "robustness_afa_baseline")

    # --- Three-way (positive / negative / null) — symmetry test ---
    tw_rows: list[dict] = []
    t1 = run_probit("pub ~ pos + neg", df, "AFA 3-way (1): Bivariate")
    t2 = run_probit("pub ~ pos + neg + year_c", df, "AFA 3-way (2): + Year")
    t3 = run_probit("pub ~ pos + neg + year_c + log_nauth", df,
                    "AFA 3-way (3): + Authors")

    for r in [t1, t2, t3]:
        if r:
            tw_rows.extend(r)

    if tw_rows:
        save_table(tw_rows, "robustness_afa_threeway")

    # --- Summary stats table ---
    pub_rates_rows = []
    for direction_val, label in [(1, "Positive"), (-1, "Negative"), (0, "Null")]:
        sub = df[df["direction"] == direction_val]
        if len(sub) == 0:
            continue
        pub_rates_rows.append({
            "Direction": label,
            "N_AFA": len(sub),
            "N_Published": int(sub["pub"].sum()),
            "Pub_Rate": round(sub["pub"].mean(), 3),
        })
    if pub_rates_rows:
        pub_rates_df = pd.DataFrame(pub_rates_rows)
        pub_csv = OUT_DIR / "table2_afa_pub_rates.csv"
        pub_rates_df.to_csv(pub_csv, index=False)
        print(f"\n  AFA publication rates by direction:")
        print(pub_rates_df.to_string(index=False))
        print(f"  Saved {pub_csv.name}")

    # --- Comparison to NBER baseline ---
    print("\n  Key comparison:")
    if all_rows:
        # Extract AFA bivariate directional coefficient
        afa_biv = next(
            (r for r in all_rows
             if r["Specification"] == "AFA (1): Bivariate"
             and r["Variable"] == "directional"),
            None
        )
        if afa_biv:
            print(f"  AFA  bivariate directional β = {afa_biv['Coefficient']:.4f}"
                  f" (s.e. {afa_biv['Std Error']:.4f}){afa_biv['Significance']}")
    print("  NBER bivariate directional β = 1.2793 (s.e. 0.1623)***")
    print("  (If AFA β is similar, the significance-sorting pattern is robust"
          " to the choice of pre-publication baseline.)")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(step: str, no_api: bool) -> None:
    VALID_STEPS = {"classify", "direction", "match", "analyze", "all"}
    if step not in VALID_STEPS:
        raise ValueError(f"Unknown step '{step}'. Choose from: {VALID_STEPS}")

    if no_api:
        print("Running in --no-api mode: keyword + rule-based coding (free, no API calls)")

    run_all = (step == "all")

    if run_all or step == "classify":
        step_classify(no_api=no_api)

    if run_all or step == "direction":
        step_direction(no_api=no_api)

    if run_all or step == "match":
        step_match()

    if run_all or step == "analyze":
        step_analyze()

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AFA conference baseline robustness pipeline"
    )
    parser.add_argument(
        "--step",
        default="all",
        choices=["classify", "direction", "match", "analyze", "all"],
        help="Pipeline step to run (default: all)"
    )
    parser.add_argument(
        "--no-api", action="store_true",
        help="Free mode: keyword filter for scope + rule-based direction coding. "
             "No Anthropic API calls. Direction coding is noisier but costs nothing."
    )
    args = parser.parse_args()
    main(args.step, args.no_api)
