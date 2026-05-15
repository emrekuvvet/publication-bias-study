# AFA Abstract Enrichment Plan
## Making the AFA and NBER keyword filters equivalent

**Date:** 2026-05-14

---

## Problem

The NBER in-scope classification (script 03) applies the keyword filter to
`title + abstract` and feeds the full abstract to the LLM for scope and
direction coding.

The current AFA pipeline applies a title-only keyword filter and codes
direction from titles alone (or from matched published abstracts). This
asymmetry means:

1. AFA papers with neutral titles but policy-signal abstracts are missed
   by the keyword filter — underrepresenting in-scope AFA papers
2. Direction coding for unmatched AFA papers is noisier because titles
   rarely reveal finding direction
3. The comparison β_AFA vs β_NBER is not apples-to-apples

**Fix:** Fetch abstracts for AFA papers from OpenAlex and re-run the
pipeline with `title + abstract` filter and abstract-based LLM coding —
identical to what we do for NBER.

---

## Abstract Source

**Primary: OpenAlex** (`https://api.openalex.org/works`)
- Free, no key required (polite pool with mailto= speeds it up)
- Stores abstracts as `abstract_inverted_index` (inverted word→position map)
- Good coverage of finance papers published in journals; working paper
  coverage is lower but many AFA papers are already published or on SSRN
- Rate limit: ~10 req/s polite pool

**Secondary: Semantic Scholar** (`https://api.semanticscholar.org/graph/v1`)
- Free, covers pre-prints and working papers
- Try for papers OpenAlex misses

**Expected coverage:**
- AFA papers that were eventually published in a journal: ~60-70% will
  have OpenAlex abstracts (the published version)
- AFA working papers not yet published: lower coverage (~20-30%)
- Overall: expect abstracts for ~50% of in-scope AFA papers

---

## Implementation

### New script: `scripts/10b_enrich_afa_abstracts.py`

**Input:** `data/raw/afa_papers.parquet` (4,001 AFA papers)

**Steps:**
1. For each AFA paper, search OpenAlex by title + year window (±2 years)
2. If match found (fuzzy title score ≥ 80), extract and decode abstract
3. For papers without OA abstract, try Semantic Scholar
4. Cache all results in `data/raw/afa_abstract_cache.json` (avoid re-fetching)
5. Save enriched dataset with `abstract` column populated where found

**Output:** `data/raw/afa_papers_enriched.parquet`
Columns: all original columns + `abstract`, `abstract_source`
         (`openalex` | `semantic_scholar` | `none`)

**Runtime estimate:**
- ~4,000 papers × 0.12s per request = ~8 minutes (OpenAlex polite pool)
- ~500 SS fallback queries × 4s = ~33 minutes
- Total: ~40 minutes first run; subsequent runs use cache (seconds)

**Cost:** Free (no API key required)

---

### Updated script: `scripts/11_afa_analysis.py`

**Step A (classify) changes:**
- Load `afa_papers_enriched.parquet` instead of `afa_papers.parquet`
- Apply keyword filter to `title + abstract` (same `combined_text` logic
  as script 03, line 193-195)
- LLM scope prompt uses abstract when available, falls back to title-only
  prompt when abstract is missing (with lower confidence threshold)

**Step B (direction) changes:**
- For papers with abstract: use the standard `DIRECTION_PROMPT` from
  script 04 (abstract-based, same prompt as NBER)
- For papers without abstract: fall back to title-only direction prompt
  (flag with `title_only=True`)
- `--no-api` mode: rule-based from abstract when available, title keywords
  when not

**Expected improvement:**
- In-scope AFA papers: from ~251 → ~350-450 (more papers caught by
  abstract keyword hits)
- Directional AFA papers: from ~8 → ~30-50 (abstracts reveal direction)
- R10 probit N: from 251 → 350-450
- R10 standard errors: narrower (more observations)
- Direction coding quality: comparable to NBER for the ~50% with abstracts

---

## Output changes

Same output files as before:
```
output/tables/robustness_afa_baseline.{csv,tex}
output/tables/robustness_afa_threeway.{csv,tex}
output/tables/table2_afa_pub_rates.{csv,tex}
```

Paper text updated with new N, β, and a note that abstracts were sourced
from OpenAlex for papers where available.

---

## Execution Order

| Step | Command | Time | Cost |
|---|---|---|---|
| 1 | `python scripts/10b_enrich_afa_abstracts.py` | ~40 min | Free |
| 2 | `python scripts/11_afa_analysis.py --step all --no-api` | ~2 min | Free |
| 3 | `python scripts/11_afa_analysis.py --step all` | ~45 min | ~$15 API |
| 4 | Update paper + recompile | ~2 min | — |

Step 2 (no-api) gives a quick free result using abstract-based keyword
filter + rule-based direction from abstracts.
Step 3 (with LLM) gives the most accurate result.

---

## Limitations

1. **Coverage gap:** AFA papers that were never published and never posted
   to SSRN/arXiv will not have OpenAlex abstracts. These remain title-only.
   We flag them with `abstract_source = 'none'`.

2. **Version mismatch:** The OpenAlex abstract may be the published version,
   which may differ from the AFA conference draft. This is the same issue
   as the NBER-to-published version mismatch; we accept it as unavoidable.

3. **Residual asymmetry:** The ~50% of AFA papers without abstracts still
   rely on title-only classification, whereas all NBER papers have abstracts.
   We report results separately for (a) AFA papers with abstracts and
   (b) full AFA sample, to show the asymmetry does not drive the result.
