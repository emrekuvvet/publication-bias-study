# AFA Conference Baseline Robustness Plan
## Addressing the "NBER population is too different" concern

**Date:** 2026-05-14

---

## Motivation

The central identification critique is that NBER working papers and published
JF/RFS/JFE articles are two fundamentally different populations selected through
different institutional processes. Only 14.2% of published papers are matched to
an NBER counterpart, meaning we compare largely non-overlapping cross-sections.

**The AFA fix:** Papers presented at the American Finance Association annual
meeting are drawn from essentially the same elite research network as papers
published in JF/RFS/JFE. AFA acceptance itself is competitive and selective on
quality, not direction. The vast majority of JF/RFS/JFE papers were first
presented at AFA. This gives us a pre-publication reference population that
is:

1. **Closer in quality** to published papers than NBER (both AFA and JF/RFS/JFE
   require elite-institution affiliation, fully developed drafts, peer scrutiny)
2. **More likely to overlap** with the published sample — we expect 40–60% of
   JF/RFS/JFE papers to have been presented at AFA, versus only 14% matched to NBER
3. **Not selected on direction** — AFA program committees evaluate research design
   and importance, not whether the estimated coefficient is positive, negative, or null

If the significance-sorting pattern survives with AFA as the baseline, it is
harder to attribute to NBER-specific selection and substantially more credible.

---

## Data Source

**Website:** https://afajof.org/past-meetings/

**Coverage by year:**

| Years | URL Pattern | Format |
|---|---|---|
| 2006–2023 | `https://afajof.org/{YEAR}-preliminary-program/` | HTML |
| 2024–2026 | `https://afajof.org/management/full-program{YEAR}.html` | HTML |
| 2003–2005 | ASSA PDF links (via past-meetings index) | PDF |
| 2000–2002 | Not available on AFA website | — |

**Usable range for this study: 2006–2024** (19 years, all HTML).

We accept a truncated window: our NBER sample covers 2000–2024 but the AFA
baseline will cover 2006–2024. This covers 85%+ of the published papers in our
sample by paper count (most published papers are from 2006 onward).

---

## HTML Structure

**2006–2023 pages** (`{YEAR}-preliminary-program/`):
```
<h3>Session time block (e.g. "Saturday 8:00am - 10:00am")</h3>
  [Session: Capital Structure / Chair: ...]
  <h5>Paper Title Here</h5>
  <p>Author Name (University), Author Name 2 (University)</p>
  <p>Discussant: Author Name (University)</p>
  <hr>  ← session separator
```

**2024+ pages** (`full-program{YEAR}.html`):
```
<h5>Session Title</h5>
  <h6>Paper Title Here</h6>
  <p>Author Name (University), Author Name 2 (University)</p>
  <p>Discussant: ...</p>
```

---

## Pipeline Design

```
10_collect_afa.py          → data/raw/afa_papers.parquet
         ↓
11_afa_analysis.py
   Step A: keyword filter  (reuse keyword taxonomy from script 03)
   Step B: LLM scope class (reuse prompt from script 03)
   Step C: LLM direction   (reuse prompt from script 04)
   Step D: fuzzy matching  (reuse matching logic from script 05)
   Step E: probit analysis (same specs as script 06)
         ↓
output/tables/robustness_afa_baseline.{csv,tex}
output/tables/table2_afa_pub_rates.{csv,tex}
```

---

## Step A — Keyword Filter

Reuse the exact same keyword taxonomy from `scripts/03_classify_inscope.py`.
Apply to AFA paper titles (abstracts not available on AFA program pages).

**Important limitation:** AFA programs list titles only, no abstracts. This
means the keyword filter and LLM scope classifier must work from title alone.
The LLM prompt for AFA papers must explicitly acknowledge this and code on a
lower-confidence basis. Papers classified as in-scope from title alone should be
flagged with `title_only = True`.

---

## Step B — LLM In-Scope Classification (Title Only)

Use a modified prompt:
```
"The following is the TITLE of a finance paper presented at the AFA annual
meeting. No abstract is available. Based on the title alone, classify whether
this paper's primary research question is likely the causal effect of a
government law, regulation, or public policy on financial markets, firms, or
investors. Return in_scope=true only if the title clearly indicates a
government-policy causal-effect study. If uncertain, return in_scope=false.
Return JSON: {in_scope: bool, confidence: 'high'|'medium'|'low', reason: str}"
```

Because we have only titles, expect:
- Higher false-negative rate (genuinely in-scope papers missed due to indirect titles)
- Lower false-positive rate (unclear titles won't be flagged)
- This biases toward fewer in-scope AFA papers, which makes our comparison
  conservative (harder to find the significance-sorting pattern)

---

## Step C — LLM Direction Coding (Title Only)

For in-scope AFA papers identified in Step B, code direction from title.
Use a modified direction prompt that acknowledges title-only input.

**Key concern:** Direction coding from titles is substantially noisier than
from abstracts. Many titles are neutral ("The Effect of X on Y") without
revealing direction. Expect a higher null/unclear rate in AFA papers coded
from titles. This makes the AFA baseline harder to use — more papers will
be uncoded.

**Mitigation strategy:**
1. For AFA papers that are **matched to JF/RFS/JFE** (via fuzzy title matching
   in Step D), use the direction code from the published paper's abstract instead
2. For **unmatched AFA papers**, use title-only direction coding with
   conservative null assignment for uncertain cases
3. Report results separately for (a) matched papers and (b) full sample

---

## Step D — Matching AFA Papers to Published Papers

Use the same fuzzy title matching logic from `scripts/05_link_nber_to_published.py`
with `RapidFuzz token_sort_ratio ≥ 85` (slightly tighter than NBER threshold to
reduce false positives since AFA titles are often identical to published titles).

For each AFA paper, search against:
- `data/raw/jf_papers.parquet`
- `data/raw/rfs_papers.parquet`
- `data/raw/jfe_papers.parquet`

AFA papers with a match → `published_top3 = True`
AFA papers without a match → `published_top3 = False`

**Expected match rate:** 40–60% (substantially higher than 14% for NBER).
A high match rate is precisely what makes AFA a better baseline — it confirms
that AFA and JF/RFS/JFE draw from overlapping populations.

---

## Step E — Probit Analysis

Run the same binary directional probit as the primary analysis:

```
Prob(Published_in_top3 = 1) = Φ(α + β · Directional_i + γ X_i)
```

Where:
- `Published_in_top3 = 1` if AFA paper is matched to JF/RFS/JFE publication
- `Directional = 1` if direction code ≠ null
- Controls: year trend, log(author count), quasi_exp indicator

Also run the three-way specification (positive/negative/null) for the symmetry
test.

**Key comparison:** β_AFA vs β_NBER. If both are similar and significant, the
significance-sorting pattern is robust to the choice of pre-publication baseline.
If β_AFA is smaller, this could reflect genuine quality selection into AFA
(directional papers at AFA are already higher quality on average) or noisier
direction coding from titles.

---

## Expected Outputs

```
output/tables/robustness_afa_baseline.csv/.tex    ← main robustness table
output/tables/table2_afa_pub_rates.csv/.tex       ← raw AFA publication rates
output/tables/afa_sample_summary.csv/.tex         ← summary stats for AFA sample
```

And additions to the paper's Robustness section (Section V):

```
R_AFA — AFA Conference Baseline

To address the concern that NBER working papers represent a fundamentally
different population from published papers, we construct an alternative
pre-publication baseline using papers presented at the AFA annual meeting.
AFA presentations are selected by a competitive program committee on research
quality, not finding direction, and are drawn from the same elite-institution
network as JF/RFS/JFE publications. We collect [N] papers presented at AFA
annual meetings from 2006–2024, classify [M] as in-scope government intervention
papers, and link [K] (∼X%) to JF/RFS/JFE publications.

The binary directional probit using the AFA baseline yields a coefficient of
[β_AFA]*** (s.e. [se]), consistent with the NBER-baseline estimate of 1.279***
(s.e. 0.162). Directional AFA papers are [X]× more likely to appear in a top
journal than null-finding AFA papers. This result is robust to using a
substantially more comparable pre-publication reference population and
addresses the central identification concern about NBER selection.
```

---

## Limitations of the AFA Approach

1. **Title-only coding** increases direction-coding noise substantially.
   Some papers will be uncoded (neutral titles don't reveal direction).
   This biases toward finding fewer in-scope papers and more nulls.

2. **Coverage starts 2006.** Papers published 2000–2005 in our JF/RFS/JFE
   sample cannot be compared to an AFA baseline.

3. **AFA itself is selective.** AFA acceptance requires a competitive application.
   If AFA selects on "interesting" (i.e., directional) findings, the AFA
   distribution is also shifted toward directional papers relative to the
   full pre-submission universe. However, this is a smaller concern than
   NBER selection because AFA program review is more transparent and
   focused on methodology.

4. **Abstracts required for reliable direction coding.** After matching AFA
   papers to published papers, we can get direction codes from published
   abstracts. For unmatched AFA papers, title-only coding is noisy.
   The most credible sub-analysis is therefore among matched AFA-journal pairs.

5. **Year window mismatch.** An AFA paper presented in January 2015 might be
   published in JF in 2016 or 2017. A ±2 year window should be used when
   matching AFA presentations to publications.

---

## Execution Order

| Step | Script | Estimated Time |
|---|---|---|
| 1 | `python scripts/10_collect_afa.py` | 5–10 min (HTTP scraping) |
| 2 | `python scripts/11_afa_analysis.py --step classify` | 15–30 min + $5 API |
| 3 | `python scripts/11_afa_analysis.py --step direction` | 15–30 min + $10 API |
| 4 | `python scripts/11_afa_analysis.py --step match` | 2–3 min (fuzzy matching) |
| 5 | `python scripts/11_afa_analysis.py --step analyze` | 1 min |
| 6 | Update `paper/main.tex` with new robustness section | Manual |
| 7 | Recompile PDF | 1 min |

Or run all steps in sequence:
```bash
python scripts/10_collect_afa.py
python scripts/11_afa_analysis.py --step all
```
