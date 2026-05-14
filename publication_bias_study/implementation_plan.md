# Publication Bias in Top Finance Journals: Implementation Plan

**Research Question:** Do the Journal of Finance (JF), Review of Financial Studies (RFS), and Journal of Financial Economics (JFE) publish disproportionately more papers showing *positive* effects of government intervention in financial markets, relative to what the underlying working paper population (NBER) would predict?

**PI:** Professor of Finance  
**Status:** Pre-implementation planning  
**Last updated:** 2026-05-13

---

## Hypothesis

Top finance journals exhibit directional publication bias: papers finding positive effects of government laws, regulation, and public policy are accepted at higher rates than papers finding negative or null effects, conditional on paper quality and topic.

---

## 1. Sample Construction

### 1a. Published Papers (Treatment Sample)

Collect all papers published in JF, RFS, and JFE from **1990 to 2024** that study the effects of government intervention.

**Journals and data sources:**

| Journal | Source | Coverage |
|---|---|---|
| Journal of Finance | Wiley Online Library / Web of Science | 1990–2024 |
| Review of Financial Studies | Oxford Academic / Web of Science | 1990–2024 |
| Journal of Financial Economics | Elsevier ScienceDirect / Web of Science | 1990–2024 |

**Fields to pull per paper:**
- Title, authors, affiliations, year, volume, issue, DOI
- Abstract (full text)
- JEL codes (if available)
- Number of citations (for quality controls later)

### 1b. Working Paper Control Sample

Collect all NBER working papers tagged under relevant programs from **1990 to 2024**.

**Relevant NBER programs:**
- Corporate Finance (CF)
- Asset Pricing (AP)
- Monetary Economics (ME)
- Economic Fluctuations and Growth (EFG)
- Public Economics (PE) — partial overlap

**Data source:** NBER Working Paper API (https://api.nber.org/papers) — free, no authentication required.

**Fields to pull:** title, authors, year, abstract, program codes, working paper number.

---

## 2. Identifying "Government Intervention" Papers

### 2a. Keyword Taxonomy

Papers are in-scope if their primary research question concerns the effect of a government law, regulation, or public policy on financial markets, firms, or investors.

**Inclusion keywords (apply to title + abstract):**

| Category | Example Keywords |
|---|---|
| Financial regulation | "regulation", "regulatory", "Dodd-Frank", "Basel", "Glass-Steagall", "capital requirements", "leverage ratio" |
| Securities law | "SEC", "disclosure", "Reg FD", "Sarbanes-Oxley", "SOX", "insider trading law", "short-selling ban" |
| Banking policy | "FDIC", "deposit insurance", "lender of last resort", "bailout", "TARP", "stress test" |
| Monetary/fiscal policy | "quantitative easing", "QE", "Fed intervention", "government guarantee", "subsidy" |
| Market microstructure rules | "circuit breaker", "tick size", "trading halt", "uptick rule" |
| Corporate governance law | "board independence requirement", "say-on-pay", "proxy access", "mandatory audit" |

### 2b. Classification Pipeline

1. **First pass:** Python keyword filter on title + abstract → candidate set
2. **Second pass:** LLM classifier (Claude API) reads full abstract and classifies: `in_scope = True/False` with a one-sentence justification
3. **Validation:** Human review on a random sample of 300 papers (150 flagged in, 150 flagged out) to compute precision/recall
4. **Target:** precision ≥ 0.85, recall ≥ 0.80 before proceeding

---

## 3. Coding the Direction of Findings

This is the most critical step. For each in-scope paper, assign:

| Code | Meaning |
|---|---|
| `+1` | Main finding: government intervention had a **positive/beneficial** effect on the outcome studied |
| `-1` | Main finding: government intervention had a **negative/harmful** effect |
| `0` | Main finding: **null**, mixed, or inconclusive |

### 3a. LLM Coding Protocol

Use Claude API to read abstract + conclusion paragraphs (if available via full text) and return:
- `direction`: +1 / -1 / 0
- `confidence`: high / medium / low
- `rationale`: one sentence

Prompt template (to be refined):
> "The following is the abstract of a finance paper studying a government intervention. Classify the main finding as positive (+1: intervention improved outcomes), negative (-1: intervention harmed outcomes), or null/mixed (0). Return JSON with keys: direction, confidence, rationale."

### 3b. Human Validation

- Two independent research assistants (RAs) code a random sample of 400 in-scope papers
- Compute Cohen's Kappa for inter-rater agreement — target κ ≥ 0.70
- Resolve disagreements by discussion, then update LLM prompt if systematic errors are found
- Use human codes as ground truth for the 400-paper validation set; report LLM accuracy

### 3c. Ambiguity Rules

- If a paper studies multiple interventions with conflicting signs, code the *primary* intervention (the one named in the title or first sentence of abstract)
- Theoretical papers (no empirical test) → exclude
- Papers that *only* describe a regulation without estimating its effect → exclude

---

## 4. Statistical Tests

### 4a. Primary Test — Publication Probability by Finding Direction

Estimate a probit/logit model at the paper level:

```
Published_in_top3(i) = α + β₁ Positive(i) + β₂ Negative(i) + γ X(i) + ε(i)
```

Where:
- `Published_in_top3` = 1 if NBER working paper was eventually published in JF, RFS, or JFE
- `Positive` = 1 if finding direction coded +1
- `Negative` = 1 if finding direction coded -1
- Omitted category = null/mixed (0)
- `X(i)` = controls: year FE, NBER program FE, number of authors, top-institution affiliation dummy

**Key coefficient:** β₁ > 0 and statistically significant → positive-finding papers are more likely to be published in top journals.

**Matching approach (robustness):** Match NBER papers to published papers on topic and year; compare direction distributions within matched pairs.

### 4b. Cross-Journal Comparison

Run the same probit separately for JF, RFS, JFE to test whether bias is concentrated in one journal or shared across all three.

Also compare publication rates *within* the same NBER program (e.g., Corporate Finance papers) across journals.

### 4c. t-Statistic Distribution Test (Brodeur et al. Approach)

Following Brodeur, Lé, Sangnier & Zylberberg (2016):
- Extract the primary test statistic (t-stat or z-stat) from each published paper
- Plot the empirical distribution
- Test for a discontinuity/heap just above t = 1.96 (one-sided) or |t| = 1.96 (two-sided)
- A heap at the threshold in *positive-direction* papers indicates p-hacking or selective reporting

This test is applied to published papers only — no need for working paper matching.

### 4d. Time Trend Analysis

Run the primary regression separately for:
- Pre-2008 (1990–2007)
- Post-crisis (2008–2015)
- Post-Dodd-Frank / recent (2016–2024)

Hypothesis: bias may have intensified after the financial crisis as policy-relevance became a stronger selection criterion.

### 4e. Editor Fixed Effects

Collect the editorial roster for each journal (editors change every 4–6 years). Include editor fixed effects to test whether bias varies with editorial leadership.

---

## 5. Controls and Robustness Checks

| Control | Why it matters |
|---|---|
| Identification strategy (RDD, IV, DID, OLS) | Better-identified papers may have different direction distributions |
| Sample size / number of observations | Larger samples detect smaller effects |
| Top-institution affiliation | Network effects in publishing |
| Number of NBER citations at submission | Proxy for paper quality before journal decision |
| JEL codes | Control for subfield differences |
| Cross-listing in multiple NBER programs | Broader appeal papers may differ |

**Robustness checks:**
- Restrict to papers with quasi-experimental identification only (RDD, IV, natural experiment) — does bias persist?
- Re-run with a stricter "government intervention" definition (exclude monetary policy, keep only explicit legislation/regulation)
- Use RFS/JFE as the comparison journals (is JF uniquely biased relative to peer journals?)

---

## 6. Data Pipeline (Technical Implementation)

### 6a. Environment

```
Python 3.11+
Libraries: requests, pandas, numpy, scikit-learn, anthropic, statsmodels, matplotlib, seaborn
Database: DuckDB (parquet files, consistent with existing HMDA stack)
```

### 6b. File Structure

```
publication_bias_study/
├── implementation_plan.md          ← this file
├── data/
│   ├── raw/
│   │   ├── nber_papers.parquet
│   │   ├── jf_papers.parquet
│   │   ├── rfs_papers.parquet
│   │   └── jfe_papers.parquet
│   ├── classified/
│   │   ├── inscope_papers.parquet
│   │   └── coded_direction.parquet
│   └── final/
│       └── analysis_dataset.parquet
├── scripts/
│   ├── 01_collect_nber.py          ← NBER API scraper
│   ├── 02_collect_journals.py      ← WoS/journal scrapers
│   ├── 03_classify_inscope.py      ← LLM keyword + scope filter
│   ├── 04_code_direction.py        ← LLM direction coding
│   ├── 05_link_nber_to_published.py ← fuzzy match working paper → journal paper
│   ├── 06_analysis_main.py         ← primary probit regressions
│   ├── 07_tstat_distribution.py    ← Brodeur et al. test
│   └── 08_robustness.py            ← all robustness checks
├── validation/
│   ├── human_coding_sheet.xlsx     ← RA coding instrument
│   └── interrater_reliability.py
└── output/
    ├── tables/
    └── figures/
```

### 6c. Linking NBER Papers to Published Papers

This is technically the hardest step. Approach:
1. For each in-scope journal paper, search NBER for the same title using fuzzy string matching (RapidFuzz library, threshold ≥ 85%)
2. Cross-check author names (at least one author overlap required)
3. Flag high-confidence matches (score ≥ 90%) as automatic links
4. Flag medium-confidence matches (75–90%) for human review
5. Unmatched published papers are dropped from the NBER-comparison analysis (kept for t-stat analysis)

---

## 7. Potential Limitations and Responses

| Limitation | Response |
|---|---|
| Self-censorship bias: researchers don't submit negative results to top journals | Show that NBER papers themselves show balanced direction distribution — the gap emerges only for published papers |
| LLM coding errors | Validate against human codes; report accuracy; sensitivity analysis with human-only subsample |
| "Positive effect" is ambiguous | Publish the coding rubric; report κ; show results are robust to collapsing categories |
| Paper quality confound | Include identification strategy as a control; show bias holds within high-quality subsample |
| NBER is not a random sample | Acknowledge; note it is the standard pre-publication repository in top finance research |

---

## 8. Literature to Engage

- Brodeur, Lé, Sangnier & Zylberberg (2016, AER) — t-stat distribution test for publication bias
- Andrews & Kasy (2019, AER) — structural model of publication bias
- Franco, Malhotra & Simonovits (2014, Science) — file drawer problem in social science
- Christensen & Miguel (2018, JEL) — transparency and reproducibility in economics
- Ioannidis, Stanley & Doucouliagos (2017, EJ) — power and bias in economics research

---

## 9. Sequence of Steps to Start

1. [ ] Set up Python environment and DuckDB schema
2. [ ] Run `01_collect_nber.py` — pull all NBER papers for CF, AP, ME programs 1990–2024
3. [ ] Run `02_collect_journals.py` — pull JF, RFS, JFE paper lists (Web of Science institutional access required)
4. [ ] Run `03_classify_inscope.py` — keyword filter + LLM scope classification
5. [ ] Build `human_coding_sheet.xlsx` and have RAs code 400-paper validation sample
6. [ ] Calibrate LLM coding against RA ground truth
7. [ ] Run `04_code_direction.py` on full in-scope sample
8. [ ] Run `05_link_nber_to_published.py` — fuzzy matching
9. [ ] Run `06_analysis_main.py` — primary regressions
10. [ ] Run `07_tstat_distribution.py` — Brodeur test
11. [ ] Run `08_robustness.py` — all checks
12. [ ] Write paper

---

## Notes

- Web of Science access is needed for journal paper collection — confirm institutional access before starting script 02
- Claude API key needed for LLM classification steps (scripts 03, 04) — budget ~$50–100 for full classification run depending on abstract length
- Consider pre-registering the study design (AEA RCT Registry or OSF) before data collection to strengthen credibility of findings
