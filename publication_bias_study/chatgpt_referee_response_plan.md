# Revision Plan — ChatGPT Referee Report
## "Significance Bias in Government Intervention Research: Evidence from Top Finance Journals"
**Date:** 2026-05-14  
**Referee verdict:** Reject (with encouragement to revise and resubmit elsewhere or reframe substantially)

---

## Summary of Referee's Core Objections

| # | Referee Concern | Severity | Disposition |
|---|---|---|---|
| C1 | NBER vs published = two different populations, not filtered cohort | Fatal for causal claims | Reframe + new text |
| C2 | NBER is itself highly selected — not a neutral pre-publication baseline | Fatal for causal claims | Reframe + new text |
| C3 | Positive/negative/null coding is normatively ambiguous in finance regulation | Moderate | Demote; promote binary spec |
| C4 | Abstract endogeneity: R8 shows effects vanish with length control | Serious | Acknowledge + reframe |
| C5 | Causal language ("editors favor", "significance filter") unsupported | Serious | Systematic language audit |
| C6 | Quality controls (author count, quasi-exp dummy) inadequate | Moderate | Strengthen or limit claims |
| C7 | Wald test H₀: β₁=β₂ underpowered; silence ≠ evidence | Moderate | Power calculations + hedging |
| C8 | JF journal-level cells too small (n=38) for cross-journal claims | Moderate | Label exploratory |
| C9 | Crisis-period story speculative and overinterpreted | Moderate | Label descriptive |

---

## Part I — The Two Biggest Structural Changes

### I.A  Promote Binary Directional Specification to Primary Analysis

**Referee quote:** *"In my view, the binary specification is substantially more credible than the positive-versus-negative decomposition and should probably become the paper's primary analysis."*

This is the single most important structural change the referee requests. The referee is correct: the normatively-coded positive/negative split introduces ambiguity (is reduced lending "negative" or "positive"?) while adding little beyond the directional vs null contrast.

**Actions:**

1. **Table III restructure**: Make Column (1) the binary directional probit (currently R4/robustness). Push the three-way positive/negative/null split to an appendix table.
2. **Abstract/introduction rewrite**: Lead with "we find that directional findings—regardless of sign—are substantially more likely to appear in top finance journals than null findings." Remove language framing the paper around positive-vs-negative symmetry.
3. **Section IV.A rewrite**: Rename "Direction Distribution" subsection to "Directional vs Null Findings" and reorder the discussion to put binary results first.
4. **Appendix**: Keep three-way results as Appendix Table A (currently Table III) for completeness but explicitly note: "The three-way decomposition requires normative assignment of intervention direction; results are reported for completeness but the binary specification is the paper's primary analysis."

**Files:** `paper/main.tex`, `paper/tables/table3_probit_main.tex`, `scripts/06_analysis_main.py`

---

### I.B  Reframe the Entire Paper as a Descriptive/Correlational Study

**Referee quote:** *"In effect, the paper documents an equilibrium difference between two selected populations. It does not identify where in the research-production pipeline the selection occurs."*

The paper currently oscillates between cautious hedging and causal editorial claims. The referee wants consistent, disciplined framing.

**Actions:**

1. **Title**: Consider changing "Significance Bias" (implies a mechanism) to something like "Significance Sorting in Government Intervention Research" or "Directional Findings and Publication in Top Finance Journals." This signals that the paper documents a pattern rather than identifies a cause.
2. **Abstract**: Replace all causal/mechanistic language with observational language. Draft revision:
   - BEFORE: "Top finance journals exhibit strong significance bias…"
   - AFTER: "Papers with statistically directional findings are substantially more likely to appear in top finance journals than papers with null findings, controlling for observable quality measures."
3. **Section III — Research Design**: Add a new subsection III.A "Identification and Interpretation" (≈400 words) that:
   - States formally what the design identifies: a cross-sectional association between direction and publication status
   - States the key assumption required for causal interpretation (NBER = representative pre-publication distribution) and explains why this assumption is unlikely to hold exactly
   - Names the three competing explanations the data cannot distinguish: (a) editorial selection, (b) author self-selection in submission, (c) author strategic search over specifications
   - Explicitly benchmarks against stronger designs (Franco et al. 2014 NSF grants, DellaVigna & Linos 2022 RCT registry) and explains why those are infeasible here
4. **Section V — Mechanisms**: Restructure. New subsection V.A "What the Data Can and Cannot Tell Us" before any mechanism discussion. Then V.B "Supply-Side vs Demand-Side" with equal treatment of both channels. V.C "Policy Implications Under Each Mechanism."
5. **Systematic language audit** (see Part II.C below for full word list).

**Files:** `paper/main.tex`

---

## Part II — Text Revisions (No New Code Required)

### II.A  Section III.A — New "Research Design and Identification" Subsection

Write a ~400-word subsection with the following elements:

```
Design: We compare the distribution of directional codes across two cross-sections:
(i) NBER working papers on government intervention in finance, and (ii) papers published
in JF, RFS, and JFE on the same topic. These populations are largely non-overlapping
(only 26 papers appear in both).

Identifying assumption: The NBER working paper distribution approximates the pre-publication
distribution of empirical findings in this domain. This assumption would be violated if
NBER working papers are themselves selected toward directional findings relative to the
full population of ongoing research.

Limitations: NBER affiliation is correlated with author prestige, institutional affiliation,
and project ambition. NBER authors likely submit stronger, more credible projects that may
independently have higher directional rates. Furthermore, many finance papers circulate
through SSRN or institutional series and never enter NBER. The comparison therefore
reflects differences between two selected populations rather than an editorial filter applied
to a common set of papers.

Interpretation: Our results should be read as documenting a pattern in observed distributions.
We cannot attribute the pattern specifically to editorial behavior, author self-selection in
submission, or specification search.
```

### II.B  Strengthen the Limitations Section (Section VI)

Current limitations section buries the identification concern. Revise to:
- Lead paragraph of limitations: "The most fundamental constraint on our design is that NBER working papers and published journal articles are not different states of the same population." (~150 words)
- Add: "Future work with submission-stage data — desk rejection logs, conference submission archives, revise-and-resubmit histories — could meaningfully sharpen the causal interpretation."
- Add: "SSRN preprint histories for authors in our sample would allow a more comprehensive pre-publication baseline."

### II.C  Systematic Language Audit

Search `main.tex` for each of the following phrases and replace with the suggested alternative:

| Current phrase | Replacement |
|---|---|
| "journals favor" | "papers with X are more likely to appear in journals" |
| "editorial demand" | "the publication gap we observe" |
| "significance filter" | "the association between significance and publication" |
| "publication premium" | "higher publication rate for directional findings" |
| "editors reward" | "directional findings are associated with higher publication rates" |
| "our results imply" | "our results are consistent with" |
| "our findings show that journals" | "we find that the published sample contains more" |
| "the bias we document" | "the pattern we document" |

**Implementation:** Run `grep -n "journals favor\|editorial demand\|significance filter\|publication premium\|editors reward\|our results imply" paper/main.tex` to find all instances, then edit manually.

### II.D  Section IV.D — Label Time-Period Analysis Explicitly Exploratory

1. Change section heading: "D. Time-Period Heterogeneity" → "D. Time-Period Heterogeneity (Exploratory)"
2. Add opening sentence: *"The following analysis is exploratory. Sub-period sample sizes range from 47 to 91 papers, and coefficients should be interpreted as descriptive evidence rather than confirmatory tests."*
3. Remove or heavily qualify any sentence attributing the 2008–2015 pattern to editorial demand for anti-regulatory findings. Replace with: *"The 2008–2015 period coincides with the global financial crisis; whether editors, referees, authors, or the research environment itself drove this pattern cannot be determined from our data."*
4. Add 95% confidence intervals (in brackets) to all period-specific coefficient values cited in-text.

### II.E  Section IV.C (Journal-Level) — Add Underpowered Caveat

1. Add footnote to Table IV header or note: *"The JF subsample contains 38 papers; cross-journal comparisons are underpowered and should be read as descriptive."*
2. Add Wald tests for cross-journal equality of β_pos (JF vs RFS, JF vs JFE, RFS vs JFE). These are expected to be insignificant but their inclusion is methodologically required.
3. Change any in-text language claiming JF differs from RFS/JFE to hedged language: *"Point estimates suggest JF exhibits a larger premium, though the difference is not statistically distinguishable given sample sizes."*

---

## Part III — Empirical Additions (New Code Required)

### III.A  Promote and Expand Binary Directional Probit

**Script:** `scripts/06_analysis_main.py`

- Move the binary directional probit (currently R4) into the main analysis as Column (1) of Table III
- Re-label original three-way columns as Columns (2)–(4) in an appendix table
- Add a binary × journal interaction table as new Table III Panel B (replacing current Table IV structure or adding a panel)

### III.B  Add Formal Power Calculations

**Script:** `scripts/06_analysis_main.py`

- Compute power of Wald test H₀: β_pos = β_neg at observed N, variance components, and range of plausible effect sizes
- Report: minimum detectable difference (MDE) at 80% power
- Add as a footnote in Table III: *"Power of the Wald test H₀: β_pos = β_neg is X% at the observed sample size; minimum detectable difference at 80% power is Y percentage points."*
- Use `scipy.stats` or `statsmodels` for the calculation

### III.C  Abstract Length Robustness — Address Referee Directly

**Note:** R8 (abstract length control) already exists in `scripts/08_robustness.py`. The referee found this the most revealing result and calls the paper's dismissal of it inadequate. The paper must engage with it more seriously.

**New approach:**
- Do NOT dismiss the abstract-length control as merely endogenous. Instead, present it as a bounding exercise:
  - Column A (no length control): upper bound on the publication-direction association if abstract polish drives the finding
  - Column B (with log abstract length): lower bound if all abstract-length variation is editorial
  - True effect lies between A and B
- Add this framing explicitly to the robustness discussion in Section V.B

**Files:** `paper/main.tex`, `paper/tables/robustness_all.tex` (add framing to notes)

### III.D  Minimal Human Validation Audit

The referee specifically requests human validation of the coding. This is the most labor-intensive addition but also the most important for credibility.

**Scope (minimal but credible):** Hand-code 40 papers — 20 drawn randomly from the published sample and 20 from NBER — for:
1. In-scope (yes/no)
2. Direction (positive/negative/null/mixed)
3. Binary directional (yes/no)

**Process:**
- Use `scripts/04_code_direction.py` output to identify the 40 papers
- Code manually using original abstracts (not LLM output)
- Report Cohen's κ between human codes and LLM codes for each category
- Add as a new subsection in Section II.D ("Validation")
- If κ < 0.6 for null category (which is plausible given the existing flip-rate evidence), add it as a limitation and argue the binary spec is preferred precisely because it is more reliable

**Note:** The existing evidence that *15 of 20 null-coded papers flip in replication* is damaging and currently understated. The revision should: (a) report this prominently rather than burying it, (b) use it to motivate the binary specification (null vs directional is harder to flip), and (c) add human-coded validation to establish a floor on reliability.

---

## Part IV — What NOT to Do

- **Do not** add SSRN data or submission-stage data in this revision. The referee suggests it as ideal future work, not as a required revision. Attempting it would delay the paper and may not be feasible.
- **Do not** change the core empirical strategy. The referee's identification concerns are structural; addressing them requires reframing, not a new design.
- **Do not** add new journal coverage (JFQA, RoF) without substantial new data work. Out of scope for this revision.
- **Do not** drop the NBER comparison entirely. It remains the paper's most useful contribution; it just needs to be framed as a cross-sectional descriptive comparison, not a causal filter test.

---

## Execution Order

| Step | Task | Output | Est. Effort |
|---|---|---|---|
| 1 | Language audit: find all causal phrases in main.tex | List of line numbers | 30 min |
| 2 | Rewrite Abstract, Introduction opening paragraph | Revised text in main.tex | 1 hr |
| 3 | Write new Section III.A "Research Design and Identification" | ~400 new words in main.tex | 1.5 hr |
| 4 | Rewrite Section V mechanisms with supply/demand framing | Revised section in main.tex | 1 hr |
| 5 | Add exploratory labels to IV.D and caveats to IV.C | Small edits in main.tex | 30 min |
| 6 | Rewrite Limitations section to lead with identification concern | Revised section in main.tex | 45 min |
| 7 | Add power calculations to 06_analysis_main.py | Footnote for Table III | 1 hr |
| 8 | Add cross-journal Wald tests to 06_analysis_main.py | New column in Table IV | 45 min |
| 9 | Restructure Table III — binary spec as Column (1) | Revised table3_probit_main.tex | 1 hr |
| 10 | Add abstract-length bounding framing to robustness section | Revised text + table notes | 45 min |
| 11 | Hand-code 40-paper validation sample | New subsection II.D | 3-4 hr |
| 12 | Recompile PDF and verify all cross-references | Final PDF | 30 min |

**Total estimated effort:** ~14–15 hours  
**Suggested order:** Steps 1–6 first (pure text), then 7–10 (code + tables), then 11 (validation), then 12 (compile).

---

## Decision Point: Target Journal After Revision

The referee explicitly suggests the paper may be better suited to *"a top-field or interdisciplinary journal focused on meta-science, research methodology, or the economics of science"* rather than the Journal of Finance.

Candidate venues if JF rejects after revision:
- **Journal of Financial Economics** (less methods-focused than JF)
- **Review of Finance**
- **Journal of Financial and Quantitative Analysis**
- **Quarterly Journal of Economics / AER** (if reframed as economics-of-science paper)
- **PLOS ONE / eLife** (if reframed as meta-science with validation)
- **Journal of Accounting Research** (publication bias in accounting is an adjacent literature)

The revision plan above is designed to be appropriate for a JF resubmission *or* a resubmission elsewhere. The reframing toward descriptive/correlational language is necessary in either case.
