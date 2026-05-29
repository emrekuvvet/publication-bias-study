# Referee Response Implementation Plan
## Journal of Finance — "Significance Bias in Government Intervention Research"
**Date:** 2026-05-14

---

## Overview

Five critical referee points require (a) new empirical work, (b) new robustness checks,
and (c) targeted rewriting. The table below maps each point to the deliverable.

| Referee Point | Type | Primary File(s) |
|---|---|---|
| P1: NBER not a valid control | Text + robustness | main.tex, 06_analysis_main.py |
| P2: Asymmetric LLM measurement error | Text + new robustness | main.tex, 06_analysis_main.py |
| P3: Quality alternative inadequate | New LLM coding + new spec | 09_code_id_strategy.py, 06_analysis_main.py, main.tex |
| P4: Supply vs demand indistinguishable | Text rewrite | main.tex |
| P5: Sample too small for subsidiary claims | Power calcs + labeling | 06_analysis_main.py, main.tex |

---

## Point 1 — NBER Sample Is Not a Valid Cohort Control

**Problem:** 85.8% of published papers have no NBER match. The design compares two
largely non-overlapping populations, not papers tracked through the editorial filter.

**Changes:**

### 1a. Methods section — new "Design and Identification" subsection
- Add explicit subsection III.A (before the probit spec) titled "Research Design and Identification Assumptions"
- State clearly: this is a cross-sectional comparison between the published distribution and
  the NBER-circulated distribution — not a within-paper tracked cohort design
- State the identifying assumption formally: NBER working papers constitute a draw from
  the pre-publication distribution of findings in this domain, conditional on NBER affiliation
- Contrast with stronger designs (Franco 2014 NSF grants; DellaVigna 2022 RCT registry)
  and explain why those are infeasible here

### 1b. Limitations — strengthen existing paragraph
- Rewrite "Low NBER-to-journal match rate" to lead with the design limitation rather than
  bury it; acknowledge that this is the most fundamental constraint
- Add sentence: estimates should be interpreted as associations in a cross-section,
  not causal publication filter effects

### 1c. New robustness R8 — matched-pairs-only analysis
- Among the 26 matched papers, compare direction codes between their NBER and
  published versions to test whether direction shifts between draft and publication
- Add to robustness table and script 06

---

## Point 2 — LLM Measurement Error Is Asymmetric and Could Generate the Finding Spuriously

**Problem:** Published abstracts are ~2.4x longer than NBER abstracts (109 vs 44 words).
The LLM may code published papers as directional not because they ARE directional but because
polished abstract language signals directionality. This is an upward bias concern, not just
attenuating.

**Changes:**

### 2a. New robustness R9 — abstract-length control
- Add `log(abstract_len)` as a covariate in the probit
- Note: this control is endogenous (abstract length is partly shaped by editorial revision),
  so the coefficient is not interpretable causally — but if direction coefficients are
  unchanged, it provides reassurance that abstract polish does not drive the result
- Add to script 06_analysis_main.py and robustness table

### 2b. Rewrite R5 in paper (abstract language section)
- Acknowledge the two-sided nature of the concern:
  (i) attenuating direction: LLM codes directional NBER papers as null → understates premium
  (ii) inflating direction: LLM codes published papers as directional due to polish → overstates premium
- Show which direction dominates using the replication exercise data
- The robustness with log(abstract_len) control addresses (ii)

---

## Point 3 — Quality Alternative Requires a Stronger Control Than Author Count

**Problem:** Author count is unrelated to paper quality in finance. The referee demands
control for identification strategy (RDD, DiD, IV vs. OLS).

**Changes:**

### 3a. New script: 09_code_id_strategy.py
- Use Claude API to classify identification strategy for each of the 296 papers
- Classification scheme:
  - `quasi_exp`: Regression Discontinuity (RDD), Instrumental Variables (IV),
    Difference-in-Differences (DiD), or Event Study with causal interpretation
  - `ols_reduced`: OLS, reduced-form correlation, panel regression without instrument
  - `structural`: Structural/calibration model
  - `unclear`: Cannot determine from abstract
- Save results to `data/final/analysis_dataset.parquet` (populate `id_strategy` column)
- Validation: check agreement across 20 random re-runs

### 3b. New probit Spec 5 — identification strategy control
- Add `quasi_exp` dummy as covariate in main probit
- If quasi-experimental papers are more likely to find directional results (stronger
  identification → cleaner estimates → more likely significant), this would be a quality
  confound; the dummy absorbs it
- Add as Column (5) in Table III
- Add robustness restricting to quasi-experimental papers only

### 3c. Update paper text
- Replace "author count is our quality proxy" with results from both controls
- Add discussion: quasi-experimental dummy is still imperfect (strategy coded from abstract;
  quality within RDD papers varies widely) — acknowledge residual concern

---

## Point 4 — Supply vs Demand Mechanisms Cannot Be Distinguished

**Problem:** Strategic non-submission (supply side) and editorial selection (demand side)
generate identical patterns. Policy recommendations assume demand-side mechanism.

**Changes:**

### 4a. Rewrite Mechanisms subsection (Section V.C)
- Add explicit paragraph: "Two distinct mechanisms could generate our findings..."
- Supply side: researchers with null results self-select out of top-journal submissions,
  anticipating rejection. Under this interpretation, the data reflect efficient sorting
  rather than editorial distortion.
- Demand side: editors/referees differentially advance directional papers through review.
- The time-variation evidence (Section IV.D) is described as "more consistent with" demand
  side but explicitly cannot rule out supply side.

### 4b. Rewrite Policy Implications subsection (Section V.A)
- Soften language: "our results are consistent with" rather than "our results imply"
- Add sentence: if the mechanism is primarily supply-side, editorial interventions
  (registered reports, results-blind review) may have limited effect — addressing the
  author-incentive side would be more important
- Add explicit statement of what additional data would be needed (desk-rejection rates,
  submission-level data from editors) to distinguish the channels

---

## Point 5 — Power and Exploratory Labeling

**Problem:** Wald test for β₁=β₂ has low power at N=296. Sub-period and cross-journal
results presented as findings but are severely underpowered.

**Changes:**

### 5a. Add power calculations to script 06
- Compute power of Wald test H₀: β₁=β₂ at observed sample size, variance, and effect sizes
- Use asymptotic chi-squared distribution; report minimum detectable difference at 80% power
- Add as footnote in Table III

### 5b. Add cross-journal Wald tests to Table IV
- For each pair of journals (JF vs RFS, JF vs JFE, RFS vs JFE), test whether β_pos
  coefficients are statistically distinguishable
- These tests are expected to be insignificant (power is low) but reporting them is
  methodologically required

### 5c. Label sub-period analysis as exploratory
- Change Section IV.D heading from "Time-Period Heterogeneity" to
  "Time-Period Heterogeneity (Exploratory)"
- Add opening sentence: "The following sub-period analysis is exploratory; sub-sample
  sizes are small and the results should be interpreted as descriptive evidence, not
  confirmatory tests."
- Add 95% confidence intervals to coefficient values in the sub-period discussion

---

## Execution Order

1. Run `09_code_id_strategy.py` — LLM calls (~296 papers, ~5 min)
2. Update `06_analysis_main.py` — new specs (abstract_len, id_strategy, power calc, Wald tests)
3. Update `08_robustness.py` — matched-pairs robustness (R8), abstract-len robustness (R9)
4. Run both scripts to regenerate all output tables
5. Rewrite `main.tex` — all text changes (P1, P2, P4, P5 text; new table columns; new footnotes)
6. Recompile PDF
