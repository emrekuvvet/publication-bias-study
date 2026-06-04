# Referee Response Plan — Round 5 (New Referee)
**Journal:** Journal of Corporate Finance
**Recommendation:** Major Revisions Required
**Date:** 2026-06-04

---

## Strategic Assessment

This is a new referee seeing the current (caliper-first) version. The tone is constructive and the paper structure is acknowledged as "innovative and highly policy-relevant." The rejection is conditional: five major concerns, most addressable without new data.

The critical path:

1. **Comment 2.2 (caliper reversal in coef+SE rows)** — The referee calls this a "massive red flag." This is the single most important thing to fix. The right response is: (a) star-consistency validation of z-stat-only rows to rule out extraction error; (b) a compelling theoretical mechanism for why z-stat-only rows show more bunching; (c) reframe the headline as "bunching in z-stat-only rows, absent in coef+SE rows" rather than "bunching overall."

2. **Comment 2.3 (power reframing)** — Text-only fix: replace "null result" framing with explicit MDE-bounded language throughout.

3. **Comment 2.4 (null omitted category)** — Switch to binary probit (Column 1) as primary specification.

4. **Comment 2.5 (QuasiExp confounding)** — `n_authors` already controlled; `top_institution` and `nber_citations` are empty in the dataset. Add the one available proxy (log author count) explicitly to the discussion; acknowledge institutional controls as future work.

5. **Comment 2.1 (NBER non-overlap)** — Already prominently discussed. The referee is asking for SSRN scraping — this requires new data collection beyond this session; flag as future work.

---

## Change Log

### R5-T1. Script 27: Z-stat-only star-consistency validation (Critical)
**Problem:** Referee alleges caliper reversal in coef+SE rows suggests extraction error in z-stat-only rows.
**Action:** New script that:
1. Cross-validates extracted z_abs against reported significance stars in z-stat-only rows
2. Reports fraction of z-stat-only rows with star-consistent z-values (expected: high)
3. Checks distribution of stars in the caliper window (1.645, 1.96) — if mostly '*', extraction is valid
4. Compares consistency rates: published z-stat-only vs NBER z-stat-only vs coef+SE rows
5. Table-type breakdown: what categories produce z-stat-only rows?
**Expected output:** `output/tables/zonly_star_consistency.csv`

---

### R5-T2. Add theoretical mechanism for z-stat-only bunching (Critical)
**Problem:** Referee asks "why would specification search only infect standalone statistics?"
**Theory:** Z-stat-only rows disproportionately represent *selectively reported* test statistics — from robustness panels, alternative specification summaries, and "additional tests" sections where authors have already selected which result to highlight. Main regression tables (coef+SE rows) systematically report ALL coefficients for all specifications in that table; supplementary test statistics are reported selectively. Selective reporting of individual test statistics is itself a form of within-paper specification search: the author chose which test to report and where to put it.
**Action:** Add 2-paragraph subsection (or `\noindent\textbf{Mechanism}` block) in §4.1 explaining:
(a) z-stat-only rows come disproportionately from supplementary/robustness tests, not from primary regression tables
(b) these selectively-reported statistics face an implicit significance filter (authors report them because they're noteworthy)
(c) coef+SE rows from full regression tables are systematically reported — not selectively filtered
(d) this explains the asymmetric bunching pattern without requiring "one-sided" p-hacking of primary estimates

---

### R5-T3. Power reframing throughout text (High)
**Problem:** Referee says the probit null result is described as "not systematically distorted" but the CI is wide enough to encompass large biases.
**Action:**
- Find all "no detectable directional sorting" / "null result" / "no evidence of direction-based sorting" language
- Add explicit MDE language: "The design can rule out effects ≥18 percentage points at 80% power (probit coefficient ≥0.466); moderate directional asymmetries below this threshold cannot be distinguished from zero."
- Change "null result" → "null result within the power of this design"
- Change "not systematically distorted" → language that acknowledges limited power explicitly

---

### R5-T4. Switch to binary probit as primary specification (High)
**Problem:** Referee says "a more reliable approach would be to rely primarily on the binary directional framework (Column 1)" given the 25% null/mixed agreement rate.
**Action:**
- Rewrite §4.2 intro to explicitly say: "Because the Null/Mixed omitted category has only 25% replication agreement, we treat Column (1) — the binary directional specification — as our primary specification. The three-way direction specification (Column 2) serves as robustness."
- Update abstract to reference binary directional coefficient as primary result
- Update probit table notes to relabel Column (1) as "[Primary]"

---

### R5-T5. JF sub-sample statistics distribution table (Medium)
**Problem:** Referee says the JF anomaly (caliper ratio 0.88 but density-discontinuity p<0.001) could be driven by a single outlier paper with massive tables.
**Action:** Add a short table or figure showing the distribution of extracted statistics across the 26 JF papers: mean, median, max, and identify any single paper with >100 statistics (potential outlier).

---

### R5-T6. AFA baseline divergence discussion (Medium)
**Problem:** AFA gives β=+0.327, NBER gives β=−0.020. Referee wants the opposing point estimates explained.
**Theory:** (1) AFA selection: conference presenters are a positively selected group — papers presented at AFA have already survived a competitive selection step before submission, so the AFA baseline is more "results-ready" than NBER WPs. (2) Sample composition: AFA papers cover a wider range of finance topics; NBER papers in our sample are restricted to government-intervention topics. (3) Both are underpowered to reject zero; the sign difference is within sampling variability.
**Action:** Add 3-4 sentence paragraph in the AFA results section (§5.4 or robustness) explicitly discussing the diverging signs.

---

### R5-T7. Add abstract hedging analysis (Medium)
**Problem:** Referee notes the 158-word/62% confidence vs. 109-word/81% confidence asymmetry is a "fascinating finding" and suggests quantifying it.
**Action:** Write a short analysis using `analysis_dataset.parquet` (has `abstract` text and `abstract_len`):
1. Count hedging words per abstract (define hedging word list: "suggest," "appear," "consistent with," "may," "could," "seem," "evidence suggests," "we find that," "implies," etc.)
2. Report: hedging word count per 100 words, by corpus (NBER vs published)
3. Add as a 2-sentence note in §3 (Data) or §2 (Limitations), with the counts in a small table or inline

---

### R5-T8. List CONTROL_RE regex in appendix (Low)
**Problem:** Referee asks for the exact regex taxonomy used to exclude control variables.
**Action:** Add the CONTROL_RE pattern from `scripts/25_primary_coeff_caliper.py` verbatim to the appendix (Appendix J), formatted as a verbatim/code block.

---

### R5-T9. SSRN/CEPR broader baseline (Deferred — requires new data)
**Status:** Requires web scraping of SSRN/CEPR for same author/topic corpus. Outside scope of current revision. Acknowledge as important future work in the limitations section; add 2-3 sentences describing what such a comparison would accomplish.

---

### R5-T10. Institutional quality controls (Partial — limited data)
**Status:** `top_institution` and `nber_citations` columns are empty in the current dataset. `n_authors` is already included as a control (Column 5). Add explicit text acknowledging this limitation and framing institutional controls as future work. Reaffirm that `n_authors` and `quasi_exp` are the only quality proxies currently available.

---

## Execution Order
1. R5-T1: Script 27 (z-stat-only star consistency) — generates validation table
2. R5-T2: Theoretical mechanism for z-stat-only (add to §4.1)
3. R5-T3: Power reframing (text changes)
4. R5-T4: Binary probit as primary (text changes)
5. R5-T5: JF distribution analysis + table
6. R5-T6: AFA divergence discussion (text)
7. R5-T7: Abstract hedging analysis (new analysis)
8. R5-T8: List CONTROL_RE regex in appendix (text)
9. R5-T9, R5-T10: Add deferred-work acknowledgment to limitations
10. Recompile + sync PDF
