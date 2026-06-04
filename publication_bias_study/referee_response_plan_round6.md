# Referee Response Plan — Round 6 (New Referee, Major Revision)
**Journal:** Journal of Corporate Finance
**Recommendation:** Major revision with significant reservations
**Date:** 2026-06-04

---

## Strategic Assessment

This referee is more skeptical than previous ones and raises structurally important points. The paper's headline is in trouble: the entire caliper finding lives in z-stat-only rows (unverifiable), and the coef+SE-only comparison (verifiable) shows NO gap (published 0.81, NBER 0.95). The referee is not wrong. The right response is a structural pivot:

**New headline:** "We find that systematically-reported primary regression rows (coef+SE) show no caliper gap between published and NBER corpora. Bunching is concentrated in standalone z-statistics from supplementary tables — a pattern consistent with selective reporting of which individual test results to highlight, not systematic manipulation of primary regression estimates."

This is actually a sharper and more defensible finding. The paper becomes about selective reporting of supplementary statistics, not about revision-stage p-hacking of primary estimates.

The critical path:

1. **Comment 3.1 (coef+SE as primary)** — Most important. Structural rewrite of the caliper results section. Coef+SE comparison becomes the headline; z-stat-only bunching is the mechanism/secondary finding. Add decomposition separating composition effect from within-type effect.
2. **Comment 3.2 (bootstrap inference)** — Foreground paper-level bootstrap CI [0.81, 2.67] throughout. Drop "sharp contrast," "strikingly asymmetric." Be even-handed across all three inference procedures.
3. **Comment 3.5 (human-vs-LLM κ in main text)** — The human validation κ must appear in the paper body, not just appendix.
4. **Comment 3.4/3.5 (QE variable)** — Downgrade substantially or add genuine validation. We cannot run a full hand-coding exercise, but we can (a) compute κ-equivalent for QE coding from the already-existing replication run and (b) add much stronger downgrade language.
5. **Comment 3.6 (window sensitivity to main text)** — Move from Appendix I to main text.
6. **Comments 3.1/3.3 (title, abstract)** — New title removing "Pre-Publication Evidence" and leading with the two-corpus design rather than the bunching result.
7. **Comment 3.5 (prompt-perturbation)** — Add explicit discussion that second LLM run tests model sensitivity, not prompt sensitivity; acknowledge limitation.
8. **Comment: Appendices not received** — Our PDF is 75 pages and includes appendices. Flag in cover letter that appendices are included from p. 59 onward.

---

## Change Log

### R6-T1. Restructure caliper results: coef+SE as primary specification (Critical)
**Problem:** Referee says verifiable coef+SE rows show no bunching; the entire result lives in the unverifiable z-stat-only subset.
**Action:**
- Rewrite §4.1 (caliper results) opening to lead with coef+SE comparison as primary
- New narrative: "Our primary caliper specification restricts to rows with verifiable coefficient-SE pairs (5,510 published, 5,149 NBER). In this verifiable subsample, the published caliper ratio is 0.81 — below 1, statistically indistinguishable from the NBER ratio of 0.95 (cluster-bootstrap p = [X]). **No systematic gap exists in primary regression rows.**"
- Secondary finding: "In standalone z-statistics (z-stat-only rows), bunching is present in both corpora and substantially larger in published papers (2.55 vs. 1.49) — consistent with selective reporting of which supplementary statistics to highlight."
- Add decomposition paragraph: separate the overall 1.20 published ratio into (a) within-type effect and (b) composition effect (different z-stat-only shares: 29.0% published vs. 18.7% NBER)
- Rewrite conclusion's caliper summary to reflect this reframing

### R6-T2. Decomposition script: composition vs. within-type caliper gap (Critical)
**Problem:** Referee specifically asks for re-weighting to separate composition effect from within-row-type effect.
**Action:** Write Python script (`scripts/28_caliper_decomposition.py`) that computes:
- Observed overall ratios: published 1.20, NBER 1.01
- Counterfactual published ratio: using NBER's z-stat-only share (18.7%) with published within-type ratios
- Counterfactual NBER ratio: using published's z-stat-only share (29.0%) with NBER within-type ratios
- Decompose overall gap into: (a) within-type effect, (b) composition effect
- Output a clean table for the paper

### R6-T3. Title change (Critical)
**Problem:** "Pre-Publication Evidence" oversells; "Threshold Bunching" leads with fragile result.
**New title:** "Selective Reporting and Methodological Sorting in Published Finance Research: A Two-Corpus Analysis of Government-Intervention Studies"
OR: "Within-Paper Selective Reporting in Finance: Evidence from a Two-Corpus Caliper Analysis"
**Selected:** "Selective Reporting in Published Finance Research: A Two-Corpus Comparison of Government-Intervention Studies"

### R6-T4. Abstract rewrite (Critical)
**Problem:** Abstract leads with "sharp contrast," includes ≈46% corpus-membership rate, buries bootstrap CI, and still oversells the bunching result.
**Action:** Rewrite abstract to:
- Lead with the primary coef+SE null finding
- Present z-stat-only bunching as the secondary mechanism finding
- Foreground cluster-bootstrap CI [0.81, 2.67] for the full-sample comparison
- Remove ≈46% corpus-membership rate (or relegate to one clause)
- Remove "sharp contrast," "strikingly asymmetric"
- Shorten overall; move inline caveats to a single hedging sentence at end

### R6-T5. Foreground bootstrap inference throughout (High)
**Problem:** Referee says the paper leans on permutation test; bootstrap gives full-sample p = 0.150 and negative-finding p = 0.135 (both insignificant).
**Action:**
- In caliper results section: lead with bootstrap CI before permutation p
- Rewrite the directional (negative-finding) claim: "The directional result is statistically fragile: while the within-paper permutation p < 0.001, the paper-level cluster bootstrap gives p = 0.135 with CI [0.81, 2.67]; this should be treated as suggestive rather than definitive."
- Add a sentence to the summary box/paragraph making clear all three inference methods and their conclusions
- Update Conclusion to reflect this nuance

### R6-T6. Human-vs-LLM κ in main text (High)
**Problem:** The human-validation κ (against a human gold standard) is referenced but never reported in the main text.
**Action:**
- Read current Appendix H (human validation) to extract the κ statistic
- Add to §3.4 (or wherever direction coding is described): "Our validation against a hand-coded gold standard of [N] papers yields a human-LLM linear weighted κ of [X], with the confusion matrix reported in Appendix H. Agreement is near-perfect for directional papers but drops to 25% for null/mixed papers (see Section 3.3)."
- The χ² joint-equality test (p = 0.992) should be foregrounded as the most robust probit result given null baseline instability

### R6-T7. QE variable: validate or substantially downgrade (High)
**Problem:** Referee says QE variable has no validation at all (unlike direction codes). Either hand-code a sample or dramatically downgrade claims.
**Action:**
- Check whether the LLM replication run produced a QE code alongside direction; if so, compute QE inter-run κ
- Add strong downgrade text in the probit results section: "We emphasize that the quasi-experimental dummy has received no validation comparable to the direction coding. We cannot rule out that it proxies for author quality, institutional affiliation, or topic fashionability rather than identification rigor. This coefficient should be treated as a preliminary association, not a validated causal claim."
- Move this variable out of the abstract's primary results; it currently reads as a headline finding
- In the abstract, replace the unhedged 0.467*** claim with a heavily caveated version or remove

### R6-T8. Window sensitivity table to main text (High)
**Problem:** Appendix I (window sensitivity) shows that widening the lower bound to |z| = 1.50 reverses the sign for published papers. Referee says this must be in the main text.
**Action:**
- Move the window sensitivity table from Appendix I to a subsection in §5 (Robustness)
- Add text: "The caliper result is window-dependent. Using the Brodeur et al. (2016) window (1.645, 1.96) vs. (1.96, 2.275), the published ratio is 1.20. Widening the lower bound to |z| = 1.50 reverses the sign, because density is still rising below 1.645. This confirms the result is specific to the standard caliper window and should not be interpreted as evidence of a general discontinuity."

### R6-T9. JF internal tension: reconcile caliper vs. density-discontinuity (Medium)
**Problem:** JF caliper ratio = 0.88 (p = 0.423, no bunching) but density-discontinuity p < 0.001. Referee says this looks like two noisy tests disagreeing.
**Action:** Add explicit reconciliation paragraph: "These two tests answer different questions. The caliper ratio tests whether mass in (1.645, 1.96) exceeds mass in (1.96, 2.275); a ratio of 0.88 indicates there is actually slightly less mass just below 1.96 than above. The density-discontinuity test asks whether the density drops sharply at 1.96 regardless of the adjacent mass comparison. The JF pattern — smooth density with a sharp step at 1.96 — is geometrically consistent with both: mass may be shifted further below 1.96 rather than concentrated just below it. We caution that with 26 JF papers the subsample is too small for confident inference from either test."

### R6-T10. Multiple comparisons: state total test count (Medium)
**Problem:** Referee asks how many caliper/discontinuity comparisons were run in total and which were pre-specified.
**Action:** Add a paragraph in §4 or §5 (Robustness):
"Multiple testing: we run 12 caliper and 12 density-discontinuity comparisons across journal and direction subgroups [list them]. Of these, the full-sample and the negative-finding published comparison were pre-specified as primary; all journal-level and additional direction subgroups are exploratory. Bonferroni-adjusted significance threshold for the 2 pre-specified tests is p < 0.025; for all 12, p < 0.004."

### R6-T11. Prompt-perturbation discussion (Medium)
**Problem:** Referee points out that the second LLM "rater" (Haiku) uses the same prompt as the first — it tests model sensitivity, not true inter-rater reliability.
**Action:** Add to §3.4 or limitations: "We note that our inter-rater reliability estimate uses a different model (Claude Haiku) with the same prompt, which tests model-choice sensitivity but not prompt sensitivity. Two LLMs sharing a prompt are not independent raters in the same way as two trained human coders; our reported κ = 0.674 should be interpreted as a model-stability check, not as a true inter-rater agreement. Prompt-perturbation robustness (varying the instruction wording systematically) is a natural extension that we leave for future work."

### R6-T12. LLM model citation and reproducibility caveat (Medium)
**Problem:** Need precise model version, access date, and a reproducibility warning.
**Action:** In methodology: "Direction coding was performed using Claude Opus 4.7 (model ID: claude-opus-4-7, accessed [date]) for the primary run and Claude Haiku 4.5 (model ID: claude-haiku-4-5-20251001) for the replication run. Closed-weight models evolve over deployment and may not produce identical outputs in the future; this is a genuine threat to exact replication and we therefore provide all input abstracts and output codes in our replication package alongside the exact model IDs used."

### R6-T13. Policy claims softening in Section 7.1 (Medium)
**Problem:** "Not systematically distorted" is too strong given 22pp MDE and the coef+SE null for caliper.
**Action:**
- Rewrite the policy-implications paragraph to: "Within the power constraints of this design, we find no evidence of large-scale directional distortion. Moderate asymmetries — below the 22-percentage-point detectability threshold — remain entirely consistent with the data. Policymakers relying on this literature should treat the directional-neutrality finding as an upper bound on detectable bias, not as evidence that no bias exists."

### R6-T14. Journal fit: re-center abstract and introduction on corporate finance (Medium)
**Problem:** Referee says the paper feels like meta-science rather than corporate finance; Section 7.1 feels "bolted on."
**Action:**
- Restructure the introduction to begin with the corporate-finance-specific interventions (Sarbanes-Oxley, Dodd-Frank, capital requirements, say-on-pay, proxy access) before broadening
- Move §7.1 (Corporate Finance Regulation) content earlier — incorporate key numbers into the introduction rather than siloing them in a late section
- In the abstract, lead with "corporate finance regulation literature" rather than "government-intervention literature"

### R6-T15. Appendices note (Low)
**Problem:** Referee states the PDF received ended at Conclusion (p. 58) without appendices.
**Action:**
- Verify our compiled PDF is 75 pages (it is) with appendices A–K
- Add note in cover letter: "The full manuscript including Appendices A–K begins at p. 59 of the attached PDF. If only 58 pages were received, please use the resubmitted file."
- No change to main.tex needed

---

## Execution Order
1. R6-T2: Decomposition script (generates numbers for T1 text)
2. R6-T1: Restructure caliper results (text changes — needs T2 numbers)
3. R6-T3: Title change
4. R6-T4: Abstract rewrite
5. R6-T5: Bootstrap inference foregrounding
6. R6-T6: Human-vs-LLM κ in main text (needs reading Appendix H)
7. R6-T7: QE variable downgrade
8. R6-T8: Window sensitivity to main text
9. R6-T9: JF reconciliation paragraph
10. R6-T10: Multiple comparisons accounting
11. R6-T11: Prompt-perturbation discussion
12. R6-T12: LLM model citation
13. R6-T13: Policy claims softening
14. R6-T14: Journal fit / recentering
15. Recompile + sync PDF
