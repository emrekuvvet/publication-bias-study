# Referee Response Plan — Round 3 (New Referee)
**Journal:** Journal of Corporate Finance
**Recommendation:** Reject (encourage redesign)
**Date:** 2026-06-04

---

## Strategic Assessment

The referee has four substantive objections:
1. The NBER/journal samples are non-overlapping populations, not the same papers tracked pre/post.
2. The 46% "publication rate" is a corpus-membership ratio, not an actual acceptance rate.
3. The title and prose ("editorial neutrality," "revision-stage pressure") claim more than the evidence shows.
4. The caliper analysis is the genuine contribution; the probit is secondary.

Critically: none of these require new data. Almost all are framing and interpretation issues the paper already acknowledges partially, but not prominently enough. The path to acceptance is:
- Lead with what is directly observed (threshold bunching in published test statistics vs. NBER working papers)
- Demote the probit to "supporting context" (it already is in the abstract; needs to match throughout)
- Rename/reframe "publication rate" and "editorial neutrality" throughout
- Move underpowered cross-journal probit to appendix
- Strengthen existing identification caveats

---

## Change Log

### T1. Title (Critical)
**Problem:** "Editorial Neutrality and Revision-Stage Pressure" — both are inferred mechanisms, not directly observed findings. Referee explicitly flags this.

**Old:** "Editorial Neutrality and Revision-Stage Pressure in Top Finance Journals: Evidence from Government Intervention Research"

**New:** "Threshold Bunching in Published Finance Research: Pre-Publication Evidence from Government-Intervention Studies"

---

### T2. Abstract — paragraph 1 (Critical)
**Problem:** "pre-publication counterparts" without noting the samples are non-overlapping.
**Action:** Add "Note that the NBER and journal samples are largely non-overlapping cross-sections (only ~12% of published papers are matched to NBER counterparts), so this comparison reflects differences between corpora, not the same papers tracked pre- and post-publication."

---

### T3. Abstract — paragraph 2 (Critical)
**Problem 1:** "publication rates" is not what is measured.
**Problem 2:** "editorial selection is directionally neutral" overclaims.
**Actions:**
- Change "publication rates" → "empirical corpus-membership probabilities in this sample"
- Change "editorial selection is directionally neutral" → "no detectable directional sorting between corpora in this cross-section"

---

### T4. Introduction — line ~211 (High)
**Problem:** "we observe actual pre-publication findings" — not true, we observe NBER working papers.
**Action:** Replace with "we observe findings from working papers that circulated before potential journal publication, rather than merely inferring the pre-publication distribution from published p-value shapes"

---

### T5. Introduction — early framing (High)
**Problem:** Introduction discusses journal-specific probit results in detail (lines 259-268) from a table being moved to appendix.
**Action:** Compress journal-specific intro discussion to one sentence; redirect to appendix.

---

### T6. Move Cross-Journal Probit (§4.2) to Appendix (High)
**Problem:** Referee says "I would move these analyses to an appendix." JF has only 26 papers; results are highly underpowered.
**Action:**
- Move subsection text and table (tab:probit_by_journal) to Appendix
- Replace in main text with one-sentence pointer: "Journal-level estimates are reported in Appendix~\ref{app:by_journal}; no journal shows a significant direction coefficient, consistent with the pooled result, though the JF sub-sample (26 papers) has very limited power."

---

### T7. "Publication rate" language (High)
**Problem:** The 46% "publication rate" is mechanical; depends entirely on corpus construction. Referee says this has no economic meaning as presented.
**Action:** At first introduction (§3 results section), add explicit disclaimer: "We emphasize that this rate is a corpus-membership ratio determined by our sample construction, not an estimate of true acceptance probabilities. It depends on the relative sizes of the NBER and journal corpora and on NBER program coverage."
- In Table notes and text: change "publication rate" → "corpus membership rate" (or "empirical publication probability in this sample") consistently

---

### T8. "Editorial neutrality" → "no detectable directional sorting" (High)
**Instances to fix:** title (done), abstract paragraph 2, introduction, conclusion.
**Action:** Systematic replacement throughout.

---

### T9. "Revision-stage pressure" → calibrated language (Medium)
**Problem:** "revision-stage pressure" is an inferred mechanism, not directly observed.
**Action:** Replace with "threshold bunching concentrated in the published record, consistent with revision-stage specification pressure" — keeping the observation and softening the mechanism claim.

---

### T10. Power argument (Medium)
**Problem:** Referee says "equal significance rates do not imply equal statistical power."
**Action:** Add paragraph in §5.2 (NBER comparison results) explicitly noting: (a) equal significance rates are one relevant data point; (b) the referee is correct that this alone doesn't fully rule out power differences; (c) the published/NBER bunching *gap*, not just the levels, is the relevant statistic — if NBER papers had lower power, we would expect *lower* significance rates but *no bunching* below 1.96 (threshold-clearing search) — the two pieces of evidence together constrain the interpretation.

---

### T11. NBER benchmark caveats (Medium)
**Problem:** Current limitations paragraph is adequate but doesn't emphasize enough that NBER selection is a core concern, not a minor caveat.
**Action:** Rewrite the NBER limitations paragraph to put researcher-selection, timing-selection, and prestige-selection concerns upfront rather than as a list of minor qualifications.

---

### T12. Move journal-specific text in intro to appendix pointer (Low)
**Problem:** Lines 259-268 in intro discuss journal-specific coefficients in detail before they get to a section being moved.
**Action:** Compress to one sentence.

---

### T13. "Directly observes pre-publication findings" language (Low)
**Instances:** Introduction (~line 211), possibly elsewhere.
**Action:** Fix to "observes findings from NBER working papers, which circulated before potential publication."

---

## Execution Order
1. Title change
2. Abstract rewrites (T2, T3)
3. Introduction fixes (T4, T5, T12, T13)
4. Move §4.2 + table to appendix (T6)
5. "Publication rate" fixes (T7)
6. "Editorial neutrality" fixes (T8)
7. "Revision-stage pressure" fixes (T9)
8. Power paragraph (T10)
9. NBER caveats (T11)
10. Recompile + sync PDF
