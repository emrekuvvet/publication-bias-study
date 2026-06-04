# Referee Response Plan — Round 4 (Same Referee, Second Round)
**Journal:** Journal of Corporate Finance
**Recommendation:** Reject
**Date:** 2026-06-04

---

## Strategic Assessment

The referee acknowledges the round-3 improvements are "substantial" and thanks us for the transparency. The rejection is narrower than before and rests on one structural message:

> "The within-paper test-statistic analysis remains potentially publishable with further development, but the publication-sorting analysis does not yet provide sufficiently persuasive evidence."

The referee explicitly says: "I would encourage them to reorient the manuscript around the test-statistic evidence." The path to acceptance at this journal is therefore:

1. **Make the caliper analysis structurally primary** — move it to §4.1 (first in Results)
2. **Demote the probit to "Supporting Context"** — explicitly rename/reframe
3. **Fix "directly observes" → "proxy for"** — referee explicitly flags this as still present
4. **Narrow the QuasiExp interpretation** — LLM classification captures more than identification quality alone
5. **Strengthen null/mixed instability discussion** — 25% agreement complicates the omitted-category interpretation; add dedicated paragraph
6. **Update intro and conclusion to signal caliper-first** throughout

None of these require new data. All are framing and structural changes.

---

## Change Log

### R4-T1. Reorder §4 Results: caliper first (Critical)
**Problem:** Currently §4 Results has probit analysis (§4.1–§4.5) before caliper (§4.6). This structurally signals probit as primary, contradicting the new framing.
**Action:** Move `\subsection{Within-Paper Test Statistic Distributions}` block (lines ~1268–1693) to come before `\subsection{Directional vs. Null Findings}` (line ~978). Result: caliper is §4.1, probit subsections are §4.2+.

---

### R4-T2. Rewrite caliper section opening sentence (Critical)
**Problem:** Current opening ("A limitation of the cross-sectional design is that it cannot detect within-paper selective reporting...") frames caliper as a supplement to an already-discussed probit. Once caliper comes first, this sentence is wrong.
**New opening:** "We begin with the paper's primary empirical test: whether published papers in the government-intervention literature show excess mass just below conventional significance thresholds, relative to NBER working papers on the same topic. We apply the \citet{Brodeur2016} caliper test..."

---

### R4-T3. Add "Supporting Context" framing before probit (Critical)
**Problem:** The probit section currently begins with no signal that it is secondary.
**Action:** Before `\subsection{Directional vs. Null Findings}`, insert a paragraph:
"As supporting context, we examine whether the caliper bunching pattern is accompanied by directional sorting at the corpus-membership level. If threshold bunching were driven by editorial selection favoring certain directions, we would also expect direction to predict corpus membership. As the following analysis shows, it does not."

---

### R4-T4. Fix "directly observes" → "proxy for" language (High)
**Problem:** Referee Point 3: "The paper should consistently use language such as 'proxy for the pre-publication distribution' rather than 'direct observation of the pre-publication distribution.'"
**Instances to fix:**
- Line ~399: "Our paper sidesteps this problem by directly observing the pre-publication distribution through NBER working paper full texts"
- Line ~1296: "The NBER sample therefore represents the pre-publication distribution of test statistics from the same research domain, allowing a direct..."
**Action:** Replace with "proxy for" / "closest available approximation to" language.

---

### R4-T5. Narrow QuasiExp interpretation (High)
**Problem:** Referee Point 4: LLM classification captures topic choice, author quality, correlation with other quality proxies — not just identification quality. "The interpretation should be narrower."
**Action:** At the point where we interpret the QuasiExp coefficient as "evidence journals reward methodological rigor," add qualifier: "though we caution that the LLM-coded quasi-experimental dummy is likely correlated with author quality, topic type, and other dimensions of paper quality that our probit cannot separate from identification strategy alone. The coefficient should be interpreted as reflecting a bundle of correlated attributes, with identification strategy as a plausible—but not the only—contributing factor."

---

### R4-T6. Null/mixed instability — omitted-category implications (High)
**Problem:** Referee Point 5: "Given that null papers are the omitted category in the probit models, instability in this category complicates interpretation considerably." Current text discusses directional bias from misclassification but not specifically the omitted-category interpretation problem.
**Action:** Add a dedicated paragraph in §5 (Limitations) explicitly addressing: (a) null papers are the probit omitted category, (b) with only 25% replication agreement, the omitted-category baseline is uncertain, (c) this means direction dummies absorb any instability in the null baseline — inflating apparent precision of direction coefficients without necessarily inflating their magnitude, (d) the conservative-bias argument still holds mechanically.

---

### R4-T7. Introduction: restructure contributions paragraph (High)
**Problem:** Contributions paragraph currently lists caliper and probit as roughly co-equal. Needs caliper-first ordering with probit explicitly labeled as supporting.
**Action:** Rewrite contribution paragraph so:
(1) First bullet: caliper finding (primary contribution)
(2) Second bullet: probit null result (supporting context)
(3) Third bullet: QuasiExp / LLM pipeline (methodology)

---

### R4-T8. Conclusion: caliper-first (Medium)
**Problem:** Conclusion discusses both contributions roughly equally; caliper should be the headline.
**Action:** Rewrite first two paragraphs of conclusion to lead with caliper finding, then present probit null as supporting.

---

### R4-T9. Add roadmap sentence to Results section intro (Medium)
**Problem:** No explicit signal at the start of §4 that caliper is primary.
**Action:** Add one sentence at the start of §4 Results: "We organize results caliper-first: Section~\ref{subsec:pvalue_dist} presents the primary within-paper test-statistic analysis; Section~\ref{subsec:direction_results} presents the cross-sectional direction-sorting analysis as supporting context."

---

### R4-T10. Fix NBER as "proxy" in empirical strategy section (Low)
**Problem:** §3 Empirical Strategy already calls NBER "closest available proxy" (good!) but other instances remain.
**Action:** Scan and fix any remaining "direct" / "directly observes" language in §3.

---

## Execution Order
1. R4-T1: Section reorder (Python script)
2. R4-T2: Caliper intro rewrite
3. R4-T3: Supporting context framing
4. R4-T4: "Directly observes" → "proxy for"
5. R4-T5: QuasiExp narrowing
6. R4-T6: Null/mixed instability paragraph
7. R4-T7: Intro contributions rewrite
8. R4-T8: Conclusion rewrite
9. R4-T9: Roadmap sentence
10. Recompile + sync PDF
