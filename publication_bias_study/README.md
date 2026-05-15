# Publication Bias in Top Finance Journals — Replication Package

**Paper:** *Significance Sorting in Government Intervention Research: Evidence from Top Finance Journals*

This repository contains the complete data pipeline and analysis code to replicate
all results in the paper. The paper documents a cross-sectional association between
finding direction and publication status in top finance journals. **All regression
coefficients are associations, not causal editorial-filter effects**; the design
cannot distinguish editorial selection from author self-selection in submission or
specification search.

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.11+ |
| DuckDB | 0.10.3 |
| Anthropic API key | Required for scripts 03, 04, 09, and 11 (without --no-api) |
| Web of Science API key | Required for script 02 (CSV fallback available) |

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys.

---

## Directory Structure

```
publication_bias_study/
├── README.md
├── implementation_plan.md
├── chatgpt_referee_response_plan.md   ← revision plan (ChatGPT referee report)
├── requirements.txt
├── setup_db.py              ← run first
├── .env.example
├── data/
│   ├── raw/                 ← populated by scripts 01–02
│   ├── classified/          ← populated by scripts 03–04
│   └── final/               ← populated by script 05
├── scripts/
│   ├── 01_collect_nber.py
│   ├── 02_collect_journals.py
│   ├── 03_classify_inscope.py
│   ├── 04_code_direction.py
│   ├── 05_link_nber_to_published.py
│   ├── 06_analysis_main.py
│   ├── 07_tstat_distribution.py
│   ├── 08_robustness.py
│   ├── 09_code_id_strategy.py
│   ├── 10_collect_afa.py          ← AFA conference program scraper
│   └── 11_afa_analysis.py         ← AFA baseline robustness pipeline
├── validation/
│   ├── generate_coding_sheet.py
│   ├── human_coding_sheet.xlsx   ← generated; filled by RAs
│   └── interrater_reliability.py
├── paper/
│   ├── main.tex
│   └── references.bib
└── output/
    ├── tables/
    └── figures/
```

---

## Replication Steps

### Step 0 — Initialise database

```bash
python setup_db.py
```

Creates `publication_bias.duckdb` with views over all Parquet files.

---

### Step 1 — Collect NBER working papers

```bash
python scripts/01_collect_nber.py
```

- Source: `https://www.nber.org/api/v1/search?facet=contentType:working_paper` (free, no key required)
- All programs — program code filtering is handled downstream in script 03 via keyword/LLM
- Years: 1990–2024 (~25,000 papers)
- Output: `data/raw/nber_papers.parquet`
- Expected runtime: ~10–12 minutes

---

### Step 2 — Collect journal papers (JF, RFS, JFE)

```bash
python scripts/02_collect_journals.py
```

- **Primary source: OpenAlex** (`https://api.openalex.org`) — free, no key required
- JF: ~4,200 papers | RFS: ~2,800 papers | JFE: ~3,500 papers (1990–2024)
- Expected runtime: ~5–8 minutes total
- Output: `data/raw/{jf,rfs,jfe}_papers.parquet`

**Abstract coverage note:** The Journal of Financial Economics (JFE) restricts
abstract access through Elsevier's agreement with OpenAlex. Most JFE abstracts
(~2001–2020) are unavailable from any free API. JFE papers without abstracts
are classified using title-only in script 03; the paper acknowledges this as a
data limitation. For full coverage, use the Web of Science CSV fallback:

```bash
# Manual WoS CSV fallback (requires institutional access)
python scripts/02_collect_journals.py --csv path/to/jfe_export.csv --journal jfe
```

---

### Step 3 — Classify in-scope papers

Requires `ANTHROPIC_API_KEY` in `.env`.

```bash
python scripts/03_classify_inscope.py
```

For a quick test without API calls:
```bash
python scripts/03_classify_inscope.py --skip-llm --sample 500
```

Output: `data/classified/inscope_papers.parquet`
Estimated API cost: $10–20 depending on corpus size.

---

### Step 4 — Code direction of findings

```bash
python scripts/04_code_direction.py
```

If interrupted, resume without re-coding:
```bash
python scripts/04_code_direction.py --resume
```

Output: `data/classified/coded_direction.parquet`
Estimated API cost: $30–80.

**Reliability note:** Agreement between two independent LLM runs is near-perfect
for directional papers (positive: 90%, negative: 100%) but lower for null papers
(25%). The null-coding instability is conservative — ambiguous papers flip toward
directional codes in replication, which would *attenuate* the estimated gap.
The binary directional specification (script 06) is preferred precisely because
it does not depend on the null/directional boundary stability.

---

### Step 5 — Human validation (RAs)

Generate the coding sheet:
```bash
python validation/generate_coding_sheet.py
```

Open `validation/human_coding_sheet.xlsx`, complete columns
`ra1_code` and `ra2_code` for each paper (codes: +1, -1, 0).

Then compute inter-rater reliability:
```bash
python validation/interrater_reliability.py
```

Target: Cohen's κ ≥ 0.70 before proceeding.

---

### Step 6 — Link NBER papers to published papers

```bash
python scripts/05_link_nber_to_published.py
```

Output: `data/final/analysis_dataset.parquet`
Review queue: `validation/match_review_queue.parquet`

Human review of medium-confidence matches (score 75–90) is recommended
before running analysis.

**Design note:** Only 26 of 183 published papers (14.2%) are matched to
NBER working papers. The design therefore compares two largely non-overlapping
cross-sections, not papers tracked through an editorial filter. Direction
distributions of matched vs. unmatched published papers are nearly identical
(76.9% vs. 73.2% directional), suggesting no systematic direction-specific
selection into the NBER-matched sub-sample.

---

### Step 6b — Classify identification strategies (quality control)

Requires `ANTHROPIC_API_KEY` in `.env`.

```bash
python scripts/09_code_id_strategy.py
```

Populates the `quasi_exp` column in `data/final/analysis_dataset.parquet`
by classifying each paper's identification strategy from its abstract
(quasi_exp / ols_reduced / structural / unclear).

This must be run **before** script 06 to enable Column (6) of Table III
(the ID-strategy quality control specification).

Estimated API cost: ~$2–5 for 296 papers.

---

### Step 7 — Primary analysis

```bash
python scripts/06_analysis_main.py --output-format latex
```

Outputs:

| File | Description |
|---|---|
| `table1_summary_stats.{csv,tex}` | Summary statistics by sub-sample |
| `table2_publication_rates.{csv,tex}` | Raw publication rates by direction |
| `table3_binary_primary.{csv,tex}` | **PRIMARY result**: binary directional probit |
| `table3_probit_threeway.{csv,tex}` | Secondary: three-way direction (symmetry test) |
| `table4_probit_by_journal.{csv,tex}` | By-journal analysis (exploratory; underpowered) |
| `appendix_time_trends.{csv,tex}` | Sub-period analysis (exploratory; small N) |
| `appendix_lpm.{csv,tex}` | Linear probability model robustness |
| `appendix_llm_confidence.{csv,tex}` | LLM confidence by publication status |

**Primary result** is the binary directional probit (`table3_binary_primary`).
This collapses positive and negative findings into a single directional indicator,
avoiding normative coding of the direction sign. The three-way probit
(`table3_probit_threeway`) is used to test the symmetry hypothesis
H₀: β_pos = β_neg and is included as the secondary specification.

---

### Step 8 — t-statistic distribution test

**Note:** This script requires the column `primary_tstat` to be
populated in `data/final/analysis_dataset.parquet`. This column must
be filled manually by extracting the primary test statistic from the
main result table of each published paper. A structured data
collection form is provided in `validation/tstat_extraction_guide.md`.

Demo mode (simulated data):
```bash
python scripts/07_tstat_distribution.py --demo
```

Real data:
```bash
python scripts/07_tstat_distribution.py
```

Output: `output/figures/fig_tstat_distribution_{all,positive,null}.pdf`

---

### Step 9 — Robustness checks

```bash
python scripts/08_robustness.py
```

Output: `output/tables/robustness_all.csv`

**R8 interpretation:** The abstract-length control is a *bounding exercise*.
The R8 estimates (with log(abstract length) control) are a lower bound on the
true direction–publication association; the main estimates are an upper bound.
The true effect lies between these bounds. See Section V.B in the paper.

---

### Step 10 — AFA conference baseline robustness (R10)

Addresses the critique that NBER working papers are too different a population
from published papers. Uses AFA annual meeting presentations as a pre-publication
baseline — the same elite-institution network as JF/RFS/JFE publications,
selected on research quality rather than finding direction.

**Step 10a — Scrape AFA programs**

```bash
python scripts/10_collect_afa.py
```

- Scrapes `afajof.org` program pages for 2006–2024; 2013 is fetched from
  `aeaweb.org/conference/2013/preliminary.php` (AFA sessions filtered by
  `sessionSource` tag) because the AFA site returns 404 for that year
- Handles four distinct HTML formats across years automatically
- Caches fetched HTML in `data/raw/afa_html_cache/` — re-runs use cache
- Output: `data/raw/afa_papers.parquet` (~4,000 papers, full 2006–2024)
- Runtime: ~45 seconds (network) or ~2 seconds (cached)

**Step 10b — Enrich AFA abstracts from OpenAlex and CrossRef**

```bash
python scripts/10b_enrich_afa_abstracts.py
```

- Fetches abstracts from OpenAlex (primary) and CrossRef (fallback) for AFA papers
- 27.4% overall coverage (1,095/4,001); ~31% hit rate for in-scope papers
  - OpenAlex: 427 papers; CrossRef: 668 additional papers
- Runtime: ~3–4 hours (first run with CrossRef); instant on re-runs (cache)
- Flags: `--no-ss` (skip Semantic Scholar), `--no-arxiv` (skip arXiv), `--retry-misses`
- Output: `data/raw/afa_papers_enriched.parquet` (adds `abstract`, `abstract_source`)
- Cache: `data/raw/afa_abstract_cache.json` (re-runs are instant)

**Step 10c — Run AFA analysis (free, no API)**

```bash
python scripts/11_afa_analysis.py --step all --no-api
```

Automatically loads `afa_papers_enriched.parquet` when available and applies the
same `title + abstract` keyword filter as the NBER pipeline (script 03). Falls
back to `afa_papers.parquet` with title-only filtering if enrichment hasn't been run.

`--no-api` mode runs entirely free — no Anthropic API calls:
- Scope classification: title+abstract keyword filter (no LLM)
- Direction coding: abstract-based codes from existing analysis dataset for
  matched papers; conservative title-keyword rules for unmatched papers

To run with LLM classification (more accurate, ~$15):
```bash
python scripts/11_afa_analysis.py --step all
```

Outputs:

| File | Description |
|---|---|
| `robustness_afa_baseline.{csv,tex}` | Binary directional probit, AFA baseline |
| `robustness_afa_threeway.{csv,tex}` | Three-way (pos/neg/null) symmetry test |
| `table2_afa_pub_rates.csv` | AFA publication rates by direction |

**R10 result:** Binary directional probit β = 1.136*** (s.e. 0.312, p<0.001, N = 268),
consistent with and statistically indistinguishable from the NBER-baseline estimate
of 1.279*** (s.e. 0.162). Directional AFA papers are published at a 50% rate vs 13%
for null-finding AFA papers. Match rate: 15.3% (41/268), N_directional=18.
Abstracts retrieved from OpenAlex (427) and CrossRef (668) for 1,095 total papers;
84 in-scope. Direction coding uses the same three-tier hierarchy as NBER:
dataset match → abstract rules → title rules.

**Caveat:** Confidence intervals remain wide given N_directional=18. Direction
codes for journal-matched papers come from published-abstract coding and are
reliable; unmatched papers use abstract keyword rules (when available) or
default to null (conservative). The AFA estimate should be treated as a
directional consistency check, not a precise magnitude.

---

### Step 11 — Compile the paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The paper uses `plainnat` bibliography style. All tables are inline in `main.tex`
and do not use `\input{}` from the output directory; update table values in
`main.tex` to match the generated output files after running the analysis.

---

## DuckDB Queries

After running `setup_db.py`, you can query all data interactively:

```python
import duckdb
con = duckdb.connect("publication_bias.duckdb")

# Publication rates by direction
con.execute("""
    SELECT direction,
           AVG(published_top3::INT) AS pub_rate,
           COUNT(*) AS n
    FROM analysis_dataset
    GROUP BY direction
    ORDER BY direction
""").df()

# Binary directional vs null
con.execute("""
    SELECT CASE WHEN direction != 0 THEN 'directional' ELSE 'null' END AS group,
           AVG(published_top3::INT) AS pub_rate,
           COUNT(*) AS n
    FROM analysis_dataset
    WHERE direction IS NOT NULL
    GROUP BY 1
""").df()

# Year trend in positive-finding share
con.execute("""
    SELECT year,
           AVG(positive::INT) AS pct_positive,
           COUNT(*) AS n
    FROM analysis_dataset
    WHERE source = 'nber'
    GROUP BY year
    ORDER BY year
""").df()
```

---

## Estimated Timeline

| Step | Time |
|---|---|
| Scripts 01–02 (data collection) | 1–2 hours |
| Script 03 (scope classification) | 1–2 hours + $15 API |
| Script 04 (direction coding) | 2–4 hours + $60 API |
| Script 09 (ID strategy coding) | 15 minutes + $3 API |
| Human validation (RAs) | 1–2 weeks |
| Scripts 05–08 (analysis) | 30 minutes |
| Script 10 (AFA scraper) | 45 seconds (network) / 2 seconds (cached) |
| Script 11 --no-api (AFA analysis) | 2 minutes, free |
| Script 11 with LLM (AFA analysis) | 30–60 minutes + $15 API |
| Paper writing | — |

---

## Pre-registration

Before collecting data, consider pre-registering the study design at:
- **OSF:** https://osf.io/prereg
- **AEA RCT Registry:** https://www.socialscienceregistry.org

Pre-registration strengthens the credibility of the findings and addresses
supply-side vs. demand-side mechanism identification concerns.

---

## Citation

If you use this code, please cite:

```bibtex
@unpublished{Author2026,
  author = {[Author Name]},
  title  = {Significance Sorting in Government Intervention Research:
             Evidence from Top Finance Journals},
  year   = {2026},
  note   = {Working paper}
}
```
