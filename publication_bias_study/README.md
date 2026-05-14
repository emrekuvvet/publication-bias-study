# Publication Bias in Top Finance Journals — Replication Package

**Paper:** *The Positive Slant: Publication Bias in Government Intervention Research at Top Finance Journals*

This repository contains the complete data pipeline and analysis code to replicate all results.

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.11+ |
| DuckDB | 0.10.3 |
| Anthropic API key | Required for scripts 03 & 04 |
| Web of Science API key | Required for script 02 (CSV fallback available) |

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys.

---

## Directory Structure

```
publication_bias_study/
├── implementation_plan.md
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
│   └── 08_robustness.py
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

---

### Step 7 — Primary analysis

```bash
python scripts/06_analysis_main.py --output-format latex
```

Output: `output/tables/table{1-4}*.{csv,tex}`

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

---

### Step 10 — Compile the paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

After running the analysis scripts, insert the output tables and
figures into `main.tex` where `\placeholder{...}` markers appear.

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
| Human validation (RAs) | 1–2 weeks |
| Scripts 05–09 (analysis) | 30 minutes |
| Paper writing | — |

---

## Pre-registration

Before collecting data, consider pre-registering the study design at:
- **OSF:** https://osf.io/prereg
- **AEA RCT Registry:** https://www.socialscienceregistry.org

Pre-registration strengthens the credibility of directional bias findings.

---

## Citation

If you use this code, please cite:

```bibtex
@unpublished{Author2026,
  author = {[Author Name]},
  title  = {The Positive Slant: Publication Bias in Government
             Intervention Research at Top Finance Journals},
  year   = {2026},
  note   = {Working paper}
}
```
