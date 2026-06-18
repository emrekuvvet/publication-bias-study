"""
Counts two separate families of hedging terms in NBER vs. published abstracts:

  (i)  Style-only hedges  -- author tentativeness without naming a finding
  (ii) Null-result phrases -- explicit references to insignificant/null findings

Reports per-abstract and per-100-words counts so that the writing-style component
and the substantive-finding component can be disentangled (referee comment #17).

Inputs:
  data/final/analysis_dataset.parquet  (must have 'abstract' and 'published_top3' columns)
"""

import re
from pathlib import Path

import pandas as pd

STYLE_ONLY = [
    "may", "might", "could", "possibly", "perhaps",
    "suggests", "appears", "tends to", "seems", "unclear",
]

NULL_PHRASES = [
    "no significant", "do not find", "no evidence",
    "insignificant", "statistically insignificant",
    "null", "null result",
]


def count_terms(text: str, terms: list[str]) -> int:
    if not isinstance(text, str):
        return 0
    pad = " " + text.lower() + " "
    n = 0
    for t in terms:
        n += len(re.findall(r"\b" + re.escape(t) + r"\b", pad))
    return n


def summarise(df: pd.DataFrame, label: str) -> dict:
    n_words = df["abstract"].fillna("").str.split().str.len()
    style = df["abstract"].apply(lambda x: count_terms(x, STYLE_ONLY))
    nullp = df["abstract"].apply(lambda x: count_terms(x, NULL_PHRASES))
    return {
        "label":              label,
        "N":                  int(len(df)),
        "avg_words":          round(float(n_words.mean()), 1),
        "style_per_abstract": round(float(style.mean()), 3),
        "style_per_100w":     round(float((100 * style / n_words).mean()), 3),
        "null_per_abstract":  round(float(nullp.mean()), 3),
        "null_per_100w":      round(float((100 * nullp / n_words).mean()), 3),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    df = pd.read_parquet(root / "data" / "final" / "analysis_dataset.parquet")
    rows = [
        summarise(df[df["published_top3"] == False], "NBER"),
        summarise(df[df["published_top3"] == True],  "Published"),
    ]
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out_path = root / "output" / "tables" / "abstract_hedging.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
