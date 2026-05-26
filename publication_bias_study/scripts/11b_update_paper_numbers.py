"""
11b_update_paper_numbers.py
----------------------------
Reads analysis outputs and patches the corresponding numbers in paper/main.tex.
Safe to re-run: prints a diff of every substitution made.

Run after 11_afa_analysis.py completes:
    python scripts/11b_update_paper_numbers.py
"""

import pathlib
import re
import sys

import pandas as pd

ROOT    = pathlib.Path(__file__).parent.parent.resolve()
TEX     = ROOT / "paper" / "main.tex"
OUT_DIR = ROOT / "output" / "tables"


def fmt_coef(val: float, sig: str) -> str:
    stars = {"***": "^{***}", "**": "^{**}", "*": "^{*}"}.get(sig.strip(), "")
    return f"{val:.3f}{stars}"


def pval_str(sig: str) -> str:
    """Return LaTeX p-value string for robustness table (e.g. $<$0.01)."""
    return {"***": "$<$0.01", "**": "$<$0.05", "*": "$<$0.10"}.get(sig.strip(), "---")


def num(x) -> str:
    """Format integer with thousands separator."""
    return f"{int(x):,}"


def load_numbers() -> dict:
    d = {}

    # --- Abstract coverage ---
    enr = pd.read_parquet(ROOT / "data" / "raw" / "afa_papers_enriched.parquet")
    has = enr["abstract"].notna() & (enr["abstract"].str.strip() != "")
    d["total_papers"] = len(enr)          # should stay 4,080
    d["abs_count"]    = int(has.sum())
    d["abs_pct"]      = has.mean() * 100

    # --- In-scope / direction counts ---
    dir_df = pd.read_parquet(ROOT / "data" / "classified" / "afa_direction.parquet")
    d["n_inscope"]      = len(dir_df)
    d["n_with_abs"]     = int(dir_df["has_abstract"].sum())
    d["n_directional"]  = int((dir_df["direction"].abs() == 1).sum())
    d["dir_pct"]        = d["n_directional"] / d["n_inscope"] * 100

    # --- Publication rates ---
    pub = pd.read_csv(OUT_DIR / "table2_afa_pub_rates.csv")
    dir_pub = pub[pub["Direction"].isin(["Positive", "Negative"])]
    d["n_published_dir"]   = int(dir_pub["N_Published"].sum())
    d["n_published_total"] = int(pub["N_Published"].sum())
    d["pub_rate_dir"]      = d["n_published_dir"] / d["n_directional"] * 100
    d["pub_pct_total"]     = d["n_published_total"] / d["n_inscope"] * 100

    # --- Regression results ---
    reg = pd.read_csv(OUT_DIR / "robustness_afa_baseline.csv")
    biv  = reg[(reg["Specification"].str.contains("Bivariate")) &
               (reg["Variable"] == "directional")].iloc[0]
    year = reg[(reg["Specification"].str.contains(r"\+ Year", regex=True) |
                reg["Specification"].str.contains("Year")) &
               (reg["Variable"] == "directional")].iloc[0]

    d["biv_coef"]  = biv["Coefficient"]
    d["biv_se"]    = biv["Std Error"]
    d["biv_sig"]   = biv["Significance"]
    d["biv_N"]     = int(biv["N"])
    d["year_coef"] = year["Coefficient"]
    d["year_se"]   = year["Std Error"]
    d["year_sig"]  = year["Significance"]

    return d


def patch(text: str, old: str, new: str, label: str) -> str:
    if old == new:
        return text
    count = text.count(old)
    if count == 0:
        print(f"  WARN [{label}]: pattern not found — skipping")
        return text
    if count > 1:
        print(f"  WARN [{label}]: {count} matches — replacing all")
    result = text.replace(old, new)
    print(f"  [{label}]  '{old}'  →  '{new}'")
    return result


def main():
    print("Loading analysis outputs ...")
    try:
        d = load_numbers()
    except FileNotFoundError as e:
        sys.exit(f"Missing output file: {e}")

    print(f"\nKey numbers:")
    print(f"  Abstract coverage : {num(d['abs_count'])} / {num(d['total_papers'])} ({d['abs_pct']:.1f}%)")
    print(f"  In-scope N        : {d['n_inscope']}")
    print(f"  In-scope w/ abs   : {d['n_with_abs']}")
    print(f"  Directional N     : {d['n_directional']} ({d['dir_pct']:.1f}%)")
    print(f"  Published (total) : {d['n_published_total']} ({d['pub_pct_total']:.1f}%)")
    print(f"  Published (dir.)  : {d['n_published_dir']} of {d['n_directional']} ({d['pub_rate_dir']:.1f}%)")
    print(f"  Bivariate coef    : {d['biv_coef']:.3f}{d['biv_sig']} SE={d['biv_se']:.3f}")
    print(f"  + Year coef       : {d['year_coef']:.3f}{d['year_sig']} SE={d['year_se']:.3f}")

    tex = TEX.read_text(encoding="utf-8")
    original = tex

    print("\nPatching main.tex ...")

    _pub_total = d["n_published_total"]
    _pub_words = {
        35: "Thirty-five", 36: "Thirty-six", 34: "Thirty-four", 33: "Thirty-three",
        37: "Thirty-seven", 30: "Thirty", 40: "Forty", 25: "Twenty-five",
    }.get(_pub_total, str(_pub_total))

    # Null publication count and rate (total minus directional)
    n_null_published = _pub_total - d["n_published_dir"]
    n_null = d["n_inscope"] - d["n_directional"]
    null_rate = n_null_published / n_null * 100 if n_null else 0.0

    biv_pval = pval_str(d["biv_sig"])
    year_pval = pval_str(d["year_sig"])
    # Determine inline p-value label from significance
    biv_p_inline = "0.01" if d["biv_sig"].strip() == "***" else "0.05"
    year_p_inline = "0.01" if d["year_sig"].strip() == "***" else "0.05"

    # 1. Abstract coverage sentence
    tex = patch(tex,
        f"2,775 of the 4,080 AFA papers (68.0\\% overall",
        f"{num(d['abs_count'])} of the {num(d['total_papers'])} AFA papers ({d['abs_pct']:.1f}\\% overall",
        "abs coverage sentence")

    # 2. Abstract count in robustness note
    tex = patch(tex,
        "(2,775 total; 125 in-scope)",
        f"({num(d['abs_count'])} total; {d['n_with_abs']} in-scope)",
        "abs count robustness note")

    # 3. Total in-scope N (inline prose)
    tex = patch(tex,
        "yielding 156",
        f"yielding {d['n_inscope']}",
        "N inscope prose")

    # 4. Published/matched count + pct
    tex = patch(tex,
        "Thirty-five AFA papers (22.4\\%)",
        f"{_pub_words} AFA papers ({d['pub_pct_total']:.1f}\\%)",
        "published/matched count prose")

    # 5. Bivariate regression (inline)
    tex = patch(tex,
        f"$\\hat\\beta = 0.513^{{**}}$ (s.e.\\ 0.245, $p < 0.05$, $N = 156$)",
        f"$\\hat\\beta = {fmt_coef(d['biv_coef'], d['biv_sig'])}$ (s.e.\\ {d['biv_se']:.3f}, $p < {biv_p_inline}$, $N = {d['biv_N']}$)",
        "bivariate coef inline")

    # 6. Publication rate directional
    tex = patch(tex,
        "27.8\\% rate (27 of 97)",
        f"{d['pub_rate_dir']:.1f}\\% rate ({d['n_published_dir']} of {d['n_directional']})",
        "pub rate directional")

    # 7. Null publication rate (vs. directional)
    tex = patch(tex,
        "13.6\\% for null-finding papers (8 of 59)",
        f"{null_rate:.1f}\\% for null-finding papers ({n_null_published} of {n_null})",
        "null pub rate")

    # 8. +Year regression (inline)
    tex = patch(tex,
        f"($\\hat\\beta = 0.506^{{**}}$, s.e.\\ 0.248)",
        f"($\\hat\\beta = {fmt_coef(d['year_coef'], d['year_sig'])}$, s.e.\\ {d['year_se']:.3f})",
        "year coef inline")

    # 9. Robustness table row — bivariate
    tex = patch(tex,
        f"$0.513^{{**}}$ & $<$0.05 & 156 \\\\",
        f"${fmt_coef(d['biv_coef'], d['biv_sig'])}$ & {biv_pval} & {d['biv_N']} \\\\",
        "bivariate table row")

    # 10. Robustness table row — +year
    tex = patch(tex,
        f"$0.506^{{**}}$ & $<$0.05 & 156 \\\\",
        f"${fmt_coef(d['year_coef'], d['year_sig'])}$ & {year_pval} & {d['biv_N']} \\\\",
        "year table row")

    # 11. Table caption N
    tex = patch(tex,
        "$N=156$ in-scope",
        f"$N={d['n_inscope']}$ in-scope",
        "caption N")

    # 12. Match rate in robustness note
    tex = patch(tex,
        f"22.4\\% match rate; $N_{{\\text{{directional}}}}=97$",
        f"{d['pub_pct_total']:.1f}\\% match rate; $N_{{\\text{{directional}}}}={d['n_directional']}$",
        "match rate note")

    if tex == original:
        print("\nNo changes — all numbers already up to date.")
        return

    TEX.write_text(tex, encoding="utf-8")
    changed = sum(1 for a, b in zip(original.splitlines(), tex.splitlines()) if a != b)
    print(f"\nWrote {TEX.name} ({changed} lines changed).")


if __name__ == "__main__":
    main()
