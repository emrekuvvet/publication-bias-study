"""
10c_fix_2014_2016_parsing.py
----------------------------
Re-parse the cached AFA HTML for 2014, 2015, 2016 with corrected parsers,
replace corrupt rows in afa_papers.parquet, and update
afa_papers_enriched.parquet (new rows get abstract='', abstract_source='none').

Root cause: parse_old_v2() in 10_collect_afa.py did not understand the
<i>/<a>-wrapped title structure of these years, so it concatenated the
title with the following author text.

Fixed parsers:
  2014/2015: <p> with <i> child  → title = <i>.text; authors = rest of <p>.
             If authors empty, look at next <p> (no <i>/<strong>/<b>).
  2016:      <p> with <a> child but NOT <strong>  → title = <a>.text;
             authors = rest of <p>.  Skip if p also has <strong> (session header).

Usage:
    python scripts/10c_fix_2014_2016_parsing.py
"""

import hashlib
import pathlib
import re

import pandas as pd
from bs4 import BeautifulSoup

# ------------------------------------------------------------------ #
# Paths                                                                #
# ------------------------------------------------------------------ #

ROOT        = pathlib.Path(__file__).parent.parent.resolve()
CACHE_DIR   = ROOT / "data" / "raw" / "afa_html_cache"
PAPERS_PATH = ROOT / "data" / "raw" / "afa_papers.parquet"
ENRICHED_PATH = ROOT / "data" / "raw" / "afa_papers_enriched.parquet"

URLS = {
    2014: "https://afajof.org/2014-preliminary-program/",
    2015: "https://afajof.org/2015-preliminary-program/",
    2016: "https://afajof.org/2016-preliminary-program/",
}

FIX_YEARS = [2014, 2015, 2016]

# ------------------------------------------------------------------ #
# Text helpers                                                         #
# ------------------------------------------------------------------ #

def clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def make_paper_id(year: int, title: str) -> str:
    slug = re.sub(r"[^a-z0-9]", "", title.lower())[:40]
    return f"afa_{year}_{slug}"


_INST_WORDS = re.compile(
    r"\b(university|school|college|institute|faculty|department|business|"
    r"insead|nber|stern|booth|sloan|fuqua|tuck|haas|gsb|hbs|federal reserve)\b",
    re.IGNORECASE,
)


def looks_like_institution(text: str) -> bool:
    return bool(_INST_WORDS.search(text))


def is_skip_line(text: str) -> bool:
    t = text.strip().lower()
    return (
        t.startswith("discussant")
        or t.startswith("session:")
        or t.startswith("session chair")
        or t.startswith("chair:")
        or t.startswith("president")
    )


def is_garbage_title(title: str) -> bool:
    """Reject titles that are too short or look like session/location strings."""
    words = title.split()
    if len(words) < 3:
        return True
    t_lower = title.lower()
    if (t_lower.startswith("discussant")
            or t_lower.startswith("session")
            or t_lower.startswith("session chair")
            or t_lower.startswith("president")
            or t_lower.startswith("location:")):
        return True
    # Don't treat real paper titles as institutions just because they mention
    # a university name — only reject if they look like a pure affiliation line
    # (no verb-like words, very short, etc.)
    return False


# ------------------------------------------------------------------ #
# Parser: 2014 / 2015                                                  #
# ------------------------------------------------------------------ #

def parse_2014_2015(html: str, year: int) -> list[dict]:
    """
    HTML structure: <p><i><u>Title</u></i> Author, Affil Author2, Affil<br/></p>
    Sometimes the <p> has only the <i> tag, and the next <p> has the authors.
    Session headers are in <p><b>Session…</b></p> or <p><strong>…</strong></p>.
    """
    soup = BeautifulSoup(html, "lxml")
    content = (soup.find("article")
               or soup.find("div", class_="entry-content")
               or soup)
    ps = content.find_all("p")

    papers: list[dict] = []
    current_session = ""
    i = 0

    while i < len(ps):
        p = ps[i]

        # ---- Session header: <p> with <b> or <strong> ----
        b_tag = p.find("b") or p.find("strong")
        if b_tag and not p.find("i"):
            txt = clean(p.get_text(separator=" "))
            if txt and not is_skip_line(txt):
                current_session = txt
            i += 1
            continue

        # ---- Paper: <p> with <i> child ----
        i_tag = p.find("i")
        if i_tag:
            title = clean(i_tag.get_text(separator=" "))

            # Authors: rest of <p> text after the <i> text
            full_text = clean(p.get_text(separator=" "))
            i_text = clean(i_tag.get_text(separator=" "))
            authors = ""
            if full_text.startswith(i_text):
                authors = full_text[len(i_text):].strip().lstrip(",").lstrip(";").strip()
            else:
                # Fallback: strip the title from full text
                authors = full_text.replace(i_text, "", 1).strip().lstrip(",").lstrip(";").strip()

            # If still no authors, check next <p> (must have no <i>/<strong>/<b>)
            i += 1
            if not authors and i < len(ps):
                nxt = ps[i]
                nxt_txt = clean(nxt.get_text(separator=" "))
                has_structural_tag = nxt.find("i") or nxt.find("strong") or nxt.find("b")
                if (nxt_txt
                        and not has_structural_tag
                        and not is_skip_line(nxt_txt)
                        and (looks_like_institution(nxt_txt)
                             or "," in nxt_txt)):
                    authors = nxt_txt
                    i += 1   # consume the authors <p>

            if is_garbage_title(title):
                continue

            papers.append({
                "paper_id": make_paper_id(year, title),
                "year":     year,
                "title":    title,
                "authors":  authors,
                "session":  current_session,
                "source":   "afa",
            })
            continue

        # ---- Everything else: might be a date/time line ----
        txt = clean(p.get_text(separator=" "))
        if txt and re.match(r"0[1-9]/\d{2}/20\d{2}", txt):
            # date line like "01/03/2014 – 8:00 AM"
            pass  # just skip
        i += 1

    return papers


# ------------------------------------------------------------------ #
# Parser: 2016                                                         #
# ------------------------------------------------------------------ #

def parse_2016(html: str) -> list[dict]:
    """
    HTML structure: <p><a>Title</a> Author; Affil Author2; Affil<br/></p>
    Session headers: <p><strong>Session: …</strong></p>  (also has <a> for year)
    Discussants:     <p><strong>Discussant: …</strong></p>
    Session chairs:  <p>Session Chair: …</p>  (plain text, no children)
    """
    soup = BeautifulSoup(html, "lxml")
    content = (soup.find("article")
               or soup.find("div", class_="entry-content")
               or soup)
    ps = content.find_all("p")

    papers: list[dict] = []
    current_session = ""

    for p in ps:
        strong_tag = p.find("strong")
        a_tag = p.find("a")

        # ---- Session header / discussant: has <strong> ----
        if strong_tag:
            txt = clean(p.get_text(separator=" "))
            if txt.lower().startswith("session"):
                current_session = txt
            # Discussants and session chairs: skip for paper extraction
            continue

        # ---- Paper: <p> with <a> child but no <strong> ----
        if a_tag:
            title = clean(a_tag.get_text(separator=" "))

            # Authors: rest of <p> text after the <a> text
            full_text = clean(p.get_text(separator=" "))
            a_text = clean(a_tag.get_text(separator=" "))
            authors = ""
            if full_text.startswith(a_text):
                authors = full_text[len(a_text):].strip().lstrip(";").lstrip(",").strip()
            else:
                authors = full_text.replace(a_text, "", 1).strip().lstrip(";").lstrip(",").strip()

            if is_garbage_title(title):
                continue

            papers.append({
                "paper_id": make_paper_id(2016, title),
                "year":     2016,
                "title":    title,
                "authors":  authors,
                "session":  current_session,
                "source":   "afa",
            })

    return papers


# ------------------------------------------------------------------ #
# Load cached HTML                                                     #
# ------------------------------------------------------------------ #

def load_cached_html(year: int) -> str:
    url = URLS[year]
    cache_key = hashlib.md5(url.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.html"
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")
    return cache_file.read_text(encoding="utf-8", errors="replace")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    # ---- Load existing parquet ----
    df = pd.read_parquet(PAPERS_PATH)
    print("=== BEFORE FIX ===")
    print("Papers per year:")
    print(df.groupby("year").size().to_string())
    print()

    # ---- Parse corrected data for 2014, 2015, 2016 ----
    new_rows: list[dict] = []

    for year in FIX_YEARS:
        html = load_cached_html(year)
        if year in (2014, 2015):
            parsed = parse_2014_2015(html, year)
        else:
            parsed = parse_2016(html)

        # Deduplicate by paper_id (keep last occurrence)
        seen_ids: set[str] = set()
        unique_parsed = []
        for row in parsed:
            pid = row["paper_id"]
            if pid not in seen_ids:
                seen_ids.add(pid)
                unique_parsed.append(row)

        print(f"{year}: {len(unique_parsed)} unique papers parsed")
        print("  Sample titles:")
        for row in unique_parsed[:5]:
            print(f"    - {row['title'][:80]}")
        print()

        new_rows.extend(unique_parsed)

    new_df = pd.DataFrame(new_rows)

    # ---- Replace rows in main parquet ----
    df_other = df[~df["year"].isin(FIX_YEARS)].copy()
    df_fixed = pd.concat([df_other, new_df], ignore_index=True)
    df_fixed = df_fixed.sort_values(["year", "paper_id"]).reset_index(drop=True)

    df_fixed.to_parquet(PAPERS_PATH, index=False)
    print("=== AFTER FIX ===")
    print("Papers per year (afa_papers.parquet):")
    print(df_fixed.groupby("year").size().to_string())
    print()

    # ---- Update enriched parquet ----
    if ENRICHED_PATH.exists():
        df_enr = pd.read_parquet(ENRICHED_PATH)
        df_enr_other = df_enr[~df_enr["year"].isin(FIX_YEARS)].copy()

        # Build enriched rows with empty abstracts
        new_enr = new_df.copy()
        new_enr["abstract"] = ""
        new_enr["abstract_source"] = "none"

        # Ensure same column order as enriched file
        for col in df_enr.columns:
            if col not in new_enr.columns:
                new_enr[col] = None
        new_enr = new_enr[df_enr.columns]

        df_enr_fixed = pd.concat([df_enr_other, new_enr], ignore_index=True)
        df_enr_fixed = df_enr_fixed.sort_values(["year", "paper_id"]).reset_index(drop=True)
        df_enr_fixed.to_parquet(ENRICHED_PATH, index=False)
        print("Papers per year (afa_papers_enriched.parquet):")
        print(df_enr_fixed.groupby("year").size().to_string())
        print()
        print("Abstract coverage by year (after fix — 2014-2016 reset to 0):")
        cov = df_enr_fixed.groupby("year")["abstract"].apply(
            lambda x: round((x != "").sum() / len(x) * 100, 1)
        )
        print(cov.to_string())
    else:
        print(f"Enriched file not found at {ENRICHED_PATH} — skipping.")

    # ---- Final sample ----
    print()
    print("=== SAMPLE CLEAN TITLES ===")
    for year in FIX_YEARS:
        subset = df_fixed[df_fixed["year"] == year]
        print(f"\n{year} ({len(subset)} papers):")
        for _, row in subset.head(5).iterrows():
            print(f"  [{row['paper_id'][-15:]}] {row['title'][:80]}")
            print(f"    authors: {row['authors'][:70]}")


if __name__ == "__main__":
    main()
