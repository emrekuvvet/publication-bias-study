"""
setup_db.py
-----------
Initialise the DuckDB database (publication_bias.duckdb) and create
persistent views over the Parquet files.  Run once before any analysis.

Usage:
    python setup_db.py
"""

import pathlib
import duckdb

ROOT = pathlib.Path(__file__).parent.resolve()
DB_PATH = ROOT / "publication_bias.duckdb"
DATA = ROOT / "data"


def main():
    con = duckdb.connect(str(DB_PATH))

    # ------------------------------------------------------------------ #
    # Raw layer — one view per source file                                 #
    # ------------------------------------------------------------------ #
    _create_view(con, "raw_nber",
                 DATA / "raw" / "nber_papers.parquet",
                 """
                 wp_number     VARCHAR,
                 title         VARCHAR,
                 authors       VARCHAR,
                 year          INTEGER,
                 month         INTEGER,
                 abstract      VARCHAR,
                 programs      VARCHAR,   -- comma-separated program codes
                 url           VARCHAR
                 """)

    for journal in ("jf", "rfs", "jfe"):
        _create_view(con, f"raw_{journal}",
                     DATA / "raw" / f"{journal}_papers.parquet",
                     """
                     doi           VARCHAR,
                     title         VARCHAR,
                     authors       VARCHAR,
                     year          INTEGER,
                     volume        INTEGER,
                     issue         VARCHAR,
                     journal       VARCHAR,
                     abstract      VARCHAR,
                     jel_codes     VARCHAR,
                     citations     INTEGER
                     """)

    # ------------------------------------------------------------------ #
    # Classified layer                                                     #
    # ------------------------------------------------------------------ #
    _create_view(con, "inscope_papers",
                 DATA / "classified" / "inscope_papers.parquet",
                 """
                 paper_id      VARCHAR,   -- doi OR wp_number
                 source        VARCHAR,   -- 'nber' | 'jf' | 'rfs' | 'jfe'
                 title         VARCHAR,
                 abstract      VARCHAR,
                 year          INTEGER,
                 in_scope      BOOLEAN,
                 scope_conf    VARCHAR,   -- 'high' | 'medium' | 'low'
                 scope_reason  VARCHAR,
                 keyword_hit   BOOLEAN
                 """)

    _create_view(con, "coded_direction",
                 DATA / "classified" / "coded_direction.parquet",
                 """
                 paper_id      VARCHAR,
                 direction     TINYINT,   -- +1 / -1 / 0
                 confidence    VARCHAR,
                 rationale     VARCHAR,
                 human_code    TINYINT,   -- NULL until RA validation
                 agreed        BOOLEAN
                 """)

    # ------------------------------------------------------------------ #
    # Final analysis dataset                                               #
    # ------------------------------------------------------------------ #
    _create_view(con, "analysis_dataset",
                 DATA / "final" / "analysis_dataset.parquet",
                 """
                 paper_id           VARCHAR,
                 source             VARCHAR,
                 published_top3     BOOLEAN,   -- TRUE = published in JF/RFS/JFE
                 journal            VARCHAR,
                 year               INTEGER,
                 direction          TINYINT,
                 positive           BOOLEAN,
                 negative           BOOLEAN,
                 nber_program       VARCHAR,
                 id_strategy        VARCHAR,   -- 'RDD' | 'IV' | 'DID' | 'OLS' | 'Other'
                 top_institution    BOOLEAN,
                 n_authors          SMALLINT,
                 nber_citations     INTEGER,
                 jel_codes          VARCHAR,
                 nber_wp_number     VARCHAR,   -- NULL if unmatched
                 match_score        FLOAT      -- fuzzy match score
                 """)

    # ------------------------------------------------------------------ #
    # Convenience table macro: publication probability by direction        #
    # ------------------------------------------------------------------ #
    con.execute("""
        CREATE OR REPLACE MACRO pub_rate_by_direction(direction_val) AS TABLE
            SELECT
                AVG(published_top3::INT) AS pub_rate,
                COUNT(*)                 AS n
            FROM analysis_dataset
            WHERE direction = direction_val
    """)

    con.close()
    print(f"Database initialised: {DB_PATH}")
    print("Views created: raw_nber, raw_jf, raw_rfs, raw_jfe,")
    print("               inscope_papers, coded_direction, analysis_dataset")


def _create_view(con: duckdb.DuckDBPyConnection,
                 name: str,
                 parquet_path: pathlib.Path,
                 schema_comment: str) -> None:
    """Create a view if the Parquet file exists; otherwise create an empty stub."""
    if parquet_path.exists():
        con.execute(f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT * FROM read_parquet('{parquet_path}')
        """)
        print(f"  [OK]  view '{name}' → {parquet_path.name}")
    else:
        # Create an empty typed table as a placeholder
        # (so downstream scripts don't crash before data collection)
        con.execute(f"DROP TABLE IF EXISTS {name}")
        con.execute(f"CREATE TABLE {name} ({schema_comment})")
        print(f"  [STUB] table '{name}' (parquet not yet present)")


if __name__ == "__main__":
    main()
