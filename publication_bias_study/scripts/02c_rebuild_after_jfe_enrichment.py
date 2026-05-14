"""
02c_rebuild_after_jfe_enrichment.py
-------------------------------------
After running 02b_enrich_jfe_abstracts.py, re-runs the full downstream
pipeline: classify → direction-code → link → analyse → robustness.

Usage:
    python scripts/02c_rebuild_after_jfe_enrichment.py
"""

import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()

STEPS = [
    ("03 Classify in-scope",        ["python3", "scripts/03_classify_inscope.py"]),
    ("04 Code direction",            ["python3", "scripts/04_code_direction.py"]),
    ("05 Link NBER→published",       ["python3", "scripts/05_link_nber_to_published.py"]),
    ("06 Main analysis",             ["python3", "scripts/06_analysis_main.py"]),
    ("08 Robustness checks",         ["python3", "scripts/08_robustness.py"]),
]

def main() -> None:
    for label, cmd in STEPS:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"\nStep failed: {label}")
            sys.exit(result.returncode)
    print("\nAll steps complete.")

if __name__ == "__main__":
    main()
