"""
Reorder main.tex so the caliper subsection (Within-Paper Test Statistic Distributions)
comes BEFORE the probit subsections in the Results section.

Before:
  §4 Results
    §4.1 Directional vs. Null Findings           (probit, lines ~978)
    §4.2 Probit Regression Results               (probit, lines ~1035)
    §4.3 Economic Magnitude                      (probit, lines ~1204)
    §4.4 Cross-Journal Analysis                  (pointer, lines ~1230)
    §4.5 Time-Period Heterogeneity               (probit, lines ~1239)
    §4.6 Within-Paper Test Statistic Dists       (caliper, lines ~1268)

After:
  §4 Results
    §4.1 Within-Paper Test Statistic Dists       (caliper — PRIMARY)
    §4.2 Directional vs. Null Findings           (supporting context)
    §4.3 Probit Regression Results
    §4.4 Economic Magnitude
    §4.5 Cross-Journal Analysis
    §4.6 Time-Period Heterogeneity
"""

from pathlib import Path

TEX = Path('paper/main.tex')
text = TEX.read_text()
lines = text.split('\n')

# --- Locate key boundaries -------------------------------------------------

def find_line(marker, start=0):
    for i, l in enumerate(lines[start:], start):
        if marker in l:
            return i
    raise ValueError(f'Marker not found: {marker!r}')

# Start of probit block (first subsection after \section{Results})
results_sec   = find_line(r'\section{Results}')
probit_start  = find_line(r'\subsection{Directional vs.\ Null Findings}', results_sec)

# Start of caliper block
caliper_start = find_line(r'\subsection{Within-Paper Test Statistic Distributions}', results_sec)

# End of caliper block: the blank line just before \section{Robustness}
robustness_sec = find_line(r'\section{Robustness}')
# caliper block ends at the last non-empty line before \section{Robustness}
caliper_end = robustness_sec  # exclusive: lines[caliper_start:caliper_end]

print(f'results_sec={results_sec}, probit_start={probit_start}, '
      f'caliper_start={caliper_start}, caliper_end={caliper_end}')
assert results_sec < probit_start < caliper_start < caliper_end

# --- Extract blocks --------------------------------------------------------
before_probit  = lines[:probit_start]          # everything up to (not incl.) first probit sub
probit_block   = lines[probit_start:caliper_start]  # all probit subsections
caliper_block  = lines[caliper_start:caliper_end]   # caliper subsection
after_caliper  = lines[caliper_end:]               # Robustness onwards

# --- Reassemble: caliper first, then probit --------------------------------
new_lines = before_probit + caliper_block + probit_block + after_caliper

TEX.write_text('\n'.join(new_lines))
print('Done. Section order reversed: caliper now first in Results.')
print(f'  Caliper block: {len(caliper_block)} lines')
print(f'  Probit block:  {len(probit_block)} lines')
