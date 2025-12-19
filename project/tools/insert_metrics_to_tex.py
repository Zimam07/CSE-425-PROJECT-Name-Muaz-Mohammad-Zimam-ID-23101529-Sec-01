"""Insert metrics from a CSV into the Demo metrics table in `project/report/report.tex`.

Usage:
    python project/tools/insert_metrics_to_tex.py results/metrics_summary.csv --analysis results/conv_focus/conv_ld64_hc32_analysis

This will find the "Demo experiment" metrics table in `report.tex` and replace the values with those from the selected analysis row.
"""
import csv
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print('Usage: python project/tools/insert_metrics_to_tex.py <metrics_csv> [--analysis <analysis_path>]')
    raise SystemExit(1)

csv_path = Path(sys.argv[1])
analysis_filter = None
if '--analysis' in sys.argv:
    idx = sys.argv.index('--analysis')
    if idx+1 < len(sys.argv):
        analysis_filter = sys.argv[idx+1]

if not csv_path.exists():
    print('CSV not found:', csv_path)
    raise SystemExit(1)

rows = []
with csv_path.open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

row = None
if analysis_filter:
    for r in rows:
        if r.get('analysis','').endswith(analysis_filter) or r.get('analysis','')==analysis_filter:
            row = r
            break
if row is None and rows:
    # fallback: pick the first row
    row = rows[0]

if row is None:
    print('No rows in CSV')
    raise SystemExit(1)

s = Path('project/report/report.tex').read_text()
old_block_start = '\\subsection{Metrics}'
# locate the table start
start = s.find(old_block_start)
if start == -1:
    print('Could not find Demo metrics subsection')
    raise SystemExit(1)
# naive replace of the small table following the subsection
import re
new_table = f"""\\subsection{{Metrics}}
\\begin{{tabular}}{{l r}}
\\toprule
Metric & Value \\\\
\\midrule
Silhouette & {row.get('silhouette','N/A')} \\\\
Calinski-Harabasz & {row.get('ch','N/A')} \\\\
Davies-Bouldin & {row.get('db','N/A')} \\\\
\\bottomrule
\\end{{tabular}}
"""
print('DEBUG new_table repr:\n', repr(new_table))
# replace the first occurrence of the old table
# Replace the metrics table robustly (avoid escape issues with re.sub replacements)
start = s.find('\\subsection{Metrics}')
end = s.find('\\end{tabular}', start)
if start == -1 or end == -1:
    print('Could not find Demo metrics subsection or table end')
    raise SystemExit(1)
end += len('\\end{tabular}')
s = s[:start] + new_table + s[end:]
Path('project/report/report.tex').write_text(s)
print('Inserted metrics from', csv_path, 'into project/report/report.tex')
