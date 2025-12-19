from pathlib import Path
import csv
rows = list(csv.DictReader(Path('results/metrics_summary.csv').open()))
row = rows[0]
new_table = '\\subsection{Metrics}\n\\begin{tabular}{l r}\n\\toprule\nMetric & Value \\\\\n\\midrule\nSilhouette & %s \\\\\nCalinski-Harabasz & %s \\\\\nDavies-Bouldin & %s \\\\\n\\bottomrule\n\\end{tabular}\n' % (row.get('silhouette','N/A'), row.get('ch','N/A'), row.get('db','N/A'))
print('RAW REPR:')
print(repr(new_table))
print('RAW PRINT:')
print(new_table)
