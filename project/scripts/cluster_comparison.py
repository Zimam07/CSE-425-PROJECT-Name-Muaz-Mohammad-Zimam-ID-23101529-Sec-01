"""
Run quick_eval clustering with multiple clusterers and generate a comparison plot and CSV.

Usage:
  python scripts/cluster_comparison.py --feat_dir data/features/multimodal --k 4 --out results/cluster_comparison.csv --plot results/cluster_comparison.png
"""
import argparse
import csv
import os
import subprocess
import json
import tempfile
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--feat_dir', default='data/features/multimodal')
parser.add_argument('--k', type=int, default=4)
parser.add_argument('--out', default='results/cluster_comparison.csv')
parser.add_argument('--plot', default='results/cluster_comparison.png')
args = parser.parse_args()

clusterers = ['kmeans', 'agglomerative', 'dbscan']
results = []
os.makedirs(os.path.dirname(args.out), exist_ok=True)
for c in clusterers:
    out_tmp = tempfile.mktemp(suffix='.csv')
    cmd = ['python', 'scripts/quick_eval.py', '--feat_dir', args.feat_dir, '--k', str(args.k), '--clusterer', c, '--out', out_tmp]
    print('Running:', ' '.join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print('Clusterer', c, 'failed; skipping')
        continue
    # read metrics
    with open(out_tmp, 'r', encoding='utf-8') as fh:
        r = csv.reader(fh)
        next(r, None)
        m = {row[0]: row[1] for row in r}
    results.append((c, m))

# write combined csv
with open(args.out, 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    header = ['clusterer'] + list(results[0][1].keys())
    w.writerow(header)
    for c, m in results:
        row = [c] + [m.get(k, '') for k in header[1:]]
        w.writerow(row)

# plot silhouettes (if available)
sil = []
labels = []
for c, m in results:
    val = m.get('silhouette', '')
    try:
        sil.append(float(val) if val != '' else float('nan'))
    except Exception:
        sil.append(float('nan'))
    labels.append(c)
plt.figure(figsize=(6,3))
plt.bar(labels, sil, color='C0')
plt.ylabel('Silhouette')
plt.title('Clusterer comparison (PCA->clustering)')
plt.tight_layout()
plt.savefig(args.plot)
print('Wrote', args.out, args.plot)