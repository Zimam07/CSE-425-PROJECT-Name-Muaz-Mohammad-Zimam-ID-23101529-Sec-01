import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

paths = glob.glob('results/**/metrics.csv', recursive=True)
rows = []
for p in paths:
    try:
        df = pd.read_csv(p)
        metrics = {r['metric']: r['value'] for _, r in df.iterrows()}
        name = os.path.dirname(p)
        rows.append({'analysis': name, 'silhouette': float(metrics.get('silhouette', 'nan')), 'ch': float(metrics.get('calinski_harabasz', 'nan')), 'db': float(metrics.get('davies_bouldin', 'nan'))})
    except Exception:
        continue
if not rows:
    print('No metrics found')
    exit(0)
agg = pd.DataFrame(rows)
agg = agg.sort_values('silhouette', ascending=False)
agg.to_csv('results/metrics_summary.csv', index=False)

# plot silhouettes
plt.figure(figsize=(8,4))
plt.barh(agg['analysis'], agg['silhouette'])
plt.xlabel('Silhouette')
plt.title('Silhouette by analysis')
plt.tight_layout()
plt.savefig('results/silhouette_summary.png')
print('Wrote results/metrics_summary.csv and results/silhouette_summary.png')
