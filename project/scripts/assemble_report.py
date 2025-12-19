import os
import glob
import csv


def read_metrics_csv(path):
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, 'r', encoding='utf-8') as fh:
        r = csv.reader(fh)
        next(r, None)
        for row in r:
            if len(row) >= 2:
                d[row[0]] = row[1]
    return d


def find_analysis_dirs(root='results'):
    dirs = []
    for p in glob.glob(os.path.join(root, '**', '*_analysis'), recursive=True):
        if os.path.isdir(p):
            dirs.append(p)
    return sorted(dirs)


def assemble(output='results/final_report.html'):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    analyses = find_analysis_dirs('results')
    parts = []
    parts.append('<!doctype html><html><head><meta charset="utf-8"><title>Project Report</title></head><body>')
    parts.append('<h1>VAE Hybrid Audio+Lyrics Clustering — Report (Auto-assembled)</h1>')

    # include exported notebook if present
    nb_html = None
    for p in glob.glob('results/**/latent_analysis.html', recursive=True):
        nb_html = p
        break
    if nb_html:
        parts.append(f'<h2>Notebook</h2><p>Full notebook export: <a href="{nb_html}">{nb_html}</a></p>')

    parts.append('<h2>Analyses</h2>')
    # include sweep summary if present
    summary_csv = 'results/metrics_summary.csv'
    summary_img = 'results/silhouette_summary.png'
    if os.path.exists(summary_csv):
        parts.append('<h3>Sweep summary</h3>')
        parts.append(f'<p>Summary table: <a href="{summary_csv}">{summary_csv}</a></p>')
        if os.path.exists(summary_img):
            parts.append(f'<div><img src="{summary_img}" style="max-width:800px;width:80%;height:auto;"></div>')
    if not analyses:
        parts.append('<p>No analysis directories found under results/</p>')
    for a in analyses:
        parts.append(f'<h3>{a}</h3>')
        mpath = os.path.join(a, 'metrics.csv')
        metrics = read_metrics_csv(mpath)
        if metrics:
            parts.append('<ul>')
            for k, v in metrics.items():
                parts.append(f'<li><b>{k}</b>: {v}</li>')
            parts.append('</ul>')
        else:
            parts.append('<p>No metrics found.</p>')

        # images
        for name in ('pca.png', 'tsne.png', 'umap.png'):
            ipath = os.path.join(a, name)
            if os.path.exists(ipath):
                parts.append(f'<div><img src="{ipath}" style="max-width:800px;width:80%;height:auto;"></div>')

        # link to latent file
        lat = os.path.join(os.path.dirname(a), 'latents.npy')
        if os.path.exists(lat):
            parts.append(f'<p>Latents: <a href="{lat}">{lat}</a></p>')

    parts.append('<hr><p>Generated automatically by <code>scripts/assemble_report.py</code>.</p>')
    parts.append('</body></html>')

    with open(output, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(parts))
    print('Wrote', output)


if __name__ == '__main__':
    assemble()
