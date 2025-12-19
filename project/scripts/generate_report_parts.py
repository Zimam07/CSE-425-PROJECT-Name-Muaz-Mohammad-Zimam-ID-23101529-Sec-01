"""
Generate small metric text files consumed by the LaTeX report and export the notebook to HTML.
Usage:
    python scripts/generate_report_parts.py --metrics results/demo_analysis/metrics.csv --notebook ../notebooks/latent_analysis.ipynb --out_dir results/report_parts
"""
import os
import argparse
import pandas as pd
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', default='results/demo_analysis/metrics.csv')
    parser.add_argument('--notebook', default='../notebooks/latent_analysis.ipynb')
    parser.add_argument('--out_dir', default='results/report_parts')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.exists(args.metrics):
        df = pd.read_csv(args.metrics)
        # write separate small files for LaTeX \input
        def write_metric(name, filename):
            row = df[df['metric'] == name]
            val = ''
            if not row.empty:
                val = str(row['value'].values[0])
            with open(os.path.join(args.out_dir, filename), 'w', encoding='utf-8') as fh:
                fh.write(val)

        write_metric('silhouette', 'metrics_silhouette.txt')
        write_metric('calinski_harabasz', 'metrics_ch.txt')
        write_metric('davies_bouldin', 'metrics_db.txt')
        print('Wrote metric parts to', args.out_dir)
    else:
        print('Metrics CSV not found:', args.metrics)

    # Export notebook to HTML
    try:
        nbpath = os.path.join('project', args.notebook) if not os.path.isabs(args.notebook) else args.notebook
        subprocess.run(['jupyter', 'nbconvert', '--to', 'html', args.notebook, '--output-dir', args.out_dir], check=True)
        print('Exported notebook to HTML in', args.out_dir)
    except Exception as e:
        print('Failed to export notebook to HTML:', e)

if __name__ == '__main__':
    main()
