"""Finalize the report: optionally run demo, insert metrics, and compile report.tex.

Usage:
  python project/tools/finalize_report.py [--run-demo] [--analysis <analysis_path>] [--compile]

- --run-demo: run `scripts/run_demo_pipeline.py` (creates demo outputs).
- --analysis: choose which analysis row from `results/metrics_summary.csv` to insert into the Demo metrics table.
- --compile: compile `project/report/report.tex` using the bundled tectonic engine.

This script is conservative: it does not delete files.
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--run-demo', action='store_true', help='Run the demo pipeline before finalizing')
parser.add_argument('--analysis', type=str, help='Analysis path or name to pick metrics from')
parser.add_argument('--compile', action='store_true', help='Compile the LaTeX report after inserting metrics')
args = parser.parse_args()

root = Path(os.getcwd()).resolve()
print('Workspace root:', root)

if args.run_demo:
    print('Running demo pipeline...')
    cmd = [sys.executable, str(root / 'scripts' / 'run_demo_pipeline.py')]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print('Demo pipeline failed with exit code', proc.returncode)
        sys.exit(proc.returncode)
    print('Demo completed')

# Insert metrics if CSV exists
metrics_csv = root / 'results' / 'metrics_summary.csv'
if metrics_csv.exists():
    ins = root / 'project' / 'tools' / 'insert_metrics_to_tex.py'
    cmd = [sys.executable, str(ins), str(metrics_csv)]
    if args.analysis:
        cmd += ['--analysis', args.analysis]
    print('Inserting metrics using:', ' '.join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print('Metric insertion script failed')
else:
    print('No metrics CSV found at', metrics_csv)

if args.compile:
    tectonic = root / 'project' / 'tools' / 'tectonic' / 'tectonic.exe'
    if not tectonic.exists():
        print('Tectonic engine not found; please run install script or set --compile=false')
        sys.exit(1)
    print('Compiling report.tex...')
    rdir = root / 'project' / 'report'
    cmd = [str(tectonic), '--outdir', str(root / 'results'), 'report.tex']
    proc = subprocess.run(cmd, cwd=str(rdir))
    if proc.returncode != 0:
        print('LaTeX compile failed')
        sys.exit(proc.returncode)
    print('Compiled results/report.pdf')

print('Finalize step complete')
