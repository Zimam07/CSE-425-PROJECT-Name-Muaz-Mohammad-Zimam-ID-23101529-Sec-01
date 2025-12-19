import sys
import time
import glob
import os
import subprocess

EXPECTED = 6
ROOT = 'results/conv_sweep'

print('Watching for conv sweep analyses...')
while True:
    metrics = glob.glob(os.path.join(ROOT, '**', '*_analysis', 'metrics.csv'), recursive=True)
    n = len(metrics)
    print(f'Found {n}/{EXPECTED} analysis metrics')
    if n >= EXPECTED:
        print('All analyses present, running assemble_report.py')
        subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), 'assemble_report.py')])
        print('Assembled report; exiting')
        break
    time.sleep(10)
