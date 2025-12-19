"""Clean generated and intermediate files.

Usage:
  python project/tools/clean_workspace.py [--dry-run] [--yes] [--all-results]

- --dry-run: show what would be deleted
- --yes: actually delete
- --all-results: allow deletion of result directories (use with caution)
"""
import argparse
from pathlib import Path
import shutil
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--yes', action='store_true')
parser.add_argument('--all-results', action='store_true')
args = parser.parse_args()

root = Path.cwd()
deletes = [
    root / 'results' / 'report_base64.txt',
    root / 'project' / 'report' / 'report_with_figures.pdf',
    root / 'results' / 'report_figures_page_render.png',
]
# add extracted bins
deletes += list((root / 'results').glob('extracted_image_*.bin'))
# rendered pages
deletes += list((root / 'results').glob('report_page*_render.png'))
# temporary test files
deletes += [root / 'project' / 'report' / 'test_img.tex', root / 'project' / 'report' / 'test_img.pdf']
# fixed/verification files
deletes += [root / 'project' / 'report' / 'report_fixed.tex', root / 'results' / 'report_fixed.pdf']
# LaTeX intermediates
deletes += list((root / 'results').glob('*.aux'))
deletes += list((root / 'results').glob('*.toc'))

def do_delete(p):
    if p.exists():
        if args.dry_run:
            print('[DRY-RUN] Would delete:', p)
        elif args.yes:
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                print('Deleted', p)
            except Exception as e:
                print('Failed to delete', p, e)
    else:
        print('Not found:', p)

print('Running clean_workspace: dry-run=' , args.dry_run, 'yes=', args.yes)
for p in deletes:
    do_delete(p)

if not args.dry_run and not args.yes:
    print('No deletions performed (use --yes to delete).')
else:
    print('Done')
