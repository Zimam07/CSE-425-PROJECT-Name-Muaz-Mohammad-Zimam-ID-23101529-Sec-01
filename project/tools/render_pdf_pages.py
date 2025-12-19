import fitz
from pathlib import Path
p=Path('results/report.pdf')
d=fitz.open(str(p))
for i in range(min(4, d.page_count)):
    out = p.parent / f'report_page{i+1}_render.png'
    pix = d.load_page(i).get_pixmap(dpi=150)
    pix.save(str(out))
    print('Wrote', out)
