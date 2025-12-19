import fitz  # PyMuPDF
from pathlib import Path
pdf = Path('results/report.pdf')
out = Path('results/report_page1_render.png')
if not pdf.exists():
    print('PDF not found:', pdf); raise SystemExit(1)
doc = fitz.open(str(pdf))
page = doc.load_page(0)
pix = page.get_pixmap(dpi=150)
pix.save(str(out))
print('Wrote', out)
