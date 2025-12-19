from PIL import Image
import os
report_dir = os.path.join(os.getcwd(), 'project', 'report')
pngs = [p for p in os.listdir(report_dir) if p.lower().endswith('.png')]
for p in pngs:
    path = os.path.join(report_dir, p)
    pdf_name = os.path.splitext(p)[0] + '.pdf'
    pdf_path = os.path.join(report_dir, pdf_name)
    im = Image.open(path).convert('RGB')
    im.save(pdf_path, 'PDF', resolution=150)
    print('Saved', pdf_name)
print('All done')
