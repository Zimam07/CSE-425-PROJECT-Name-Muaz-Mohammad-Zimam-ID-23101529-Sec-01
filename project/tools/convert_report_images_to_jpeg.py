from PIL import Image
import os
report_dir = os.path.join(os.getcwd(), 'project', 'report')
pngs = [p for p in os.listdir(report_dir) if p.lower().endswith('.png')]
for p in pngs:
    path = os.path.join(report_dir, p)
    jpg_name = os.path.splitext(p)[0] + '.jpg'
    jpg_path = os.path.join(report_dir, jpg_name)
    im = Image.open(path).convert('RGB')
    im.save(jpg_path, 'JPEG', quality=95)
    print('Saved', jpg_name)
print('Done')
